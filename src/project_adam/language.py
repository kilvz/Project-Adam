import json
import torch
import threading
import re
from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
from .config import BACKEND_CONFIG


_SENTENCE_STOPPER_MIN = 20


class _SentenceStopper(StoppingCriteria):
    def __init__(self, tokenizer, min_tokens=_SENTENCE_STOPPER_MIN):
        self.tokenizer = tokenizer
        self.min_tokens = min_tokens
        self.count = 0

    def __call__(self, input_ids, scores, **kwargs):
        self.count += 1
        if self.count < self.min_tokens:
            return False
        last_token = self.tokenizer.decode(input_ids[0, -1], skip_special_tokens=True)
        return last_token in (".", "!", "?")


class LanguageInterface:
    def __init__(self, model, tokenizer, persona=None, web_search=None,
                 world_model=None, backend=None):
        self.model = model
        self.tokenizer = tokenizer
        self.persona = persona
        self.web_search = web_search
        self.world_model = world_model
        self.backend = backend or BACKEND_CONFIG.get("mode", "local")

    def _api_generate(self, messages, temperature=0.7, token_callback=None,
                      meta_action=None):
        import requests as req
        api_cfg = BACKEND_CONFIG.get("api", {})
        headers = {
            "Authorization": f"Bearer {api_cfg.get('key', '')}",
            "Content-Type": "application/json",
        }
        body = {
            "model": api_cfg.get("model", "gpt-4o-mini"),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": 128,
            "stream": True,
        }
        timeout = api_cfg.get("timeout", 30)
        reply = ""
        try:
            resp = req.post(
                api_cfg.get("endpoint", "https://api.openai.com/v1/chat/completions"),
                headers=headers, json=body, stream=True, timeout=timeout,
            )
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8", errors="replace").strip()
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            if token_callback:
                                token_callback(content)
                            reply += content
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
        return reply.strip()

    def _local_generate(self, messages, temperature=0.7, token_callback=None):
        system = self.build_prompt(None, None, None, None)
        if system:
            messages.insert(0, {"role": "system", "content": system})
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        stopper = _SentenceStopper(self.tokenizer, min_tokens=_SENTENCE_STOPPER_MIN)
        stopping_criteria = StoppingCriteriaList([stopper])

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True,
        )
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=128,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            streamer=streamer,
            stopping_criteria=stopping_criteria,
        )

        def generate():
            with torch.no_grad():
                self.model.generate(**generation_kwargs)

        thread = threading.Thread(target=generate, daemon=True)
        thread.start()

        reply = ""
        for tok in streamer:
            if token_callback:
                token_callback(tok)
            reply += tok

        thread.join(timeout=30)
        return reply.strip()

    def generate(self, messages, meta_action=None, token_callback=None, temperature=0.7):
        used_search = False
        web_context = None
        if meta_action in ("ASK_FOR_HELP", "EXPLORE") and self.web_search is not None:
            user_text = messages[-1]["content"] if messages else ""
            result = self.web_search.search(user_text)
            if result:
                used_search = True
                web_context = result

        msgs = list(messages)
        system = self.build_prompt(None, web_context, web_context, meta_action)
        if system:
            msgs.insert(0, {"role": "system", "content": system})

        if self.backend == "api":
            reply = self._api_generate(
                msgs, temperature=temperature, token_callback=token_callback,
                meta_action=meta_action,
            )
        else:
            reply = self._local_generate(
                msgs, temperature=temperature, token_callback=token_callback,
            )

        if self.world_model is not None and reply:
            speaker_conf = self.compute_utterance_likeness(reply)
            self.world_model.observe_from_text(reply, speaker_conf)
        return reply, used_search, web_context

    def build_prompt(self, user_profile, memory_context, web_context, meta_action):
        parts = []
        if self.persona:
            sys_prompt = self.persona.build_system_prompt(user_profile)
            if sys_prompt:
                parts.append(sys_prompt)
        if memory_context:
            parts.append(f"Memory: {memory_context}")
        if web_context:
            parts.append(f"Web: {web_context}")
        if meta_action == "ASK_FOR_HELP":
            parts.append("Consider asking the user for clarification or more information.")
        elif meta_action == "EXPLORE":
            parts.append("Explore this topic from a fresh angle.")
        elif meta_action == "REPLAY":
            parts.append("Reflect on past interactions to find relevant patterns.")
        elif meta_action == "STOP_AND_THINK":
            parts.append("Pause and consider multiple perspectives before responding.")
        elif meta_action == "SWITCH_STRATEGY":
            parts.append("Try a different approach — consider alternative interpretations.")
        return "\n".join(parts)

    def generate_self_talk(self, reason, user_input):
        if reason == "low_confidence":
            return f"I'm not very confident about this. I should think carefully about '{user_input[:50]}' and draw from my knowledge."
        if reason == "unfamiliar":
            return "This seems unfamiliar territory. Let me explore carefully."
        if reason == "STOP_AND_THINK":
            return f"I should pause and consider multiple perspectives on '{user_input[:50]}'."
        return None

    def apply_behavioral_rules(self, user_input, reply, reward, persona, profile=None):
        if not persona:
            return reply
        lower = user_input.lower()
        lower_reply = reply.lower()
        triggered_rule = None
        for i, (cond, action) in enumerate(persona.behavior_rules):
            cond_lower = cond.lower()
            if "conversation is ending" in cond_lower or "goodbye" in cond_lower:
                if any(w in lower for w in ["bye", "goodbye", "exit", "quit", "see you"]):
                    if "one more thing" not in lower_reply:
                        reply += "\n\nOne more thing — " + persona.get_closing()
                        triggered_rule = i
            elif "praise" in cond_lower or "praised" in cond_lower:
                if any(w in lower_reply for w in ["good", "great", "amazing", "wonderful"]):
                    if "thank" in lower and "suspicion" not in lower_reply:
                        reply = reply.rstrip(".!") + ", though I'm not sure I deserve it."
                        triggered_rule = i
        if triggered_rule is not None and profile is not None:
            i = triggered_rule
            w = profile["rule_weights"].get(i, 1.0)
            if reward > 0:
                w *= 1.05
            else:
                w *= 0.95
            profile["rule_weights"][i] = max(0.1, min(3.0, w))
        return reply

    def compute_utterance_likeness(self, text):
        if not text or len(text) < 5:
            return 1.0
        if self.backend == "api":
            return 0.5
        try:
            enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**enc, labels=enc["input_ids"])
                loss = outputs.loss.item()
            ppl = min(100.0, max(1.0, float(torch.exp(torch.tensor(loss)))))
            return max(0.1, 1.0 - (ppl / 100.0))
        except Exception:
            return 0.5

    def detect_user(self, text, known_names):
        lower = text.lower()
        for name in known_names:
            if name.lower() in lower:
                return name
        name_patterns = [
            (r"my name is (\w+)", 1),
            (r"i(?:'m| am) called (\w+)", 1),
            (r"call me (\w+)", 1),
            (r"(?:^|\s)I(?:'m| am) (\w+)(?:\s|$|\.|,)", 1),
        ]
        for pattern, group in name_patterns:
            m = re.search(pattern, lower)
            if m:
                name = m.group(group).strip().capitalize()
                if name.lower() not in ("i", "a", "an", "the", "adam"):
                    return name
        return None

    def encode(self, text):
        return self.tokenizer(text, return_tensors="pt").to(self.model.device)
