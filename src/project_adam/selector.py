import torch
import threading
from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList


class _SentenceStopper(StoppingCriteria):
    def __init__(self, tokenizer, min_tokens=20):
        self.tokenizer = tokenizer
        self.min_tokens = min_tokens
        self.count = 0

    def __call__(self, input_ids, scores, **kwargs):
        self.count += 1
        if self.count < self.min_tokens:
            return False
        last_token = self.tokenizer.decode(input_ids[0, -1], skip_special_tokens=True)
        if last_token in (".", "!", "?"):
            return True
        return False


class ActionSelector:
    def __init__(self, tokenizer, model, episodic_memory, semantic_memory,
                 web_search, metacognitive, persona=None):
        self.tokenizer = tokenizer
        self.model = model
        self.episodic = episodic_memory
        self.semantic = semantic_memory
        self.web_search = web_search
        self.metacognitive = metacognitive
        self.persona = persona
        self._last_meta_action = "proceed"
        self._streamer = None

    def select(self, user_input, context, user_profile=None,
               sfl_q=None, token_callback=None):
        confidence = 0.5
        meta_action = self.metacognitive.act(confidence, None, sfl_q)
        self._last_meta_action = meta_action

        used_search = False
        web_context = None

        if meta_action == "search" and self.web_search:
            result = self.web_search.search(user_input)
            if result:
                used_search = True
                web_context = result

        recall = self.episodic.search(user_input, k=3)
        memory_context = ""
        if recall:
            memory_context = "Related: " + " | ".join(text for text, _, _ in recall[:2])

        system = self._build_prompt(user_profile, memory_context, web_context, meta_action)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_input},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        stopper = _SentenceStopper(self.tokenizer, min_tokens=20)
        stopping_criteria = StoppingCriteriaList([stopper])

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True,
        )

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
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

        return reply.strip(), used_search, web_context, meta_action

    def _build_prompt(self, user_profile, memory_context, web_context, meta_action):
        parts = []
        if self.persona:
            sys_prompt = self.persona.build_system_prompt(user_profile)
            if sys_prompt:
                parts.append(sys_prompt)
        if memory_context:
            parts.append(f"Memory: {memory_context}")
        if web_context:
            parts.append(f"Web: {web_context}")
        if meta_action == "clarify":
            parts.append("The user might need clarification. Ask a thoughtful follow-up.")
        elif meta_action == "explore":
            parts.append("Explore this topic from a fresh angle.")
        elif meta_action == "replay":
            parts.append("Reflect on past interactions.")
        return "\n".join(parts)
