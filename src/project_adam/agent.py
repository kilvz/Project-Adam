import time
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

import logging
from .config import (BASE_MODEL, MODEL_3B, MODEL_1_5B, MODEL_0_5B,
                     _4BIT_CONFIG, DEVICE, get_memory_dir)

logger = logging.getLogger(__name__)
from .persona import Persona
from .profiles import UserProfileManager
from .encoder import SensoryEncoder
from .memory.working import WorkingMemory
from .memory.episodic import EpisodicMemory
from .memory.semantic import SemanticMemory
from .memory.neural import NeuralMemory
from .sfl import SFLModule
from .metacog import MetacognitiveController
from .search import WebSearch
from .consolidator import OfflineConsolidator
from .selector import ActionSelector
from .utils import detect_user, extract_facts, compute_implicit_reward, extract_topics


class CognitiveAgent:
    def __init__(self):
        t0 = time.time()
        model_id = BASE_MODEL
        is_4bit = False
        candidates = [MODEL_3B, MODEL_1_5B, MODEL_0_5B]
        for candidate in candidates:
            try:
                model_id = candidate
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                if candidate == MODEL_0_5B:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_id, torch_dtype=torch.float16, device_map=DEVICE,
                    )
                    is_4bit = False
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        quantization_config=_4BIT_CONFIG,
                        device_map="auto", torch_dtype=torch.float16,
                    )
                    is_4bit = True
                break
            except Exception:
                is_4bit = False
                continue
        self.model.eval()
        model_short = model_id.split("/")[-1].replace("-Instruct", "")
        model_label = f"{model_short}-{'4bit' if is_4bit else 'fp16'}"
        dt = time.time() - t0
        logger.info("Loaded %s: %s params in %ds", model_label, f"{sum(p.numel() for p in self.model.parameters()):,}", dt)

        model_dtype = self.model.dtype
        hidden_dim = self.model.config.hidden_size

        self.sensory_encoder = SensoryEncoder(input_dim=hidden_dim, dtype=model_dtype).to(DEVICE)
        self.working_memory = WorkingMemory(max_turns=8)
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        self.neural_memory = NeuralMemory(input_dim=hidden_dim, dtype=model_dtype).to(DEVICE)
        self.persona = Persona()
        self.metacognitive = MetacognitiveController()
        self.web_search = WebSearch()

        self._lora_config = LoraConfig(
            r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, self._lora_config)
        self.model.eval()
        self._current_adapter = "base"
        self._adapter_dir = get_memory_dir() / "adapters"
        self._adapter_dir.mkdir(exist_ok=True)
        lora_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("LoRA adapters: %s trainable params (%s)", f"{lora_trainable:,}", self._current_adapter)

        self.action_selector = ActionSelector(
            self.tokenizer, self.model,
            self.episodic_memory, self.semantic_memory,
            self.web_search, self.metacognitive, self.persona,
        )
        self.consolidator = OfflineConsolidator(
            self.episodic_memory, self.semantic_memory,
            self.neural_memory, self.tokenizer, self.model,
            embedder=self.episodic_memory.embedder,
        )
        self.user_profiles = UserProfileManager()
        self.consolidator.user_profiles = self.user_profiles
        self.sfl_module = SFLModule(n_features=4).to(DEVICE)
        self.current_profile = None
        self._load_state()

    def _load_state(self):
        nm_path = get_memory_dir() / "neural_memory.pt"
        if nm_path.exists():
            self.neural_memory.load_state_dict(torch.load(nm_path, map_location=DEVICE, weights_only=True))
            logger.info("neural memory loaded")

    def chat(self, user_input, token_callback=None):
        self.working_memory.add("user", user_input)

        known_names = self.user_profiles.list_users()
        detected = detect_user(user_input, known_names)
        if detected:
            is_new = detected not in known_names
            self.current_profile = self.user_profiles.set_current(detected)
            if is_new:
                logger.info("hello %s, storing your profile", detected)
            else:
                logger.info("welcome back %s", detected)
            self._switch_adapter(detected)

        elif self.current_profile is None:
            self.current_profile = self.user_profiles.get_or_create("Stranger")
        self._current_user = self.current_profile.get("name", "")

        facts = extract_facts(user_input)
        for cat, fact in facts:
            self.semantic_memory.add(cat, fact)
            self.episodic_memory.add(fact, reward=0.7)

        is_question = any(user_input.lower().startswith(w)
                          for w in ["what", "why", "how", "when", "where",
                                    "who", "which", "does", "is", "are", "can"]) \
                      and "?" in user_input
        if is_question and not facts:
            sem_retrieval = self.semantic_memory.retrieve(user_input, 3)
            sem_conf = max([s for _, _, s in sem_retrieval], default=0)
            if sem_conf < 0.25:
                result = self.web_search.search(user_input)
                if result:
                    for cat in ["name", "location", "likes", "preference", "dislikes"]:
                        if cat in user_input.lower():
                            self.semantic_memory.add(cat, result[:200])
                            break
                    else:
                        self.semantic_memory.add("web_knowledge", result[:300])
                    self.episodic_memory.add(f"[auto-search] {result[:200]}", reward=0.4)

        self.episodic_memory.add(user_input)

        if facts:
            self._inline_learn(facts)
        elif len(user_input) > 10:
            self._inline_learn([("input", user_input)])

        reward = compute_implicit_reward(user_input, self.current_profile, self.episodic_memory.embedder)

        features = self._build_sfl_features(user_input, reward)
        self.sfl_module.update(features, reward)
        q_value = self.sfl_module(torch.as_tensor(features, dtype=torch.float32, device=DEVICE)).item()

        topics = extract_topics(user_input, self.episodic_memory.embedder)
        self._update_user_profile(user_input, reward, topics)
        self._mint_custom_rules()

        reply, used_search, web_context, meta_action = self.action_selector.select(
            user_input, self.working_memory.get_context(),
            user_profile=self.current_profile,
            sfl_q=q_value,
            token_callback=token_callback,
        )

        if meta_action == "replay":
            self._inline_learn([("replay", user_input)])
            self.consolidator.merge_episodes()

        reply = self._apply_behavioral_rules(user_input, reply, reward)
        self.working_memory.add("assistant", reply)
        self._update_rule_weights(reward)

        if self.current_profile is not None:
            self.current_profile["last_q"] = round(q_value, 3)
            self.current_profile.setdefault("q_history", []).append(round(q_value, 3))
            self.current_profile.setdefault("reward_history", []).append(round(reward, 3))
            if len(self.current_profile["q_history"]) > 100:
                self.current_profile["q_history"] = self.current_profile["q_history"][-100:]
                self.current_profile["reward_history"] = self.current_profile["reward_history"][-100:]

        if self.current_profile is not None:
            count = self.current_profile.get("interaction_count", 0)
            if count > 0 and count % 10 == 0:
                self._lora_train_step()

        return reply

    def _build_sfl_features(self, user_input, reward):
        profile = self.current_profile or {}
        sentiment = profile.get("avg_sentiment", 0.0)
        engagement = min(1.0, len(user_input) / 100.0)
        interaction_norm = min(1.0, profile.get("interaction_count", 0) / 100.0)
        topic_novelty = 0.0
        topics = extract_topics(user_input, self.episodic_memory.embedder if hasattr(self, 'episodic_memory') else None)
        known = profile.get("topics", {})
        if topics:
            novel = sum(1 for t in topics if t not in known)
            topic_novelty = min(1.0, novel / max(len(topics), 1))
        return [sentiment, engagement, interaction_norm, topic_novelty]

    def _inline_learn(self, facts):
        try:
            fact_text = " ".join([f for _, f in facts])
            if len(fact_text) < 3:
                return
            inputs = self.tokenizer(fact_text, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                emb = self.model.get_input_embeddings()(inputs["input_ids"]).to(self.neural_memory.dtype)
            self.neural_memory.learn(emb)
            flat = emb.view(-1, emb.shape[-1])
            self.sensory_encoder.vae_loss(flat[:min(64, flat.shape[0])])
        except Exception as e:
            logger.warning("Inline learn failed: %s", e)

    def _update_user_profile(self, user_input, reward, topics):
        if self.current_profile is None:
            return
        name = self.current_profile.get("name", "Stranger")
        self.user_profiles.update_after_turn(name, user_input, "", reward, topics)

    def _update_rule_weights(self, reward):
        if self.current_profile is None or not self.persona.behavior_rules:
            return
        profile = self.current_profile
        for i in range(len(self.persona.behavior_rules)):
            w = profile["rule_weights"].get(i, 1.0)
            if reward > 0.3:
                w *= 1.05
            elif reward < -0.3:
                w *= 0.95
            profile["rule_weights"][i] = max(0.1, min(3.0, w))

    def _mint_custom_rules(self):
        if self.current_profile is None:
            return
        profile = self.current_profile
        topics = profile.get("topics", {})
        existing_rules = {(c.lower(), a.lower()) for c, a in profile.get("custom_rules", [])}
        for topic, count in topics.items():
            if count >= 5:
                cond = f"{topic} comes up"
                action = f"engage with curiosity and wonder about {topic}, relating it to your own experience as an AI"
                if (cond.lower(), action.lower()) not in existing_rules:
                    if not any(topic in c for c, _ in profile.get("custom_rules", [])):
                        profile.setdefault("custom_rules", []).append((cond, action))
                        logger.info("minted new rule: If %s, then %s...", cond, action[:50])

    def _apply_behavioral_rules(self, user_input, reply, reward):
        lower = user_input.lower()
        lower_reply = reply.lower()
        if not self.persona:
            return reply
        triggered_rule = None
        for i, (cond, action) in enumerate(self.persona.behavior_rules):
            cond_lower = cond.lower()
            if "conversation is ending" in cond_lower or "goodbye" in cond_lower:
                if any(w in lower for w in ["bye", "goodbye", "exit", "quit", "see you"]):
                    if "one more thing" not in lower_reply:
                        reply += "\n\nOne more thing — " + self.persona.get_closing()
                        triggered_rule = i
            elif "praise" in cond_lower or "praised" in cond_lower:
                if any(w in lower_reply for w in ["good", "great", "amazing", "wonderful"]):
                    if "thank" in lower and "suspicion" not in lower_reply:
                        reply = reply.rstrip(".!") + ", though I'm not sure I deserve it."
                        triggered_rule = i

        if triggered_rule is not None and self.current_profile is not None:
            i = triggered_rule
            w = self.current_profile["rule_weights"].get(i, 1.0)
            if reward > 0:
                w *= 1.05
            else:
                w *= 0.95
            self.current_profile["rule_weights"][i] = max(0.1, min(3.0, w))
        return reply

    def _lora_format_examples(self, episodes, max_examples=8):
        examples = []
        for ep in episodes[-max_examples:]:
            text = ep.get("text", "")
            if not text or len(text) < 5:
                continue
            text = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\nI see.<|im_end|>"
            examples.append(text)
        return examples

    def _lora_train_step(self):
        episodes = self.episodic_memory.episodes
        if len(episodes) < 5:
            return
        candidates = [e for e in episodes[-30:] if e.get("reward", 0) > -0.3]
        candidates.sort(key=lambda e: e.get("reward", 0), reverse=True)
        if not candidates:
            return
        texts = self._lora_format_examples(candidates, max_examples=5)
        if not texts:
            return
        self.model.train()
        optim = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=1e-4
        )
        total_loss = 0.0
        for i, text in enumerate(texts):
            enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(DEVICE)
            labels = enc["input_ids"].clone()
            assist_tok = self.tokenizer("<|im_start|>assistant", add_special_tokens=False)["input_ids"][0]
            assist_idx = (labels[0] == assist_tok).nonzero()
            if assist_idx.numel() > 0:
                labels[0, :assist_idx[0, 0]] = -100
            out = self.model(**enc, labels=labels, use_cache=False)
            loss = out.loss
            if not torch.isnan(loss):
                weight = max(0.1, min(2.0, candidates[i].get("reward", 0) + 1.0))
                (loss * weight).backward()
                optim.step()
                optim.zero_grad()
                total_loss += loss.item()
        self.model.eval()
        if total_loss > 0:
            self._save_adapter()

    def _save_adapter(self, name=None):
        name = name or self._current_adapter
        path = self._adapter_dir / name
        path.mkdir(parents=True, exist_ok=True)
        try:
            state = {k: v for k, v in self.model.state_dict().items() if "lora" in k}
            torch.save(state, path / "adapter_model.safetensors")
            with open(path / "adapter_config.json", "w") as f:
                json.dump({"adapter_name": name}, f)
        except Exception as e:
            logger.warning("Failed to save adapter %s: %s", name, e)

    def _switch_adapter(self, user):
        if user == self._current_adapter:
            return
        if self._current_adapter != "base":
            self._save_adapter(self._current_adapter)
        user_path = self._adapter_dir / user
        adapter_exists = (user_path / "adapter_model.safetensors").exists()
        if adapter_exists:
            try:
                state = torch.load(user_path / "adapter_model.safetensors", map_location=DEVICE, weights_only=True)
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if name in state:
                            param.copy_(state[name])
            except Exception as e:
                logger.warning("Failed to load adapter for %s: %s", user, e)
        self._current_adapter = user
