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
from .memory.procedural import ProceduralMemory
from .memory.spatial import SpatialMemory
from .language import LanguageInterface
from .sfl import SFLModule
from .metacog import MetacognitiveController
from .search import WebSearch
from .consolidator import OfflineConsolidator
from .selector import ActionSelector
from .rl_core import TDCore
from .world_model import WorldModel
from .rl_core import TDCore as _TDCore
from .config import BACKEND_CONFIG


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

        self.world_model = WorldModel()
        self.web_search = WebSearch()
        self.backend = BACKEND_CONFIG.get("mode", "local")
        self.language = LanguageInterface(self.model, self.tokenizer, persona=None, web_search=self.web_search, world_model=self.world_model, backend=self.backend)
        enc_input_dim = 384
        self.sensory_encoder = SensoryEncoder(input_dim=enc_input_dim, latent_dim=64, dtype=model_dtype).to(DEVICE)
        self.working_memory = WorkingMemory(max_turns=64)
        self.episodic_memory = EpisodicMemory()
        self.semantic_memory = SemanticMemory()
        self.procedural_memory = ProceduralMemory()
        self.spatial_memory = SpatialMemory()
        self.persona = Persona()
        self.language.persona = self.persona
        self.metacognitive = MetacognitiveController()

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

        if self.episodic_memory.embedder:
            self.working_memory.set_embedder(self.episodic_memory.embedder)
        self.working_memory.set_episodic_memory(self.episodic_memory)

        self.action_selector = ActionSelector(
            self.language,
            self.episodic_memory, self.semantic_memory,
            self.metacognitive, self.world_model, self.persona,
        )
        self.sfl_module = SFLModule(n_features=7).to(DEVICE)
        self.td_core = TDCore(n_features=7)
        self.consolidator = OfflineConsolidator(
            self.episodic_memory, self.semantic_memory,
            world_model=self.world_model,
            embedder=self.episodic_memory.embedder,
            td_core=self.td_core,
            procedural_memory=self.procedural_memory,
        )
        self.user_profiles = UserProfileManager()
        self.consolidator.user_profiles = self.user_profiles
        self.current_profile = None

    def _build_td_features(self, user_input, reward):
        profile = self.current_profile or {}
        sentiment = profile.get("avg_sentiment", 0.0)
        engagement = min(1.0, len(user_input) / 100.0)
        interaction_norm = min(1.0, profile.get("interaction_count", 0) / 100.0)
        topic_count = min(1.0, len(profile.get("topics", {})) / 20.0)
        sfl_q = self.sfl_module.q_history[-1] if self.sfl_module.q_history else 0.0
        enc_sparsity = 0.0
        if hasattr(self, 'sensory_encoder') and self.episodic_memory.embedder is not None:
            emb = self.episodic_memory.encode(user_input)
            if emb.shape[-1] == self.sensory_encoder.encoder[0].in_features:
                with torch.no_grad():
                    enc_dtype = next(self.sensory_encoder.parameters()).dtype
                    x_t = torch.as_tensor(emb, dtype=enc_dtype, device=DEVICE).unsqueeze(0)
                    z, _ = self.sensory_encoder.forward(x_t)
                    enc_sparsity = float((torch.abs(z) < 0.01).float().mean())
        return [sentiment, engagement, interaction_norm, topic_count, reward, sfl_q, enc_sparsity]

    def chat(self, user_input, token_callback=None):
        self.working_memory.add("user", user_input)

        known_names = self.user_profiles.list_users()
        detected = self.language.detect_user(user_input, known_names)
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

        facts = self.semantic_memory.extract_facts(user_input)
        for cat, fact in facts:
            self.semantic_memory.add(cat, fact)
            self.episodic_memory.add(fact, reward=0.7)

        speaker_conf = self.language.compute_utterance_likeness(user_input)
        self.world_model.observe_from_text(user_input, speaker_conf)

        spatial_rels = self.spatial_memory.extract_from_text(user_input)
        for rel in spatial_rels:
            self.episodic_memory.add(f"[spatial] {rel} mentioned", reward=0.3)

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

        self.episodic_memory.add(user_input, context=str(self.working_memory.get_context(4)), backend=self.backend)

        reward = _TDCore.compute_reward(user_input, self.current_profile, self.episodic_memory.embedder)

        td_features = self._build_td_features(user_input, reward)
        rpe = self.td_core.update(reward, td_features)

        if self.episodic_memory.embedder is not None:
            emb = self.episodic_memory.encode(user_input)
            if emb.shape[-1] == self.sensory_encoder.encoder[0].in_features:
                enc_dtype = next(self.sensory_encoder.parameters()).dtype
                x_t = torch.as_tensor(emb, dtype=enc_dtype, device=DEVICE).unsqueeze(0)
                _, z = self.sensory_encoder.train_step(x_t, rpe)
                if self.episodic_memory.episodes:
                    self.episodic_memory.episodes[-1]["latent_z"] = z[0].tolist()

        features = self._build_sfl_features(user_input)
        f_t = torch.as_tensor(features, dtype=torch.float32, device=DEVICE)
        self.sfl_module.update(f_t, rpe)
        q_value = self.sfl_module(f_t).item()

        topics = self.semantic_memory.extract_topics(user_input, self.episodic_memory.embedder)
        self._update_user_profile(user_input, reward, topics)
        self._mint_custom_rules()

        self.procedural_memory.record(user_input, self.current_profile.get("name", "") if self.current_profile else "", reward)
        self.procedural_memory.update_from_rpe(rpe)

        self.metacognitive.record_confidence(abs(reward))
        confidence, _ = self.metacognitive.estimate_confidence(None)
        talk_reason = self.metacognitive.should_self_talk(confidence, q_value)
        self_talk = self.language.generate_self_talk(talk_reason, user_input)
        if self_talk:
            self.working_memory.add("assistant", f"[self-talk] {self_talk}")

        procedural_hint = self.procedural_memory.retrieve(user_input)
        if procedural_hint and self.current_profile:
            self.current_profile.setdefault("procedural_hints", []).append(procedural_hint[:100])

        temperature = self.sfl_module.compute_temperature()
        reply, used_search, web_context, meta_action = self.action_selector.select(
            user_input, self.working_memory.get_context(),
            user_profile=self.current_profile,
            sfl_q=q_value, temperature=temperature,
            token_callback=token_callback,
        )

        if meta_action == "REPLAY":
            self.consolidator.merge_episodes(rpe=rpe)
        elif meta_action == "proceed" and not used_search:
            self.action_selector.record_fast_outcome(rpe)

        reply = self.language.apply_behavioral_rules(
            user_input, reply, reward, self.persona, self.current_profile
        )
        self.episodic_memory.update_last_action(reply, backend=self.backend)
        self.working_memory.add("assistant", reply)
        self._update_rule_weights(reward)

        self.metacognitive.record_outcome(used_search, reward=reward)
        self.metacognitive.learn(reward)

        if self.current_profile is not None:
            self.current_profile["last_q"] = round(q_value, 3)
            self.current_profile.setdefault("q_history", []).append(round(q_value, 3))
            self.current_profile.setdefault("reward_history", []).append(round(reward, 3))
            self.current_profile.setdefault("rpe_history", []).append(round(rpe, 3))
            if len(self.current_profile["q_history"]) > 100:
                self.current_profile["q_history"] = self.current_profile["q_history"][-100:]
                self.current_profile["reward_history"] = self.current_profile["reward_history"][-100:]
                self.current_profile["rpe_history"] = self.current_profile["rpe_history"][-100:]

        if self.current_profile is not None:
            count = self.current_profile.get("interaction_count", 0)
            if count > 0 and count % 10 == 0:
                self._lora_train_step(rpe)

        return reply

    def _build_sfl_features(self, user_input):
        profile = self.current_profile or {}
        sentiment = profile.get("avg_sentiment", 0.0)
        engagement = min(1.0, len(user_input) / 100.0)
        interaction_norm = min(1.0, profile.get("interaction_count", 0) / 100.0)
        topic_novelty = 0.0
        topics = self.semantic_memory.extract_topics(user_input, self.episodic_memory.embedder if hasattr(self, 'episodic_memory') else None)
        known = profile.get("topics", {})
        if topics:
            novel = sum(1 for t in topics if t not in known)
            topic_novelty = min(1.0, novel / max(len(topics), 1))
        rh = profile.get("reward_history", [])
        majority_opinion = sum(rh[-10:]) / max(len(rh[-10:]), 1) if rh else 0.0
        expert_endorsement = engagement * (0.5 + 0.5 * abs(sentiment))
        popularity = min(1.0, profile.get("interaction_count", 0) / 50.0)
        return [sentiment, engagement, interaction_norm, topic_novelty,
                majority_opinion, expert_endorsement, popularity]

    def _update_user_profile(self, user_input, reward, topics):
        if self.current_profile is None:
            return
        name = self.current_profile.get("name", "Stranger")
        self.user_profiles.update_after_turn(name, user_input, reward, topics)

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

    def _lora_format_examples(self, episodes, max_examples=8):
        examples = []
        for ep in episodes[-max_examples:]:
            text = ep.get("text", "")
            if not text or len(text) < 5:
                continue
            text = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\nI see.<|im_end|>"
            examples.append(text)
        return examples

    def _lora_train_step(self, rpe=1.0):
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
        rpe_scale = max(0.5, min(2.0, abs(rpe)))
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
                weight = max(0.1, min(2.0, candidates[i].get("reward", 0) + 1.0)) * rpe_scale
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
