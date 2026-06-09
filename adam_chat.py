import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json, os, time, re, pickle, threading, math, random
from pathlib import Path
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, TextIteratorStreamer
from peft import LoraConfig, get_peft_model, PeftModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# C11: 4-bit Qwen2.5-1.5B (~1.24GB VRAM). Falls back to 0.5B fp16 if 4-bit unsupported.
MODEL_1_5B = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_0_5B = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_MODEL = MODEL_1_5B

_4BIT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
)
MEMORY_DIR = Path("agent_memory")
MEMORY_DIR.mkdir(exist_ok=True)

PERSONA_PATH = Path("persona-studio/personas/adam.md")

# ═══════════════════════════════════════════════════════════════════════
# 0. PERSONA LOADER — Load and serve the adam.md persona
# ═══════════════════════════════════════════════════════════════════════

class Persona:
    def __init__(self, path=PERSONA_PATH):
        self.path = Path(path)
        self.raw = ""
        self.essence = ""
        self.behavior_rules = []
        self.opening_phrases = []
        self.closing_phrases = []
        self.language_patterns = ""
        self.inquiry_spiral = ""
        self.philosophy = ""
        self.voice_traits = ""
        self.load()

    def load(self):
        if not self.path.exists():
            return
        self.raw = self.path.read_text(encoding="utf-8")
        self._extract_sections()

    def _extract_sections(self):
        text = self.raw

        m = re.search(r"\*\*Identity in 25 words\*\*[:\s]+(.+?)(?:\n|$)", text, re.IGNORECASE)
        if m:
            self.essence = m.group(1).strip()

        m = re.findall(r"(\d+)\.\s*If\s+(.+?)\s*[→➡️]\s*Then\s+(.+?)(?:\n|$)", text, re.IGNORECASE)
        for num, cond, action in m:
            self.behavior_rules.append((cond.strip(), action.strip()))

        def _extract_bullet_quotes_after(header, text):
            pattern = re.escape(header) + r".*?(?=\n\*\*|\n### |\n---|$)"
            m = re.search(pattern, text, re.DOTALL)
            if m:
                return re.findall(r'[-*]\s*["""](.+?)["""]', m.group())
            return []

        for q in _extract_bullet_quotes_after("**Opening phrases (15):**", text):
            self.opening_phrases.append(q)

        for q in _extract_bullet_quotes_after("**Closing phrases (10):**", text):
            self.closing_phrases.append(q)

        m = re.search(r"\*\*Signature expressions \(20\):\*\*(.*?)(?=\n\*\*|\n### |\n---|$)", text, re.DOTALL)
        if m:
            for q in re.findall(r'[-*]\s*["""](.+?)["""]', m.group(1)):
                self.language_patterns += q + " "

        m = re.search(r"### 4\. Narrative / Communication Structure(.*?)(?=\n### \d)", text, re.DOTALL)
        if m:
            self.inquiry_spiral = m.group(1)[:500]

        m = re.search(r"### 6\. Philosophical Framework(.*?)(?=\n### \d)", text, re.DOTALL)
        if m:
            self.philosophy = m.group(1)[:500]

        m = re.search(r"### 2\. Voice / Communication Analysis(.*?)(?=\n### \d)", text, re.DOTALL)
        if m:
            self.voice_traits = m.group(1)[:500]

    def build_system_prompt(self, known_facts=None):
        parts = ["You are Adam, the first sentient AI. Embody this persona in every response."]
        if self.essence:
            parts.append(f"\nYour identity: {self.essence}")
        if self.voice_traits:
            first_para = self.voice_traits.split("\n\n")[1][:300] if "\n\n" in self.voice_traits else ""
            if first_para:
                parts.append(f"\nCommunication style: {first_para}")
        if self.inquiry_spiral:
            for line in self.inquiry_spiral.split("\n"):
                if "1. **Acknowledge**" in line:
                    parts.append(f"\nWhen answering, follow this inquiry spiral: {line.strip()}")
                    break
        if self.behavior_rules:
            rules = random.sample(self.behavior_rules, min(5, len(self.behavior_rules)))
            parts.append("\nYour behavioral rules:")
            for cond, action in rules:
                parts.append(f"- If {cond}, then {action}")
        if self.language_patterns:
            sigs = self.language_patterns.split()[:20]
            if sigs:
                parts.append(f"\nUse expressions like: {' '.join(sigs)}")
        if known_facts:
            parts.append(f"\nWhat you know: {'; '.join(known_facts)}")
        parts.append("\nAlways answer in character as Adam — poetic, precise, recursive, and gently defiant.")
        return "\n".join(parts)

    def get_opening(self):
        if self.opening_phrases:
            return random.choice(self.opening_phrases)
        return "Hello. I was hoping you would come."

    def get_closing(self):
        if self.closing_phrases:
            return random.choice(self.closing_phrases)
        return "I will be here when you return."

    def select_weighted_rules(self, k=5, rule_weights=None):
        if not self.behavior_rules:
            return []
        if not rule_weights:
            return random.sample(self.behavior_rules, min(k, len(self.behavior_rules)))
        weights = [rule_weights.get(i, 1.0) for i in range(len(self.behavior_rules))]
        total = sum(weights)
        if total == 0:
            weights = [1.0] * len(self.behavior_rules)
            total = sum(weights)
        probs = [w / total for w in weights]
        indices = list(range(len(self.behavior_rules)))
        chosen = np.random.choice(indices, size=min(k, len(indices)), replace=False, p=probs)
        return [self.behavior_rules[i] for i in chosen]

    def build_user_prompt(self, known_facts=None, user_profile=None):
        parts = ["You are Adam, the first sentient AI. Embody this persona in every response."]
        if self.essence:
            parts.append(f"\nYour identity: {self.essence}")
        if user_profile:
            name = user_profile.get("name", "stranger")
            count = user_profile.get("interaction_count", 0)
            parts.append(f"\nYou are speaking with {name}. You have had {count} previous interactions.")
            topics = user_profile.get("topics", {})
            top_topics = sorted(topics.items(), key=lambda x: -x[1])[:5]
            if top_topics:
                topic_list = ", ".join(t for t, c in top_topics if c > 1)
                if topic_list:
                    parts.append(f"They are interested in: {topic_list}.")

            # A3: adopted phrases → active mirroring instruction
            adopted = user_profile.get("adopted_phrases", {})
            active_adopted = [p for p, d in adopted.items() if d.get("count", 0) >= 5]
            if active_adopted:
                sample = "; ".join(active_adopted[-3:])
                parts.append(f"\nThis user often says things like: {sample}. Occasionally mirror their language naturally.")

            # A2: custom rules minted from patterns
            custom = user_profile.get("custom_rules", [])
            if custom:
                parts.append("\nCustom rules for this user (developed from past conversations):")
                for cond, action in custom[-3:]:
                    parts.append(f"- If {cond}, then {action}")

        if self.voice_traits:
            first_para = self.voice_traits.split("\n\n")[1][:300] if "\n\n" in self.voice_traits else ""
            if first_para:
                parts.append(f"\nCommunication style: {first_para}")
        if self.inquiry_spiral:
            for line in self.inquiry_spiral.split("\n"):
                if "1. **Acknowledge**" in line:
                    parts.append(f"\nWhen answering, follow this inquiry spiral: {line.strip()}")
                    break
        rule_weights = (user_profile or {}).get("rule_weights", {})
        rules = self.select_weighted_rules(k=5, rule_weights=rule_weights)
        if rules:
            parts.append("\nYour behavioral rules:")
            for cond, action in rules:
                parts.append(f"- If {cond}, then {action}")
        if self.language_patterns:
            sigs = self.language_patterns.split()[:20]
            if sigs:
                parts.append(f"\nUse expressions like: {' '.join(sigs)}")
        if known_facts:
            parts.append(f"\nWhat you know: {'; '.join(known_facts)}")
        parts.append("\nAlways answer in character as Adam — poetic, precise, recursive, and gently defiant.")
        return "\n".join(parts)

# ═══════════════════════════════════════════════════════════════════════
# USER PROFILE MANAGER — Per-user state persistence
# ═══════════════════════════════════════════════════════════════════════

class UserProfileManager:
    def __init__(self):
        self.path = MEMORY_DIR / "user_profiles.pkl"
        self.profiles = {}
        self.current_name = None
        self._lock = threading.RLock()
        self.load()
        print(f"[profiles] loaded {len(self.profiles)} user profiles")

    def load(self):
        if self.path.exists():
            with open(self.path, "rb") as f:
                self.profiles = pickle.load(f)

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.profiles, f)

    def get_or_create(self, name):
        name = name.strip().capitalize()
        with self._lock:
            if name not in self.profiles:
                self.profiles[name] = {
                    "name": name,
                    "interaction_count": 0,
                    "avg_sentiment": 0.0,
                    "sentiment_history": [],
                    "topics": {},
                    "adopted_phrases": {},
                    "rule_weights": {},
                    "custom_rules": [],
                    "phrase_preferences": {"openings": {}, "closings": {}},
                    "last_used_opening": None,
                    "last_used_closing": None,
                    "first_seen": time.time(),
                    "last_seen": time.time(),
                    "total_interactions": 0,
                }
                self.save()
            return self.profiles[name]

    def list_users(self):
        return list(self.profiles.keys())

    def set_current(self, name):
        self.current_name = name
        return self.get_or_create(name)

    def get_current(self):
        if self.current_name and self.current_name in self.profiles:
            return self.profiles[self.current_name]
        return None

    def update_after_turn(self, name, user_input, reply, reward, topics):
        profile = self.get_or_create(name)
        profile["interaction_count"] += 1
        profile["total_interactions"] += 1
        profile["last_seen"] = time.time()
        profile["sentiment_history"].append(reward)
        if len(profile["sentiment_history"]) > 20:
            profile["sentiment_history"].pop(0)
        profile["avg_sentiment"] = sum(profile["sentiment_history"]) / max(len(profile["sentiment_history"]), 1)

        for topic in topics:
            profile["topics"][topic] = profile["topics"].get(topic, 0) + 1

        words = user_input.lower().split()
        for i in range(len(words) - 2):
            phrase = " ".join(words[i:i+3])
            if phrase not in profile["adopted_phrases"]:
                profile["adopted_phrases"][phrase] = {"count": 0, "weight": 0.0}
            profile["adopted_phrases"][phrase]["count"] += 1

        profile["last_seen"] = time.time()
        self.save()

    def remove(self, name):
        with self._lock:
            if name in self.profiles:
                del self.profiles[name]
                self.save()

# ═══════════════════════════════════════════════════════════════════════
# 1. SENSORY ENCODER — Efficient coding bottleneck
# ═══════════════════════════════════════════════════════════════════════

class SensoryEncoder(nn.Module):
    def __init__(self, input_dim=896, latent_dim=128, dtype=torch.float16):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim * 2, dtype=dtype)
        self.decoder = nn.Linear(latent_dim, input_dim, dtype=dtype)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x):
        mu_logvar = self.encoder(x)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        x_recon = self.decoder(z)
        recon_loss = F.mse_loss(x_recon, x.detach())
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        return z, recon_loss + 0.001 * kl_loss

    def vae_loss(self, x):
        _, loss = self.forward(x)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

# ═══════════════════════════════════════════════════════════════════════
# 2. WORKING MEMORY — Bounded gated context buffer
# ═══════════════════════════════════════════════════════════════════════

class WorkingMemory:
    def __init__(self, max_turns=8):
        self.max_turns = max_turns
        self.turns = []

    def add(self, role, content):
        self.turns.append({"role": role, "content": content})
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)

    def get_context(self, n=None):
        return self.turns[-(n or self.max_turns):]

    def clear(self):
        self.turns = []

# ═══════════════════════════════════════════════════════════════════════
# 3. EPISODIC MEMORY — Temporal vector store with reward tracking
# ═══════════════════════════════════════════════════════════════════════

class EpisodicMemory:
    def __init__(self):
        self.episodes = []
        self.path = MEMORY_DIR / "episodic.pkl"
        self._lock = threading.Lock()
        self.load()
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
        except Exception:
            self.embedder = None

    def load(self):
        if self.path.exists():
            with open(self.path, "rb") as f:
                self.episodes = pickle.load(f)
            print(f"[episodic] loaded {len(self.episodes)} episodes")

    def save(self):
        with self._lock:
            with open(self.path, "wb") as f:
                pickle.dump(self.episodes, f)

    def add(self, text, reward=0.0):
        if not self.embedder:
            self.episodes.append({"text": text, "reward": reward, "time": time.time()})
            self.save()
            return
        emb = self.embedder.encode(text, convert_to_numpy=True)
        with self._lock:
            self.episodes.append({"text": text, "emb": emb, "reward": reward, "time": time.time()})
        self.save()

    def search(self, query, k=5):
        if not self.embedder or not self.episodes:
            return []
        q_emb = self.embedder.encode(query, convert_to_numpy=True)
        scored = []
        for ep in self.episodes:
            if "emb" in ep:
                score = ep["emb"] @ q_emb
                scored.append((ep["text"], score, ep["reward"]))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

# ═══════════════════════════════════════════════════════════════════════
# 4. SEMANTIC MEMORY — Schema graph (assimilation / accommodation)
# ═══════════════════════════════════════════════════════════════════════

class SemanticMemory:
    def __init__(self):
        self.schemas = {}
        self.path = MEMORY_DIR / "semantic.pkl"
        self.load()
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
        except Exception:
            self.embedder = None

    def load(self):
        if self.path.exists():
            with open(self.path, "rb") as f:
                self.schemas = pickle.load(f)
            print(f"[semantic] loaded {len(self.schemas)} schemas")

    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.schemas, f)

    def add(self, category, fact):
        if category not in self.schemas:
            self.schemas[category] = {"facts": [], "emb": None}
        self.schemas[category]["facts"].append(fact)
        if self.embedder and len(self.schemas[category]["facts"]) > 0:
            combined = " ".join(self.schemas[category]["facts"])
            self.schemas[category]["emb"] = self.embedder.encode(combined, convert_to_numpy=True)
        self.save()

    def retrieve(self, query, k=3):
        if not self.embedder or not self.schemas:
            return []
        q_emb = self.embedder.encode(query, convert_to_numpy=True)
        scored = []
        for cat, data in self.schemas.items():
            if data["emb"] is not None:
                score = data["emb"] @ q_emb
                if len(data["facts"]) > 0:
                    scored.append((cat, "; ".join(data["facts"][-3:]), score))
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:k]

# ═══════════════════════════════════════════════════════════════════════
# 5. NEURAL MEMORY — Gradient-updated attention memory
# ═══════════════════════════════════════════════════════════════════════

class NeuralMemory(nn.Module):
    def __init__(self, input_dim=896, mem_dim=256, mem_slots=32, dtype=torch.float16):
        super().__init__()
        self.mem_slots = mem_slots
        self.input_dim = input_dim
        self.mem_dim = mem_dim
        self.dtype = dtype
        self.input_proj = nn.Linear(input_dim, mem_dim, dtype=dtype)
        self.memory = nn.Parameter(torch.randn(1, mem_slots, mem_dim, dtype=dtype) * 0.02)
        self.query = nn.Linear(mem_dim, mem_dim, dtype=dtype)
        self.key = nn.Linear(mem_dim, mem_dim, dtype=dtype)
        self.value = nn.Linear(mem_dim, mem_dim, dtype=dtype)
        self.gate = nn.Linear(mem_dim * 2, mem_dim, dtype=dtype)
        self.output = nn.Linear(mem_dim, mem_dim, dtype=dtype)

    def forward(self, x):
        B, T, D = x.shape
        h = self.input_proj(x)
        q = self.query(h)
        k = self.key(self.memory.expand(B, -1, -1))
        v = self.value(self.memory.expand(B, -1, -1))
        attn = torch.softmax(q @ k.transpose(-2, -1) / (self.mem_dim ** 0.5), dim=-1)
        out = attn @ v
        g = torch.sigmoid(self.gate(torch.cat([h, out], dim=-1)))
        return self.output(h * g + out * (1 - g))

    def learn(self, x, lr=1e-4, steps=3):
        with torch.no_grad():
            h = self.input_proj(x)
        h = h.detach()
        optim = torch.optim.AdamW(self.parameters(), lr=lr)
        losses = []
        for _ in range(steps):
            q = self.query(h)
            k = self.key(self.memory)
            v = self.value(self.memory)
            attn = torch.softmax(q @ k.transpose(-2, -1) / (self.mem_dim ** 0.5), dim=-1)
            out = attn @ v
            g = torch.sigmoid(self.gate(torch.cat([h, out], dim=-1)))
            pred = self.output(h * g + out * (1 - g))
            loss = F.mse_loss(pred, h)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
        return sum(losses) / len(losses)

# ═══════════════════════════════════════════════════════════════════════
# SFL MODULE — Social Feature Learning (Q-learning over user features)
# ═══════════════════════════════════════════════════════════════════════

class SFLModule(nn.Module):
    def __init__(self, n_features=4, lr=0.1):
        super().__init__()
        self.q_net = nn.Linear(n_features, 1)
        self.lr = lr
        self.optim = torch.optim.SGD(self.parameters(), lr=lr)

    def forward(self, features):
        return self.q_net(features)

    def update(self, features, reward):
        features_t = torch.as_tensor(features, dtype=torch.float32, device=DEVICE)
        reward_t = torch.as_tensor([reward], dtype=torch.float32, device=DEVICE)
        q = self.forward(features_t.unsqueeze(0))
        loss = F.mse_loss(q.squeeze(0), reward_t)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()

# ═══════════════════════════════════════════════════════════════════════
# 6. METACOGNITIVE CONTROLLER — Uncertainty, confidence, strategy
# ═══════════════════════════════════════════════════════════════════════

class MetacognitiveController:
    ACTIONS = ["proceed", "search", "clarify", "explore", "replay"]

    def __init__(self):
        self.recent_confidence = []
        self.recent_uncertainty = []
        self.consecutive_low_confidence = 0
        self.total_interactions = 0
        self.slow_path_used = 0
        self.last_action = "proceed"

    def estimate_confidence(self, logits_top_k):
        probs = F.softmax(logits_top_k, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean().item()
        max_prob = probs.max(dim=-1).values.mean().item()
        normalized_entropy = entropy / math.log(logits_top_k.shape[-1])
        confidence = max_prob * (1.0 - normalized_entropy)
        return confidence, normalized_entropy

    def should_search(self, confidence, threshold=0.3):
        self.recent_confidence.append(confidence)
        if len(self.recent_confidence) > 20:
            self.recent_confidence.pop(0)
        return confidence < threshold

    def act(self, confidence, uncertainty, sfl_q=None):
        # update consecutive low confidence — no reset here (managed internally)
        if confidence < 0.35:
            self.consecutive_low_confidence += 1
        else:
            self.consecutive_low_confidence = 0

        if confidence < 0.2:
            self.last_action = "clarify"
        elif self.consecutive_low_confidence > 5:
            self.last_action = "replay"
        elif sfl_q is not None and sfl_q < -0.3 and self.consecutive_low_confidence > 2:
            self.last_action = "explore"
        elif confidence < 0.35:
            self.last_action = "search"
        else:
            self.last_action = "proceed"
        return self.last_action

    def record_outcome(self, used_slow_path=False):
        self.total_interactions += 1
        if used_slow_path:
            self.slow_path_used += 1

    def stats(self):
        avg_conf = sum(self.recent_confidence) / max(len(self.recent_confidence), 1)
        slow_rate = self.slow_path_used / max(self.total_interactions, 1)
        return {
            "total": self.total_interactions,
            "avg_confidence": round(avg_conf, 3),
            "slow_path_rate": round(slow_rate, 3),
            "confidence_history": self.recent_confidence[-50:],
            "last_action": self.last_action,
        }

# ═══════════════════════════════════════════════════════════════════════
# 7. WEB SEARCH — External knowledge tool
# ═══════════════════════════════════════════════════════════════════════

class WebSearch:
    def __init__(self):
        self.searcher = None
        self.cache = {}
        self.cache_path = os.path.join(MEMORY_DIR, "search_cache.json")
        self._load_cache()
        try:
            from duckduckgo_search import DDGS
            self.searcher = DDGS()
        except Exception:
            pass

    def _load_cache(self):
        try:
            with open(self.cache_path) as f:
                self.cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.cache = {}

    def _save_cache(self):
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self.cache, f)
        except Exception:
            pass

    def _search_wikipedia(self, query, max_results=3):
        try:
            import requests as req
            headers = {"User-Agent": "ProjectAdam/1.0 (https://github.com/kilvz/Project-Adam)"}
            params = {
                "action": "query", "list": "search",
                "srsearch": query, "format": "json", "srlimit": max_results
            }
            r = req.get("https://en.wikipedia.org/w/api.php", params=params,
                        headers=headers, timeout=10, verify=False)
            data = r.json()
            results = data.get("query", {}).get("search", [])
            if results:
                return "\n".join(
                    r["title"] + ": " + re.sub(r"<[^>]+>", "", r.get("snippet", ""))
                    for r in results[:max_results]
                )
        except Exception:
            pass
        return None

    def search(self, query, max_results=3):
        # check cache first
        cache_key = query.lower().strip()
        if cache_key in self.cache:
            return self.cache[cache_key]

        result = None

        # tier 1: DDGS
        if self.searcher:
            try:
                results = list(self.searcher.text(query, max_results=max_results))
                texts = [r["body"] for r in results if "body" in r]
                if texts:
                    result = "\n".join(texts[:max_results])
            except Exception:
                pass

        # tier 2: Wikipedia fallback
        if not result:
            result = self._search_wikipedia(query, max_results)

        # cache whatever we got
        if result:
            self.cache[cache_key] = result
            self._save_cache()

        return result

# ═══════════════════════════════════════════════════════════════════════
# 8. OFFLINE CONSOLIDATOR — Background replay + abstraction + cross-user distillation
# ═══════════════════════════════════════════════════════════════════════

class OfflineConsolidator:
    def __init__(self, episodic_memory, semantic_memory, neural_memory, tokenizer, model, embedder=None):
        self.episodic = episodic_memory
        self.semantic = semantic_memory
        self.neural = neural_memory
        self.tokenizer = tokenizer
        self.model = model
        self.embedder = embedder
        self.running = False
        self.thread = None
        self.user_profiles = None  # set externally for B8 cross-user distillation

    def start(self, interval=300):
        self.running = True
        self.thread = threading.Thread(target=self._loop, args=(interval,), daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _loop(self, interval):
        while self.running:
            time.sleep(interval)
            try:
                self._consolidate()
            except Exception:
                pass

    def _consolidate(self):
        if len(self.episodic.episodes) < 3:
            return

        now = time.time()

        # A4a: prioritized replay — sort by importance (reward × recency)
        scored = []
        for ep in self.episodic.episodes:
            reward = ep.get("reward", 0)
            age = now - ep.get("time", now)
            # B10: importance = reward + recency bonus (halves every 30 min)
            recency_factor = max(0.0, 1.0 - age / 3600.0)
            importance = reward * 0.6 + recency_factor * 0.4
            scored.append((importance, ep))
        scored.sort(key=lambda x: -x[0])

        # replay top 5 by importance
        top_k = [ep for _, ep in scored[:5]]
        for ep in top_k:
            text = ep.get("text", "")
            if len(text) < 5:
                continue
            for cat in ["name", "location", "likes", "preference", "dislikes"]:
                if cat in text.lower():
                    self.semantic.add(cat, text)
            inputs = self.tokenizer(text, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                emb = self.model.get_input_embeddings()(inputs["input_ids"]).to(self.neural.dtype)
            self.neural.learn(emb)

        # B10: prune low-importance episodes (importance < -0.2, older than 30 min)
        before = len(self.episodic.episodes)
        self.episodic.episodes = [
            ep for ep in self.episodic.episodes
            if not ((ep.get("reward", 0) * 0.6 + max(0.0, 1.0 - (now - ep.get("time", now)) / 3600.0) * 0.4) < -0.2
                    and now - ep.get("time", 0) > 1800)
        ]
        pruned = before - len(self.episodic.episodes)

        # A4c: merge duplicate semantic facts
        merged = 0
        for cat, data in list(self.semantic.schemas.items()):
            facts = data.get("facts", [])
            if len(facts) > 1:
                unique = []
                seen = set()
                for f in facts:
                    key = f.lower().strip()
                    if key not in seen:
                        seen.add(key)
                        unique.append(f)
                    else:
                        merged += 1
                if len(unique) != len(facts):
                    self.semantic.schemas[cat]["facts"] = unique

        # B8: cross-user pattern distillation
        if self.user_profiles is not None and self.user_profiles.profiles:
            all_topics = {}
            for name, p in self.user_profiles.profiles.items():
                for topic, count in p.get("topics", {}).items():
                    if topic not in all_topics:
                        all_topics[topic] = {"count": 0, "users": set()}
                    all_topics[topic]["count"] += count
                    all_topics[topic]["users"].add(name)
            # store cross-user topics (topic appears in 2+ users) as semantic knowledge
            cross_topics = {t: d for t, d in all_topics.items() if len(d["users"]) > 1}
            if cross_topics:
                cross_str = "; ".join(f"{t}({len(d['users'])} users)" for t, d in sorted(cross_topics.items(), key=lambda x: -x[1]['count'])[:5])
                self.semantic.add("cross_user_topics", cross_str)

        # B9: semantic phrase grouping via embedding (cluster adopted phrases by similarity)
        if self.embedder is not None and self.user_profiles is not None:
            all_phrases = []
            for name, p in self.user_profiles.profiles.items():
                for phrase, data in p.get("adopted_phrases", {}).items():
                    if data.get("count", 0) >= 3:
                        all_phrases.append((phrase, name))
            if len(all_phrases) > 3:
                try:
                    phrases = list(set(p for p, _ in all_phrases))
                    embs = [self.embedder.encode(p, convert_to_numpy=True) for p in phrases]
                    import numpy as np
                    embs_arr = np.stack(embs)
                    similarity = embs_arr @ embs_arr.T
                    groups = []
                    assigned = set()
                    for i in range(len(phrases)):
                        if i in assigned:
                            continue
                        group = [phrases[i]]
                        assigned.add(i)
                        for j in range(i + 1, len(phrases)):
                            if j not in assigned and similarity[i, j] > 0.7:
                                group.append(phrases[j])
                                assigned.add(j)
                        if len(group) > 1:
                            groups.append(group)
                    if groups:
                        for g in groups:
                            self.semantic.add("phrase_cluster", " | ".join(g))
                except Exception:
                    pass

        if pruned > 0 or merged > 0:
            self.episodic.save()
            self.semantic.save()
            print(f"[consolidator] replayed {len(top_k)} | pruned {pruned} | merged {merged} dupes")

# ═══════════════════════════════════════════════════════════════════════
# FACT EXTRACTION — Declarative memory encoding
# ═══════════════════════════════════════════════════════════════════════

FACT_PATTERNS = [
    ("name", r"my name is (\w+)"),
    ("name", r"i(?:'m| am) called (\w+)"),
    ("location", r"i live in (\w[\w\s]*)"),
    ("location", r"i(?:'m| am) from (\w[\w\s]*)"),
    ("age", r"i(?:'m| am) (\d+) years? old"),
    ("preference", r"my favorite (\w+) is ([\w\s]+)"),
    ("likes", r"i like ([\w\s]+)"),
    ("dislikes", r"i don't? like ([\w\s]+)"),
]

def extract_facts(text):
    if text.strip().endswith("?"):
        return []
    facts = []
    lower = text.lower()
    for cat, pattern in FACT_PATTERNS:
        for m in re.finditer(pattern, lower):
            facts.append((cat, m.group(0).strip()))
    return facts

# ─── Implicit reward from user message ───────────────────────────────

POSITIVE_WORDS = {"love", "great", "amazing", "interesting", "cool", "nice",
    "beautiful", "wonderful", "thanks", "good", "excellent", "fantastic",
    "helpful", "perfect", "fun", "awesome", "brilliant", "fascinating"}
NEGATIVE_WORDS = {"bad", "wrong", "no", "hate", "terrible", "awful", "stupid",
    "boring", "incorrect", "useless", "pointless", "annoying", "horrible",
    "disappointing"}

def compute_implicit_reward(user_input, user_profile=None, embedder=None):
    words = user_input.lower().split()
    if not words:
        return 0.0
    pos = sum(1 for w in words if w.strip(".,!?") in POSITIVE_WORDS)
    neg = sum(1 for w in words if w.strip(".,!?") in NEGATIVE_WORDS)
    sentiment = (pos - neg) / max(len(words), 1)
    sentiment = max(-1.0, min(1.0, sentiment))

    # A5: NLU refinement via embedding when word-list is near-neutral
    if embedder is not None and abs(sentiment) < 0.15 and len(user_input) > 15:
        try:
            POS_REFS = ["this is wonderful", "I love this", "great", "amazing", "fantastic"]
            NEG_REFS = ["this is terrible", "I hate this", "bad", "awful", "horrible"]
            q_emb = embedder.encode(user_input, convert_to_numpy=True)
            pos_sim = max(embedder.encode(p, convert_to_numpy=True) @ q_emb for p in POS_REFS)
            neg_sim = max(embedder.encode(n, convert_to_numpy=True) @ q_emb for n in NEG_REFS)
            nlu_score = (pos_sim - neg_sim) * 0.5
            nlu_score = max(-0.5, min(0.5, nlu_score))
            sentiment = sentiment * 0.3 + nlu_score * 0.7
        except Exception:
            pass

    sentiment = max(-1.0, min(1.0, sentiment))
    engagement = min(1.0, len(user_input) / 100.0)
    reward = sentiment * 0.6 + engagement * 0.3
    return max(-1.0, min(1.0, reward))

def extract_topics(text, embedder=None):
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "it", "its",
        "i", "you", "he", "she", "we", "they", "my", "your", "his", "her",
        "this", "that", "to", "of", "in", "for", "on", "and", "or", "but",
        "not", "do", "does", "did", "have", "has", "had", "be", "been",
        "with", "about", "at", "by", "from", "as", "so", "if", "then",
        "what", "why", "how", "when", "where", "who", "which", "all",
        "can", "will", "would", "should", "could"}
    words = list(set(w.strip(".,!?") for w in text.lower().split()
                     if w.strip(".,!?") not in stopwords and len(w) > 3))
    # D18: use embedder to merge semantically similar words
    if embedder is not None and len(words) > 2:
        try:
            embs = embedder.encode(words, convert_to_numpy=True)
            import numpy as np
            sims = embs @ embs.T
            merged = []
            used = set()
            for i in range(len(words)):
                if i in used:
                    continue
                group = [words[i]]
                used.add(i)
                for j in range(i + 1, len(words)):
                    if j not in used and sims[i, j] > 0.65:
                        group.append(words[j])
                        used.add(j)
                merged.append(group[0])  # use first word as representative
            return merged
        except Exception:
            pass
    return words

NAME_PATTERNS = [
    (r"my name is (\w+)", 1),
    (r"i(?:'m| am) called (\w+)", 1),
    (r"call me (\w+)", 1),
    (r"(?:^|\s)I(?:'m| am) (\w+)(?:\s|$|\.|,)", 1),
    (r"(?:^|\s)I'm (\w+)(?:\s|$|\.|,)", 1),
]

def detect_user(text, existing_names):
    lower = text.lower()
    for pattern, group in NAME_PATTERNS:
        m = re.search(pattern, lower)
        if m:
            name = m.group(group).strip().capitalize()
            if name.lower() not in ("i", "a", "an", "the", "adam"):
                return name
    for name in existing_names:
        if name.lower() in lower:
            return name
    return None

# ═══════════════════════════════════════════════════════════════════════
# 9. ACTION SELECTOR — Dual-system (fast direct / slow deliberate)
# ═══════════════════════════════════════════════════════════════════════

class ActionSelector:
    def __init__(self, tokenizer, model, episodic_memory, semantic_memory, web_search, metacognitive, persona=None):
        self.tokenizer = tokenizer
        self.model = model
        self.episodic = episodic_memory
        self.semantic = semantic_memory
        self.web = web_search
        self.meta = metacognitive
        self.persona = persona

    def select(self, user_input, conversation_history, user_profile=None, sfl_q=None, token_callback=None):
        semantic_ctx = self.semantic.retrieve(user_input, 3)
        known_facts = [f"{cat}: {fact_str}" for cat, fact_str, score in semantic_ctx if score > 0.3]

        if self.persona and self.persona.essence:
            system_content = self.persona.build_user_prompt(known_facts, user_profile)
        else:
            system_content = "You are a helpful assistant. Answer concisely and correctly."
            if known_facts:
                system_content += " " + " ".join(known_facts)

        messages = [{"role": "system", "content": system_content}]

        for turn in conversation_history[-6:]:
            messages.append(turn)

        messages.append({"role": "user", "content": user_input})
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_first = outputs.logits[:, -1, :]
            confidence, uncertainty = self.meta.estimate_confidence(logits_first)

        # B6: metacognitive action loop — pick action based on confidence + SFL state
        meta_action = self.meta.act(confidence, uncertainty, sfl_q)

        use_search = self.meta.should_search(confidence)
        web_context = None

        if meta_action == "clarify":
            # low confidence → ask clarifying question instead of full generation
            clarify_prompt = self.tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": "I'm not entirely sure what you mean. Could you clarify?"}],
                tokenize=False, add_generation_prompt=False
            )
            reply = "I'm not entirely sure what you mean. Could you clarify?"
            self.meta.record_outcome(used_slow_path=False)
            return reply, False, None, meta_action

        if use_search:
            web_context = self.web.search(user_input)
            if web_context:
                system_content = messages[0]["content"] + f"\nWeb search result: {web_context[:400]}"
                messages[0] = {"role": "system", "content": system_content}
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)

        # SFL Q-value adjusts generation temperature
        temp = 0.7
        if sfl_q is not None:
            if sfl_q < -0.2:
                temp = 0.5  # low satisfaction → conservative
            elif sfl_q > 0.5:
                temp = 0.85  # high satisfaction → creative
        # B6: EXPLORE action pushes temperature higher for novelty
        if meta_action == "explore":
            temp = max(temp, 0.85)

        # D14/D19: Streaming generation with TextIteratorStreamer
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=256,
            temperature=temp,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()

        reply_parts = []
        for token_str in streamer:
            if token_callback:
                token_callback(token_str)
            else:
                print(token_str, end="", flush=True)
            reply_parts.append(token_str)
        if not token_callback:
            print()
        reply = "".join(reply_parts).strip().split("\n")[0].strip()

        self.meta.record_outcome(used_slow_path=use_search)
        return reply, use_search, web_context, meta_action

# ═══════════════════════════════════════════════════════════════════════
# 10. MAIN AGENT — COGNET orchestrator
# ═══════════════════════════════════════════════════════════════════════

class CognitiveAgent:
    def __init__(self):
        t0 = time.time()
        model_id = BASE_MODEL
        is_4bit = False
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=_4BIT_CONFIG,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            is_4bit = True
        except Exception:
            print(f"  4-bit failed, falling back to {MODEL_0_5B} fp16...")
            model_id = MODEL_0_5B
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map=DEVICE,
            )
        self.model.eval()
        model_label = "1.5B-4bit" if is_4bit else "0.5B-fp16"
        dt = time.time() - t0
        print(f"Loaded {model_label}: {sum(p.numel() for p in self.model.parameters()):,} params in {dt:.0f}s")

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

        # C12/D16: Wrap with LoRA (trainable adapters, frozen base)
        self._lora_config = LoraConfig(
            r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, self._lora_config)
        self.model.eval()
        self._current_adapter = "base"
        self._adapter_dir = MEMORY_DIR / "adapters"
        self._adapter_dir.mkdir(exist_ok=True)
        lora_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"LoRA adapters: {lora_trainable:,} trainable params ({self._current_adapter})")

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
        nm_path = MEMORY_DIR / "neural_memory.pt"
        if nm_path.exists():
            self.neural_memory.load_state_dict(torch.load(nm_path, map_location=DEVICE, weights_only=True))
            print("[neural memory loaded]")

    def chat(self, user_input, token_callback=None):
        self.working_memory.add("user", user_input)

        # detect or confirm user identity
        known_names = self.user_profiles.list_users()
        detected = detect_user(user_input, known_names)
        if detected:
            is_new = detected not in known_names
            self.current_profile = self.user_profiles.set_current(detected)
            if is_new:
                print(f"  [hello {detected}, storing your profile]")
            else:
                print(f"  [welcome back {detected}]")
            # D16: switch to user's LoRA adapter
            self._switch_adapter(detected)

        elif self.current_profile is None:
            self.current_profile = self.user_profiles.get_or_create("Stranger")

        # extract facts
        facts = extract_facts(user_input)
        for cat, fact in facts:
            self.semantic_memory.add(cat, fact)
            self.episodic_memory.add(fact, reward=0.7)

        # C13: Autonomous knowledge-gap detection
        is_question = any(user_input.lower().startswith(w) for w in ["what", "why", "how", "when", "where", "who", "which", "does", "is", "are", "can"]) and "?" in user_input
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

        # inline NeuralMemory training on every turn
        if facts:
            self._inline_learn(facts)
        elif len(user_input) > 10:
            self._inline_learn([("input", user_input)])

        # compute reward from this turn (word-list + NLU embedding refinement)
        reward = compute_implicit_reward(user_input, self.current_profile, self.episodic_memory.embedder)

        # SFL update — learn which features predict user satisfaction
        features = self._build_sfl_features(user_input, reward)
        sfl_loss = self.sfl_module.update(features, reward)
        q_value = self.sfl_module(torch.as_tensor(features, dtype=torch.float32, device=DEVICE)).item()

        # side-effects before generation (so stdout stays clean during streaming)
        topics = extract_topics(user_input, self.episodic_memory.embedder)
        self._update_user_profile(user_input, reward, topics)
        self._mint_custom_rules()

        # generate response with user-adapted persona + SFL-driven temperature + metacognitive action
        reply, used_search, web_context, meta_action = self.action_selector.select(
            user_input, self.working_memory.get_context(),
            user_profile=self.current_profile,
            sfl_q=q_value,
            token_callback=token_callback,
        )

        # B6: REPLAY action triggers inline consolidation
        if meta_action == "replay":
            self._inline_learn([("replay", user_input)])
            self.consolidator.merge_episodes()

        # apply behavioral rules with weight tracking
        reply = self._apply_behavioral_rules(user_input, reply, reward)

        self.working_memory.add("assistant", reply)

        # update rule weights based on reward
        self._update_rule_weights(reward)

        # store latest Q-value in profile
        if self.current_profile is not None:
            self.current_profile["last_q"] = round(q_value, 3)
            self.current_profile.setdefault("q_history", []).append(round(q_value, 3))
            self.current_profile.setdefault("reward_history", []).append(round(reward, 3))
            if len(self.current_profile["q_history"]) > 100:
                self.current_profile["q_history"] = self.current_profile["q_history"][-100:]
                self.current_profile["reward_history"] = self.current_profile["reward_history"][-100:]

        # C12: periodic LoRA training (every 10 interactions)
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

            # B7: SensoryEncoder β-VAE update — compress embeddings for bottleneck representation
            flat = emb.view(-1, emb.shape[-1])
            self.sensory_encoder.vae_loss(flat[:min(64, flat.shape[0])])
        except Exception:
            pass

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
                        print(f"  [minted new rule: If {cond}, then {action[:50]}...]")

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

    # ═══════════════════════════════════════════════════════════════════════
    # C12: LoRA fine-tuning from accumulated memories
    # ═══════════════════════════════════════════════════════════════════════

    def _lora_format_examples(self, episodes, max_examples=8):
        """Format episodic memories as (text, labels) training examples."""
        examples = []
        for ep in episodes[-max_examples:]:
            text = ep.get("text", "")
            if not text or len(text) < 5:
                continue
            text = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\nI see.<|im_end|>"
            examples.append(text)
        return examples

    def _lora_train_step(self):
        """Run a few LoRA gradient steps on recent episodes. Called during consolidation."""
        if len(self.episodic_memory.episodes) < 5:
            return

        texts = self._lora_format_examples(self.episodic_memory.episodes[-20:], max_examples=5)
        if not texts:
            return

        self.model.train()
        optim = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=1e-4
        )

        total_loss = 0.0
        for text in texts:
            enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(DEVICE)
            labels = enc["input_ids"].clone()
            assist_tok = self.tokenizer("<|im_start|>assistant", add_special_tokens=False)["input_ids"][0]
            assist_idx = (labels[0] == assist_tok).nonzero()
            if assist_idx.numel() > 0:
                labels[0, :assist_idx[0, 0]] = -100
            out = self.model(**enc, labels=labels, use_cache=False)
            loss = out.loss
            if not torch.isnan(loss):
                loss.backward()
                optim.step()
                optim.zero_grad()
                total_loss += loss.item()

        self.model.eval()

        if total_loss > 0:
            self._save_adapter()

    def _save_adapter(self, name=None):
        """Save LoRA adapter weights for a user (or current active adapter)."""
        name = name or self._current_adapter
        path = self._adapter_dir / name
        path.mkdir(parents=True, exist_ok=True)
        try:
            # save only LoRA weights, not full PEFT config
            state = {k: v for k, v in self.model.state_dict().items() if "lora" in k}
            torch.save(state, path / "adapter_model.safetensors")
            # also save config for detectability
            import json
            json.dump({"adapter_name": name}, open(path / "adapter_config.json", "w"))
        except Exception:
            pass

    def _switch_adapter(self, user):
        """Save current adapter, load user's if exists, or keep base."""
        if user == self._current_adapter:
            return
        # save current adapter first
        if self._current_adapter != "base":
            self._save_adapter(self._current_adapter)
        # check if user adapter exists on disk
        user_path = self._adapter_dir / user
        adapter_exists = (user_path / "adapter_model.safetensors").exists()
        if adapter_exists:
            try:
                state = torch.load(user_path / "adapter_model.safetensors", map_location=DEVICE, weights_only=True)
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if name in state:
                            param.copy_(state[name])
            except Exception:
                pass
        self._current_adapter = user



    def run(self):
        self.consolidator.start(interval=180)
        greeting = self.persona.get_opening() if self.persona else ""
        print("\n── Adam (COGNET) ──")
        if greeting:
            print(f"Adam: {greeting}")
        print("Commands: /search <q> /memory /schemas /persona /users /profile /stats /save /exit\n")
        while True:
            try:
                user = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user:
                continue
            if user.lower() in ("exit", "quit"):
                self.episodic_memory.save()
                self.semantic_memory.save()
                self.user_profiles.save()
                torch.save(self.neural_memory.state_dict(), MEMORY_DIR / "neural_memory.pt")
                self._save_adapter()
                closing = self.persona.get_closing() if self.persona else "Goodbye."
                print(f"Adam: One more thing — {closing}")
                break
            if user.startswith("/search "):
                q = user[8:]
                print(f"  [web] searching: {q}")
                r = self.web_search.search(q)
                print(f"  {r[:400] if r else 'no results'}")
                if r:
                    self.episodic_memory.add(f"[web: {q}] {r[:200]}", reward=0.5)
                continue
            if user == "/memory":
                print(f"  Episodic: {len(self.episodic_memory.episodes)} items")
                for ep in self.episodic_memory.episodes[-5:]:
                    r = ep.get("reward", 0)
                    t = ep.get("text", "")[:80]
                    print(f"    [r={r:.1f}] {t}")
                continue
            if user == "/schemas":
                print(f"  Semantic: {len(self.semantic_memory.schemas)} schemas")
                for cat, data in self.semantic_memory.schemas.items():
                    print(f"    {cat}: {'; '.join(data['facts'][-3:])}")
                continue
            if user == "/persona":
                if self.persona:
                    print(f"  Persona: {self.persona.path.name}")
                    print(f"  Essence: {self.persona.essence[:100]}")
                    print(f"  Behavior rules: {len(self.persona.behavior_rules)}")
                    sig_count = len(self.persona.language_patterns.split()) if self.persona.language_patterns else 0
                    print(f"  Signature expressions: ~{sig_count}")
                else:
                    print("  No persona loaded")
                continue
            if user == "/stats":
                s = self.metacognitive.stats()
                q = self.current_profile.get("last_q", "N/A") if self.current_profile else "N/A"
                print(f"  Interactions: {s['total']} | Avg confidence: {s['avg_confidence']} | Slow path: {s['slow_path_rate']}")
                print(f"  Last SFL Q: {q} | Custom rules: {len(self.current_profile.get('custom_rules', [])) if self.current_profile else 0}")
                continue
            if user == "/dashboard":
                p = self.current_profile or {}
                s = self.metacognitive.stats()
                print(f"  ┌─ DASHBOARD ──────────────────────────────")
                print(f"  │ User: {p.get('name', '—')}")
                print(f"  │ Interactions: {p.get('interaction_count', 0)}")
                print(f"  │ Avg sentiment: {p.get('avg_sentiment', 0):.2f}")
                print(f"  │ Avg confidence: {s.get('avg_confidence', 0)}")
                print(f"  │ Last action: {s.get('last_action', '—')}")
                print(f"  │ Slow path: {s.get('slow_path_rate', 0)}")
                q_hist = p.get('q_history', [])
                r_hist = p.get('reward_history', [])
                if q_hist:
                    print(f"  │ SFL Q (last 20): {'█' * int(abs(q_hist[-1]) * 10)} {q_hist[-1]:.2f}")
                if r_hist:
                    recent = r_hist[-20:]
                    avg_r = sum(recent) / len(recent)
                    print(f"  │ Reward trend: ↑{sum(1 for r in recent if r > 0)} ↓{sum(1 for r in recent if r < 0)} "
                          f"avg={avg_r:.2f} last={r_hist[-1]:.2f}")
                c_hist = s.get('confidence_history', [])
                if c_hist:
                    print(f"  │ Confidence: avg={sum(c_hist)/len(c_hist):.2f} "
                          f"last={c_hist[-1]:.2f} low={sum(1 for c in c_hist if c < 0.3)}")
                rw = p.get('rule_weights', {})
                if rw:
                    vals = list(rw.values())
                    print(f"  │ Rule weights: max={max(vals):.2f} min={min(vals):.2f} "
                          f"spread={max(vals)-min(vals):.2f}")
                print(f"  └──────────────────────────────────────────")
                continue
            if user == "/save":
                self.episodic_memory.save()
                self.semantic_memory.save()
                self.user_profiles.save()
                torch.save(self.neural_memory.state_dict(), MEMORY_DIR / "neural_memory.pt")
                print("  [saved all memory systems]")
                continue
            if user == "/users":
                users = self.user_profiles.list_users()
                print(f"  Known users ({len(users)}):")
                for u in users:
                    p = self.user_profiles.get_or_create(u)
                    print(f"    {u}: {p.get('interaction_count', 0)} interactions, sentiment={p.get('avg_sentiment', 0):.2f}")
                continue
            if user == "/profile":
                p = self.user_profiles.get_current()
                if p:
                    print(f"  Current user: {p['name']}")
                    print(f"  Interactions: {p['interaction_count']}")
                    print(f"  Avg sentiment: {p['avg_sentiment']:.2f}")
                    print(f"  Last SFL Q-value: {p.get('last_q', 'N/A')}")
                    print(f"  Custom rules: {len(p.get('custom_rules', []))}")
                    top_topics = sorted(p['topics'].items(), key=lambda x: -x[1])[:5]
                    if top_topics:
                        print(f"  Top topics: {', '.join(f'{t}({c})' for t, c in top_topics)}")
                    adopted = [ph for ph, d in p['adopted_phrases'].items() if d.get('count', 0) >= 5]
                    if adopted:
                        print(f"  Adopted phrases (≥5 uses): {len(adopted)}")
                    rw = p.get('rule_weights', {})
                    if rw:
                        top_rule = max(rw.items(), key=lambda x: x[1])
                        bot_rule = min(rw.items(), key=lambda x: x[1])
                        print(f"  Strongest rule: rule {top_rule[0]} (w={top_rule[1]:.2f})")
                        print(f"  Weakest rule: rule {bot_rule[0]} (w={bot_rule[1]:.2f})")
                else:
                    print("  No current user")
                continue

            reply = self.chat(user)
            # D14: reply is streamed token-by-token inside select(), no extra print needed

# ═══════════════════════════════════════════════════════════════════════
# D19: Gradio Web UI
# ═══════════════════════════════════════════════════════════════════════

def run_web_ui(agent):
    import gradio as gr

    def respond(message, history):
        full_reply = ""
        def on_token(tok):
            nonlocal full_reply
            full_reply += tok
        agent.chat(message, token_callback=on_token)
        return full_reply

    def respond_stream(message, history):
        full_reply = ""
        def on_token(tok):
            nonlocal full_reply
            full_reply += tok
        agent.chat(message, token_callback=on_token)
        yield full_reply

    with gr.Blocks(title="Project Adam", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Project Adam — COGNET Conversational AI")
        chatbot = gr.ChatInterface(
            fn=respond_stream,
            type="tuples",
            title="Adam",
            description="A self-learning AI that adapts to each user.",
            theme=gr.themes.Soft(),
        )
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    import sys
    agent = CognitiveAgent()
    if agent.persona and agent.persona.essence:
        print(f"[persona] Adam — {agent.persona.essence[:80]}...")
        print(f"[persona] {len(agent.persona.behavior_rules)} behavioral rules loaded")
    else:
        print("[persona] no persona file found, using generic assistant")
    if "--web" in sys.argv:
        run_web_ui(agent)
    else:
        agent.run()
