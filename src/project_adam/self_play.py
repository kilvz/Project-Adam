"""Self-play learner — generates (query, teacher_response) pairs into episodic memory."""

import json
import logging
import threading
import time
import numpy as np
from collections import deque

from .config import get_memory_dir

logger = logging.getLogger(__name__)


class SelfPlayLearner:
    def __init__(self, agent, config):
        self.agent = agent
        self._running = threading.Event()
        self._thread = None

        self.interval = config.get("interval_seconds", 120)
        self.batch_size = config.get("batch_size", 8)
        self.reward = config.get("reward", 0.85)
        self.strategies = config.get("strategies",
            ["schema", "world_model", "procedural", "creative"])
        self.checkpoint_interval = config.get("checkpoint_interval", 50)

        self._query_history = deque(maxlen=config.get("max_recent_queries", 200))
        self._checkpoint_path = get_memory_dir() / "self_play_checkpoint.json"
        self._load_checkpoint()
        self._run_immediately = threading.Event()
        self._stats_lock = threading.Lock()

        self.stats = {
            "total_queries": 0,
            "total_trained": 0,
            "current_strategy": None,
            "running": False,
            "started_at": None,
            "last_error": None,
        }

    def start(self):
        if self._running.is_set():
            return
        self._running.set()
        self.stats["running"] = True
        self.stats["started_at"] = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Self-play: thread started (interval=%ds, batch=%d)", self.interval, self.batch_size)

    def stop(self):
        self._running.clear()
        self._run_immediately.set()  # wake sleeping thread
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self.stats["running"] = False
        self._save_checkpoint()
        logger.info("Self-play: thread stopped")

    # ── Main loop ──────────────────────────────────────────────────

    def get_stats(self):
        with self._stats_lock:
            return dict(self.stats)

    def _loop(self):
        while self._running.is_set():
            # Metacog gate: only generate when metacog would choose exploration
            if self.agent.metacognitive.last_action not in ("EXPLORE", "ASK_FOR_HELP"):
                if not self._run_immediately.is_set():
                    time.sleep(self.interval)
                    continue
                self._run_immediately.clear()

            for strategy in self.strategies:
                if not self._running.is_set():
                    break
                with self._stats_lock:
                    self.stats["current_strategy"] = strategy
                n = max(1, self.batch_size // len(self.strategies))
                queries = self._generate_queries(strategy, n)
                queries = self._dedup(queries)
                for q in queries:
                    if not self._running.is_set():
                        break
                    resp = self._call_teacher(q)
                    if not resp:
                        continue
                    self.agent.episodic_memory.add(
                        text=q,
                        reward=self.reward,
                        action=resp,
                        context="self_play",
                    )
                    # Compute RPE immediately for consolidation prioritization
                    baseline = [0.0] * 8
                    rpe = self.agent.td_core.update(self.reward, baseline)
                    if self.agent.episodic_memory.episodes:
                        self.agent.episodic_memory.episodes[-1]["rpe"] = rpe
                    self._query_history.append(q)
                    with self._stats_lock:
                        self.stats["total_queries"] += 1
                        self.stats["total_trained"] += 1
                with self._stats_lock:
                    if self.stats["total_queries"] % self.checkpoint_interval == 0:
                        self._save_checkpoint()
            time.sleep(self.interval)

    # ── Query generation (4 strategies) ─────────────────────────────

    def _generate_queries(self, strategy, n):
        if strategy == "schema":
            return self._queries_from_schemas(n)
        elif strategy == "world_model":
            return self._queries_from_world_model(n)
        elif strategy == "procedural":
            return self._queries_from_procedural_gaps(n)
        elif strategy == "creative":
            return self._queries_creative(n)
        return []

    def _queries_from_schemas(self, n):
        schemas = self.agent.semantic_memory.schemas
        if not schemas:
            return ["Teach me something interesting about science."]
        candidates = []
        for sid, s in schemas.items():
            pe = s.get("prediction_error", 1.0)
            oc = s.get("observed_count", 0)
            if pe > 0.2 or oc < 3:
                candidates.append((pe, s.get("category", "unknown")))
        if not candidates:
            return ["Teach me something interesting about science."]
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [f"Explain {cat} to me. Focus on what I don't know yet."
                for _, cat in candidates[:n]]

    def _queries_from_world_model(self, n):
        entities = self.agent.world_model.entities
        if not entities:
            return []
        scored = []
        for entity, attrs in entities.items():
            uncertainties = []
            for attr in attrs:
                u = self.agent.world_model.uncertainty(entity, attr)
                uncertainties.append(u)
            avg_u = sum(uncertainties) / max(len(uncertainties), 1)
            scored.append((avg_u, entity))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f"What is {entity}? I want to understand it better."
                for _, entity in scored[:n]]

    def _queries_from_procedural_gaps(self, n):
        skills = self.agent.procedural_memory.skills
        if not skills:
            return []
        candidates = []
        for sid, skill in skills.items():
            if skill.q_value < 0.3 or skill.success_rate < 0.5:
                keywords = " ".join(skill.keywords)
                candidates.append((skill.q_value, keywords))
        if not candidates:
            return []
        candidates.sort(key=lambda x: x[0])
        return [f"How do I handle {kw}? I've struggled with this."
                for _, kw in candidates[:n]]

    def _queries_creative(self, n):
        queries = []
        max_attempts = n * 2
        attempts = 0
        while len(queries) < n and attempts < max_attempts:
            attempts += 1
            topic_query = "Suggest a topic for me to learn about. Give me one specific concept."
            topic = self._call_teacher(topic_query)
            if not topic:
                continue
            topic_clean = topic.split(".")[0].split(":")[0].strip()
            if len(topic_clean) < 3:
                continue
            # Dedup check against existing query history
            topic_dup = False
            if self.agent.episodic_memory.embedder:
                t_emb = self.agent.episodic_memory.encode(topic_clean)
                for recent in list(self._query_history):
                    r_emb = self.agent.episodic_memory.encode(recent)
                    sim = float(t_emb @ r_emb / (np.linalg.norm(t_emb) * np.linalg.norm(r_emb) + 1e-8))
                    if sim > 0.85:
                        topic_dup = True
                        break
            if topic_dup:
                continue
            self._query_history.append(topic_clean)
            queries.append(f"Explain {topic_clean} to me. What should I know about it?")
        return queries[:n]

    # ── Teacher call ────────────────────────────────────────────────

    def _call_teacher(self, query):
        try:
            return self.agent.teacher_generate(query)
        except Exception as e:
            logger.warning("Self-play teacher call failed: %s", e)
            self.stats["last_error"] = str(e)
            return None

    # ── Dedup ───────────────────────────────────────────────────────

    def _dedup(self, queries):
        if not self._query_history or not self.agent.episodic_memory.embedder:
            return queries
        result = []
        for q in queries:
            q_emb = self.agent.episodic_memory.encode(q)
            is_dup = False
            for recent in list(self._query_history):
                r_emb = self.agent.episodic_memory.encode(recent)
                sim = float(q_emb @ r_emb / (np.linalg.norm(q_emb) * np.linalg.norm(r_emb) + 1e-8))
                if sim > 0.85:
                    is_dup = True
                    break
            if not is_dup:
                result.append(q)
        return result

    # ── Checkpointing ───────────────────────────────────────────────

    def _save_checkpoint(self):
        try:
            data = {
                "query_history": list(self._query_history),
                "stats": {k: v for k, v in self.stats.items() if k != "started_at"},
            }
            with open(self._checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning("Self-play checkpoint save failed: %s", e)

    def _load_checkpoint(self):
        try:
            if self._checkpoint_path.exists():
                with open(self._checkpoint_path, encoding="utf-8") as f:
                    data = json.load(f)
                self._query_history.extend(data.get("query_history", []))
                saved = data.get("stats", {})
                for k in ("total_queries", "total_trained"):
                    if k in saved:
                        self.stats[k] = saved[k]
        except Exception as e:
            logger.warning("Self-play checkpoint load failed: %s", e)
