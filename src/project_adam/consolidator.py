import threading
import time


class OfflineConsolidator:
    def __init__(self, episodic_memory, semantic_memory, neural_memory,
                 tokenizer, model, embedder=None):
        self.episodic = episodic_memory
        self.semantic = semantic_memory
        self.neural = neural_memory
        self.tokenizer = tokenizer
        self.model = model
        self.embedder = embedder
        self.user_profiles = None
        self._running = False
        self._thread = None
        self._stop = False

    def start(self, interval=180):
        if self._running:
            return
        self._running = True
        self._stop = False
        self._thread = threading.Thread(target=self._loop, args=(interval,), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        self._running = False

    def _loop(self, interval):
        while not self._stop:
            time.sleep(interval)
            try:
                self._consolidate()
            except Exception:
                pass

    def _consolidate(self):
        episodes = list(self.episodic.episodes)
        if not episodes:
            return

        candidates = sorted(episodes, key=lambda e: e.get("reward", 0), reverse=True)
        top = candidates[:10]

        for ep in top:
            text = ep.get("text", "")
            if len(text) < 10:
                continue
            if not any(w in text.lower() for w in ["my", "i", "you", "is", "are"]):
                continue
            category = "experience"
            self.semantic.add(category, text[:200])
            self.episodic.add(f"[consolidated] {text[:100]}", reward=ep.get("reward", 0) * 0.8)

        self.semantic.consolidate()

        if self.user_profiles is not None:
            all_profiles = self.user_profiles.profiles
            patterns = self.semantic.cross_user_distill(all_profiles)
            if patterns:
                for keyword, users in patterns:
                    msg = f"[distilled] '{keyword}' mentioned by {', '.join(users)}"
                    self.episodic.add(msg, reward=0.6)

        if self.embedder is not None and episodes:
            texts = [e.get("text", "") for e in episodes[-20:] if len(e.get("text", "")) > 10]
            if len(texts) >= 3:
                try:
                    embs = self.embedder.encode(texts, convert_to_numpy=True)
                    sims = embs @ embs.T
                    clusters = []
                    used = set()
                    for i in range(len(texts)):
                        if i in used:
                            continue
                        group = [texts[i]]
                        used.add(i)
                        for j in range(i + 1, len(texts)):
                            if j not in used and sims[i, j] > 0.7:
                                group.append(texts[j])
                                used.add(j)
                        clusters.append(group[0])
                    for phrase in clusters[:5]:
                        self.episodic.add(f"[pattern] {phrase[:100]}", reward=0.5)
                except Exception:
                    pass

        self.episodic.prune(threshold=0.3)

    def merge_episodes(self):
        episodes = self.episodic.episodes
        if len(episodes) < 5:
            return
        recent = episodes[-5:]
        texts = [e.get("text", "") for e in recent if len(e.get("text", "")) > 20]
        if not texts:
            return
        merged = " ".join(texts[:3])
        self.semantic.add("merged_experience", merged[:300])
        self.episodic.add(f"[merged] {merged[:100]}", reward=0.5)
