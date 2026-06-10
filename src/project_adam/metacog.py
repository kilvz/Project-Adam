import math
from collections import deque


class MetacognitiveController:
    def __init__(self):
        self.confidence_history = deque(maxlen=50)
        self.recent_confidence = deque(maxlen=20)
        self.strategy_history = deque(maxlen=20)
        self.total = 0
        self.slow_path = 0
        self.consecutive_low_confidence = 0
        self.last_action = "proceed"
        self.total_interactions = 0
        self.slow_path_used = 0

    def estimate_confidence(self, logits):
        probs = logits.softmax(dim=-1)
        entropies = (-probs * (probs + 1e-8).log()).sum(dim=-1).mean().item()
        max_entropy = math.log(logits.shape[-1])
        uncertainty = entropies / max_entropy if max_entropy > 0 else 0.5
        confidence = max(0.0, min(1.0, 1.0 - uncertainty))
        self.confidence_history.append(confidence)
        return confidence, uncertainty

    def should_search(self, confidence, threshold=0.3):
        self.recent_confidence.append(confidence)
        return confidence < threshold

    def act(self, confidence, uncertainty, sfl_q=None):
        self.total += 1
        self.strategy_history.append("act")
        if confidence >= 0.7:
            self.consecutive_low_confidence = 0
            self.last_action = "proceed"
            return "proceed"
        self.consecutive_low_confidence += 1
        if uncertainty is not None and uncertainty > 0.7:
            self.slow_path += 1
            self.last_action = "clarify"
            return "clarify"
        if sfl_q is not None and sfl_q < 0.2:
            self.slow_path += 1
            self.last_action = "explore"
            return "explore"
        if self.consecutive_low_confidence >= 5:
            self.slow_path += 1
            self.last_action = "replay"
            return "replay"
        self.last_action = "search"
        return "search"

    def record_outcome(self, used_slow_path):
        self.total_interactions += 1
        if used_slow_path:
            self.slow_path_used += 1

    def stats(self):
        return {
            "total": self.total,
            "avg_confidence": sum(self.confidence_history) / max(len(self.confidence_history), 1),
            "slow_path_rate": self.slow_path_used / max(self.total_interactions, 1),
            "last_action": self.last_action,
            "confidence_history": list(self.recent_confidence),
        }
