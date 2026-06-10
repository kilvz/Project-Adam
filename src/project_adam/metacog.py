import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


CANONICAL_ACTIONS = ("EXPLORE", "REPLAY", "ASK_FOR_HELP", "STOP_AND_THINK", "SWITCH_STRATEGY")
_N_ACTIONS = len(CANONICAL_ACTIONS)


class MetacogPolicy(nn.Module):
    def __init__(self, n_features=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Linear(16, _N_ACTIONS),
        )

    def forward(self, x):
        return self.fc(x)


class MetacognitiveController:
    def __init__(self, lr=1e-3):
        self.confidence_history = deque(maxlen=50)
        self.recent_confidence = deque(maxlen=20)
        self.recent_rewards = deque(maxlen=20)
        self.strategy_history = deque(maxlen=20)
        self.total = 0
        self.slow_path = 0
        self.consecutive_low_confidence = 0
        self.last_action = "proceed"
        self.total_interactions = 0
        self.slow_path_used = 0
        self._policy = MetacogPolicy(n_features=5)
        self._optimizer = torch.optim.Adam(self._policy.parameters(), lr=lr)
        self._last_features = None
        self._last_logits = None

    def _build_features(self, confidence, uncertainty, sfl_q):
        c = confidence if confidence is not None else 0.5
        u = uncertainty if uncertainty is not None else 0.5
        q = sfl_q if sfl_q is not None else 0.5
        cl = min(1.0, self.consecutive_low_confidence / 10.0)
        lp = 0.0
        if len(self.recent_rewards) >= 5:
            recent = list(self.recent_rewards)
            lp = (recent[-1] - recent[0]) / max(len(recent), 1)
            lp = max(-1.0, min(1.0, lp))
        return torch.tensor([c, u, q, cl, lp], dtype=torch.float32)

    def record_confidence(self, value):
        self.confidence_history.append(value)

    def estimate_confidence(self, logits):
        if logits is None:
            avg = sum(self.confidence_history) / max(len(self.confidence_history), 1) if self.confidence_history else 0.5
            return avg, 1.0 - avg
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

        if confidence is not None and confidence >= 0.7:
            self.consecutive_low_confidence = 0
            self.last_action = "proceed"
            return "proceed"

        self.consecutive_low_confidence += 1
        rule_action = self._rule_act(confidence, uncertainty, sfl_q)

        features = self._build_features(confidence, uncertainty, sfl_q)
        self._last_features = features
        logits = self._policy(features.unsqueeze(0)).squeeze(0)
        self._last_logits = logits.detach().requires_grad_(True)
        probs = F.softmax(logits, dim=-1)

        mixing = min(0.9, self.total / (self.total + 100))
        if torch.rand(1).item() < mixing:
            action_idx = torch.multinomial(probs.detach(), 1).item()
            chosen = CANONICAL_ACTIONS[action_idx]
        else:
            chosen = rule_action

        self.last_action = chosen
        return chosen

    def _rule_act(self, confidence, uncertainty, sfl_q=None):
        if uncertainty is not None and uncertainty > 0.7:
            return "STOP_AND_THINK"
        if sfl_q is not None and sfl_q < 0.2:
            return "EXPLORE"
        if self.consecutive_low_confidence >= 5:
            return "REPLAY"
        if confidence is not None and confidence < 0.3:
            return "ASK_FOR_HELP"
        return "SWITCH_STRATEGY"

    def learn(self, reward):
        if self._last_features is None or self._last_logits is None:
            return
        logits = self._policy(self._last_features.unsqueeze(0)).squeeze(0)
        advantages = torch.tensor([reward], dtype=torch.float32)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            action_idx = torch.argmax(self._last_logits)
        policy_loss = -(log_probs[action_idx] * advantages)
        self._optimizer.zero_grad()
        policy_loss.backward()
        self._optimizer.step()
        self._last_features = None
        self._last_logits = None

    def should_self_talk(self, confidence, sfl_q=None):
        if confidence < 0.4:
            return "low_confidence"
        if sfl_q is not None and sfl_q < 0.2:
            return "unfamiliar"
        return None

    def record_outcome(self, used_slow_path, reward=None):
        self.total_interactions += 1
        if used_slow_path:
            self.slow_path_used += 1
        if reward is not None:
            self.recent_rewards.append(reward)

    def stats(self):
        return {
            "total": self.total,
            "avg_confidence": sum(self.confidence_history) / max(len(self.confidence_history), 1),
            "slow_path_rate": self.slow_path_used / max(self.total_interactions, 1),
            "slow_path_abs": self.slow_path,
            "last_action": self.last_action,
            "strategy_count": len(self.strategy_history),
            "confidence_history": list(self.recent_confidence),
        }
