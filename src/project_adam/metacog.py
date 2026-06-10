import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import logging

logger = logging.getLogger(__name__)

CANONICAL_ACTIONS = ("EXPLORE", "REPLAY", "ASK_FOR_HELP", "STOP_AND_THINK", "SWITCH_STRATEGY")
_N_ACTIONS = len(CANONICAL_ACTIONS)


class MetacogPolicy(nn.Module):
    """
    Neural network that learns which metacognitive action to take.
    
    Inputs (5 features):
    - confidence: model confidence in current answer
    - uncertainty: inverse entropy of model logits
    - sfl_q: social feature learning value (did people like this style?)
    - consecutive_low_conf: how many turns low confidence in a row
    - learning_progress: reward trend (improving or not?)
    
    Output: 5 action logits
    - EXPLORE: try new approaches
    - REPLAY: consolidate/sleep learning
    - ASK_FOR_HELP: query for information
    - STOP_AND_THINK: pause for deliberation
    - SWITCH_STRATEGY: change approach
    """
    
    def __init__(self, n_features=5, n_actions=_N_ACTIONS):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_actions),
        )

    def forward(self, x):
        return self.fc(x)


class MetacognitiveController:
    """
    Learns when to use which metacognitive strategy.
    
    IMPORTANT: This should be PRIMARILY LEARNED (not rule-based).
    The old code was 90% rule-based; this fixes it to be 90% learned.
    
    Key fix (Gap #5):
    - Old: 90% rule → 10% learned (backwards)
    - New: 10% rule → 90% learned (correct)
    
    The learning process:
    1. Take action (rule or learned)
    2. Observe outcome (reward)
    3. Update policy with REINFORCE: ∇J ∝ log π(a|s) * R
    """
    
    def __init__(self, lr=1e-3, rule_phase_length=100):
        # Confidence tracking
        self.confidence_history = deque(maxlen=50)
        self.recent_confidence = deque(maxlen=20)
        self.recent_rewards = deque(maxlen=20)
        self.strategy_history = deque(maxlen=20)
        
        # Interaction tracking
        self.total = 0
        self.consecutive_low_confidence = 0
        self.last_action = "proceed"
        self.total_interactions = 0
        self.slow_path_used = 0
        
        # Learning components
        self._policy = MetacogPolicy(n_features=5, n_actions=_N_ACTIONS)
        self._optimizer = torch.optim.Adam(self._policy.parameters(), lr=lr)
        self._last_features = None
        self._last_action_idx = None
        self._rule_phase_length = rule_phase_length  # How long to run rules before learning
        
        # Learning statistics
        self._action_counts = {a: 0 for a in CANONICAL_ACTIONS}
        self._action_rewards = {a: [] for a in CANONICAL_ACTIONS}
        self._policy_loss_history = deque(maxlen=100)

    def _build_features(self, confidence, uncertainty, sfl_q):
        """Extract 5 features for the metacognitive policy."""
        c = confidence if confidence is not None else 0.5
        u = uncertainty if uncertainty is not None else 0.5
        q = sfl_q if sfl_q is not None else 0.5
        
        # How many consecutive low-confidence turns
        cl = min(1.0, self.consecutive_low_confidence / 10.0)
        
        # Learning progress: is reward trend improving?
        lp = 0.0
        if len(self.recent_rewards) >= 5:
            recent = list(self.recent_rewards)
            lp = (recent[-1] - recent[0]) / max(len(recent), 1)
            lp = max(-1.0, min(1.0, lp))
        
        return torch.tensor([c, u, q, cl, lp], dtype=torch.float32)

    def record_confidence(self, value):
        """Record model confidence estimate."""
        self.confidence_history.append(value)

    def estimate_confidence(self, logits):
        """Estimate confidence and uncertainty from model logits."""
        if logits is None:
            avg = (
                sum(self.confidence_history) / max(len(self.confidence_history), 1)
                if self.confidence_history else 0.5
            )
            return avg, 1.0 - avg
        
        # Entropy from softmax probabilities
        probs = logits.softmax(dim=-1)
        entropies = (-probs * (probs + 1e-8).log()).sum(dim=-1).mean().item()
        max_entropy = math.log(logits.shape[-1])
        uncertainty = entropies / max_entropy if max_entropy > 0 else 0.5
        confidence = max(0.0, min(1.0, 1.0 - uncertainty))
        
        self.confidence_history.append(confidence)
        return confidence, uncertainty

    def should_search(self, confidence, threshold=0.3):
        """Query for external information if confidence too low."""
        self.recent_confidence.append(confidence)
        return confidence < threshold

    def act(self, confidence, uncertainty, sfl_q=None):
        """
        Select a metacognitive action.
        
        KEY FIX (Gap #5):
        - Gradually increase learned policy contribution
        - mixing = min(0.9, total / (total + rule_phase_length))
        - After 100 interactions: 50% learned, 50% rule
        - After 1000 interactions: 90% learned, 10% rule
        """
        self.total += 1
        self.strategy_history.append("act")

        # High confidence → always proceed (fast path)
        if confidence is not None and confidence >= 0.7:
            self.consecutive_low_confidence = 0
            self.last_action = "proceed"
            return "proceed"

        self.consecutive_low_confidence += 1
        
        # Build features for policy input
        features = self._build_features(confidence, uncertainty, sfl_q)
        self._last_features = features

        # Get policy logits
        with torch.no_grad():
            logits = self._policy(features.unsqueeze(0)).squeeze(0)
        
        probs = F.softmax(logits, dim=-1)

        # CRITICAL FIX: Gradually increase learned policy contribution
        # mixing = P(learned policy is selected)
        mixing = min(0.9, self.total / (self.total + self._rule_phase_length))
        
        if torch.rand(1).item() < mixing:
            # Use learned policy (increasingly dominant over time)
            action_idx = torch.multinomial(probs.detach(), 1).item()
            chosen = CANONICAL_ACTIONS[action_idx]
            is_learned = True
        else:
            # Use rule-based fallback (fallback only)
            action_idx = CANONICAL_ACTIONS.index(self._rule_act(confidence, uncertainty, sfl_q))
            chosen = CANONICAL_ACTIONS[action_idx]
            is_learned = False
        
        # Store for learning
        self._last_action_idx = action_idx
        
        # Track for stats
        self._action_counts[chosen] += 1
        
        logger.debug(
            f"[metacog] {chosen} (learned={is_learned}, conf={confidence:.2f}, "
            f"mix={mixing:.1%})"
        )
        
        self.last_action = chosen
        return chosen

    def _rule_act(self, confidence, uncertainty, sfl_q=None):
        """
        Rule-based fallback policy.
        
        These rules are defaults when policy hasn't learned yet.
        Over time (after ~100-200 interactions), policy should override these.
        """
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
        """
        Update the metacognitive policy using REINFORCE.
        
        REINFORCE: ∇J ∝ ∇log π(a|s) * R
        Where R is the return (cumulative discounted reward).
        
        High reward → increase probability of chosen action
        Low reward → decrease probability of chosen action
        """
        if self._last_features is None or self._last_action_idx is None:
            return
        
        # Forward pass
        logits = self._policy(self._last_features.unsqueeze(0)).squeeze(0)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # REINFORCE loss: -log_prob(action) * reward
        # (negative because we're doing gradient ascent)
        action_log_prob = log_probs[self._last_action_idx]
        policy_loss = -action_log_prob * reward
        
        # Backprop
        self._optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._policy.parameters(), max_norm=1.0)
        self._optimizer.step()
        
        # Track learning
        self._action_rewards[CANONICAL_ACTIONS[self._last_action_idx]].append(reward)
        self._policy_loss_history.append(policy_loss.item())
        
        logger.debug(
            f"[metacog-learn] action={CANONICAL_ACTIONS[self._last_action_idx]} "
            f"reward={reward:.3f} loss={policy_loss.item():.4f}"
        )
        
        # Clean up
        self._last_features = None
        self._last_action_idx = None

    def should_self_talk(self, confidence, sfl_q=None):
        """Should Adam think out loud?"""
        if confidence < 0.4:
            return "low_confidence"
        if sfl_q is not None and sfl_q < 0.2:
            return "unfamiliar"
        return None

    def record_outcome(self, used_slow_path, reward=None):
        """Record interaction outcome for tracking."""
        self.total_interactions += 1
        if used_slow_path:
            self.slow_path_used += 1
        if reward is not None:
            self.recent_rewards.append(reward)

    def get_mixing_rate(self):
        """What % of actions come from learned policy?"""
        return min(0.9, self.total / (self.total + self._rule_phase_length))

    def stats(self):
        """Return statistics about metacognitive learning."""
        learned_contribution = self.get_mixing_rate()
        
        # Best and worst actions
        best_action = max(
            self._action_rewards.keys(),
            key=lambda a: (
                sum(self._action_rewards[a]) / max(len(self._action_rewards[a]), 1)
                if self._action_rewards[a] else 0
            )
        )
        
        avg_policy_loss = (
            sum(self._policy_loss_history) / len(self._policy_loss_history)
            if self._policy_loss_history else 0.0
        )
        
        return {
            "total_actions": self.total,
            "avg_confidence": (
                sum(self.confidence_history) / max(len(self.confidence_history), 1)
                if self.confidence_history else 0.0
            ),
            "slow_path_rate": (
                self.slow_path_used / max(self.total_interactions, 1)
                if self.total_interactions > 0 else 0.0
            ),
            "last_action": self.last_action,
            "learned_policy_contribution": round(learned_contribution, 2),
            "best_action": best_action,
            "avg_policy_loss": round(avg_policy_loss, 4),
            "action_counts": dict(self._action_counts),
        }
