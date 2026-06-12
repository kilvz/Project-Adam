import torch
import torch.nn as nn
import torch.nn.functional as F


_POSITIVE_WORDS = {"love", "great", "amazing", "interesting", "cool", "nice",
    "beautiful", "wonderful", "thanks", "good", "excellent", "fantastic",
    "helpful", "perfect", "fun", "awesome", "brilliant", "fascinating"}
_NEGATIVE_WORDS = {"bad", "wrong", "no", "hate", "terrible", "awful", "stupid",
    "boring", "incorrect", "useless", "pointless", "annoying", "horrible",
    "disappointing"}
_POS_REFS = ["this is wonderful", "I love this", "great", "amazing", "fantastic"]
_NEG_REFS = ["this is terrible", "I hate this", "bad", "awful", "horrible"]


class ValueNetwork(nn.Module):
    def __init__(self, n_features=6):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.fc(x)


_N_ACTIONS = 5


class ActorNetwork(nn.Module):
    def __init__(self, n_features=6, n_actions=_N_ACTIONS):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.fc(x)


ALPHA = 1e-3


class TDCore:
    def __init__(self, n_features=6, gamma=0.95, lmbda=0.8, lr=1e-3, actor_lr=1e-4):
        self.value_net = ValueNetwork(n_features)
        self.actor_net = ActorNetwork(n_features)
        self.gamma = gamma
        self.lmbda = lmbda
        self.lr = lr
        self.eligibility = {}
        self.last_features = None
        self.rpe_history = []
        self._rpe_listeners = []
        self._actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=actor_lr)

    def register_rpe_listener(self, fn):
        self._rpe_listeners.append(fn)

    def _broadcast_rpe(self, rpe):
        for fn in self._rpe_listeners:
            try:
                fn(rpe)
            except Exception:
                pass

    def predict(self, features):
        t = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return self.value_net(t).item()

    def get_policy(self, features):
        t = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.actor_net(t).squeeze(0)
            return F.softmax(logits, dim=-1).numpy()

    def update(self, reward, next_features):
        f_now = torch.as_tensor(next_features, dtype=torch.float32).unsqueeze(0)

        if self.last_features is None:
            self.last_features = next_features
            return 0.0

        f_prev = torch.as_tensor(self.last_features, dtype=torch.float32).unsqueeze(0)
        f_prev.requires_grad_(True)
        v_s = self.value_net(f_prev).squeeze(-1)

        with torch.no_grad():
            v_s_prime = self.value_net(f_now).item()

        td_error = reward + self.gamma * v_s_prime - v_s
        rpe = td_error.item()
        self.rpe_history.append(rpe)
        if len(self.rpe_history) > 100:
            self.rpe_history.pop(0)

        grad_v = torch.autograd.grad(v_s, self.value_net.parameters(), create_graph=False)

        if not self.eligibility:
            for i, p in enumerate(self.value_net.parameters()):
                self.eligibility[i] = torch.zeros_like(p)

        for i, g in enumerate(grad_v):
            self.eligibility[i] = self.gamma * self.lmbda * self.eligibility[i] + g.detach()

        delta = td_error.detach()
        with torch.no_grad():
            for i, p in enumerate(self.value_net.parameters()):
                if i in self.eligibility:
                    p.add_(self.lr * delta * self.eligibility[i])

        prev_logits = self.actor_net(f_prev).squeeze(0)
        actor_loss = -F.log_softmax(prev_logits, dim=-1).mean() * delta.detach()
        self._actor_optimizer.zero_grad()
        actor_loss.backward()
        self._actor_optimizer.step()

        self.last_features = next_features
        self._broadcast_rpe(rpe)
        return rpe

    @staticmethod
    def compute_reward(user_input, user_profile=None, embedder=None):
        words = user_input.lower().split()
        if not words:
            return 0.0
        pos = sum(1 for w in words if w.strip(".,!?") in _POSITIVE_WORDS)
        neg = sum(1 for w in words if w.strip(".,!?") in _NEGATIVE_WORDS)
        sentiment = (pos - neg) / max(len(words), 1)
        sentiment = max(-1.0, min(1.0, sentiment))

        if embedder is not None and abs(sentiment) < 0.15 and len(user_input) > 15:
            try:
                q_emb = embedder.encode(user_input, convert_to_numpy=True)
                pos_sim = max(embedder.encode(p, convert_to_numpy=True) @ q_emb for p in _POS_REFS)
                neg_sim = max(embedder.encode(n, convert_to_numpy=True) @ q_emb for n in _NEG_REFS)
                nlu_score = (pos_sim - neg_sim) * 0.5
                nlu_score = max(-0.5, min(0.5, nlu_score))
                sentiment = sentiment * 0.3 + nlu_score * 0.7
            except Exception:
                pass

        sentiment = max(-1.0, min(1.0, sentiment))
        engagement = min(1.0, len(user_input) / 100.0)
        reward = sentiment * 0.6 + engagement * 0.3
        return max(-1.0, min(1.0, reward))

    def reset(self):
        self.last_features = None
        self.eligibility = {}
