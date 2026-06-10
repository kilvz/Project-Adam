import torch
import torch.nn as nn


class SFLModule(nn.Module):
    def __init__(self, n_features=7, lr=0.1):
        super().__init__()
        self.fc = nn.Linear(n_features, 1)
        self.lr = lr
        self.q_history = []

    def forward(self, features):
        return self.fc(features)

    def update(self, features, reward):
        with torch.no_grad():
            features = torch.as_tensor(features, dtype=torch.float32,
                                       device=next(self.parameters()).device)
            q = self.forward(features)
            delta = reward - q.item()
            self.fc.weight.add_(self.lr * delta * features.unsqueeze(0))
            self.fc.bias.add_(self.lr * delta)
            new_q = self.forward(features)
            self.q_history.append(new_q.item())
        return delta

    def compute_temperature(self):
        if not self.q_history:
            return 0.7
        q = self.q_history[-1]
        return max(0.4, min(0.9, 0.7 - q * 0.2))
