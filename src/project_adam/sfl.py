import torch
import torch.nn as nn
import torch.nn.functional as F


class SFLModule(nn.Module):
    def __init__(self, n_features=4, lr=0.01):
        super().__init__()
        self.fc = nn.Linear(n_features, 1)
        self.lr = lr
        self.q_history = []
        self.confidence_history = []

    def forward(self, features):
        return self.fc(features)

    def update(self, features, reward):
        features = torch.as_tensor(features, dtype=torch.float32, device=next(self.parameters()).device)
        q = self.forward(features)
        target = torch.as_tensor([reward], dtype=torch.float32, device=q.device)
        loss = F.mse_loss(q, target)
        loss.backward()
        with torch.no_grad():
            self.fc.weight -= self.lr * self.fc.weight.grad
            self.fc.bias -= self.lr * self.fc.bias.grad
        self.fc.zero_grad()
        self.q_history.append(q.item())
        return loss.item()

    def select_action(self, temperature=1.0):
        if not self.q_history:
            return 0.7
        q = self.q_history[-1]
        t = max(0.4, min(0.9, 0.7 - q * 0.2))
        return t
