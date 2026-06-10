import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralMemory(nn.Module):
    def __init__(self, input_dim=896, mem_dim=256, mem_slots=32, dtype=torch.float32):
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
        self.train()
        for _ in range(steps):
            q = self.query(h)
            k = self.key(self.memory.expand(x.size(0), -1, -1))
            v = self.value(self.memory.expand(x.size(0), -1, -1))
            attn = torch.softmax(q @ k.transpose(-2, -1) / (self.mem_dim ** 0.5), dim=-1)
            out = attn @ v
            g = torch.sigmoid(self.gate(torch.cat([h, out], dim=-1)))
            recon = self.output(h * g + out * (1 - g))
            loss = F.mse_loss(recon, h)
            loss.backward()
            optim.step()
            optim.zero_grad()
        self.eval()
        return loss.item()

    def read(self, query_emb):
        with torch.no_grad():
            return self.forward(query_emb)

    def consolidate(self):
        with torch.no_grad():
            self.memory.data = torch.randn_like(self.memory) * 0.02
