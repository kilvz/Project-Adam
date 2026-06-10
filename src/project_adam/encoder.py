import torch
import torch.nn as nn
import torch.nn.functional as F


class SensoryEncoder(nn.Module):
    def __init__(self, input_dim=896, latent_dim=128, beta=0.001, dtype=torch.float32):
        super().__init__()
        self.beta = beta
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
        return z, recon_loss + self.beta * kl_loss

    def vae_loss(self, x):
        _, loss = self.forward(x)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
