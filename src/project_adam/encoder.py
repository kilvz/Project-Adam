import torch
import torch.nn as nn
import torch.nn.functional as F


_SPARSITY_KEEP_FRAC = 0.1


class SensoryEncoder(nn.Module):
    def __init__(self, input_dim=896, latent_dim=128, beta=0.001, dtype=torch.float32):
        super().__init__()
        self.beta = beta
        self.sparsity_weight = 1e-3
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256, dtype=dtype),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2, dtype=dtype),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256, dtype=dtype),
            nn.ReLU(),
            nn.Linear(256, input_dim, dtype=dtype),
        )
        self.prior_mu = nn.Parameter(torch.zeros(1, latent_dim, dtype=dtype))
        self.prior_logvar = nn.Parameter(torch.zeros(1, latent_dim, dtype=dtype))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def _kl_with_learned_prior(self, mu, logvar):
        p_mu = self.prior_mu
        p_logvar = self.prior_logvar
        p_var = torch.exp(p_logvar)
        q_var = torch.exp(logvar)
        kl = 0.5 * (q_var / p_var + (mu - p_mu).pow(2) / p_var - 1 + p_logvar - logvar)
        return kl.sum(dim=-1).mean()

    def _topk_sparse(self, z):
        k = max(1, int(z.shape[-1] * _SPARSITY_KEEP_FRAC))
        vals, _ = torch.topk(torch.abs(z), k, dim=-1)
        threshold = vals[..., -1].unsqueeze(-1)
        mask = (torch.abs(z) >= threshold).float()
        return z * mask

    def forward(self, x):
        mu_logvar = self.encoder(x)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        z = self._topk_sparse(z)
        x_recon = self.decoder(z)
        recon_loss = F.mse_loss(x_recon, x.detach())
        kl_loss = self._kl_with_learned_prior(mu, logvar)
        sparsity_loss = self.sparsity_weight * torch.abs(z).mean()
        return z, recon_loss + self.beta * kl_loss + sparsity_loss

    def compute_loss(self, x, rpe=1.0):
        _, vae_loss_val = self.forward(x)
        task_loss = max(0.0, -rpe)
        return vae_loss_val + task_loss

    def train_step(self, x, rpe=1.0):
        loss = self.compute_loss(x, rpe)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def vae_loss(self, x, rpe_weight=1.0):
        return self.train_step(x, rpe_weight)
