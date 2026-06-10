import pytest
import torch

from project_adam import SensoryEncoder

@pytest.fixture
def encoder():
    return SensoryEncoder(input_dim=32, latent_dim=8, dtype=torch.float32)

def test_encoder_init(encoder):
    assert encoder.encoder[0].in_features == 32
    assert encoder.encoder[0].out_features == 256
    assert encoder.decoder[0].in_features == 8
    assert encoder.decoder[0].out_features == 256

def test_encoder_forward_shape(encoder):
    x = torch.randn(4, 32)
    z, loss = encoder.forward(x)
    assert z.shape == (4, 8)
    assert loss.shape == ()

def test_encoder_forward_single_sample(encoder):
    x = torch.randn(1, 32)
    z, loss = encoder.forward(x)
    assert z.shape == (1, 8)

def test_encoder_forward_batched(encoder):
    x = torch.randn(16, 32)
    z, loss = encoder.forward(x)
    assert z.shape == (16, 8)

def test_encoder_forward_loss_differentiable(encoder):
    x = torch.randn(4, 32)
    z, loss = encoder.forward(x)
    assert loss.requires_grad
    loss.backward()
    for param in encoder.parameters():
        assert param.grad is not None

def test_vae_loss_returns_float(encoder):
    x = torch.randn(4, 32)
    loss = encoder.vae_loss(x)
    assert isinstance(loss, float)
    assert loss > 0

def test_vae_loss_optimizes(encoder):
    x = torch.randn(8, 32)
    losses = []
    for _ in range(20):
        loss = encoder.vae_loss(x)
        losses.append(loss)
    assert losses[-1] <= losses[0] + 0.5

def test_encoder_reconstruction(encoder):
    x = torch.randn(4, 32)
    z, _ = encoder.forward(x)
    x_recon = encoder.decoder(z)
    assert x_recon.shape == (4, 32)

def test_encoder_different_latent_dim():
    enc = SensoryEncoder(input_dim=64, latent_dim=16, dtype=torch.float32)
    x = torch.randn(2, 64)
    z, loss = enc.forward(x)
    assert z.shape == (2, 16)

def test_encoder_dtype_preserved():
    enc = SensoryEncoder(input_dim=16, latent_dim=4, dtype=torch.float64)
    x = torch.randn(2, 16, dtype=torch.float64)
    z, loss = enc.forward(x)
    assert z.dtype == torch.float64
    assert loss.dtype == torch.float64

def test_encoder_reparameterization(encoder):
    x = torch.randn(4, 32)
    z1, _ = encoder.forward(x)
    z2, _ = encoder.forward(x)
    diff = (z1 - z2).abs().mean().item()
    assert diff > 1e-6

def test_encoder_mu_logvar_separation():
    enc = SensoryEncoder(input_dim=16, latent_dim=4, dtype=torch.float32)
    x = torch.randn(1, 16)
    mu_logvar = enc.encoder(x)
    mu, logvar = mu_logvar.chunk(2, dim=-1)
    assert mu.shape == (1, 4)
    assert logvar.shape == (1, 4)

def test_encoder_optimizer_steps(encoder):
    old_w = encoder.encoder[0].weight.clone()
    x = torch.randn(4, 32)
    for _ in range(3):
        encoder.vae_loss(x)
    new_w = encoder.encoder[0].weight
    assert not torch.allclose(old_w, new_w)

def test_encoder_zero_input(encoder):
    x = torch.zeros(2, 32)
    z, loss = encoder.forward(x)
    assert z.shape == (2, 8)
    assert not torch.isnan(loss)

def test_encoder_grad_enabled(encoder):
    x = torch.randn(2, 32)
    z, loss = encoder.forward(x)
    assert loss.requires_grad
