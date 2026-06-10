import pytest
import torch

from project_adam import SFLModule, DEVICE

@pytest.fixture
def sfl():
    return SFLModule(n_features=4, lr=0.1).to(DEVICE)

def test_sfl_init(sfl):
    assert sfl.fc.in_features == 4
    assert sfl.fc.out_features == 1
    assert sfl.lr == 0.1

def test_sfl_forward(sfl):
    features = torch.tensor([0.5, 0.3, 0.1, 0.7], dtype=torch.float32, device=DEVICE)
    q = sfl.forward(features)
    assert q.shape == (1,)
    assert q.dtype == torch.float32

def test_sfl_forward_batched(sfl):
    features = torch.randn(3, 4, device=DEVICE)
    q = sfl.forward(features)
    assert q.shape == (3, 1)

def test_sfl_update(sfl):
    features = [0.5, 0.3, 0.1, 0.7]
    reward = 0.8
    loss = sfl.update(features, reward)
    assert isinstance(loss, float)
    assert loss >= 0

def test_sfl_update_negative_reward(sfl):
    features = [0.1, 0.2, 0.3, 0.4]
    loss = sfl.update(features, -0.5)
    assert isinstance(loss, float)
    assert loss >= 0

def test_sfl_update_zero_features(sfl):
    features = [0.0, 0.0, 0.0, 0.0]
    loss = sfl.update(features, 1.0)
    assert isinstance(loss, float)

def test_sfl_q_changes_after_update(sfl):
    features = [0.5, 0.3, 0.1, 0.7]
    q_before = sfl.forward(torch.tensor(features, dtype=torch.float32, device=DEVICE)).item()
    for _ in range(2):
        sfl.update(features, 0.9)
    q_after = sfl.forward(torch.tensor(features, dtype=torch.float32, device=DEVICE)).item()
    assert q_after != q_before

def test_sfl_multiple_updates(sfl):
    for i in range(5):
        features = [i * 0.1, i * 0.2, i * 0.3, i * 0.4]
        loss = sfl.update(features, 0.5)
        assert loss >= 0

def test_sfl_n_features_mismatch():
    sfl = SFLModule(n_features=3)
    features = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
    q = sfl.forward(features)
    assert q.shape == (1,)

def test_sfl_device_matches():
    sfl = SFLModule(n_features=4).to(DEVICE)
    params = list(sfl.parameters())
    assert params[0].device.type == DEVICE
