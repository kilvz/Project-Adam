import pytest
import torch
import math

from project_adam import MetacognitiveController

@pytest.fixture
def meta():
    return MetacognitiveController()

def test_estimate_confidence_high(meta):
    logits = torch.tensor([[10.0, 0.0, 0.0], [8.0, 1.0, 1.0]], dtype=torch.float32)
    confidence, uncertainty = meta.estimate_confidence(logits)
    assert 0.0 < confidence <= 1.0
    assert 0.0 <= uncertainty <= 1.0

def test_estimate_confidence_low(meta):
    logits = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
    confidence, uncertainty = meta.estimate_confidence(logits)
    assert confidence < 0.5
    assert uncertainty > 0.5

def test_estimate_confidence_uniform(meta):
    logits = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
    confidence, uncertainty = meta.estimate_confidence(logits)
    assert confidence < 0.5
    assert uncertainty > 0.5

def test_should_search_low_confidence(meta):
    assert meta.should_search(0.2, threshold=0.3) is True

def test_should_search_high_confidence(meta):
    assert meta.should_search(0.9, threshold=0.3) is False

def test_should_search_tracks_history(meta):
    meta.should_search(0.2)
    meta.should_search(0.8)
    meta.should_search(0.9)
    assert len(meta.recent_confidence) == 3

def test_should_search_bounded_history(meta):
    for i in range(25):
        meta.should_search(0.5)
    assert len(meta.recent_confidence) <= 20

def test_act_proceed_high_confidence(meta):
    action = meta.act(confidence=0.9, uncertainty=0.1)
    assert action == "proceed"

def test_act_search_moderate_confidence(meta):
    action = meta.act(confidence=0.3, uncertainty=0.5)
    assert action == "search"

def test_act_clarify_low_confidence(meta):
    action = meta.act(confidence=0.15, uncertainty=0.8)
    assert action == "clarify"

def test_act_replay_after_consecutive_low(meta):
    for _ in range(6):
        meta.act(confidence=0.2, uncertainty=0.7)
    # After 6 consecutive low, should be "replay"
    assert meta.last_action == "replay"

def test_act_explore_with_low_sfl(meta):
    meta.consecutive_low_confidence = 3
    action = meta.act(confidence=0.3, uncertainty=0.5, sfl_q=-0.5)
    assert action == "explore"

def test_act_explore_high_sfl_no_trigger(meta):
    meta.consecutive_low_confidence = 0
    action = meta.act(confidence=0.3, uncertainty=0.5, sfl_q=0.5)
    assert action == "search"

def test_record_outcome(meta):
    meta.record_outcome(used_slow_path=False)
    assert meta.total_interactions == 1
    assert meta.slow_path_used == 0

def test_record_outcome_slow_path(meta):
    meta.record_outcome(used_slow_path=True)
    assert meta.total_interactions == 1
    assert meta.slow_path_used == 1

def test_stats_basic(meta):
    meta.act(confidence=0.8, uncertainty=0.2)
    meta.record_outcome(used_slow_path=True)
    stats = meta.stats()
    assert "total" in stats
    assert "avg_confidence" in stats
    assert "slow_path_rate" in stats
    assert "last_action" in stats
    assert stats["total"] == 1
    assert stats["slow_path_rate"] == 1.0

def test_stats_zero_division(meta):
    stats = meta.stats()
    assert stats["total"] == 0
    assert stats["slow_path_rate"] == 0.0
    assert stats["avg_confidence"] == 0.0

def test_confidence_resets_on_high(meta):
    for _ in range(3):
        meta.act(confidence=0.2, uncertainty=0.7)
    assert meta.consecutive_low_confidence == 3
    meta.act(confidence=0.9, uncertainty=0.1)
    assert meta.consecutive_low_confidence == 0

def test_estimate_confidence_single_token(meta):
    logits = torch.tensor([[5.0, 3.0, 1.0]], dtype=torch.float32)
    confidence, uncertainty = meta.estimate_confidence(logits)
    assert isinstance(confidence, float)
    assert isinstance(uncertainty, float)

def test_estimate_confidence_no_nan(meta):
    logits = torch.tensor([[1e8, 0.0, 0.0]], dtype=torch.float32)
    confidence, uncertainty = meta.estimate_confidence(logits)
    assert not math.isnan(confidence)
    assert not math.isnan(uncertainty)

def test_act_default_last_action(meta):
    assert meta.last_action == "proceed"

def test_confidence_history_in_stats(meta):
    meta.should_search(0.5)
    meta.should_search(0.6)
    meta.act(confidence=0.5, uncertainty=0.3)
    stats = meta.stats()
    assert len(stats["confidence_history"]) == 2
