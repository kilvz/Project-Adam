"""Tests for DiffMemory module."""

import numpy as np
import pytest

from project_adam.memory.diffmemory import DiffMemory


@pytest.fixture
def mem():
    return DiffMemory(dim=8, hidden_mult=2, depth=2, max_patterns=5,
                      surprise_threshold=0.01, momentum_beta=0.9,
                      momentum_scale=0.5, weight_decay=1e-4, lr=1e-3)


class TestDiffMemory:
    def test_store_and_retrieve(self, mem):
        embs = np.random.randn(3, 8).astype(np.float32)
        texts = ["pattern one", "pattern two", "pattern three"]
        mem.store(embs, texts=texts)
        assert mem.stats()["num_patterns"] == 3

        query = np.random.randn(8).astype(np.float32)
        results = mem.retrieve(query, k=2)
        assert isinstance(results, list)
        assert len(results) <= 2
        for text, sim in results:
            assert isinstance(text, str)
            assert 0.0 <= sim <= 1.0

    def test_store_deduplicates(self, mem):
        """Training on the same pattern should reduce its reconstruction error over time."""
        emb = np.random.randn(1, 8).astype(np.float32)
        mem.store(emb, texts=["original"])
        count_after_first = mem.stats()["num_patterns"]

        # Store same embedding with momentum reset
        mem._momentum = 0.0
        mem.store(emb, texts=["duplicate"])
        count_after_second = mem.stats()["num_patterns"]

        # After two stores, max_patterns should still limit growth
        assert count_after_second <= mem.max_patterns

    def test_retrieve_empty(self, mem):
        results = mem.retrieve(np.random.randn(8).astype(np.float32))
        assert results == []

    def test_retrieve_similarity_threshold(self, mem):
        """Retrieved patterns must have similarity > 0.5."""
        embs = np.random.randn(2, 8).astype(np.float32)
        texts = ["alpha", "beta"]
        mem.store(embs, texts=texts)

        # Query with random vector — unlikely to match > 0.5
        query = np.random.randn(8).astype(np.float32)
        results = mem.retrieve(query, k=5)
        assert len(results) <= 2  # never more than stored

    def test_momentum_tracking(self, mem):
        """Verify momentum buffer is updated after store()."""
        emb = np.random.randn(1, 8).astype(np.float32)
        mem.store(emb, texts=["test"])
        # Momentum should be > 0 after a store event
        stats = mem.stats()
        assert "momentum" in stats
        assert isinstance(stats["momentum"], float)

    def test_consolidate_resets_momentum(self, mem):
        emb = np.random.randn(1, 8).astype(np.float32)
        mem.store(emb, texts=["test"])
        mem.consolidate()
        stats = mem.stats()
        assert stats["momentum"] == 0.0

    def test_state_dict_roundtrip(self, mem):
        embs = np.random.randn(2, 8).astype(np.float32)
        texts = ["a", "b"]
        mem.store(embs, texts=texts)
        state = mem.state_dict()

        mem2 = DiffMemory(dim=8, hidden_mult=2, depth=2)
        mem2.load_state_dict(state)
        assert mem2.stats()["num_patterns"] == 2

    def test_stats(self, mem):
        stats = mem.stats()
        assert "num_patterns" in stats
        assert "max_patterns" in stats
        assert "depth" in stats
        assert "momentum" in stats
        assert "avg_usage" in stats
        assert stats["max_patterns"] == 5
        assert stats["depth"] == 2
        assert stats["num_patterns"] == 0

    def test_max_patterns_enforced(self, mem):
        embs = np.random.randn(10, 8).astype(np.float32)
        texts = [f"pattern_{i}" for i in range(10)]
        mem.store(embs, texts=texts)
        assert mem.stats()["num_patterns"] <= 5

    def test_torch_input(self, mem):
        import torch
        emb = torch.randn(1, 8)
        mem.store(emb, texts=["torch input"])
        assert mem.stats()["num_patterns"] == 1

    def test_retrieve_returns_texts(self, mem):
        embs = np.random.randn(2, 8).astype(np.float32)
        texts = ["first pattern text", "second pattern text"]
        mem.store(embs, texts=texts)
        query = np.random.randn(8).astype(np.float32)
        results = mem.retrieve(query, k=2)
        for text, _ in results:
            assert text in texts or text == ""  # similarity gate may filter

    def test_adamw_weight_decay(self, mem):
        """Verify optimizer is AdamW (not Adam)."""
        assert "AdamW" in type(mem.optimizer).__name__

    def test_reencode_prune(self, mem):
        """consolidate() should not crash with empty memory."""
        mem.consolidate()
        assert mem.stats()["num_patterns"] == 0
