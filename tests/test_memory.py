import pytest
import torch
import numpy as np
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from adam_chat import (
    EpisodicMemory, SemanticMemory, NeuralMemory, WorkingMemory,
    MEMORY_DIR,
)


@pytest.fixture(autouse=True)
def temp_memory_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("adam_chat.MEMORY_DIR", tmp_path)
    return tmp_path


def test_working_memory_basic():
    wm = WorkingMemory(max_turns=3)
    assert len(wm.turns) == 0

    wm.add("user", "hello")
    wm.add("assistant", "hi there")
    assert len(wm.turns) == 2

    ctx = wm.get_context()
    assert ctx[0]["role"] == "user"
    assert ctx[1]["content"] == "hi there"


def test_working_memory_bounded():
    wm = WorkingMemory(max_turns=2)
    wm.add("user", "a")
    wm.add("assistant", "b")
    wm.add("user", "c")
    assert len(wm.turns) == 2
    assert wm.turns[0]["content"] == "b"
    assert wm.turns[1]["content"] == "c"


def test_working_memory_get_context_n():
    wm = WorkingMemory(max_turns=5)
    for i in range(5):
        wm.add("user", str(i))
    ctx = wm.get_context(n=2)
    assert len(ctx) == 2


def test_working_memory_clear():
    wm = WorkingMemory(max_turns=3)
    wm.add("user", "hello")
    wm.clear()
    assert len(wm.turns) == 0


@patch("sentence_transformers.SentenceTransformer")
def test_episodic_memory_save_load(mock_st, tmp_path):
    mock_st.return_value = MagicMock()
    mem = EpisodicMemory()
    mem.path = tmp_path / "episodic.pkl"
    mem.embedder = None
    mem.episodes = []

    mem.add("hello world", reward=0.5)
    assert len(mem.episodes) == 1
    assert mem.episodes[0]["text"] == "hello world"
    assert mem.episodes[0]["reward"] == 0.5

    mem2 = EpisodicMemory()
    mem2.path = tmp_path / "episodic.pkl"
    mem2.embedder = None
    mem2.load()
    assert len(mem2.episodes) == 1


def test_episodic_memory_no_embedder(tmp_path):
    mem = EpisodicMemory()
    mem.path = tmp_path / "episodic.pkl"
    mem.embedder = None
    mem.episodes = []

    mem.add("test entry", reward=1.0)
    assert "emb" not in mem.episodes[0]


@patch("sentence_transformers.SentenceTransformer")
def test_episodic_memory_with_embedder(mock_st):
    mock_emb = MagicMock()
    mock_emb.encode.return_value = np.array([0.1, 0.2, 0.3])
    mock_st.return_value = mock_emb

    mem = EpisodicMemory()
    mem.path = Path("/tmp/_test_ep_mem.pkl")
    mem.embedder = mock_emb
    mem.episodes = []

    mem.add("test with embedding", reward=0.8)
    assert "emb" in mem.episodes[0]


@patch("sentence_transformers.SentenceTransformer")
def test_episodic_search(mock_st):
    mock_emb = MagicMock()
    emb1 = np.array([1.0, 0.0, 0.0])
    emb2 = np.array([0.0, 1.0, 0.0])
    mock_emb.encode.side_effect = [emb1, emb2, emb1]
    mock_st.return_value = mock_emb

    mem = EpisodicMemory()
    mem.path = Path("/tmp/_test_ep_search.pkl")
    mem.embedder = mock_emb
    mem.episodes = []

    mem.add("cats are great", reward=0.9)
    mem.add("dogs are fun", reward=0.5)

    mock_emb.encode.side_effect = [np.array([1.0, 0.0, 0.0])]
    results = mem.search("cats", k=2)
    assert len(results) >= 1
    assert "cats" in results[0][0]


def test_episodic_search_no_embedder(tmp_path):
    mem = EpisodicMemory()
    mem.path = tmp_path / "episodic.pkl"
    mem.embedder = None
    mem.episodes = []
    mem.add("test", reward=0.0)
    results = mem.search("test")
    assert results == []


@patch("sentence_transformers.SentenceTransformer")
def test_semantic_memory_save_load(mock_st, tmp_path):
    mock_st.return_value = MagicMock()
    mem = SemanticMemory()
    mem.path = tmp_path / "semantic.pkl"
    mem.embedder = None
    mem.schemas = {}

    mem.add("likes", "I like pizza")
    assert "likes" in mem.schemas
    assert "I like pizza" in mem.schemas["likes"]["facts"]

    mem2 = SemanticMemory()
    mem2.path = tmp_path / "semantic.pkl"
    mem2.embedder = None
    mem2.load()
    assert "likes" in mem2.schemas


def test_semantic_memory_no_embedder(tmp_path):
    mem = SemanticMemory()
    mem.path = tmp_path / "semantic.pkl"
    mem.embedder = None
    mem.schemas = {}

    mem.add("name", "Kilv")
    assert mem.schemas["name"]["emb"] is None


@patch("sentence_transformers.SentenceTransformer")
def test_semantic_retrieve(mock_st):
    mock_emb = MagicMock()
    mock_emb.encode.return_value = np.array([1.0, 0.0, 0.0])
    mock_st.return_value = mock_emb

    mem = SemanticMemory()
    mem.path = Path("/tmp/_test_sem_ret.pkl")
    mem.embedder = mock_emb
    mem.schemas = {}

    mem.add("likes", "I like programming")
    results = mem.retrieve("coding", k=3)
    assert isinstance(results, list)


def test_semantic_retrieve_empty(tmp_path):
    mem = SemanticMemory()
    mem.path = tmp_path / "semantic.pkl"
    mem.embedder = None
    mem.schemas = {}
    results = mem.retrieve("anything")
    assert results == []


def test_neural_memory_forward():
    nm = NeuralMemory(input_dim=32, mem_dim=16, mem_slots=4, dtype=torch.float32)
    x = torch.randn(2, 5, 32)
    out = nm.forward(x)
    assert out.shape == (2, 5, 16)


def test_neural_memory_learn():
    nm = NeuralMemory(input_dim=32, mem_dim=16, mem_slots=4, dtype=torch.float32)
    x = torch.randn(2, 5, 32)
    loss = nm.learn(x, lr=1e-3, steps=2)
    assert isinstance(loss, float)
    assert loss > 0


def test_neural_memory_learn_reduces_loss():
    nm = NeuralMemory(input_dim=8, mem_dim=4, mem_slots=2, dtype=torch.float32)
    x = torch.randn(1, 3, 8)
    losses = []
    for _ in range(5):
        loss = nm.learn(x, lr=1e-2, steps=1)
        losses.append(loss)
    assert losses[-1] <= losses[0] + 1e-6


def test_neural_memory_different_dtype():
    nm = NeuralMemory(input_dim=16, mem_dim=8, mem_slots=4, dtype=torch.float64)
    x = torch.randn(1, 3, 16, dtype=torch.float64)
    out = nm.forward(x)
    assert out.dtype == torch.float64


def test_neural_memory_no_grad_learn():
    nm = NeuralMemory(input_dim=16, mem_dim=8, mem_slots=4, dtype=torch.float32)
    x = torch.randn(1, 3, 16)
    loss = nm.learn(x, steps=1)
    assert loss > 0


def test_working_memory_types():
    wm = WorkingMemory(max_turns=4)
    wm.add("user", "hello")
    wm.add("assistant", "world")
    for t in wm.turns:
        assert "role" in t
        assert "content" in t
