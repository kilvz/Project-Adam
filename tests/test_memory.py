import torch
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from project_adam import (
    WorkingMemory, EpisodicMemory, SemanticMemory, SQLiteStore,
)

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

    # E23: persistence now via SQLite; load is automatic in __init__
    mem2 = EpisodicMemory()
    assert len(mem2.episodes) >= 1

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
    mem.embedder = None
    mem.schemas = {}
    mem._next_id = 0

    mem.add("likes", "I like pizza")
    # schemas are now keyed by numeric ID; find by category
    schema = next(s for s in mem.schemas.values() if s["category"] == "likes")
    assert "I like pizza" in schema["facts"]

    # persistence via SQLite; load is automatic in __init__
    mem2 = SemanticMemory()
    schema2 = next((s for s in mem2.schemas.values() if s["category"] == "likes"), None)
    assert schema2 is not None
    assert "I like pizza" in schema2.get("facts", [])


def test_semantic_memory_no_embedder(tmp_path):
    mem = SemanticMemory()
    mem.embedder = None
    mem.schemas = {}
    mem._next_id = 0

    mem.add("name", "Kilv")
    schema = next(s for s in mem.schemas.values() if s["category"] == "name")
    assert schema["emb"] is None

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



def test_working_memory_types():
    wm = WorkingMemory(max_turns=4)
    wm.add("user", "hello")
    wm.add("assistant", "world")
    for t in wm.turns:
        assert "role" in t
        assert "content" in t
