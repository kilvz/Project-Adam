import pytest
import time
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from adam_chat import UserProfileManager, compute_implicit_reward, extract_topics


@pytest.fixture
def manager(tmp_path, monkeypatch):
    monkeypatch.setattr("adam_chat.MEMORY_DIR", tmp_path)
    prof_dir = tmp_path / "user_profiles.pkl"
    m = UserProfileManager()
    m.path = prof_dir
    m.profiles = {}
    return m


def test_get_or_create_new(manager):
    profile = manager.get_or_create("Alice")
    assert profile["name"] == "Alice"
    assert profile["interaction_count"] == 0
    assert profile["avg_sentiment"] == 0.0
    assert profile["sentiment_history"] == []
    assert profile["topics"] == {}
    assert "first_seen" in profile
    assert "last_seen" in profile


def test_get_or_create_existing(manager):
    profile1 = manager.get_or_create("Bob")
    profile2 = manager.get_or_create("Bob")
    assert profile1 is profile2


def test_get_or_create_case_insensitive(manager):
    p1 = manager.get_or_create("alice")
    p2 = manager.get_or_create("Alice")
    assert p1 is p2


def test_set_current(manager):
    profile = manager.set_current("Charlie")
    assert manager.current_name == "Charlie"
    assert profile["name"] == "Charlie"


def test_get_current(manager):
    assert manager.get_current() is None
    manager.set_current("Dave")
    assert manager.get_current()["name"] == "Dave"


def test_list_users(manager):
    manager.get_or_create("Alice")
    manager.get_or_create("Bob")
    users = manager.list_users()
    assert "Alice" in users
    assert "Bob" in users


def test_update_after_turn(manager):
    profile = manager.get_or_create("Eve")
    manager.update_after_turn("Eve", "I love this", "", 0.8, ["AI", "tech"])
    assert profile["interaction_count"] == 1
    assert profile["total_interactions"] == 1
    assert profile["avg_sentiment"] == 0.8
    assert profile["topics"]["AI"] == 1
    assert profile["topics"]["tech"] == 1


def test_update_after_turn_avg_sentiment(manager):
    manager.update_after_turn("Frank", "good", "", 0.5, [])
    manager.update_after_turn("Frank", "bad", "", -0.3, [])
    profile = manager.profiles["Frank"]
    assert abs(profile["avg_sentiment"] - 0.1) < 1e-6


def test_sentiment_history_bounded(manager):
    profile = manager.get_or_create("Grace")
    for i in range(25):
        manager.update_after_turn("Grace", "msg", "", 0.5, [])
    assert len(profile["sentiment_history"]) <= 20


def test_adopted_phrases(manager):
    manager.update_after_turn("Helen", "hello how are you doing today", "", 0.0, [])
    profile = manager.profiles["Helen"]
    assert len(profile["adopted_phrases"]) > 0
    assert profile["adopted_phrases"]["hello how are"]["count"] >= 1


def test_remove(manager):
    manager.get_or_create("Isaac")
    assert "Isaac" in manager.profiles
    manager.remove("Isaac")
    assert "Isaac" not in manager.profiles


def test_remove_nonexistent(manager):
    manager.remove("Nobody")
    assert "Nobody" not in manager.profiles


def test_save_load(tmp_path):
    m1 = UserProfileManager()
    m1.path = tmp_path / "user_profiles.pkl"
    m1.profiles = {}
    m1.get_or_create("Alice")
    m1.get_or_create("Bob")

    m2 = UserProfileManager()
    m2.path = tmp_path / "user_profiles.pkl"
    m2.profiles = {}
    m2.load()
    assert "Alice" in m2.profiles
    assert "Bob" in m2.profiles


def test_update_topics_accumulates(manager):
    manager.update_after_turn("Jack", "I love AI", "", 0.3, ["AI"])
    manager.update_after_turn("Jack", "AI is great", "", 0.5, ["AI"])
    assert manager.profiles["Jack"]["topics"]["AI"] == 2


def test_last_seen_updated(manager):
    t1 = time.time()
    manager.update_after_turn("Kate", "hi", "", 0.0, [])
    t2 = manager.profiles["Kate"]["last_seen"]
    assert t2 >= t1


def test_phrase_preferences_initialized(manager):
    profile = manager.get_or_create("Leo")
    assert "phrase_preferences" in profile
    assert "openings" in profile["phrase_preferences"]
    assert "closings" in profile["phrase_preferences"]
