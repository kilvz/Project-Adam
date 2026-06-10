import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from project_adam.api import app


@pytest.fixture(autouse=True)
def mock_agent():
    agent = MagicMock()
    agent.chat.return_value = "Hello from Adam."
    agent._current_user = None

    agent.user_profiles.profiles = {
        "Alice": {
            "name": "Alice",
            "interaction_count": 5,
            "avg_sentiment": 0.3,
            "topics": {"AI": 3, "music": 2},
        },
        "Bob": {
            "name": "Bob",
            "interaction_count": 2,
            "avg_sentiment": -0.1,
            "topics": {"cooking": 1},
        },
    }

    agent.episodic_memory.search.return_value = [
        ("hello world", 0.95, 0.5),
        ("test entry", 0.80, 0.3),
    ]

    agent.semantic_memory.retrieve.return_value = [
        ("likes", ["I like pizza", "I like coding"], 0.92),
        ("dislikes", ["I hate spam"], 0.71),
    ]

    with patch("project_adam.api.get_agent", return_value=agent):
        yield agent


class TestHealth:
    def test_health_ok(self):
        with TestClient(app) as client:
            resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestChat:
    def test_chat_returns_reply(self, mock_agent):
        with TestClient(app) as client:
            resp = client.post("/chat", json={"message": "hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["reply"] == "Hello from Adam."
        mock_agent.chat.assert_called_once_with("hello")

    def test_chat_with_user_id(self, mock_agent):
        with TestClient(app) as client:
            resp = client.post("/chat", json={"message": "hi", "user_id": "Alice"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["reply"] == "Hello from Adam."

    def test_chat_returns_user_id_from_agent(self, mock_agent):
        mock_agent._current_user = "Alice"
        with TestClient(app) as client:
            resp = client.post("/chat", json={"message": "hello"})
        assert resp.status_code == 200
        assert resp.json()["user_id"] == "Alice"


class TestChatStream:
    def test_chat_stream_yields_tokens(self, mock_agent):
        def side_effect(msg, token_callback=None):
            if token_callback:
                token_callback("Hel")
                token_callback("lo")
                token_callback("!")
            return "Hello!"

        mock_agent.chat.side_effect = side_effect

        with TestClient(app) as client:
            resp = client.post("/chat/stream", json={"message": "hi"})
        assert resp.status_code == 200
        lines = [line for line in resp.iter_lines() if line]
        tokens = []
        for line in lines:
            if line.startswith("data: "):
                tokens.append(json.loads(line[6:]))
        assert {"token": "Hel"} in tokens
        assert {"token": "lo"} in tokens
        assert {"token": "!"} in tokens

    def test_chat_stream_error(self, mock_agent):
        mock_agent.chat.side_effect = RuntimeError("boom")
        with TestClient(app) as client:
            resp = client.post("/chat/stream", json={"message": "x"})
        lines = [line for line in resp.iter_lines() if line]
        errors = [line for line in lines if '"error"' in line]
        assert len(errors) >= 1


class TestUsers:
    def test_list_users(self, mock_agent):
        with TestClient(app) as client:
            resp = client.get("/users")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        names = [u["name"] for u in data]
        assert "Alice" in names
        assert "Bob" in names

    def test_get_user_found(self, mock_agent):
        with TestClient(app) as client:
            resp = client.get("/users/Alice")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Alice"
        assert data["interaction_count"] == 5

    def test_get_user_not_found(self, mock_agent):
        with TestClient(app) as client:
            resp = client.get("/users/Unknown")
        assert resp.status_code == 404


class TestMemory:
    def test_episodic_search(self, mock_agent):
        with TestClient(app) as client:
            resp = client.get("/memory/episodic", params={"query": "hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["text"] == "hello world"
        assert data[0]["similarity"] == 0.95

    def test_episodic_search_custom_k(self, mock_agent):
        with TestClient(app) as client:
            resp = client.get("/memory/episodic", params={"query": "test", "k": 1})
        assert resp.status_code == 200
        mock_agent.episodic_memory.search.assert_called_with("test", k=1)

    def test_semantic_search(self, mock_agent):
        with TestClient(app) as client:
            resp = client.get("/memory/semantic", params={"query": "likes"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2
        assert data[0]["category"] == "likes"

    def test_semantic_search_custom_k(self, mock_agent):
        with TestClient(app) as client:
            resp = client.get("/memory/semantic", params={"query": "x", "k": 1})
        assert resp.status_code == 200
        mock_agent.semantic_memory.retrieve.assert_called_with("x", k=1)
