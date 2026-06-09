import pytest
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from adam_chat import WebSearch


@pytest.fixture
def search(tmp_path, monkeypatch):
    monkeypatch.setattr("adam_chat.MEMORY_DIR", tmp_path)
    ws = WebSearch()
    ws.cache_path = os.path.join(str(tmp_path), "search_cache.json")
    ws.cache = {}
    return ws


def test_cache_hit(search):
    search.cache["hello world"] = "cached result"
    result = search.search("Hello World")
    assert result == "cached result"


def test_cache_miss(search):
    result = search.search("some unique query that wont be found")
    assert result is None or isinstance(result, str)


def test_cache_persists(tmp_path):
    cache_file = tmp_path / "search_cache.json"
    cache_file.write_text(json.dumps({"test": "value"}), encoding="utf-8")
    ws = WebSearch()
    ws.cache_path = str(cache_file)
    ws._load_cache()
    assert ws.cache.get("test") == "value"


def test_cache_save_creates_file(search):
    search.cache["key"] = "value"
    search._save_cache()
    assert os.path.exists(search.cache_path)
    with open(search.cache_path) as f:
        data = json.load(f)
    assert data["key"] == "value"


def test_cache_save_failure_does_not_raise(search):
    search.cache_path = "/nonexistent_dir/cache.json"
    search.cache["key"] = "value"
    search._save_cache()


def test_load_cache_file_not_found(search):
    search.cache_path = "/nonexistent/cache.json"
    search._load_cache()
    assert search.cache == {}


def test_load_cache_invalid_json(tmp_path, search):
    bad_file = tmp_path / "bad_cache.json"
    bad_file.write_text("not json", encoding="utf-8")
    search.cache_path = str(bad_file)
    search._load_cache()
    assert search.cache == {}


@patch("adam_chat.WebSearch._search_wikipedia")
def test_fallback_to_wikipedia(mock_wiki, search):
    search.searcher = None
    mock_wiki.return_value = "Wikipedia result"
    result = search.search("test query")
    assert result == "Wikipedia result"
    mock_wiki.assert_called_once()


@patch("adam_chat.WebSearch._search_wikipedia")
def test_ddgs_fallback_on_error(mock_wiki, search):
    mock_searcher = MagicMock()
    mock_searcher.text.side_effect = Exception("DDGS failed")
    search.searcher = mock_searcher
    mock_wiki.return_value = "Wikipedia fallback"
    result = search.search("test")
    assert result == "Wikipedia fallback"


@patch("adam_chat.WebSearch._search_wikipedia")
def test_both_fail_return_none(mock_wiki, search):
    search.searcher = None
    mock_wiki.return_value = None
    result = search.search("something random")
    assert result is None


@patch("requests.get")
def test_wikipedia_search(mock_get, search):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "query": {
            "search": [
                {"title": "AI", "snippet": "Artificial intelligence is <b>intelligence</b>."},
            ]
        }
    }
    mock_get.return_value = mock_resp
    result = search._search_wikipedia("AI")
    assert "AI" in result
    assert "Artificial intelligence" in result


@patch("requests.get")
def test_wikipedia_search_empty(mock_get, search):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"query": {"search": []}}
    mock_get.return_value = mock_resp
    result = search._search_wikipedia("xyznonexistent12345")
    assert result is None


@patch("requests.get")
def test_wikipedia_search_error(mock_get, search):
    mock_get.side_effect = Exception("network error")
    result = search._search_wikipedia("test")
    assert result is None


def test_searcher_init_no_ddgs(search):
    assert hasattr(search, "searcher")


def test_cache_key_lowercase(search):
    search.cache["hello world"] = "result"
    r1 = search.search("Hello World")
    r2 = search.search("hello world")
    assert r1 == r2 == "result"
