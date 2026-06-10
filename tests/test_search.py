import pytest
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from project_adam import WebSearch

@pytest.fixture
def search(tmp_path):
    ws = WebSearch()
    ws.cache = {}
    return ws

def test_cache_hit(search):
    search.cache["ddg:hello world"] = "cached result"
    result = search.search("Hello World")
    assert result == "cached result"

def test_cache_miss(search):
    result = search.search("some unique query that wont be found")
    assert result is None or isinstance(result, str)

def test_cache_persists(tmp_path):
    cache_file = tmp_path / "search_cache.json"
    cache_file.write_text(json.dumps({"test": "value"}), encoding="utf-8")
    ws = WebSearch(cache_path=cache_file)
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
    search.cache_path = Path("/nonexistent_dir/cache.json")
    search.cache["key"] = "value"
    search._save_cache()

def test_load_cache_file_not_found(search):
    search.cache_path = Path("/nonexistent/cache.json")
    search._load_cache()
    assert search.cache == {}

def test_load_cache_invalid_json(tmp_path, search):
    bad_file = tmp_path / "bad_cache.json"
    bad_file.write_text("not json", encoding="utf-8")
    search.cache_path = bad_file
    search._load_cache()
    assert search.cache == {}

@patch("project_adam.WebSearch._search_wikipedia")
def test_no_fallback_to_wikipedia(mock_wiki, search):
    search.ddgs = None
    mock_wiki.return_value = "Wikipedia result"
    result = search.search("test query")
    assert result is None
    mock_wiki.assert_not_called()

@patch("project_adam.WebSearch._search_wikipedia")
def test_ddgs_returns_none_on_error(mock_wiki, search):
    mock_ddgs = MagicMock()
    mock_ddgs.text.side_effect = Exception("DDGS failed")
    search.ddgs = mock_ddgs
    mock_wiki.return_value = "Wikipedia fallback"
    result = search.search("test")
    assert result is None

@patch("project_adam.WebSearch._search_wikipedia")
def test_both_fail_return_none(mock_wiki, search):
    search.ddgs = None
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
    assert hasattr(search, "ddgs")

def test_cache_key_lowercase(search):
    search.cache["ddg:hello world"] = "result"
    r1 = search.search("Hello World")
    r2 = search.search("hello world")
    assert r1 == r2 == "result"

def test_search_knowledge_calls_wikipedia(search):
    search.ddgs = None
    search._search_wikipedia = MagicMock(return_value="Wiki result")
    result = search.search_knowledge("some topic")
    assert result == "Wiki result"
    search._search_wikipedia.assert_called_once_with("some topic", 3)

def test_search_knowledge_cache_hit(search):
    search.cache["wiki:python"] = "cached wiki result"
    search._search_wikipedia = MagicMock(return_value="new result")
    result = search.search_knowledge("Python")
    assert result == "cached wiki result"
    search._search_wikipedia.assert_not_called()

def test_search_knowledge_returns_none(search):
    search._search_wikipedia = MagicMock(return_value=None)
    result = search.search_knowledge("xyznonexistent12345")
    assert result is None

def test_search_and_knowledge_independent_cache(search):
    search._search_ddgs = MagicMock(return_value="ddg result")
    search._search_wikipedia = MagicMock(return_value="wiki result")
    ddg_result = search.search("test query")
    wiki_result = search.search_knowledge("test query")
    assert ddg_result == "ddg result"
    assert wiki_result == "wiki result"
    assert "ddg:test query" in search.cache
    assert "wiki:test query" in search.cache
