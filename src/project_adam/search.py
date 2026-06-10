import json
import logging
import threading
from pathlib import Path

from .config import get_memory_dir

logger = logging.getLogger(__name__)

CACHE_PATH = get_memory_dir() / "search_cache.json"


class WebSearch:
    def __init__(self, cache_path=None):
        self.ddgs = None
        self._init_ddgs()
        self._lock = threading.Lock()
        self.cache_path = Path(cache_path) if cache_path else CACHE_PATH
        self.cache = self._load_cache()

    def _init_ddgs(self):
        try:
            from ddgs import DDGS
            self.ddgs = DDGS()
        except ImportError:
            try:
                from duckduckgo_search import DDGS
                self.ddgs = DDGS()
            except ImportError:
                logger.warning("No DDGS available — web search disabled")

    def _load_cache(self):
        try:
            if self.cache_path.exists():
                with open(self.cache_path, encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f)
        except Exception:
            pass

    def search(self, query):
        q = query.lower().strip()
        with self._lock:
            if q in self.cache:
                return self.cache[q]
        result = self._search_ddgs(q) or self._search_wikipedia(q, max_results=3)
        if result:
            with self._lock:
                self.cache[q] = result
                self._save_cache()
        return result

    def _search_ddgs(self, query):
        if self.ddgs is None:
            return None
        try:
            results = list(self.ddgs.text(query, max_results=3))
            if results:
                snippets = [r.get("body", "") for r in results[:3]]
                return " | ".join(s for s in snippets if s)
        except Exception:
            pass
        return None

    def _search_wikipedia(self, query, max_results=3):
        try:
            import requests as req
            headers = {"User-Agent": "ProjectAdam/1.0 (https://github.com/kilvz/Project-Adam)"}
            params = {
                "action": "query", "list": "search",
                "srsearch": query, "format": "json", "srlimit": max_results,
            }
            resp = req.get(
                "https://en.wikipedia.org/w/api.php",
                params=params, headers=headers, timeout=10,
            )
            data = resp.json()
            results = data.get("query", {}).get("search", [])
            if results:
                import re
                return "\n".join(
                    r["title"] + ": " + re.sub(r"<[^>]+>", "", r.get("snippet", ""))
                    for r in results[:max_results]
                )
        except Exception:
            pass
        return None
