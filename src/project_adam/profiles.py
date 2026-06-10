import time
import threading
from .config import get_memory_dir
from .memory.store import SQLiteStore


class UserProfileManager:
    def __init__(self):
        self.current_name = None
        self._lock = threading.RLock()
        self._store = SQLiteStore(
            "user_profiles",
            path=get_memory_dir() / "user_profiles.pkl",
            pickle_fallback=lambda: {},
        )
        self.profiles = self._store.load(default={})

    def save(self):
        with self._lock:
            self._store.save(self.profiles)

    def get_or_create(self, name):
        name = name.strip().capitalize()
        with self._lock:
            if name not in self.profiles:
                self.profiles[name] = {
                    "name": name,
                    "interaction_count": 0,
                    "avg_sentiment": 0.0,
                    "sentiment_history": [],
                    "topics": {},
                    "rule_weights": {},
                    "custom_rules": [],
                    "first_seen": time.time(),
                    "last_seen": time.time(),
                }
                self.save()
            return self.profiles[name]

    def set_current(self, name):
        with self._lock:
            self.current_name = name
        return self.get_or_create(name)

    def get_current(self):
        with self._lock:
            if self.current_name and self.current_name in self.profiles:
                return self.profiles[self.current_name]
        return None

    def update_after_turn(self, name, user_input, reward, topics):
        profile = self.get_or_create(name)
        profile["interaction_count"] += 1
        profile["last_seen"] = time.time()
        profile["sentiment_history"].append(reward)
        if len(profile["sentiment_history"]) > 20:
            profile["sentiment_history"].pop(0)
        profile["avg_sentiment"] = (
            sum(profile["sentiment_history"])
            / max(len(profile["sentiment_history"]), 1)
        )
        for topic in topics:
            profile["topics"][topic] = profile["topics"].get(topic, 0) + 1

    def list_users(self):
        with self._lock:
            return list(self.profiles.keys())

    def remove(self, name):
        with self._lock:
            self.profiles.pop(name, None)
