import sqlite3
import threading
import pickle
from pathlib import Path
from ..config import get_memory_dir


def _init_db():
    db_path = get_memory_dir() / "memory.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("""CREATE TABLE IF NOT EXISTS kv (
        key TEXT PRIMARY KEY, value BLOB
    )""")
    conn.commit()
    return conn


class SQLiteStore:
    def __init__(self, key, path=None, pickle_fallback=None):
        self.key = key
        self.pickle_path = Path(path) if path else None
        self.pickle_fallback = pickle_fallback
        self._conn = _init_db()
        self._lock = threading.Lock()

    def load(self, default=None):
        with self._lock:
            cur = self._conn.execute("SELECT value FROM kv WHERE key=?", (self.key,))
            row = cur.fetchone()
            if row:
                return pickle.loads(row[0])
        if self.pickle_path and self.pickle_path.exists():
            with open(self.pickle_path, "rb") as f:
                data = pickle.load(f)
            self.save(data)
            return data
        if self.pickle_fallback:
            return self.pickle_fallback()
        return default if default is not None else {}

    def save(self, data):
        with self._lock:
            blob = pickle.dumps(data)
            self._conn.execute("REPLACE INTO kv (key, value) VALUES (?,?)", (self.key, blob))
            self._conn.commit()

    def close(self):
        self._conn.close()
