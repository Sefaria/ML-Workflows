import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional


_cache_lock = threading.RLock()
_initialized_paths: set[str] = set()


def _cache_key(prompt: str, llm_string: str) -> str:
    return hashlib.sha256(f"{llm_string}\n{prompt}".encode("utf-8")).hexdigest()


def _connect(cache_path: str) -> sqlite3.Connection:
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(cache_path, timeout=60.0)
    connection.execute("PRAGMA busy_timeout = 60000")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    return connection


def configure_cache(cache_path: str) -> None:
    with _cache_lock:
        if cache_path in _initialized_paths:
            return
        with _connect(cache_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    cache_key TEXT PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    llm_string TEXT NOT NULL,
                    response TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_embedding_cache_llm ON embedding_cache(llm_string)"
            )
        _initialized_paths.add(cache_path)


def cache_lookup(prompt: str, llm_string: str, cache_path: str) -> Optional[list[float]]:
    configure_cache(cache_path)
    key = _cache_key(prompt, llm_string)
    with _cache_lock:
        with _connect(cache_path) as connection:
            row = connection.execute(
                "SELECT response FROM embedding_cache WHERE cache_key = ?",
                (key,),
            ).fetchone()
    if row is None:
        return None
    return json.loads(row[0])


def cache_update(prompt: str, llm_string: str, values: list[float], cache_path: str) -> None:
    configure_cache(cache_path)
    key = _cache_key(prompt, llm_string)
    response = json.dumps(values)
    with _cache_lock:
        with _connect(cache_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO embedding_cache (cache_key, prompt, llm_string, response)
                VALUES (?, ?, ?, ?)
                """,
                (key, prompt, llm_string, response),
            )
