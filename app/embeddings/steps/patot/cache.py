import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional


_cache_lock = threading.RLock()
_initialized_paths: set[str] = set()


def _cache_key(prompt: str, llm_string: str) -> str:
    return hashlib.sha256(f"{llm_string}\n{prompt}".encode("utf-8")).hexdigest()


def _access_marker() -> str:
    return str(time.time_ns())


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
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TEXT NOT NULL DEFAULT '0'
                )
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(embedding_cache)").fetchall()
            }
            if "last_accessed" not in columns:
                connection.execute(
                    "ALTER TABLE embedding_cache ADD COLUMN last_accessed TEXT"
                )
                connection.execute(
                    "UPDATE embedding_cache SET last_accessed = COALESCE(last_accessed, '0')"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_embedding_cache_llm ON embedding_cache(llm_string)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_embedding_cache_last_accessed "
                "ON embedding_cache(last_accessed)"
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
            if row is not None:
                connection.execute(
                    "UPDATE embedding_cache SET last_accessed = ? WHERE cache_key = ?",
                    (_access_marker(), key),
                )
    if row is None:
        return None
    return json.loads(row[0])


def _enforce_max_entries(connection: sqlite3.Connection, max_entries: Optional[int]) -> int:
    if max_entries is None:
        return 0
    if max_entries < 1:
        raise ValueError("max_entries must be at least 1 when provided.")

    row_count = int(connection.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0])
    overflow = row_count - max_entries
    if overflow <= 0:
        return 0

    connection.execute(
        """
        DELETE FROM embedding_cache
        WHERE cache_key IN (
            SELECT cache_key
            FROM embedding_cache
            ORDER BY COALESCE(last_accessed, created_at) ASC, created_at ASC
            LIMIT ?
        )
        """,
        (overflow,),
    )
    return overflow


def cache_update(
    prompt: str,
    llm_string: str,
    values: list[float],
    cache_path: str,
    max_entries: Optional[int] = None,
) -> None:
    configure_cache(cache_path)
    key = _cache_key(prompt, llm_string)
    response = json.dumps(values)
    with _cache_lock:
        with _connect(cache_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO embedding_cache
                    (cache_key, prompt, llm_string, response, last_accessed)
                VALUES (?, ?, ?, ?, ?)
                """,
                (key, prompt, llm_string, response, _access_marker()),
            )
            _enforce_max_entries(connection, max_entries)
