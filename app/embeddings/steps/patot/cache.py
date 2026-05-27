import json
import threading
from pathlib import Path
from typing import Optional

from langchain_community.cache import SQLiteCache
from langchain_core.globals import get_llm_cache, set_llm_cache
from langchain_core.outputs import Generation


_cache_lock = threading.Lock()
_configured_cache_path: Optional[str] = None


def configure_cache(cache_path: str) -> None:
    global _configured_cache_path
    with _cache_lock:
        if _configured_cache_path == cache_path and isinstance(get_llm_cache(), SQLiteCache):
            return
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        set_llm_cache(SQLiteCache(database_path=cache_path))
        _configured_cache_path = cache_path


def cache_lookup(prompt: str, llm_string: str, cache_path: str) -> Optional[list[float]]:
    configure_cache(cache_path)
    cache = get_llm_cache()
    if cache is None:
        return None
    generations = cache.lookup(prompt, llm_string)
    if not generations:
        return None
    first = generations[0]
    text = getattr(first, "text", None)
    if not text:
        return None
    return json.loads(text)


def cache_update(prompt: str, llm_string: str, values: list[float], cache_path: str) -> None:
    configure_cache(cache_path)
    cache = get_llm_cache()
    if cache is None:
        return
    cache.update(prompt, llm_string, [Generation(text=json.dumps(values))])
