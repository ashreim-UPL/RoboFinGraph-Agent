# agents/utils.py

import atexit
import shelve
from typing import Any, Dict

class Cache:
    """A simple disk-based cache."""
    _cache_instances: Dict[str, Any] = {}

    @classmethod
    def disk(cls, cache_name: str = "default_cache"):
        if cache_name not in cls._cache_instances:
            cls._cache_instances[cache_name] = shelve.open(f".cache/{cache_name}.db", writeback=True)
            atexit.register(cls._cache_instances[cache_name].close)
        return cls._cache_instances[cache_name]