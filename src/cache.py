import hashlib
import time
from collections import OrderedDict


class PromptCache:
    """
    LRU cache with TTL, keyed by a hash of (prompt, temperature, max_tokens).
    Exact-match only — not semantic. A prompt reworded slightly won't hit.
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.store: OrderedDict[str, tuple[dict, float]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def _key(self, prompt: str, temperature: float, max_tokens: int) -> str:
        raw = f"{prompt}|{temperature}|{max_tokens}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, prompt: str, temperature: float, max_tokens: int) -> dict | None:
        key = self._key(prompt, temperature, max_tokens)
        entry = self.store.get(key)
        if entry is None:
            self.misses += 1
            return None

        result, stored_at = entry
        if time.monotonic() - stored_at > self.ttl_seconds:
            del self.store[key]  # expired
            self.misses += 1
            return None

        self.store.move_to_end(key)  # mark as recently used
        self.hits += 1
        return result

    def set(self, prompt: str, temperature: float, max_tokens: int, result: dict):
        key = self._key(prompt, temperature, max_tokens)
        self.store[key] = (result, time.monotonic())
        self.store.move_to_end(key)
        if len(self.store) > self.max_size:
            self.store.popitem(last=False)  # evict oldest (LRU)

    @property
    def stats(self) -> dict:
        total = self.hits + self.misses
        hit_rate = (self.hits / total) if total > 0 else 0.0
        return {
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "hit_rate": round(hit_rate, 3),
            "cache_size": len(self.store),
        }


prompt_cache = PromptCache(max_size=100, ttl_seconds=3600)