import time
from collections import defaultdict


class TokenBucket:
    """
    One bucket per client. Refills at a fixed rate; each request
    costs 1 token. Empty bucket = rate limited.
    """
    def __init__(self, capacity: int, refill_rate_per_sec: float):
        self.capacity = capacity
        self.refill_rate = refill_rate_per_sec
        self.tokens = capacity
        self.last_refill = time.monotonic()

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def try_consume(self) -> tuple[bool, float]:
        """Returns (allowed, retry_after_seconds)."""
        self._refill()
        if self.tokens >= 1:
            self.tokens -= 1
            return True, 0.0
        needed = 1 - self.tokens
        retry_after = needed / self.refill_rate
        return False, retry_after


class RateLimiter:
    """Manages one TokenBucket per client key (e.g. IP address)."""

    def __init__(self, capacity: int = 5, refill_rate_per_sec: float = 0.5):
        # capacity=5, refill 0.5/sec => burst of 5, sustained 1 request/2sec
        self.capacity = capacity
        self.refill_rate = refill_rate_per_sec
        self.buckets: dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(capacity, refill_rate_per_sec)
        )

    def check(self, client_key: str) -> tuple[bool, float]:
        return self.buckets[client_key].try_consume()


rate_limiter = RateLimiter(capacity=5, refill_rate_per_sec=0.5)