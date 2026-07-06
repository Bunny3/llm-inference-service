from prometheus_client import Counter, Histogram, Gauge

# Counters — monotonically increasing totals
REQUESTS_TOTAL = Counter(
    "inference_requests_total", "Total inference requests", ["endpoint", "status"]
)
CACHE_HITS_TOTAL = Counter("inference_cache_hits_total", "Total cache hits")
CACHE_MISSES_TOTAL = Counter("inference_cache_misses_total", "Total cache misses")
RATE_LIMITED_TOTAL = Counter("inference_rate_limited_total", "Total 429s")

# Histogram — buckets latency into ranges, enabling p50/p95/p99 queries later
INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "Time spent in actual Ollama inference",
    buckets=[0.1, 0.25, 0.5, 1, 2, 5, 10],
)
QUEUE_WAIT_TIME = Histogram(
    "inference_queue_wait_seconds",
    "Time spent waiting in queue before processing",
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 5],
)

# Gauge — current point-in-time value, can go up or down
QUEUE_DEPTH = Gauge("inference_queue_depth", "Current number of queued requests")