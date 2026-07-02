# Benchmark Results

## Baseline — llama3.2:3b (CPU, no caching, no queue)
Date: today
Hardware: MacBook (Apple Silicon)

| Metric | Value |
|---|---|
| Avg throughput | 45.7 tok/s |
| Min throughput | 41.5 tok/s |
| Max throughput | 51.4 tok/s |
| Avg latency | 5025ms |
| P95 latency | 8902ms |

### Key observation
Latency scales with output token count, not input complexity.
Short answers (8 tokens) = 379ms. Long answers (512 tokens) = 12161ms.

### Interview talking point
"P95 latency was 8.9 seconds — nearly 3x the average. This is why I track
percentiles, not averages, in production. An SLO of '95% of requests under
10 seconds' is meaningful; 'average under 5 seconds' hides the tail."
