# LLM Inference Microservice

A production-style LLM inference service built on Ollama, demonstrating
the core patterns used in real inference servers (vLLM, TGI, TensorRT-LLM):
async request queuing, token streaming, rate limiting, caching, durability,
load testing, and observability.

## Why this exists

Most "call an LLM API" tutorials skip the part that actually matters in
production: a single model instance can only process one generation
efficiently at a time, failures need to degrade gracefully, and you need
visibility into what's actually happening under load. This project builds
that up milestone by milestone, with each one tested against real traffic
before moving to the next.

## Architecture
Client → FastAPI → Rate Limiter → Cache → Queue (in-memory or Redis) → Ollama (llama3.2:3b)
↓
Prometheus metrics

- **Single background worker** drains the queue one request at a time,
  running each blocking Ollama call in a thread pool (`run_in_executor`)
  so the event loop stays free to accept new connections.
- **Streaming requests share the same queue slot** as sync requests — a
  stream holds its place in line for its full duration, same as a sync
  call would. Deliberate tradeoff: keeps ordering/resource guarantees
  simple, at the cost of disallowing concurrent streams.
- **Dual queue backends, toggleable via config**: in-memory (`asyncio.Queue`,
  fast, lost on restart) or Redis-backed (`RPUSH`/`BLPOP`, durable across
  restarts, polling-based result delivery). Streaming currently only
  supports the in-memory backend — a known scope boundary, not a bug.
- **Worker resilience**: the Redis worker's loop wraps the entire body
  (including `BLPOP` and job deserialization) in try/except with backoff,
  so a single malformed job or dropped connection can't silently kill the
  background worker — an earlier version of this had exactly that bug.

## Milestones

- [x] **M1** — Benchmarked `llama3.2:3b` via Ollama (latency, tokens/sec)
- [x] **M2** — FastAPI service with asyncio request queue
- [x] **M3** — Token streaming via Server-Sent Events (SSE)
- [x] **M4** — Streaming reconnected to the shared queue, serialization verified
- [x] **M5** — Rate limiting (token bucket, per-client)
- [x] **M6** — LRU prompt cache with TTL, exact-match
- [x] **M7** — Redis-backed durable queue with config toggle
- [x] **M8** — Load tested with Locust
- [x] **M9** — Prometheus metrics endpoint

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/generate` | POST | Submit a prompt, get the full response back |
| `/generate/stream` | POST | Submit a prompt, receive tokens as SSE |
| `/health` | GET | Service + queue + cache stats |
| `/metrics` | GET | Prometheus-format metrics |
| `/models` | GET | List locally available Ollama models |

## Rate limiting

Token bucket algorithm, per client IP: burst capacity of 5 requests,
sustained refill of 0.5 req/sec. Chosen over a fixed-window counter
because it allows short bursts without letting a client blast all their
quota in the first second of a window.

## Caching

Exact-match LRU cache (max 100 entries, 1hr TTL), keyed on a hash of
`(prompt, temperature, max_tokens)`. Cache hits skip the queue and rate
limiter entirely. Known extension: semantic caching via embeddings +
cosine similarity (infra for this already exists in a sibling project,
doc-qa-chatbot) so near-duplicate prompts ("What's 2+2?" vs "What is 2 + 2?")
could also hit — not built here to keep this milestone's scope tight.

## Durability (Redis-backed queue)

Set `USE_REDIS_QUEUE=true` to switch `/generate` from the in-memory queue
to a Redis-backed one. Jobs are pushed via `RPUSH` and consumed via
`BLPOP`; results are polled from a per-request Redis key (50ms interval)
since the original in-process `Future` pattern can't survive a restart.
Known tradeoff: polling instead of Redis pub/sub, chosen deliberately to
avoid the added complexity of a second connection and message-ordering
concerns, at the cost of up to 50ms result-delivery latency.

## Load testing (Locust, 5 concurrent users, ~50s)

- 143 total requests, 75 rejected with clean `429` (rate limit) — no 500s,
  no crashes, no connection errors
- p50: 2ms · p95: 5ms · p99: 1200ms
- `/health` had 0 failures — stays responsive even while `/generate` is saturated
- The rate limiter (capacity 5, refill 0.5/sec) is the binding constraint
  at ~2.7 RPS, not queue depth (max 10) — confirmed via Locust's Failures
  tab showing 429s rather than 503s

## Observability

`/metrics` exposes Prometheus-format counters, histograms, and gauges:
- `inference_requests_total{endpoint, status}` — request outcomes by type
- `inference_cache_hits_total` / `inference_cache_misses_total`
- `inference_rate_limited_total`
- `inference_latency_seconds` (histogram) — actual Ollama inference time
- `inference_queue_wait_seconds` (histogram) — time spent queued
- `inference_queue_depth` (gauge) — current queue size

Histograms (not just averages) matter here specifically because this
service's own load test showed a p50 of 2ms next to a p99 of 1200ms —
an average would have hidden that gap entirely.

## Running locally

```bash
# start Ollama
ollama serve

# start Redis (only needed if USE_REDIS_QUEUE=true)
brew services start redis

# start the service
uvicorn src.api:app --reload --port 8001

# or, with the Redis-backed queue:
USE_REDIS_QUEUE=true uvicorn src.api:app --reload --port 8001
```

## Testing

```bash
# basic generation
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" -d '{"prompt": "What is 2+2?"}'

# streaming
curl -N -X POST http://localhost:8001/generate/stream \
  -H "Content-Type: application/json" -d '{"prompt": "Write a haiku about databases"}'

# metrics (note: requires -L to follow the trailing-slash redirect)
curl -L http://localhost:8001/metrics

# load test
locust -f locustfile.py --host http://localhost:8001
```

## Tech stack

Python, FastAPI, asyncio, Ollama, Redis, Prometheus, Locust