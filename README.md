# LLM Inference Microservice

A production-style LLM inference service built on Ollama, demonstrating
request queuing, async serving, and token streaming patterns used in
real-world inference servers (vLLM, TGI, TensorRT-LLM).

## Why this exists

Most "call an LLM API" tutorials skip the part that actually matters in
production: a single model instance can only process one generation
efficiently at a time. This service solves that with a serialized
asyncio queue, then layers streaming on top without breaking that
guarantee.

## Architecture

Client → FastAPI → asyncio.Queue (single worker) → Ollama (llama3.2:3b)

- **Single background worker** drains the queue one request at a time,
  running each blocking Ollama call in a thread pool (`run_in_executor`)
  so the event loop stays free to accept new connections.
- **Streaming requests share the same queue slot** as sync requests —
  a stream holds its place in line for its full duration, exactly like
  a sync call would. This is a deliberate tradeoff: it keeps ordering
  and resource guarantees simple, at the cost of not allowing concurrent
  streams. Noted here explicitly rather than left as an implicit limitation.

## Milestones

- [x] **M1** — Benchmarked `llama3.2:3b` via Ollama (latency, tokens/sec)
- [x] **M2** — FastAPI service with asyncio request queue
- [x] **M3** — Token streaming via Server-Sent Events (SSE)
- [x] **M4** — Streaming reconnected to the shared queue, with verified serialization
- [ ] **M5** — Rate limiting
- [ ] **M6** — Prompt caching
- [ ] **M7** — Redis-backed queue (survive restarts)
- [ ] **M8** — Load testing with Locust
- [ ] **M9** — Observability / metrics (Prometheus-style)

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/generate` | POST | Submit a prompt, get the full response back |
| `/generate/stream` | POST | Submit a prompt, receive tokens as SSE |
| `/health` | GET | Service + queue status |
| `/models` | GET | List locally available Ollama models |

## Running locally

```bash
# start Ollama
ollama serve

# start the service
uvicorn src.api:app --reload --port 8001
```

## Testing streaming

```bash
curl -N -X POST http://localhost:8001/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a haiku about databases"}'
```

## Load testing (Locust, 5 concurrent users)
- 143 total requests over ~50s, 75 rejected with 429 (rate limit)
- p50: 2ms, p95: 5ms, p99: 1200ms
- 0 failures on /health — monitoring stays responsive under /generate saturation
- Rate limiter (capacity=5, refill=0.5/sec) is the binding constraint at ~2.7 RPS,
  not queue depth (max=10) — confirmed via Locust FAILURES tab showing 429s, not 503s

## Tech stack

Python, FastAPI, asyncio, Ollama, requests