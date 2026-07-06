# src/api.py
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.config import config
from src.queue_worker import inference_queue
from fastapi.responses import StreamingResponse
from src.ollama_client import OllamaClient
from fastapi import Request
from src.rate_limiter import rate_limiter
from fastapi import Depends
from src.cache import prompt_cache
from src.redis_queue import RedisInferenceQueue
from src.metrics import (
    REQUESTS_TOTAL, CACHE_HITS_TOTAL, CACHE_MISSES_TOTAL,
    RATE_LIMITED_TOTAL, INFERENCE_LATENCY, QUEUE_WAIT_TIME, QUEUE_DEPTH,
)
from prometheus_client import make_asgi_app

ollama_client = OllamaClient()

redis_queue = RedisInferenceQueue(config.REDIS_URL) if config.USE_REDIS_QUEUE else None

app = FastAPI(
    title="LLM Inference Service",
    description="Production-style LLM inference with request queuing",
    version="1.0.0",
)


# --- Schemas ---
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512


class GenerateResponse(BaseModel):
    response: str
    model: str
    input_tokens: int
    output_tokens: int
    tokens_per_second: float
    latency_ms: float
    queue_wait_ms: float


# --- Lifecycle ---

@app.on_event("startup")
async def startup():
    config.validate()
    await inference_queue.start()
    if redis_queue:
        await redis_queue.start()
        print("🚀 Redis-backed queue active for /generate.")
    else:
        print("🚀 In-memory queue active for /generate.")


@app.on_event("shutdown")
async def shutdown():
    await inference_queue.stop()
    if redis_queue:
        await redis_queue.stop()


# --- Endpoints ---
def enforce_rate_limit(request: Request):
    client_key = request.client.host
    allowed, retry_after = rate_limiter.check(client_key)
    if not allowed:
        RATE_LIMITED_TOTAL.inc()
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {retry_after:.1f}s",
            headers={"Retry-After": str(round(retry_after, 1))},
        )

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest, _=Depends(enforce_rate_limit)):
    cached = prompt_cache.get(request.prompt, config.TEMPERATURE, request.max_tokens)
    if cached:
        CACHE_HITS_TOTAL.inc()
        REQUESTS_TOTAL.labels(endpoint="/generate", status="cache_hit").inc()
        return GenerateResponse(**{**cached, "queue_wait_ms": 0.0})

    CACHE_MISSES_TOTAL.inc()
    try:
        active_queue = redis_queue if redis_queue else inference_queue
        result = await active_queue.submit(request.prompt)
        prompt_cache.set(request.prompt, config.TEMPERATURE, request.max_tokens, result)

        INFERENCE_LATENCY.observe(result["latency_ms"] / 1000)
        QUEUE_WAIT_TIME.observe(result["queue_wait_ms"] / 1000)
        REQUESTS_TOTAL.labels(endpoint="/generate", status="success").inc()

        return GenerateResponse(**result)
    except RuntimeError as e:
        REQUESTS_TOTAL.labels(endpoint="/generate", status="queue_full").inc()
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        REQUESTS_TOTAL.labels(endpoint="/generate", status="error").inc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/health")
async def health():
    """Health check with queue stats."""
    queue_stats = await redis_queue.stats() if redis_queue else inference_queue.stats
    return {
        "status": "ok",
        "model": config.OLLAMA_MODEL,
        "queue_backend": "redis" if redis_queue else "in-memory",
        **queue_stats,
        **prompt_cache.stats,
    }


@app.get("/models")
async def list_models():
    """List locally available Ollama models."""
    from src.ollama_client import OllamaClient
    client = OllamaClient()
    return {"models": client.list_models()}

@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest, _=Depends(enforce_rate_limit)):
    try:
        token_gen = await inference_queue.submit_stream(request.prompt)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    async def event_generator():
        async for token in token_gen:
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")