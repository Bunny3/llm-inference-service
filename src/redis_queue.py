import json
import uuid
import time
import asyncio
import redis.asyncio as redis
from src.ollama_client import OllamaClient
from src.config import config


class RedisInferenceQueue:
    """
    Same single-worker serialization as InferenceQueue, but backed by
    Redis so queued requests survive a server restart.

    Pattern: RPUSH to enqueue (producer), BLPOP to dequeue (worker).
    Results are stored under a per-request key with a short TTL,
    since the original in-process Future can't cross a restart —
    the client instead polls/waits on that key.
    """

    QUEUE_KEY = "inference:queue"
    RESULT_KEY_PREFIX = "inference:result:"
    RESULT_TTL = 300  # seconds — results expire if never collected

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.client = OllamaClient()
        self._worker_task = None
        self.total_requests = 0
        self.total_errors = 0

    async def start(self):
        self._worker_task = asyncio.create_task(self._worker_loop())
        print("✅ Redis-backed inference queue worker started.")

    async def stop(self):
        if self._worker_task:
            self._worker_task.cancel()
        await self.redis.close()

    async def _worker_loop(self):
        while True:
            try:
                _, raw_job = await self.redis.blpop(self.QUEUE_KEY)
                job = json.loads(raw_job)
                request_id = job["request_id"]
                prompt = job["prompt"]
                enqueued_at = job["enqueued_at"]
                wait_ms = (time.time() - enqueued_at) * 1000

                loop = asyncio.get_event_loop()
                try:
                    result = await loop.run_in_executor(None, self.client.generate, prompt)
                    result["queue_wait_ms"] = round(wait_ms, 2)
                    self.total_requests += 1
                    await self._store_result(request_id, {"status": "done", "result": result})
                except Exception as e:
                    self.total_errors += 1
                    await self._store_result(request_id, {"status": "error", "error": str(e)})

            except Exception as loop_err:
                # Catches blpop/json failures too — logs instead of silently killing the worker
                print(f"⚠️ Worker loop error (job skipped, worker still alive): {loop_err}")
                await asyncio.sleep(1)  # brief backoff before retrying

    async def _store_result(self, request_id: str, payload: dict):
        key = f"{self.RESULT_KEY_PREFIX}{request_id}"
        await self.redis.set(key, json.dumps(payload), ex=self.RESULT_TTL)

    async def submit(self, prompt: str, timeout: float = 60.0) -> dict:
        """
        Enqueue a prompt and poll for its result.
        Polling (not BLPOP-on-result) keeps this simple — a pub/sub
        version would avoid the poll delay, but adds real complexity
        for a learning project. Worth naming as a known tradeoff.
        """
        request_id = str(uuid.uuid4())
        job = {"request_id": request_id, "prompt": prompt, "enqueued_at": time.time()}
        await self.redis.rpush(self.QUEUE_KEY, json.dumps(job))

        result_key = f"{self.RESULT_KEY_PREFIX}{request_id}"
        start = time.time()
        while time.time() - start < timeout:
            raw = await self.redis.get(result_key)
            if raw:
                payload = json.loads(raw)
                await self.redis.delete(result_key)
                if payload["status"] == "error":
                    raise RuntimeError(payload["error"])
                return payload["result"]
            await asyncio.sleep(0.05)  # 50ms poll interval

        raise RuntimeError("Request timed out waiting in queue")

    @property
    async def queue_depth(self) -> int:
        return await self.redis.llen(self.QUEUE_KEY)

    async def stats(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "current_queue_depth": await self.queue_depth,
        }