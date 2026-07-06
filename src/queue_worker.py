# src/queue_worker.py
import asyncio
import time
from src.ollama_client import OllamaClient
from src.config import config


class InferenceQueue:
    """
    A single asyncio queue that serializes LLM requests.

    Why a queue? LLMs running on CPU can only process one generation
    efficiently at a time. Without a queue, concurrent requests compete
    for the same resource, slowing each other down and causing timeouts.
    With a queue, requests wait their turn and each gets full throughput.

    This is the same pattern used in production inference servers like
    vLLM and TGI, just simplified for learning.
    """

    def __init__(self, max_queue_depth: int = 10):
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_depth)
        self.client = OllamaClient()
        self.max_queue_depth = max_queue_depth
        self._worker_task = None

        # Stats tracking
        self.total_requests = 0
        self.total_errors = 0

    async def start(self):
        """Start the background worker that drains the queue."""
        self._worker_task = asyncio.create_task(self._worker_loop())
        print("✅ Inference queue worker started.")

    async def stop(self):
        """Gracefully stop the worker."""
        if self._worker_task:
            self._worker_task.cancel()
            print("🛑 Inference queue worker stopped.")

    async def _worker_loop(self):
        while True:
            kind, payload, enqueued_at = await self.queue.get()
            wait_ms = (time.perf_counter() - enqueued_at) * 1000
            loop = asyncio.get_event_loop()

            if kind == "sync":
                future, prompt = payload
                try:
                    result = await loop.run_in_executor(None, self.client.generate, prompt)
                    result["queue_wait_ms"] = round(wait_ms, 2)
                    self.total_requests += 1
                    future.set_result(result)
                except Exception as e:
                    self.total_errors += 1
                    future.set_exception(e)
                finally:
                    self.queue.task_done()

            elif kind == "stream":
                output_queue, prompt = payload
                try:
                    await loop.run_in_executor(
                        None, self._drain_stream, prompt, output_queue, loop
                    )
                    self.total_requests += 1
                except Exception as e:
                    self.total_errors += 1
                    loop.call_soon_threadsafe(output_queue.put_nowait, e)
                finally:
                    self.queue.task_done()

    def _drain_stream(self, prompt: str, output_queue: asyncio.Queue, loop):
        """
        Runs in a worker thread. Pulls tokens from Ollama's blocking
        stream and hands each one to the event loop thread-safely —
        asyncio.Queue.put_nowait isn't safe to call from this thread directly.
        """
        try:
            for token in self.client.generate_stream(prompt):
                loop.call_soon_threadsafe(output_queue.put_nowait, token)
        finally:
            loop.call_soon_threadsafe(output_queue.put_nowait, None)  # sentinel

    async def submit(self, prompt: str) -> dict:
        if self.queue.full():
            raise RuntimeError(f"Queue is full ({self.max_queue_depth} requests waiting). Try again later.")
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        enqueued_at = time.perf_counter()
        await self.queue.put(("sync", (future, prompt), enqueued_at))
        return await future

    async def submit_stream(self, prompt: str):
        """Returns an async generator yielding tokens, queued alongside sync requests."""
        if self.queue.full():
            raise RuntimeError(f"Queue is full ({self.max_queue_depth} requests waiting). Try again later.")
        output_queue: asyncio.Queue = asyncio.Queue()
        enqueued_at = time.perf_counter()
        await self.queue.put(("stream", (output_queue, prompt), enqueued_at))

        async def token_generator():
            while True:
                token = await output_queue.get()
                if token is None:
                    break
                if isinstance(token, Exception):
                    raise token
                yield token

        return token_generator()

# Singleton — shared across all API requests
inference_queue = InferenceQueue(max_queue_depth=10)