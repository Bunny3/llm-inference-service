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
        """
        Continuously pulls requests from the queue and processes them.
        Runs as a background asyncio task for the lifetime of the server.
        """
        while True:
            # Wait for the next request — blocks here until one arrives
            future, prompt, enqueued_at = await self.queue.get()
            wait_ms = (time.perf_counter() - enqueued_at) * 1000

            try:
                # Run the blocking Ollama call in a thread pool so it
                # doesn't block the asyncio event loop for other tasks
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    self.client.generate,
                    prompt,
                )
                result["queue_wait_ms"] = round(wait_ms, 2)
                self.total_requests += 1
                future.set_result(result)

            except Exception as e:
                self.total_errors += 1
                future.set_exception(e)

            finally:
                self.queue.task_done()

    async def submit(self, prompt: str) -> dict:
        """
        Submit a prompt to the queue and wait for the result.
        Raises a 503 error if the queue is full.
        """
        if self.queue.full():
            raise RuntimeError(
                f"Queue is full ({self.max_queue_depth} requests waiting). "
                f"Try again later."
            )

        # A Future lets the caller await the result even though
        # it's processed by a separate background worker task
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        enqueued_at = time.perf_counter()

        await self.queue.put((future, prompt, enqueued_at))
        return await future

    @property
    def queue_depth(self) -> int:
        return self.queue.qsize()

    @property
    def stats(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "current_queue_depth": self.queue_depth,
            "max_queue_depth": self.max_queue_depth,
        }


# Singleton — shared across all API requests
inference_queue = InferenceQueue(max_queue_depth=10)