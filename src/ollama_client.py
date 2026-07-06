# src/ollama_client.py
import time
import requests
from src.config import config
import json

class OllamaClient:
    """
    Wraps the Ollama REST API.
    Ollama exposes a local HTTP server at port 11434 — same pattern
    as calling any external LLM API, just running on your machine.
    """

    def __init__(self):
        self.base_url = config.OLLAMA_BASE_URL
        self.model = config.OLLAMA_MODEL

    def generate(self, prompt: str, stream: bool = False) -> dict:
        """
        Send a prompt, get a response back.
        Returns a dict with response text + timing + token stats.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": config.TEMPERATURE,
                "num_predict": config.MAX_TOKENS,
            }
        }

        start_time = time.perf_counter()

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Ollama returns token counts in the response
        eval_count = data.get("eval_count", 0)       # output tokens
        prompt_eval_count = data.get("prompt_eval_count", 0)  # input tokens
        eval_duration_ns = data.get("eval_duration", 1)       # time in nanoseconds

        tokens_per_second = (
            eval_count / (eval_duration_ns / 1e9)
            if eval_duration_ns > 0 else 0
        )

        return {
            "response": data.get("response", ""),
            "model": data.get("model", ""),
            "latency_ms": round(latency_ms, 2),
            "input_tokens": prompt_eval_count,
            "output_tokens": eval_count,
            "tokens_per_second": round(tokens_per_second, 2),
            "done": data.get("done", False),
        }

    def health_check(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        """List all locally available models."""
        response = requests.get(f"{self.base_url}/api/tags", timeout=5)
        response.raise_for_status()
        return [m["name"] for m in response.json().get("models", [])]
    
    def generate_stream(self, prompt: str):
        """
        Stream tokens from Ollama as they're generated.
        Yields raw text chunks. Caller (FastAPI route) is responsible
        for wrapping this in a StreamingResponse.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": config.TEMPERATURE,
                "num_predict": config.MAX_TOKENS,
            }
        }

        with requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            stream=True,
            timeout=120,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break