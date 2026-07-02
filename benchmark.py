# benchmark.py
import statistics
from src.config import config
from src.ollama_client import OllamaClient


PROMPTS = [
    "What is self-attention in one sentence?",
    "Explain the difference between a list and a tuple in Python.",
    "What is the capital of Japan?",
    "Write a haiku about machine learning.",
    "What is gradient descent?",
]


def run_benchmark():
    config.validate()
    client = OllamaClient()

    if not client.health_check():
        print("❌ Ollama server not reachable. Run: ollama serve")
        return

    print(f"\n🔥 Benchmarking {config.OLLAMA_MODEL}")
    print(f"{'='*60}")

    results = []

    for i, prompt in enumerate(PROMPTS, 1):
        print(f"\n[{i}/{len(PROMPTS)}] Prompt: {prompt[:50]}...")
        result = client.generate(prompt)
        results.append(result)

        print(f"  ✅ {result['output_tokens']} tokens | "
              f"{result['tokens_per_second']} tok/s | "
              f"{result['latency_ms']}ms latency")
        print(f"  💬 {result['response'][:100]}...")

    # Summary stats
    latencies = [r["latency_ms"] for r in results]
    throughputs = [r["tokens_per_second"] for r in results]

    print(f"\n{'='*60}")
    print(f"📊 Benchmark Summary — {config.OLLAMA_MODEL}")
    print(f"{'='*60}")
    print(f"  Avg latency    : {statistics.mean(latencies):.0f}ms")
    print(f"  P95 latency    : {sorted(latencies)[int(len(latencies)*0.95)-1]:.0f}ms")
    print(f"  Avg throughput : {statistics.mean(throughputs):.1f} tok/s")
    print(f"  Min throughput : {min(throughputs):.1f} tok/s")
    print(f"  Max throughput : {max(throughputs):.1f} tok/s")


if __name__ == "__main__":
    run_benchmark()