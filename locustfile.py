from locust import HttpUser, task, between
import random

PROMPTS = [
    "What is 2+2?",
    "Write a haiku about the ocean",
    "Explain recursion in one sentence",
    "What is the capital of France?",
    "Summarize the plot of Romeo and Juliet in one line",
]


class InferenceUser(HttpUser):
    """
    Simulates a client hitting /generate repeatedly.
    wait_time models real user think-time between requests —
    without it, Locust would fire requests as fast as physically
    possible, which tests something different (max throughput)
    than realistic concurrent usage.
    """
    wait_time = between(1, 3)

    @task(3)
    def generate(self):
        prompt = random.choice(PROMPTS)
        self.client.post(
            "/generate",
            json={"prompt": prompt, "max_tokens": 50},
            name="/generate",
        )

    @task(1)
    def health_check(self):
        self.client.get("/health", name="/health")