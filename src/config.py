# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "512"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    USE_REDIS_QUEUE: bool = os.getenv("USE_REDIS_QUEUE", "false").lower() == "true"
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    @classmethod
    def validate(cls):
        print(f"✅ Config loaded — model: {cls.OLLAMA_MODEL}")

config = Config()