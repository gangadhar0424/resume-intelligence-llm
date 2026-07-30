"""
config.py — Centralized configuration, loaded from environment variables with sane defaults.
Keeping config in one place (rather than scattered constants) makes it trivial to swap
Ollama -> vLLM -> a hosted endpoint later without touching business logic.
"""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Model backend: defaults to local Ollama, matching the qwen2.5:0.5b already pulled locally.
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model_name: str = os.getenv("MODEL_NAME", "qwen2.5:0.5b")

    # If a fine-tuned LoRA adapter has been merged and pushed as its own Ollama model
    # (e.g. via a Modelfile), point MODEL_NAME at that instead, e.g. "resume-qwen2.5-3b".
    lora_adapter_path: str = os.getenv("LORA_ADAPTER_PATH", "")

    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
    max_input_chars: int = int(os.getenv("MAX_INPUT_CHARS", "20000"))  # guard against giant pasted docs

    # Generation params — low temperature for a deterministic extraction task
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))
    max_output_tokens: int = int(os.getenv("MAX_OUTPUT_TOKENS", "1024"))

    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
