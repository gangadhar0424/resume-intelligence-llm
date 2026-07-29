"""
main.py — Resume Intelligence inference API.

Calls a local Ollama-served Qwen2.5:3b (or a fine-tuned variant registered under a different
model name) to parse resume text into structured JSON, with schema validation and clean
error handling around the model call.

Run locally:
    pip install -r requirements.txt
    ollama serve                 # in a separate terminal, if not already running
    ollama pull qwen2.5:3b       # if not already pulled
    uvicorn main:app --reload --port 8000

Then:
    curl -X POST http://localhost:8000/parse-resume \
        -H "Content-Type: application/json" \
        -d '{"resume_text": "Jane Doe, jane@doe.com, Software Engineer at Acme, 2020-Present..."}'
"""
import json
import logging
import re

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from config import settings
from schemas import ParseResumeRequest, ParseResumeResponse, ParsedResume

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("resume-intelligence-api")

app = FastAPI(
    title="Resume Intelligence API",
    description="Structured resume parsing powered by a locally-served open-source LLM.",
    version="0.1.0",
)

SYSTEM_PROMPT = (
    "You are a resume parsing engine. Extract structured information from the resume "
    "text into a JSON object with exactly these fields: name, email, phone, location, "
    "summary, skills (list), education (list of {degree, institution, year}), "
    "experience (list of {title, company, duration, description}), certifications (list), "
    "projects (list of {name, description}), links (list). "
    "If a field is not present, use null for single values or [] for lists. "
    "Never invent information not present in the text. Respond with JSON only, no prose."
)


def extract_json_block(raw_text: str) -> str:
    """Small models sometimes wrap JSON in markdown fences or add a stray sentence.
    This pulls out the first {...} block so downstream json.loads() has the best shot."""
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw_text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)
    brace_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if brace_match:
        return brace_match.group(0)
    return raw_text


async def call_ollama(resume_text: str) -> str:
    """Calls the local Ollama /api/chat endpoint. Raises httpx exceptions on network/timeout
    failures, which the route handler below converts into clean HTTP error responses."""
    payload = {
        "model": settings.model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Resume text:\n{resume_text}"},
        ],
        "stream": False,
        "options": {
            "temperature": settings.temperature,
            "num_predict": settings.max_output_tokens,
        },
    }
    async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
        response = await client.post(f"{settings.ollama_host}/api/chat", json=payload)
        response.raise_for_status()
        body = response.json()
        return body["message"]["content"]


@app.get("/health")
async def health():
    """Basic liveness check; also verifies Ollama is reachable, which is the main
    failure mode operators will hit (Ollama not running, wrong host/port)."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{settings.ollama_host}/api/tags")
            r.raise_for_status()
        return {"status": "ok", "ollama_reachable": True, "model": settings.model_name}
    except Exception as e:
        return {"status": "degraded", "ollama_reachable": False, "detail": str(e)}


@app.post("/parse-resume", response_model=ParseResumeResponse)
async def parse_resume(request: ParseResumeRequest):
    resume_text = request.resume_text

    if len(resume_text) > settings.max_input_chars:
        raise HTTPException(
            status_code=413,
            detail=f"resume_text exceeds max length of {settings.max_input_chars} characters",
        )

    # --- Call the model ---
    try:
        raw_output = await call_ollama(resume_text)
    except httpx.ConnectError:
        logger.error("Could not connect to Ollama at %s", settings.ollama_host)
        raise HTTPException(
            status_code=503,
            detail=f"Model backend unreachable at {settings.ollama_host}. Is 'ollama serve' running?",
        )
    except httpx.TimeoutException:
        logger.error("Ollama request timed out after %ss", settings.request_timeout_seconds)
        raise HTTPException(status_code=504, detail="Model backend timed out")
    except httpx.HTTPStatusError as e:
        logger.error("Ollama returned an error: %s", e)
        raise HTTPException(status_code=502, detail=f"Model backend error: {e}")

    # --- Parse the model's JSON output ---
    json_text = extract_json_block(raw_output)
    try:
        parsed_dict = json.loads(json_text)
    except json.JSONDecodeError as e:
        logger.warning("Model returned invalid JSON: %s", e)
        return ParseResumeResponse(
            success=False,
            error=f"Model output was not valid JSON: {e}",
            raw_model_output=raw_output,
        )

    # --- Validate against our schema ---
    try:
        parsed_resume = ParsedResume(**parsed_dict)
    except ValidationError as e:
        logger.warning("Model output failed schema validation: %s", e)
        return ParseResumeResponse(
            success=False,
            error=f"Model output failed schema validation: {e}",
            raw_model_output=raw_output,
        )

    return ParseResumeResponse(success=True, data=parsed_resume)
