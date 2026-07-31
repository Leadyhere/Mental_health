from __future__ import annotations

import json

from openai import AsyncOpenAI

from app.config import get_settings

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncOpenAI(
            api_key=settings.grok_api_key,
            base_url=settings.grok_api_base,
        )
    return _client


async def complete(system: str, user: str, max_tokens: int = 1024) -> str:
    settings = get_settings()
    client = get_client()
    response = await client.chat.completions.create(
        model=settings.grok_model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content or ""


async def complete_json(system: str, user: str, max_tokens: int = 1024) -> dict:
    """Calls the LLM expecting a JSON object back and parses it. Malformed
    JSON is treated as a hard failure by the caller (per spec: low
    confidence / unparseable output must escalate, never silently
    de-escalate or guess)."""
    raw = await complete(system, user, max_tokens=max_tokens)
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())
