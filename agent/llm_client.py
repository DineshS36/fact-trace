"""
FactTrace LLM Client
====================
Provides a unified interface with automatic model failover across
Google Gemini, OpenAI, and Groq.
"""

import logging
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-3.5-flash-lite",
}


def call_llm(
    system_prompt: str,
    user_prompt: str = "",
    response_mime_type: str = "application/json",
) -> str:
    """
    Route prompt to the configured LLM provider and return raw response text.
    Handles single-prompt and system+user prompts across Gemini, OpenAI, and Groq.
    Provides automatic fallback to alternative candidate models upon rate-limit spikes.

    Raises:
        RuntimeError: If provider is unsupported or API key is missing.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower().strip()
    model = os.getenv(
        f"{provider.upper()}_MODEL", _DEFAULT_MODELS.get(provider, "")
    )

    if provider == "openai":
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in .env")
        client = OpenAI(api_key=api_key)
        messages = []
        if system_prompt and user_prompt:
            messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
        else:
            messages.append({"role": "user", "content": system_prompt or user_prompt})

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
        }
        if response_mime_type == "application/json":
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    if provider == "groq":
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set in .env")
        client = Groq(api_key=api_key)
        messages = []
        if system_prompt and user_prompt:
            messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
        else:
            messages.append({"role": "user", "content": system_prompt or user_prompt})

        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
        }
        if response_mime_type == "application/json":
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    if provider == "gemini":
        from google import genai
        from google.genai import types

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in .env")
        client = genai.Client(api_key=api_key)
        contents = f"{system_prompt}\n\n{user_prompt}".strip() if user_prompt else system_prompt

        candidate_models = [model]
        for fallback_mod in ("gemini-3.5-flash-lite", "gemini-3.8-flash"):
            if fallback_mod not in candidate_models:
                candidate_models.append(fallback_mod)

        last_exc = None
        for mod in candidate_models:
            try:
                resp = client.models.generate_content(
                    model=mod,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        response_mime_type=response_mime_type,
                    ),
                )
                if resp.text:
                    return resp.text
            except Exception as exc:
                last_exc = exc
                logger.warning("Gemini model '%s' failed (%s), trying fallback...", mod, exc)
                continue

        if last_exc:
            raise last_exc
        return ""

    raise RuntimeError(
        f"Unsupported LLM_PROVIDER='{provider}'. Use 'openai', 'groq', or 'gemini'."
    )
