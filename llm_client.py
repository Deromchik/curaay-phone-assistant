import os
from typing import Any, AsyncIterator, Optional

from openai import AsyncOpenAI

from llm_logger import build_request_record, set_pending_request

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.getenv("OPENROUTER_DEFAULT_MODEL", "google/gemini-2.5-flash")


def resolve_answer_llm_model(answer_model: Optional[str], expert_id: Optional[str] = None) -> str:
    """Resolve the model slug for OpenRouter."""
    _ = expert_id  # reserved for future expert-specific routing
    if answer_model and str(answer_model).strip():
        return str(answer_model).strip()
    return DEFAULT_MODEL


def _get_client() -> AsyncOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    return AsyncOpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost:8501"),
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", "EXAI Prompt Tester"),
        },
    )


async def _openrouter_stream(
    client: AsyncOpenAI,
    params: dict[str, Any],
) -> AsyncIterator[str]:
    stream = await client.chat.completions.create(**params, stream=True)
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


async def generate_response(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.4,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    reasoning: Optional[dict[str, Any]] = None,
    is_stream: bool = True,
) -> AsyncIterator[str] | str:
    """Call OpenRouter chat completions API."""
    client = _get_client()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    def build_params(include_reasoning: bool) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if top_p is not None:
            params["top_p"] = top_p
        if include_reasoning and reasoning is not None:
            params["extra_body"] = {"reasoning": reasoning}
        return params

    request_params = {
        "temperature": temperature,
    }
    if max_tokens is not None:
        request_params["max_tokens"] = max_tokens
    if top_p is not None:
        request_params["top_p"] = top_p
    if reasoning is not None:
        request_params["reasoning"] = reasoning

    set_pending_request(
        build_request_record(
            model=model,
            messages=messages,
            request_params=request_params,
        )
    )

    if is_stream:
        try:
            return _openrouter_stream(client, build_params(include_reasoning=True))
        except Exception as exc:
            if reasoning is not None:
                return _openrouter_stream(client, build_params(include_reasoning=False))
            raise exc

    try:
        response = await client.chat.completions.create(**build_params(include_reasoning=True))
    except Exception as exc:
        if reasoning is not None:
            response = await client.chat.completions.create(
                **build_params(include_reasoning=False)
            )
        else:
            raise exc

    return response.choices[0].message.content or ""


async def collect_stream(stream: AsyncIterator[str]) -> str:
    """Collect all chunks from an async stream into a single string."""
    parts: list[str] = []
    async for chunk in stream:
        parts.append(chunk)
    return "".join(parts)
