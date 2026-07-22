from datetime import datetime, timezone
from typing import Any, Optional

_pending_request: Optional[dict[str, Any]] = None


def build_request_record(
    *,
    model: str,
    messages: list[dict[str, str]],
    request_params: dict[str, Any],
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "messages": messages,
        "request_params": request_params,
    }


def set_pending_request(record: dict[str, Any]) -> None:
    global _pending_request
    _pending_request = record


def consume_pending_request() -> dict[str, Any]:
    global _pending_request
    record = _pending_request or {}
    _pending_request = None
    return record


def register_llm_exchange(
    logs: list[dict[str, Any]],
    *,
    assistant_response: str,
    generator: str,
    inputs: dict[str, Any],
    turn: int,
    user_question: str,
    conversation_history: list,
) -> None:
    """Append a full LLM exchange (request + response) to the session log."""
    request = consume_pending_request()
    logs.append({
        "timestamp": request.get(
            "timestamp", datetime.now(timezone.utc).isoformat()
        ),
        "generator": generator,
        "model": request.get("model"),
        "request_params": request.get("request_params", {}),
        "messages": request.get("messages", []),
        "inputs": inputs,
        "turn": turn,
        "user_question": user_question,
        "conversation_history": conversation_history,
        "assistant_response": assistant_response,
    })
