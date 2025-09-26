"""Runtime patches for fnllm provider integrations."""

from __future__ import annotations

import logging
import json
from collections.abc import Iterator
from typing import Any, Optional, cast

from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_message_param import (
    ChatCompletionMessageParam,
)

from fnllm.openai.llm.openai_text_chat_llm import (
    OpenAITextChatLLMImpl,
    OpenAINoChoicesAvailableError,
)
from fnllm.openai.types.chat.io import (
    OpenAIChatCompletionInput,
    OpenAIChatHistoryEntry,
    OpenAIChatOutput,
)
from fnllm.openai.types.chat.parameters import OpenAIChatParameters
from fnllm.openai.utils import build_chat_messages
from fnllm.types.generics import TJsonModel
from fnllm.types.io import LLMInput
from fnllm.types.metrics import LLMUsageMetrics

logger = logging.getLogger(__name__)

_PATCH_APPLIED = False


def _extract_response_body(raw_response: Any) -> str:
    """Return the raw HTTP response body if available."""
    try:
        return raw_response.http_response.text
    except Exception as exc:  # pragma: no cover - defensive
        return f"<failed to read response body: {exc}>"


def _build_response_details(raw_response: Any, parsed: Optional[Any] = None) -> dict[str, Any]:
    """Collect HTTP response metadata for logging."""
    http_response = getattr(raw_response, "http_response", None)
    request = getattr(http_response, "request", None)
    return {
        "status_code": getattr(http_response, "status_code", None),
        "url": str(getattr(request, "url", "")) if request else None,
        "headers": dict(getattr(http_response, "headers", {})),
        "response_body": _extract_response_body(raw_response),
        "parsed_type": type(parsed).__name__ if parsed is not None else None,
    }


def _format_detail_message(details: dict[str, Any]) -> str:
    """Convert response metadata into a human-readable string for logs."""
    status = details.get("status_code")
    url = details.get("url")
    body = details.get("response_body")
    parsed = details.get("parsed_type")
    return (
        f"status={status} url={url} parsed={parsed} body={body}"
    )


def _normalize_completion(completion: Any, raw_response: Any) -> Any:
    """Convert non-standard responses (like SSE strings) into completion objects."""
    if isinstance(completion, str):
        text = completion.strip()
        if text.startswith("data:"):
            details = _build_response_details(raw_response)
            assembled = _coalesce_stream_chunks(text)
            if assembled is not None:
                logger.warning(
                    "Coerced streaming response into chat completion; %s",
                    _format_detail_message(details),
                    extra={"details": details},
                )
                return ChatCompletion.model_validate(assembled)
    return completion


def _coalesce_stream_chunks(text: str) -> Optional[Any]:
    """Attempt to turn an SSE streaming payload into a completion-like object."""
    chunks = []
    last_payload: Optional[dict[str, Any]] = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        last_payload = parsed
        for choice in parsed.get("choices", []):
            delta = choice.get("delta", {})
            content = delta.get("content")
            if content:
                chunks.append(content)

    if last_payload is None:
        return None

    content = "".join(chunks)
    choice = {
        "index": 0,
        "message": {
            "role": "assistant",
            "content": content,
        },
        "finish_reason": "stop",
        "logprobs": None,
    }

    completion_dict: dict[str, Any] = {
        "id": last_payload.get("id", "streamed-response"),
        "object": "chat.completion",
        "model": last_payload.get("model", "unknown-model"),
        "created": last_payload.get("created") or 0,
        "choices": [choice],
        "system_fingerprint": last_payload.get("system_fingerprint"),
    }

    usage = last_payload.get("usage")
    if isinstance(usage, dict):
        completion_dict["usage"] = usage

    return completion_dict


async def _execute_llm_with_logging(
    self: OpenAITextChatLLMImpl,
    prompt: OpenAIChatCompletionInput,
    kwargs: LLMInput[TJsonModel, OpenAIChatHistoryEntry, OpenAIChatParameters],
) -> OpenAIChatOutput:
    """Execute the LLM request while emitting detailed logging on malformed responses."""
    history = kwargs.get("history", [])
    messages, prompt_message = build_chat_messages(
        prompt, history, self._special_token_behavior
    )
    local_model_parameters = kwargs.get("model_parameters")
    parameters = self._build_completion_parameters(local_model_parameters)

    raw_response = await self._client.chat.completions.with_raw_response.create(
        messages=cast(Iterator[ChatCompletionMessageParam], messages),
        **parameters,
    )

    try:
        completion = raw_response.parse()
    except Exception as exc:
        details = _build_response_details(raw_response)
        logger.error(
            "Failed to parse response from LLM; %s",
            _format_detail_message(details),
            exc_info=exc,
            extra={"details": details},
        )
        raise

    headers = raw_response.headers

    completion = _normalize_completion(completion, raw_response)

    try:
        choices = completion.choices  # type: ignore[attr-defined]
    except AttributeError:
        details = _build_response_details(raw_response, completion)
        logger.error(
            "LLM response missing 'choices' field; %s",
            _format_detail_message(details),
            extra={"details": details},
        )
        raise

    if not choices:
        details = _build_response_details(raw_response, completion)
        logger.error(
            "LLM response contained no choices; %s",
            _format_detail_message(details),
            extra={"details": details},
        )
        raise OpenAINoChoicesAvailableError

    result = choices[0].message
    usage: LLMUsageMetrics | None = None
    if completion.usage:
        usage = LLMUsageMetrics(
            input_tokens=completion.usage.prompt_tokens,
            output_tokens=completion.usage.completion_tokens,
        )

    return OpenAIChatOutput(
        raw_input=prompt_message,
        raw_output=result,
        content=result.content,
        raw_model=completion,
        usage=usage or LLMUsageMetrics(),
        headers=headers,
    )


def apply_patches() -> None:
    """Apply monkey patches for fnllm integration."""
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return
    OpenAITextChatLLMImpl._execute_llm = _execute_llm_with_logging  # type: ignore[assignment]
    _PATCH_APPLIED = True


__all__ = ["apply_patches"]
