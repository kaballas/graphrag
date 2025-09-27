# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

from __future__ import annotations

from typing import Any, cast

import httpx
import pytest
from fnllm.openai.llm.openai_text_chat_llm import OpenAITextChatLLMImpl


class _DummyRawResponse:
    def __init__(
        self,
        parsed: Any,
        *,
        status_code: int = 400,
        body: str = "bad response",
    ) -> None:
        request = httpx.Request("POST", "http://example.com/v1/chat/completions")
        self.http_response = httpx.Response(
            status_code,
            request=request,
            text=body,
        )
        self.headers = self.http_response.headers
        self._parsed = parsed

    def parse(self) -> Any:
        return self._parsed


class _DummyWithRawResponse:
    def __init__(self, raw_response: _DummyRawResponse) -> None:
        self._raw_response = raw_response

    async def create(self, *args: Any, **kwargs: Any) -> _DummyRawResponse:
        return self._raw_response


class _DummyCompletions:
    def __init__(self, raw_response: _DummyRawResponse) -> None:
        self.with_raw_response = _DummyWithRawResponse(raw_response)


class _DummyChat:
    def __init__(self, raw_response: _DummyRawResponse) -> None:
        self.completions = _DummyCompletions(raw_response)


class _DummyClient:
    def __init__(self, raw_response: _DummyRawResponse) -> None:
        self.chat = _DummyChat(raw_response)


@pytest.mark.asyncio
async def test_logs_raw_response_for_missing_choices(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("ERROR")
    raw_response = _DummyRawResponse(parsed="not-json", body="failure body")
    client = _DummyClient(raw_response)
    llm = OpenAITextChatLLMImpl(client=cast("Any", client), model="gpt-test")

    with pytest.raises(AttributeError):
        await llm._execute_llm("hello", {})  # noqa: SLF001

    messages = [record.message for record in caplog.records]
    assert any("missing 'choices'" in msg for msg in messages)

    details_entries = [
        getattr(record, "details", {})
        for record in caplog.records
        if hasattr(record, "details")
    ]
    assert any(
        entry.get("response_body") == "failure body" for entry in details_entries
    )


@pytest.mark.asyncio
async def test_coerces_sse_response(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("WARNING")
    body = (
        'data: {"choices":[{"delta":{"content":"Hello"}}]}'
        "\n"
        'data: {"choices":[{"delta":{"content":" World"}}]}'
        "\n"
        "data: [DONE]"
    )
    raw_response = _DummyRawResponse(parsed=body, status_code=200, body=body)
    client = _DummyClient(raw_response)
    llm = OpenAITextChatLLMImpl(client=cast("Any", client), model="gpt-test")

    result = await llm._execute_llm("hello", {})  # noqa: SLF001

    assert result.content == "Hello World"
    messages = [record.message for record in caplog.records]
    assert any("Coerced streaming response" in msg for msg in messages)
