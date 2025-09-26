import httpx
import pytest

from types import SimpleNamespace

from fnllm.openai.llm.openai_text_chat_llm import OpenAITextChatLLMImpl


class _DummyRawResponse:
    def __init__(
        self,
        parsed,
        *,
        status_code: int = 400,
        body: str = "bad response",
        reason: str = "Bad Request",
    ):
        request = httpx.Request("POST", "http://example.com/v1/chat/completions")
        self.http_response = httpx.Response(
            status_code,
            request=request,
            text=body,
        )
        self.headers = self.http_response.headers
        self._parsed = parsed

    def parse(self):
        return self._parsed


class _DummyWithRawResponse:
    def __init__(self, raw_response):
        self._raw_response = raw_response

    async def create(self, *args, **kwargs):
        return self._raw_response


class _DummyCompletions:
    def __init__(self, raw_response):
        self.with_raw_response = _DummyWithRawResponse(raw_response)


class _DummyChat:
    def __init__(self, raw_response):
        self.completions = _DummyCompletions(raw_response)


class _DummyClient:
    def __init__(self, raw_response):
        self.chat = _DummyChat(raw_response)


@pytest.mark.asyncio
async def test_logs_raw_response_for_missing_choices(caplog):
    caplog.set_level("ERROR")
    raw_response = _DummyRawResponse(parsed="not-json", body="failure body")
    client = _DummyClient(raw_response)
    llm = OpenAITextChatLLMImpl(client=client, model="gpt-test")

    with pytest.raises(AttributeError):
        await llm._execute_llm("hello", {})

    messages = [record.message for record in caplog.records]
    assert any("missing 'choices'" in msg for msg in messages)

    details_entries = [
        getattr(record, "details", {}) for record in caplog.records if hasattr(record, "details")
    ]
    assert any(entry.get("response_body") == "failure body" for entry in details_entries)


@pytest.mark.asyncio
async def test_coerces_sse_response(caplog):
    caplog.set_level("WARNING")
    body = "\n".join(
        [
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}",
            "data: {\"choices\":[{\"delta\":{\"content\":\" World\"}}]}",
            "data: [DONE]",
        ]
    )
    raw_response = _DummyRawResponse(parsed=body, status_code=200, reason="OK", body=body)
    client = _DummyClient(raw_response)
    llm = OpenAITextChatLLMImpl(client=client, model="gpt-test")

    result = await llm._execute_llm("hello", {})

    assert result.content == "Hello World"
    messages = [record.message for record in caplog.records]
    assert any("Coerced streaming response" in msg for msg in messages)
