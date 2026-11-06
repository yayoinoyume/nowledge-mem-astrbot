"""客户端超时、认证头、明文 HTTP 警告与错误码。"""

from __future__ import annotations

import asyncio

from core.client import NmemClient
from tests.helpers import FakeResponse, FakeSession, recording_logger


def _client(session: FakeSession, **kwargs) -> NmemClient:
    return NmemClient("http://127.0.0.1:14242/", "secret-key", session=session, **kwargs)


def test_context_bundle_uses_bearer_and_query_params() -> None:
    session = FakeSession([FakeResponse(200, {"rendered_markdown": "# H", "compiled_hash": "abc"})])
    client = _client(session)
    result = asyncio.run(client.context_bundle(agent_id="astrbot", source_app="astrbot", space_id="space"))
    assert result == {"rendered_markdown": "# H", "compiled_hash": "abc"}
    call = session.calls[0]
    assert call["method"] == "GET"
    assert call["url"] == "http://127.0.0.1:14242/context/bundle"
    assert call["kwargs"]["headers"]["Authorization"] == "Bearer secret-key"
    assert call["kwargs"]["params"] == {
        "agent_id": "astrbot",
        "source_app": "astrbot",
        "space_id": "space",
        "include_working_memory": "true",
    }


def test_search_posts_query_and_returns_list() -> None:
    payload = [{"memory": {"id": "m1"}, "similarity_score": 0.9}]
    session = FakeSession([FakeResponse(200, payload)])
    client = _client(session)
    result = asyncio.run(client.search("hello", limit=3, space_id="space", mode="fast"))
    assert result == payload
    assert session.calls[0]["kwargs"]["json"] == {
        "query": "hello",
        "limit": 3,
        "mode": "fast",
        "space_id": "space",
    }


def test_timeout_returns_none_and_empty_list() -> None:
    session = FakeSession([FakeResponse(200, error=asyncio.TimeoutError()), FakeResponse(200, error=asyncio.TimeoutError())])
    client = _client(session)
    assert asyncio.run(client.context_bundle(agent_id="a", source_app="astrbot", space_id="space")) is None
    assert asyncio.run(client.search("hello", limit=3, space_id="space")) == []


def test_import_503_returns_none_without_retry() -> None:
    session = FakeSession([FakeResponse(503, {})])
    client = _client(session)
    assert asyncio.run(client.import_thread({"thread_id": "t"})) is None
    assert len(session.calls) == 1


def test_import_400_returns_none() -> None:
    session = FakeSession([FakeResponse(400, {"success": False})])
    client = _client(session)
    assert asyncio.run(client.import_thread({"thread_id": "t"})) is None
    assert len(session.calls) == 1


def test_unconfigured_client_skips_network() -> None:
    session = FakeSession([])
    client = NmemClient("", "", session=session)
    assert client.configured is False
    assert asyncio.run(client.import_thread({"thread_id": "t"})) is None
    assert session.calls == []


def test_plain_http_remote_warns() -> None:
    logger, messages = recording_logger("plain-http")
    NmemClient("http://example.com:14242", "secret-key", logger=logger)
    assert any("明文" in item for item in messages)
