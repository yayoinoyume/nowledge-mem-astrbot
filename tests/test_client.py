"""客户端超时、认证头、明文 HTTP 警告、搜索/thread 响应归一化与错误码。"""

from __future__ import annotations

import asyncio

import pytest

from core.client import NmemClient, NmemError
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


def test_search_posts_normal_mode_and_parses_post_shape() -> None:
    payload = [
        {
            "memory": {
                "id": "m1",
                "title": "标题",
                "content": "内容",
                "unit_type": "decision",
                "importance": 0.7,
                "metadata": {"source_thread_id": "t9"},
            },
            "similarity_score": 0.9,
        }
    ]
    session = FakeSession([FakeResponse(200, payload)])
    client = _client(session)
    result = asyncio.run(client.search("hello", limit=3, space_id="space"))
    assert result == [
        {
            "id": "m1",
            "title": "标题",
            "content": "内容",
            "score": 0.9,
            "unit_type": "decision",
            "importance": 0.7,
            "labels": [],
            "source_thread": "t9",
        }
    ]
    assert session.calls[0]["kwargs"]["json"] == {
        "query": "hello",
        "limit": 3,
        "mode": "normal",
        "space_id": "space",
    }


def test_search_parses_get_shape() -> None:
    payload = {
        "memories": [
            {
                "id": "m2",
                "title": "旧记忆",
                "content": "正文",
                "relevance_score": 0.88,
                "label_ids": ["l1", "l2"],
                "source_thread": "t3",
            }
        ],
        "pagination": {"total": 1},
        "search_metadata": {},
    }
    session = FakeSession([FakeResponse(200, payload)])
    client = _client(session)
    result = asyncio.run(client.search("hello", limit=5, space_id="space", mode="deep"))
    assert result == [
        {
            "id": "m2",
            "title": "旧记忆",
            "content": "正文",
            "score": 0.88,
            "unit_type": "memory",
            "importance": 0.0,
            "labels": ["l1", "l2"],
            "source_thread": "t3",
        }
    ]


def test_search_empty_get_shape_is_empty_not_error() -> None:
    session = FakeSession([
        FakeResponse(200, {"memories": [], "pagination": {"total": 0}, "search_metadata": {}})
    ])
    client = _client(session)
    assert asyncio.run(client.search("hello", limit=5, space_id="space")) == []


def test_search_null_memories_is_empty() -> None:
    session = FakeSession([FakeResponse(200, {"memories": None, "pagination": {"total": 0}})])
    client = _client(session)
    assert asyncio.run(client.search("hello", limit=5, space_id="space")) == []


def test_search_invalid_shape_raises() -> None:
    session = FakeSession([FakeResponse(200, {"unexpected": []})])
    client = _client(session)
    with pytest.raises(NmemError):
        asyncio.run(client.search("hello", limit=5, space_id="space"))


def test_timeout_raises_nmem_error() -> None:
    session = FakeSession([
        FakeResponse(200, error=asyncio.TimeoutError()),
        FakeResponse(200, error=asyncio.TimeoutError()),
    ])
    client = _client(session)
    with pytest.raises(NmemError):
        asyncio.run(client.context_bundle(agent_id="a", source_app="astrbot", space_id="space"))
    with pytest.raises(NmemError):
        asyncio.run(client.search("hello", limit=3, space_id="space"))


def test_http_500_raises_with_status() -> None:
    session = FakeSession([FakeResponse(500, {})])
    client = _client(session)
    with pytest.raises(NmemError) as info:
        asyncio.run(client.search("hello", limit=3, space_id="space"))
    assert info.value.status == 500


def test_non_json_response_raises() -> None:
    session = FakeSession([FakeResponse(200, error=ValueError("not json"))])
    client = _client(session)
    with pytest.raises(NmemError):
        asyncio.run(client.context_bundle(agent_id="a", source_app="astrbot", space_id="space"))


def test_search_threads_uses_get_query() -> None:
    session = FakeSession([
        FakeResponse(200, {"threads": [{"thread_id": "t1"}], "total_found": 1})
    ])
    client = _client(session)
    result = asyncio.run(client.search_threads("hello", limit=3, space_id="space"))
    assert result["total_found"] == 1
    call = session.calls[0]
    assert call["method"] == "GET"
    assert call["url"] == "http://127.0.0.1:14242/threads/search"
    assert call["kwargs"]["params"] == {
        "query": "hello",
        "limit": 3,
        "space_id": "space",
    }


def test_save_memory_posts_identity_fields() -> None:
    session = FakeSession([FakeResponse(200, {"id": "m1", "memory": {"title": "标题"}})])
    client = _client(session)
    result = asyncio.run(
        client.save_memory(
            "内容",
            title="标题",
            unit_type="decision",
            importance=0.8,
            labels=["alpha", "beta"],
            space_id="space",
            agent_id="astrbot",
            source_app="astrbot",
        )
    )
    assert result["id"] == "m1"
    call = session.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == "http://127.0.0.1:14242/memories"
    assert call["kwargs"]["json"] == {
        "content": "内容",
        "importance": 0.8,
        "unit_type": "decision",
        "source": "astrbot",
        "source_app": "astrbot",
        "metadata": {"import_origin": "astrbot"},
        "title": "标题",
        "labels": ["alpha", "beta"],
        "space_id": "space",
        "agent_id": "astrbot",
    }


def test_health_get() -> None:
    session = FakeSession([
        FakeResponse(200, {"status": "ok"}),
        FakeResponse(200, {"status": "ok", "server_version": "1.2.3"}),
    ])
    client = _client(session)
    assert asyncio.run(client.health()) == {"status": "ok"}
    assert asyncio.run(client.health_details()) == {"status": "ok", "server_version": "1.2.3"}
    assert session.calls[0]["url"] == "http://127.0.0.1:14242/health"
    assert session.calls[1]["url"] == "http://127.0.0.1:14242/health/details"


def test_has_pending_reviews_parses_feed_shape() -> None:
    session = FakeSession([
        FakeResponse(200, {"events": [{"id": "r1"}], "total": 1}),
        FakeResponse(200, {"events": [], "total": 0}),
    ])
    client = _client(session)
    assert asyncio.run(client.has_pending_reviews()) is True
    assert asyncio.run(client.has_pending_reviews()) is False
    call = session.calls[0]
    assert call["method"] == "GET"
    assert call["url"] == "http://127.0.0.1:14242/agent/feed/events"
    assert call["kwargs"]["params"] == {
        "review_only": "true",
        "unresolved_only": "true",
        "last_n_days": 365,
        "limit": 1,
        "include_total": "false",
    }


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
    with pytest.raises(NmemError):
        asyncio.run(client.search("hello", limit=3, space_id="space"))
    assert session.calls == []


def test_plain_http_remote_warns() -> None:
    logger, messages = recording_logger("plain-http")
    NmemClient("http://example.com:14242", "secret-key", logger=logger)
    assert any("明文" in item for item in messages)


def test_search_threads_bare_array_is_normalized() -> None:
    session = FakeSession([FakeResponse(200, [{"thread_id": "t1", "title": "对话"}])])
    client = _client(session)
    result = asyncio.run(client.search_threads("hello", limit=3, space_id="space"))
    assert result == {"threads": [{"thread_id": "t1", "title": "对话"}]}


def test_search_threads_results_key_is_normalized() -> None:
    session = FakeSession([FakeResponse(200, {"results": [{"thread_id": "t2"}]})])
    client = _client(session)
    result = asyncio.run(client.search_threads("hello", limit=3, space_id="space"))
    assert result["threads"] == [{"thread_id": "t2"}]


def test_search_threads_error_object_raises() -> None:
    session = FakeSession([FakeResponse(200, {"error": "boom"})])
    client = _client(session)
    with pytest.raises(NmemError):
        asyncio.run(client.search_threads("hello", limit=3, space_id="space"))


def test_search_threads_non_list_raises() -> None:
    session = FakeSession([FakeResponse(200, {"threads": "not-a-list"})])
    client = _client(session)
    with pytest.raises(NmemError):
        asyncio.run(client.search_threads("hello", limit=3, space_id="space"))


def test_search_threads_missing_threads_raises() -> None:
    session = FakeSession([FakeResponse(200, {"unexpected": []})])
    client = _client(session)
    with pytest.raises(NmemError):
        asyncio.run(client.search_threads("hello", limit=3, space_id="space"))


def test_search_threads_null_threads_is_empty() -> None:
    """服务端无结果时可能返回 {"threads": null}：按空列表处理，不当成错误。"""
    session = FakeSession([FakeResponse(200, {"threads": None, "total_found": 0})])
    client = _client(session)
    result = asyncio.run(client.search_threads("hello", limit=3, space_id="space"))
    assert result["threads"] == []


def test_search_timeout_override_is_used() -> None:
    session = FakeSession([
        FakeResponse(200, {"memories": []}),
        FakeResponse(200, {"threads": []}),
    ])
    client = _client(session)
    asyncio.run(client.search("hello", limit=3, space_id="space", timeout=4.0))
    asyncio.run(client.search_threads("hello", limit=3, space_id="space", timeout=4.0))
    assert session.calls[0]["kwargs"]["timeout"].total == 4.0
    assert session.calls[1]["kwargs"]["timeout"].total == 4.0


def test_save_memory_posts_provenance_fields() -> None:
    session = FakeSession([FakeResponse(200, {"id": "m1"})])
    client = _client(session)
    asyncio.run(
        client.save_memory(
            "内容",
            title="标题",
            unit_type="fact",
            importance=0.6,
            labels=[],
            space_id="space",
            agent_id="astrbot",
            source_app="astrbot",
            source_umo="qq:group:1",
            sender_hash="0123456789abcdef",
        )
    )
    body = session.calls[0]["kwargs"]["json"]
    # 服务端没有这三个顶层字段，传了会被静默忽略；溯源必须落在 metadata 里。
    assert "source_umo" not in body
    assert "sender_hash" not in body
    assert "import_origin" not in body
    assert body["metadata"] == {
        "import_origin": "astrbot",
        "source_umo": "qq:group:1",
        "sender_hash": "0123456789abcdef",
    }


def test_get_memory_uses_get_and_space_guard() -> None:
    session = FakeSession([FakeResponse(200, {"id": "m1", "content": "正文"})])
    client = _client(session)
    result = asyncio.run(client.get_memory("m1", space_id="space"))
    assert result["id"] == "m1"
    call = session.calls[0]
    assert call["method"] == "GET"
    assert call["url"] == "http://127.0.0.1:14242/memories/m1"
    assert call["kwargs"]["params"] == {"space_id": "space"}


def test_get_memory_quotes_id_and_omits_empty_space() -> None:
    session = FakeSession([FakeResponse(200, {"id": "a/b"})])
    client = _client(session)
    asyncio.run(client.get_memory("a/b"))
    call = session.calls[0]
    assert call["url"] == "http://127.0.0.1:14242/memories/a%2Fb"
    assert call["kwargs"]["params"] is None


def test_get_memory_non_object_raises() -> None:
    session = FakeSession([FakeResponse(200, [])])
    client = _client(session)
    with pytest.raises(NmemError):
        asyncio.run(client.get_memory("m1"))


def test_get_thread_passes_range_and_returns_messages() -> None:
    session = FakeSession([
        FakeResponse(200, {"thread": {"thread_id": "t1"}, "messages": [], "total_messages": 0})
    ])
    client = _client(session)
    result = asyncio.run(
        client.get_thread("t1", offset=5, limit=10, space_id="space")
    )
    assert result["total_messages"] == 0
    call = session.calls[0]
    assert call["method"] == "GET"
    assert call["url"] == "http://127.0.0.1:14242/threads/t1"
    assert call["kwargs"]["params"] == {
        "offset": 5,
        "limit": 10,
        "space_id": "space",
    }


def test_get_thread_missing_messages_raises() -> None:
    session = FakeSession([FakeResponse(200, {"thread": {}})])
    client = _client(session)
    with pytest.raises(NmemError):
        asyncio.run(client.get_thread("t1"))


def test_expand_graph_passes_depth_limit_and_space() -> None:
    session = FakeSession([
        FakeResponse(200, {"center_node": "m1", "neighbors": [], "edges": []})
    ])
    client = _client(session)
    result = asyncio.run(
        client.expand_graph("m1", depth=2, limit=10, space_id="space")
    )
    assert result["center_node"] == "m1"
    call = session.calls[0]
    assert call["method"] == "GET"
    assert call["url"] == "http://127.0.0.1:14242/graph/expand/m1"
    assert call["kwargs"]["params"] == {
        "depth": 2,
        "limit": 10,
        "include_metadata": "true",
        "space_id": "space",
    }


def test_get_thread_drops_related_memories_and_entities() -> None:
    session = FakeSession([
        FakeResponse(
            200,
            {
                "thread": {"thread_id": "t1"},
                "messages": [{"role": "user", "content": "hi"}],
                "related_memories": [{"id": "m1"}],
                "entities": ["e1"],
                "total_messages": 1,
                "total_tokens": 0,
            },
        )
    ])
    client = _client(session)
    result = asyncio.run(client.get_thread("t1", offset=0, limit=2))
    assert set(result) == {"thread", "messages", "total_messages"}
    assert "related_memories" not in result
    assert "entities" not in result
    assert "total_tokens" not in result


def test_working_memory_uses_get_and_space() -> None:
    session = FakeSession([
        FakeResponse(
            200,
            {"rendered_markdown": "# WM", "file_path": "/srv/secret/wm.md"},
        )
    ])
    client = _client(session)
    result = asyncio.run(client.working_memory(space_id="space"))
    assert result["rendered_markdown"] == "# WM"
    assert "file_path" not in result
    call = session.calls[0]
    assert call["method"] == "GET"
    assert call["url"] == "http://127.0.0.1:14242/agent/working-memory"
    assert call["kwargs"]["params"] == {"space_id": "space"}


def test_evolves_passes_memory_id_and_limit() -> None:
    session = FakeSession([FakeResponse(200, {"edges": [], "total": 0})])
    client = _client(session)
    result = asyncio.run(client.evolves("m1", limit=5))
    assert result["total"] == 0
    call = session.calls[0]
    assert call["method"] == "GET"
    assert call["url"] == "http://127.0.0.1:14242/agent/evolves"
    assert call["kwargs"]["params"] == {"memory_id": "m1", "limit": 5}


def test_search_posts_recorded_date_window() -> None:
    session = FakeSession([FakeResponse(200, [])])
    client = _client(session)
    asyncio.run(
        client.search(
            "hello",
            limit=3,
            space_id="space",
            recorded_date_from="2026-11-01",
            recorded_date_to="2026-11-30",
        )
    )
    assert session.calls[0]["kwargs"]["json"] == {
        "query": "hello",
        "limit": 3,
        "mode": "normal",
        "space_id": "space",
        "recorded_date_from": "2026-11-01",
        "recorded_date_to": "2026-11-30",
    }


def test_search_omits_empty_recorded_date_window() -> None:
    session = FakeSession([FakeResponse(200, [])])
    client = _client(session)
    asyncio.run(client.search("hello", limit=3, space_id="space"))
    body = session.calls[0]["kwargs"]["json"]
    for key in ("recorded_date_from", "recorded_date_to"):
        assert key not in body
