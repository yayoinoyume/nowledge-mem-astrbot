"""三个 llm_tool 的参数校验、两种搜索形态、上下界、写权限、状态与失败路径。"""

from __future__ import annotations

import asyncio
import hashlib

from core.client import NmemError
from core.tools import (
    MAX_LABEL_CHARS,
    MAX_PREVIEW_CHARS,
    MAX_SAVE_CHARS,
    MAX_TITLE_CHARS,
    NmemTools,
)
from tests.helpers import FakeStore


class FakeToolClient:
    def __init__(
        self,
        *,
        memories=None,
        threads=None,
        threads_response=None,
        save_result=None,
        health_result=None,
        health_details_result=None,
        search_error=None,
        save_error=None,
        health_error=None,
        health_details_error=None,
        attention_result=None,
        attention_error=None,
        get_memory_result=None,
        thread_result=None,
        graph_result=None,
        get_memory_error=None,
        thread_error=None,
        graph_error=None,
        working_memory_result=None,
        working_memory_error=None,
        bundle_error=None,
    ) -> None:
        self.configured = True
        self.api_url = "http://mem.local:14242"
        self.memories = memories or []
        self.threads = threads or []
        self.threads_response = threads_response
        self.save_result = save_result or {"id": "m-new", "title": "新记忆"}
        self.health_result = health_result or {"status": "ok"}
        self.health_details_result = health_details_result or {
            "status": "ok",
            "server_version": "0.10.115",
            "cli_version": "0.10.113",
        }
        self.search_error = search_error
        self.save_error = save_error
        self.health_error = health_error
        self.health_details_error = health_details_error
        self.attention_result = attention_result
        self.attention_error = attention_error
        self.get_memory_result = get_memory_result or {
            "id": "m1",
            "title": "标题",
            "content": "正文",
            "unit_type": "fact",
            "importance": 0.6,
            "labels": [],
            "metadata": {},
        }
        self.thread_result = thread_result or {
            "thread": {"thread_id": "t1", "title": "对话"},
            "messages": [],
            "total_messages": 0,
        }
        self.graph_result = graph_result or {
            "center_node": "m1",
            "depth": 1,
            "neighbors": [],
            "edges": [],
            "metadata": {},
        }
        self.get_memory_error = get_memory_error
        self.thread_error = thread_error
        self.graph_error = graph_error
        self.working_memory_result = working_memory_result or {
            "rendered_markdown": "# WM"
        }
        self.working_memory_error = working_memory_error
        self.bundle_error = bundle_error
        self.working_memory_calls = 0
        self.search_calls: list[dict] = []
        self.thread_calls: list[dict] = []
        self.get_memory_calls: list[dict] = []
        self.show_thread_calls: list[dict] = []
        self.graph_calls: list[dict] = []
        self.save_calls: list[dict] = []
        self.health_calls = 0
        self.health_details_calls = 0
        self.attention_calls = 0
        self.bundle_calls = 0

    async def search(self, query, **kwargs):
        self.search_calls.append({"query": query, **kwargs})
        if self.search_error:
            raise self.search_error
        return list(self.memories)

    async def search_threads(self, query, **kwargs):
        self.thread_calls.append({"query": query, **kwargs})
        if self.search_error:
            raise self.search_error
        if self.threads_response is not None:
            return self.threads_response
        return {"threads": list(self.threads), "total_found": len(self.threads)}

    async def get_memory(self, memory_id, **kwargs):
        self.get_memory_calls.append({"memory_id": memory_id, **kwargs})
        if self.get_memory_error:
            raise self.get_memory_error
        return dict(self.get_memory_result)

    async def get_thread(self, thread_id, **kwargs):
        self.show_thread_calls.append({"thread_id": thread_id, **kwargs})
        if self.thread_error:
            raise self.thread_error
        return dict(self.thread_result)

    async def expand_graph(self, node_id, **kwargs):
        self.graph_calls.append({"node_id": node_id, **kwargs})
        if self.graph_error:
            raise self.graph_error
        return dict(self.graph_result)

    async def working_memory(self, **kwargs):
        self.working_memory_calls += 1
        if self.working_memory_error:
            raise self.working_memory_error
        return dict(self.working_memory_result)

    async def save_memory(self, content, **kwargs):
        self.save_calls.append({"content": content, **kwargs})
        if self.save_error:
            raise self.save_error
        return dict(self.save_result)

    async def health_details(self):
        self.health_details_calls += 1
        if self.health_details_error:
            raise self.health_details_error
        return dict(self.health_details_result)

    async def has_pending_reviews(self):
        self.attention_calls += 1
        if self.attention_error:
            raise self.attention_error
        return self.attention_result

    async def health(self):
        self.health_calls += 1
        if self.health_error:
            raise self.health_error
        return dict(self.health_result)

    async def context_bundle(self, **kwargs):
        self.bundle_calls += 1
        if self.bundle_error:
            raise self.bundle_error
        return {"rendered_markdown": "# H"}


def _tools(
    client: FakeToolClient | None = None,
    store: FakeStore | None = None,
    *,
    save_admin_only: bool = False,
    timeout_seconds: float = 3.0,
) -> NmemTools:
    return NmemTools(
        client or FakeToolClient(),
        store or FakeStore(),
        space_id="space",
        agent_id="astrbot",
        source_app="astrbot",
        identity="identity123",
        plugin_version="0.17.0",
        save_admin_only=save_admin_only,
        timeout_seconds=timeout_seconds,
    )


def _memory(**overrides):
    item = {
        "id": "m1",
        "title": "标题",
        "content": "内容",
        "score": 0.9,
        "unit_type": "decision",
        "importance": 0.7,
        "labels": [],
        "source_thread": "",
    }
    item.update(overrides)
    return item


def test_search_rejects_empty_query() -> None:
    result = asyncio.run(_tools().search("   "))
    assert result.startswith("error: query 不能为空")


def test_search_rejects_invalid_scope() -> None:
    result = asyncio.run(_tools().search("查询", scope="everything"))
    assert result.startswith("error: scope 必须是")


def test_search_rejects_invalid_mode() -> None:
    result = asyncio.run(_tools().search("查询", mode="turbo"))
    assert result.startswith("error: mode 必须是 normal 或 deep")


def test_search_defaults_to_limit_10_and_clamps_to_20() -> None:
    client = FakeToolClient(memories=[_memory()])
    asyncio.run(_tools(client).search("查询"))
    asyncio.run(_tools(client).search("查询", limit=99))
    assert client.search_calls[0]["limit"] == 10
    assert client.search_calls[1]["limit"] == 20


def test_search_passes_mode() -> None:
    client = FakeToolClient(memories=[_memory()])
    asyncio.run(_tools(client).search("查询", mode="deep"))
    assert client.search_calls[0]["mode"] == "deep"


def test_search_memories_formats_and_truncates() -> None:
    client = FakeToolClient(
        memories=[
            _memory(
                id="m1",
                title="决策标题",
                content="x" * 500,
                score=0.91,
                source_thread="t-source",
            )
        ]
    )
    result = asyncio.run(_tools(client).search("查询"))
    assert "matches: 1" in result
    assert "[memory] id=m1 score=0.91" in result
    assert "title: 决策标题" in result
    assert "source_thread: t-source" in result
    content_line = next(line for line in result.splitlines() if line.startswith("content: "))
    assert len(content_line) == len("content: ") + MAX_PREVIEW_CHARS
    assert content_line.endswith("…")


def test_search_threads_formats_snippets() -> None:
    client = FakeToolClient(
        threads=[
            {
                "thread_id": "t1",
                "title": "旧对话",
                "source": "astrbot",
                "relevance_score": 0.7,
                "matched_messages": [{"role": "user", "snippet": "当时说的是这件事"}],
            }
        ]
    )
    result = asyncio.run(_tools(client).search("查询", scope="threads"))
    assert "[thread] id=t1 score=0.70" in result
    assert "title: 旧对话" in result
    assert "content: 当时说的是这件事" in result
    assert "source_thread: t1" in result
    assert "source: astrbot" in result
    assert client.search_calls == []


def test_search_both_merges_and_dedupes() -> None:
    memory = _memory(id="m1", title="记忆", content="内容", score=0.9)
    client = FakeToolClient(
        memories=[memory, memory],
        threads=[
            {"thread_id": "t1", "title": "对话", "relevance_score": 0.8},
            {"thread_id": "t1", "title": "重复对话", "relevance_score": 0.5},
        ],
    )
    result = asyncio.run(_tools(client).search("查询", scope="both", limit=10))
    assert result.count("[memory] id=m1") == 1
    assert result.count("[thread] id=t1") == 1
    assert "matches: 2" in result


def test_search_both_caps_at_limit() -> None:
    memories = [
        _memory(id=f"m{i}", title=f"记忆{i}", content="内容", score=0.9 - i / 100)
        for i in range(5)
    ]
    threads = [
        {"thread_id": f"t{i}", "title": f"对话{i}", "relevance_score": 0.8 - i / 100}
        for i in range(5)
    ]
    client = FakeToolClient(memories=memories, threads=threads)
    result = asyncio.run(_tools(client).search("查询", scope="both", limit=3))
    assert "matches: 3" in result
    assert result.count(". [") == 3


def test_search_empty_returns_no_matches() -> None:
    result = asyncio.run(_tools().search("查不到", scope="both"))
    assert result.startswith('no matches: query="查不到"')


def test_search_http_500_is_reported_not_empty() -> None:
    client = FakeToolClient(search_error=NmemError("HTTP 500", status=500))
    result = asyncio.run(_tools(client).search("查询"))
    assert result.startswith("error: Nowledge Mem 搜索失败：HTTP 500")
    assert "no matches" not in result


def test_search_timeout_is_reported_not_empty() -> None:
    client = FakeToolClient(search_error=NmemError("请求失败：TimeoutError"))
    result = asyncio.run(_tools(client).search("查询"))
    assert result.startswith("error: Nowledge Mem 搜索失败：请求失败：TimeoutError")
    assert "no matches" not in result


def test_save_success_passes_identity_and_labels() -> None:
    client = FakeToolClient(save_result={"id": "m-new", "title": "标题"})
    result = asyncio.run(
        _tools(client).save(
            "值得保存的内容",
            title="标题",
            unit_type="decision",
            importance=0.8,
            labels=["alpha", "alpha", " beta "],
        )
    )
    assert result.startswith("saved: id=m-new title=标题 unit_type=decision importance=0.80")
    call = client.save_calls[0]
    assert call["content"] == "值得保存的内容"
    assert call["labels"] == ["alpha", "beta"]
    assert call["space_id"] == "space"
    assert call["agent_id"] == "astrbot"
    assert call["source_app"] == "astrbot"


def test_save_empty_title_is_omitted_for_server_generation() -> None:
    client = FakeToolClient(save_result={"id": "m-auto"})
    result = asyncio.run(_tools(client).save("内容", title="  "))
    assert "id=m-auto" in result
    assert client.save_calls[0]["title"] == ""


def test_save_missing_memory_id_is_error() -> None:
    client = FakeToolClient(save_result={"title": "没有 id"})
    result = asyncio.run(_tools(client).save("内容"))
    assert result.startswith("error: Nowledge Mem 保存响应缺少 memory id")


def test_save_rejects_empty_content() -> None:
    result = asyncio.run(_tools().save("  "))
    assert result.startswith("error: content 不能为空")


def test_save_rejects_invalid_unit_type() -> None:
    result = asyncio.run(_tools().save("内容", unit_type="unknown"))
    assert result.startswith("error: unit_type 必须是")


def test_save_truncates_long_content_and_warns() -> None:
    client = FakeToolClient()
    result = asyncio.run(_tools(client).save("x" * (MAX_SAVE_CHARS + 1)))
    assert len(client.save_calls[0]["content"]) == MAX_SAVE_CHARS
    assert "已截断后保存" in result


def test_save_clamps_importance() -> None:
    client = FakeToolClient()
    asyncio.run(_tools(client).save("内容", importance=1.8))
    asyncio.run(_tools(client).save("内容", importance=-0.2))
    asyncio.run(_tools(client).save("内容", importance="0.4"))
    assert client.save_calls[0]["importance"] == 1.0
    assert client.save_calls[1]["importance"] == 0.0
    assert client.save_calls[2]["importance"] == 0.4


def test_save_http_error_is_reported() -> None:
    client = FakeToolClient(save_error=NmemError("HTTP 400", status=400))
    result = asyncio.run(_tools(client).save("内容"))
    assert result.startswith("error: Nowledge Mem 保存失败：HTTP 400")


def test_save_timeout_is_reported() -> None:
    client = FakeToolClient(save_error=NmemError("请求失败：TimeoutError"))
    result = asyncio.run(_tools(client).save("内容"))
    assert result.startswith("error: Nowledge Mem 保存失败：请求失败：TimeoutError")


def test_status_reports_versions_attention_pending_and_last_sync() -> None:
    store = FakeStore()
    store.data[("pending", "test", "umo")] = {"attempts": 1}
    store.data[("cursor", "test", "umo")] = {"synced_at": "2026-11-07T10:00:00+00:00"}
    client = FakeToolClient(
        health_details_result={
            "status": "ok",
            "server_version": "0.10.115",
            "cli_version": "0.10.113",
            "attention": {"has_pending_reviews": True, "count_is_exact": False},
        }
    )
    result = asyncio.run(_tools(client, store).status("umo"))
    assert "connected: true" in result
    assert "api_url: http://mem.local:14242" in result
    assert "space: space" in result
    assert "agent: astrbot" in result
    assert "identity: identity123" in result
    assert "api_key: configured" in result
    assert "server_version: 0.10.115" in result
    assert "cli_version: 0.10.113" in result
    assert "attention_has_pending_reviews: true (非精确计数)" in result
    assert "pending: 1 (本会话)" in result
    assert "last_sync: 2026-11-07T10:00:00+00:00" in result
    assert "plugin_version: 0.17.0" in result
    assert client.attention_calls == 0


def test_status_attention_from_feed_is_labeled_non_exact() -> None:
    client = FakeToolClient(attention_result=True)
    result = asyncio.run(_tools(client).status("umo"))
    assert "attention_has_pending_reviews: true (非精确计数)" in result
    assert client.attention_calls == 1


def test_status_unconfigured_reports_error() -> None:
    client = FakeToolClient()
    client.configured = False
    result = asyncio.run(_tools(client).status("umo"))
    assert "connected: false" in result
    assert "api_key: missing" in result
    assert "error: 未配置 api_url 或 api_key" in result


def test_status_falls_back_to_health_then_context_bundle_on_404() -> None:
    client = FakeToolClient(
        health_details_error=NmemError("HTTP 404", status=404),
        health_error=NmemError("HTTP 404", status=404),
    )
    result = asyncio.run(_tools(client).status("umo"))
    assert "connected: true" in result
    assert "backend: /context/bundle" in result
    assert client.bundle_calls == 1


def test_status_falls_back_to_health_when_details_404() -> None:
    client = FakeToolClient(health_details_error=NmemError("HTTP 404", status=404))
    result = asyncio.run(_tools(client).status("umo"))
    assert "connected: true" in result
    assert "backend: ok" in result
    assert "cli_version: n/a (插件无 CLI)" in result
    assert client.health_calls == 1


def test_status_disconnected_on_health_details_failure() -> None:
    client = FakeToolClient(health_details_error=NmemError("HTTP 500", status=500))
    result = asyncio.run(_tools(client).status("umo"))
    assert "connected: false" in result
    assert "error: HTTP 500" in result
    assert client.health_calls == 0


def test_status_disconnected_on_timeout() -> None:
    client = FakeToolClient(health_details_error=NmemError("请求失败：TimeoutError"))
    result = asyncio.run(_tools(client).status("umo"))
    assert "connected: false" in result
    assert "error: 请求失败：TimeoutError" in result


def test_status_never_prints_api_key() -> None:
    client = FakeToolClient()
    client.api_key = "super-secret-key"
    result = asyncio.run(_tools(client).status("umo"))
    assert "super-secret-key" not in result


def test_save_clips_title_and_labels_before_sending() -> None:
    client = FakeToolClient()
    asyncio.run(
        _tools(client).save("内容", title="T" * 100_000, labels=["L" * 50_000, "ok"])
    )
    call = client.save_calls[0]
    assert len(call["title"]) == MAX_TITLE_CHARS
    assert [len(label) for label in call["labels"]] == [MAX_LABEL_CHARS, 2]


def test_save_returned_title_is_previewed() -> None:
    client = FakeToolClient(save_result={"id": "m-new", "title": "T" * 100_000})
    result = asyncio.run(_tools(client).save("内容", title="短标题"))
    returned = result.split("title=", 1)[1].split(" unit_type=", 1)[0]
    assert len(returned) == MAX_TITLE_CHARS
    assert returned.endswith("…")


def test_search_formats_long_title_with_preview() -> None:
    client = FakeToolClient(memories=[_memory(title="T" * 100_000)])
    result = asyncio.run(_tools(client).search("查询"))
    title_line = next(line for line in result.splitlines() if line.startswith("title: "))
    assert len(title_line) == len("title: ") + MAX_TITLE_CHARS
    assert title_line.endswith("…")


def test_search_threads_error_object_is_error_not_no_matches() -> None:
    client = FakeToolClient(threads_response={"error": "boom"})
    result = asyncio.run(_tools(client).search("查询", scope="threads"))
    assert result.startswith("error: ")
    assert "no matches" not in result


def test_search_threads_non_list_threads_is_error() -> None:
    client = FakeToolClient(threads_response={"threads": "not-a-list"})
    result = asyncio.run(_tools(client).search("查询", scope="threads"))
    assert result.startswith("error: thread 搜索响应缺少 threads 列表")
    assert "no matches" not in result


def test_search_threads_missing_threads_is_error() -> None:
    client = FakeToolClient(threads_response={"unexpected": []})
    result = asyncio.run(_tools(client).search("查询", scope="threads"))
    assert result.startswith("error: thread 搜索响应缺少 threads 列表")
    assert "no matches" not in result


def test_save_admin_only_rejects_member() -> None:
    client = FakeToolClient()
    result = asyncio.run(
        _tools(client, save_admin_only=True).save("内容", is_admin=False)
    )
    assert result == "error: 仅管理员可保存"
    assert client.save_calls == []


def test_save_admin_only_allows_admin() -> None:
    client = FakeToolClient()
    result = asyncio.run(
        _tools(client, save_admin_only=True).save("内容", is_admin=True)
    )
    assert result.startswith("saved: id=m-new")
    assert len(client.save_calls) == 1


def test_save_default_allows_member() -> None:
    client = FakeToolClient()
    result = asyncio.run(_tools(client).save("内容", is_admin=False))
    assert result.startswith("saved: id=m-new")
    assert len(client.save_calls) == 1


def test_save_sends_provenance_without_plaintext_sender() -> None:
    client = FakeToolClient()
    asyncio.run(_tools(client).save("内容", umo="qq:group:1", sender_id="123456789"))
    call = client.save_calls[0]
    assert call["source_umo"] == "qq:group:1"
    assert call["sender_hash"] == hashlib.sha1(b"123456789").hexdigest()[:16]
    assert len(call["sender_hash"]) == 16


def test_save_without_sender_sends_empty_provenance() -> None:
    client = FakeToolClient()
    asyncio.run(_tools(client).save("内容"))
    call = client.save_calls[0]
    assert call["source_umo"] == ""
    assert call["sender_hash"] == ""


def test_save_non_finite_importance_falls_back_to_default() -> None:
    client = FakeToolClient()
    for value in (float("nan"), float("inf"), float("-inf"), "abc"):
        asyncio.run(_tools(client).save("内容", importance=value))
    assert [call["importance"] for call in client.save_calls] == [0.6, 0.6, 0.6, 0.6]


def test_search_non_finite_limit_falls_back_to_10() -> None:
    client = FakeToolClient(memories=[_memory()])
    for value in (float("nan"), float("inf"), float("-inf"), "abc"):
        asyncio.run(_tools(client).search("查询", limit=value))
    assert [call["limit"] for call in client.search_calls] == [10, 10, 10, 10]


def test_status_redacts_api_url_userinfo_and_query() -> None:
    password = "pw-" + "123"
    client = FakeToolClient()
    client.api_url = f"http://yayoi:{password}@mem.local:14242/api?token=abc"
    result = asyncio.run(_tools(client).status("umo"))
    assert "api_url: http://mem.local:14242" in result
    assert "yayoi" not in result
    assert password not in result
    assert "token" not in result


def test_search_deep_doubles_timeout_and_caps_at_15() -> None:
    client = FakeToolClient(memories=[_memory()])
    asyncio.run(_tools(client, timeout_seconds=2).search("查询", mode="deep"))
    asyncio.run(_tools(client, timeout_seconds=10).search("查询", mode="deep"))
    asyncio.run(_tools(client, timeout_seconds=2).search("查询", mode="normal"))
    assert [call["timeout"] for call in client.search_calls] == [4, 15, 2]


def test_get_memory_formats_full_detail() -> None:
    client = FakeToolClient(
        get_memory_result={
            "id": "m1",
            "title": "决策标题",
            "content": "第一行\n第二行",
            "unit_type": "decision",
            "importance": 0.75,
            "labels": ["alpha", "beta"],
            "metadata": {"source_umo": "qq:group:1"},
            "source_thread": {"id": "t9", "title": "来源"},
        }
    )
    result = asyncio.run(_tools(client).get_memory("m1"))
    assert "memory: id=m1" in result
    assert "title: 决策标题" in result
    assert "unit_type: decision" in result
    assert "importance: 0.75" in result
    assert "labels: alpha, beta" in result
    assert '"source_umo": "qq:group:1"' in result
    assert "source_thread: t9" in result
    assert "第一行\n第二行" in result
    assert client.get_memory_calls[0] == {"memory_id": "m1", "space_id": "space"}


def test_get_memory_falls_back_to_label_ids_and_empty_metadata() -> None:
    client = FakeToolClient(
        get_memory_result={
            "id": "m1",
            "title": "",
            "content": "",
            "unit_type": "fact",
            "importance": 0.4,
            "label_ids": ["label_a"],
        }
    )
    result = asyncio.run(_tools(client).get_memory("m1"))
    assert "title: (无标题)" in result
    assert "labels: label_a" in result
    assert "metadata: {}" in result
    assert "content:\n(空)" in result


def test_get_memory_rejects_empty_id() -> None:
    client = FakeToolClient()
    result = asyncio.run(_tools(client).get_memory("  "))
    assert result == "error: memory_id 不能为空"
    assert client.get_memory_calls == []


def test_get_memory_http_error_is_reported() -> None:
    client = FakeToolClient(
        get_memory_error=NmemError("HTTP 404", status=404)
    )
    result = asyncio.run(_tools(client).get_memory("m1"))
    assert result.startswith("error: Nowledge Mem 读取记忆失败：HTTP 404")


def test_get_memory_timeout_is_reported() -> None:
    client = FakeToolClient(
        get_memory_error=NmemError("请求失败：TimeoutError")
    )
    result = asyncio.run(_tools(client).get_memory("m1"))
    assert result.startswith("error: Nowledge Mem 读取记忆失败：请求失败：TimeoutError")


def test_thread_show_formats_and_truncates_content() -> None:
    client = FakeToolClient(
        thread_result={
            "thread": {"thread_id": "t1", "title": "旧对话"},
            "messages": [
                {"role": "user", "content": "x" * 600},
                {"role": "assistant", "content": "收到"},
            ],
            "total_messages": 42,
        }
    )
    result = asyncio.run(_tools(client).thread_show("t1", offset=5, limit=10))
    assert "thread: id=t1 title=旧对话" in result
    assert "total_messages: 42" in result
    assert "range: offset=5 limit=10 returned=2" in result
    assert "1. [user] " + "x" * 499 + "…" in result
    assert "2. [assistant] 收到" in result
    assert client.show_thread_calls[0] == {
        "thread_id": "t1",
        "offset": 5,
        "limit": 10,
        "space_id": "space",
    }


def test_thread_show_clamps_range() -> None:
    client = FakeToolClient()
    asyncio.run(_tools(client).thread_show("t1", offset=-5, limit=999))
    asyncio.run(_tools(client).thread_show("t1", limit=0))
    assert client.show_thread_calls[0]["offset"] == 0
    assert client.show_thread_calls[0]["limit"] == 100
    assert client.show_thread_calls[1]["limit"] == 1


def test_thread_show_empty_messages() -> None:
    result = asyncio.run(_tools().thread_show("t1"))
    assert "messages: (empty)" in result


def test_thread_show_rejects_empty_id() -> None:
    client = FakeToolClient()
    result = asyncio.run(_tools(client).thread_show("  "))
    assert result == "error: thread_id 不能为空"
    assert client.show_thread_calls == []


def test_thread_show_http_error_is_reported() -> None:
    client = FakeToolClient(thread_error=NmemError("HTTP 500", status=500))
    result = asyncio.run(_tools(client).thread_show("t1"))
    assert result.startswith("error: Nowledge Mem 读取对话失败：HTTP 500")


def test_thread_show_timeout_is_reported() -> None:
    client = FakeToolClient(thread_error=NmemError("请求失败：TimeoutError"))
    result = asyncio.run(_tools(client).thread_show("t1"))
    assert result.startswith("error: Nowledge Mem 读取对话失败：请求失败：TimeoutError")


def test_graph_by_memory_id_formats_neighbors_and_edges() -> None:
    client = FakeToolClient(
        graph_result={
            "center_node": "m1",
            "depth": 2,
            "neighbors": [
                {"id": "e1", "label": "大肥鱼", "node_type": "entity"}
            ],
            "edges": [
                {
                    "source": "m1",
                    "target": "e1",
                    "edge_type": "MENTIONS",
                    "label": "提到",
                }
            ],
            "metadata": {},
        }
    )
    result = asyncio.run(_tools(client).graph(memory_id="m1", depth=2))
    assert "graph: center=m1 depth=2 neighbors=1 edges=1" in result
    assert "- [entity] 大肥鱼 (id=e1)" in result
    assert "- m1 -> e1 [MENTIONS] 提到" in result
    assert client.graph_calls[0] == {
        "node_id": "m1",
        "depth": 2,
        "limit": 50,
        "space_id": "space",
    }


def test_graph_by_query_searches_first() -> None:
    client = FakeToolClient(
        memories=[_memory(id="m9", title="主题")],
        graph_result={
            "center_node": "m9",
            "depth": 1,
            "neighbors": [],
            "edges": [],
            "metadata": {},
        },
    )
    result = asyncio.run(_tools(client).graph(query="主题"))
    assert "matched: id=m9 title=主题" in result
    assert "无邻居或节点不存在" in result
    assert client.search_calls[0]["limit"] == 1
    assert client.graph_calls[0]["node_id"] == "m9"


def test_graph_no_matches_does_not_expand() -> None:
    client = FakeToolClient(memories=[])
    result = asyncio.run(_tools(client).graph(query="查不到"))
    assert result.startswith('no matches: query="查不到"')
    assert client.graph_calls == []


def test_graph_requires_exactly_one_of_query_and_memory_id() -> None:
    client = FakeToolClient()
    assert asyncio.run(_tools(client).graph()).startswith("error: query 与 memory_id")
    assert asyncio.run(
        _tools(client).graph(query="主题", memory_id="m1")
    ).startswith("error: query 与 memory_id")
    assert client.search_calls == []
    assert client.graph_calls == []


def test_graph_clamps_depth() -> None:
    client = FakeToolClient()
    asyncio.run(_tools(client).graph(memory_id="m1", depth=9))
    asyncio.run(_tools(client).graph(memory_id="m1", depth=0))
    assert client.graph_calls[0]["depth"] == 5
    assert client.graph_calls[1]["depth"] == 1


def test_graph_metadata_error_is_reported() -> None:
    client = FakeToolClient(
        graph_result={
            "center_node": "m1",
            "depth": 1,
            "neighbors": [],
            "edges": [],
            "metadata": {"error": "boom"},
        }
    )
    result = asyncio.run(_tools(client).graph(memory_id="m1"))
    assert result == "error: 图扩展失败：boom"


def test_graph_http_error_is_reported() -> None:
    client = FakeToolClient(graph_error=NmemError("HTTP 500", status=500))
    result = asyncio.run(_tools(client).graph(memory_id="m1"))
    assert result.startswith("error: Nowledge Mem 图探索失败：HTTP 500")


def test_graph_timeout_is_reported() -> None:
    client = FakeToolClient(graph_error=NmemError("请求失败：TimeoutError"))
    result = asyncio.run(_tools(client).graph(memory_id="m1"))
    assert result.startswith("error: Nowledge Mem 图探索失败：请求失败：TimeoutError")


def test_skill_reads_existing_body_without_frontmatter() -> None:
    result = asyncio.run(_tools().skill("search-memory"))
    assert result.startswith("skill: search-memory")
    assert "## 什么时候搜" in result
    assert not result.startswith("---")


def test_skill_unknown_name_lists_available() -> None:
    result = asyncio.run(_tools().skill("unknown"))
    assert result.startswith("error: 未知技能 unknown")
    assert "search-memory" in result


def test_skill_empty_name_is_error() -> None:
    result = asyncio.run(_tools().skill("  "))
    assert result.startswith("error: name 不能为空")


def test_skill_missing_file_is_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("core.tools.SKILLS_DIR", tmp_path)
    result = asyncio.run(_tools().skill("status"))
    assert result == "error: 技能文件不存在：status"


def test_search_passes_recorded_date_window() -> None:
    client = FakeToolClient(memories=[_memory()])
    asyncio.run(
        _tools(client).search(
            "查询",
            recorded_date_from="2026-11-01",
            recorded_date_to="2026-11-30",
        )
    )
    call = client.search_calls[0]
    assert call["recorded_date_from"] == "2026-11-01"
    assert call["recorded_date_to"] == "2026-11-30"


def test_search_accepts_iso_datetime_and_normalizes() -> None:
    client = FakeToolClient(memories=[_memory()])
    asyncio.run(
        _tools(client).search(
            "查询", recorded_date_from="2026-11-07T10:00:00+08:00"
        )
    )
    assert client.search_calls[0]["recorded_date_from"] == "2026-11-07"


def test_search_rejects_invalid_date() -> None:
    client = FakeToolClient()
    result = asyncio.run(
        _tools(client).search("查询", recorded_date_from="2026/11/07")
    )
    assert result.startswith("error: recorded_date_from 必须是")
    assert client.search_calls == []


def test_search_rejects_reversed_date_range() -> None:
    client = FakeToolClient()
    result = asyncio.run(
        _tools(client).search(
            "查询",
            recorded_date_from="2026-12-31",
            recorded_date_to="2026-01-01",
        )
    )
    assert result == "error: recorded_date_from 不能晚于 recorded_date_to"
    assert client.search_calls == []


def test_search_defaults_empty_recorded_date_window() -> None:
    client = FakeToolClient(memories=[_memory()])
    asyncio.run(_tools(client).search("查询"))
    call = client.search_calls[0]
    assert call["recorded_date_from"] == ""
    assert call["recorded_date_to"] == ""


def test_get_memory_strips_file_path_recursively() -> None:
    client = FakeToolClient(
        get_memory_result={
            "id": "m1",
            "title": "标题",
            "content": "正文",
            "unit_type": "fact",
            "importance": 0.5,
            "metadata": {
                "file_path": "/srv/secret/a.md",
                "nested": [{"file_path": "/srv/secret/b.md", "keep": 1}],
            },
        }
    )
    result = asyncio.run(_tools(client).get_memory("m1"))
    assert "/srv/secret" not in result
    assert '"keep": 1' in result


def test_status_falls_back_to_working_memory_on_404() -> None:
    client = FakeToolClient(
        health_details_error=NmemError("HTTP 404", status=404),
        health_error=NmemError("HTTP 404", status=404),
        bundle_error=NmemError("HTTP 404", status=404),
    )
    result = asyncio.run(_tools(client).status("umo"))
    assert "connected: true" in result
    assert "backend: /agent/working-memory" in result
    assert client.working_memory_calls == 1
