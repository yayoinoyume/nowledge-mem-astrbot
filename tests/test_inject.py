"""Bundle 缓存与回退链、旧行为、召回与失败隔离。"""

from __future__ import annotations

import asyncio
import time

from core.inject import (
    GUIDANCE,
    ContextInjector,
    build_bundle_text,
    build_recall_text,
    normalize_query,
)
from tests.helpers import FakeStore, recording_logger


class FakeNmemClient:
    def __init__(self, bundle=None, results=None, fail_bundle=False) -> None:
        self.bundle = bundle
        self.results = results or []
        self.fail_bundle = fail_bundle
        self.bundle_calls = 0
        self.search_calls: list[tuple[str, dict]] = []

    async def context_bundle(self, **kwargs):
        self.bundle_calls += 1
        if self.fail_bundle:
            raise RuntimeError("bundle failed")
        return self.bundle

    async def search(self, query, **kwargs):
        self.search_calls.append((query, kwargs))
        return list(self.results)


def _injector(client: FakeNmemClient, **kwargs) -> ContextInjector:
    return ContextInjector(
        client,
        kwargs.pop("store", FakeStore()),
        agent_id="astrbot",
        source_app="astrbot",
        space_id="space",
        **kwargs,
    )


def _expire_bundle_cache(store: FakeStore, umo: str = "umo") -> None:
    store.data[("bundle", "test", umo)]["fetched_at"] = time.time() - 61


def test_normalize_query_strips_and_bounds() -> None:
    assert normalize_query("  你好   世界  ") == "你好 世界"
    assert normalize_query("  a  ") is None
    assert normalize_query("x" * 501) is None


def test_build_bundle_text_truncates_to_limit() -> None:
    text = build_bundle_text("y" * 13000)
    assert text.startswith("<nowledge_mem_context>\n")
    assert text.endswith("\n</nowledge_mem_context>")
    assert text.count("y") == 12000


def test_build_bundle_text_includes_guidance() -> None:
    assert GUIDANCE in build_bundle_text("# 上下文")


def test_build_recall_text_filters_low_score_and_truncates() -> None:
    results = [
        {"memory": {"title": "高分", "content": "a" * 500, "unit_type": "decision"}, "similarity_score": 0.9},
        {"memory": {"title": "低分", "content": "drop", "unit_type": "fact"}, "similarity_score": 0.1},
    ]
    text = build_recall_text("查询", results, 5)
    assert text is not None
    assert GUIDANCE in text
    assert "低分" not in text
    assert "[decision] 高分" in text
    assert text.split("：", 1)[1].split("\n", 1)[0].endswith("…")


def test_bundle_every_turn_injects_from_session_cache() -> None:
    store = FakeStore()
    client = FakeNmemClient(bundle={"rendered_markdown": "# 上下文", "compiled_hash": "h1"})
    logger, messages = recording_logger("bundle-every", level=20)
    injector = _injector(client, enable_recall=False, logger=logger, store=store)

    async def scenario():
        return [await injector.collect("umo", "hello") for _ in range(3)]

    results = asyncio.run(scenario())
    assert all(len(texts) == 1 for texts in results)
    assert all(GUIDANCE in texts[0] for texts in results)
    assert client.bundle_calls == 1
    assert sum("已注入 Nowledge Mem Context Bundle" in item for item in messages) == 1
    assert sum("复用会话级 Bundle 缓存并注入" in item for item in messages) == 2


def test_bundle_cache_expires_after_ttl() -> None:
    store = FakeStore()
    client = FakeNmemClient(bundle={"rendered_markdown": "# H", "compiled_hash": "h1"})
    injector = _injector(client, enable_recall=False, store=store)

    async def scenario():
        await injector.collect("umo", "hello")
        await injector.collect("umo", "hello")
        assert client.bundle_calls == 1
        _expire_bundle_cache(store)
        await injector.collect("umo", "hello")

    asyncio.run(scenario())
    assert client.bundle_calls == 2


def test_bundle_cache_refreshes_when_compiled_hash_changes() -> None:
    store = FakeStore()
    client = FakeNmemClient(bundle={"rendered_markdown": "# 旧", "compiled_hash": "h1"})
    injector = _injector(client, enable_recall=False, store=store)

    async def scenario():
        first = await injector.collect("umo", "hello")
        client.bundle = {"rendered_markdown": "# 新", "compiled_hash": "h2"}
        _expire_bundle_cache(store)
        second = await injector.collect("umo", "hello")
        return first, second

    first, second = asyncio.run(scenario())
    assert "旧" in first[0]
    assert "新" in second[0]
    assert store.data[("bundle", "test", "umo")]["hash"] == "h2"


def test_bundle_on_change_keeps_old_behavior() -> None:
    client = FakeNmemClient(bundle={"rendered_markdown": "# H", "compiled_hash": "h1"})
    injector = _injector(
        client,
        enable_recall=False,
        bundle_every_turn=False,
        bundle_on_change=True,
    )

    async def scenario():
        first = await injector.collect("umo", "hello")
        second = await injector.collect("umo", "hello")
        client.bundle["compiled_hash"] = "h2"
        third = await injector.collect("umo", "hello")
        return first, second, third

    first, second, third = asyncio.run(scenario())
    assert len(first) == 1
    assert second == []
    assert len(third) == 1


def test_bundle_disabled_does_not_fetch() -> None:
    client = FakeNmemClient(bundle={"rendered_markdown": "# H"})
    injector = _injector(client, enable_bundle=False, enable_recall=False)
    assert asyncio.run(injector.collect("umo", "hello")) == []
    assert client.bundle_calls == 0


def test_bundle_failure_without_cache_injects_degraded_notice() -> None:
    client = FakeNmemClient(fail_bundle=True)
    logger, messages = recording_logger("bundle-degraded")
    injector = _injector(client, enable_recall=False, logger=logger)
    texts = asyncio.run(injector.collect("umo", "hello"))
    assert len(texts) == 1
    assert "记忆服务暂不可用，本轮无上下文" in texts[0]
    assert GUIDANCE in texts[0]
    assert any("degraded" in item for item in messages)


def test_bundle_failure_uses_cached_bundle() -> None:
    store = FakeStore()
    client = FakeNmemClient(bundle={"rendered_markdown": "# 上次", "compiled_hash": "h1"})
    injector = _injector(client, enable_recall=False, store=store)
    asyncio.run(injector.collect("umo", "hello"))

    _expire_bundle_cache(store)
    client.bundle = None
    logger, messages = recording_logger("bundle-fallback", level=20)
    injector = _injector(client, enable_recall=False, logger=logger, store=store)
    texts = asyncio.run(injector.collect("umo", "hello"))
    assert len(texts) == 1
    assert "上次" in texts[0]
    assert any("使用本地缓存" in item for item in messages)


def test_bundle_failure_still_returns_recall() -> None:
    client = FakeNmemClient(
        results=[{"memory": {"title": "记忆", "content": "内容"}, "similarity_score": 0.9}],
        fail_bundle=True,
    )
    injector = _injector(client, enable_bundle=True, enable_recall=True)
    texts = asyncio.run(injector.collect("umo", "hello world"))
    assert len(texts) == 2
    assert any("记忆" in text for text in texts)
    assert any("记忆服务暂不可用" in text for text in texts)


def test_recall_uses_current_message_and_limits() -> None:
    client = FakeNmemClient(
        results=[{"memory": {"title": "标题", "content": "摘要", "unit_type": "fact"}, "similarity_score": 0.8}]
    )
    injector = _injector(client, enable_bundle=False, recall_limit=4)
    texts = asyncio.run(injector.collect("umo", "  帮我   查一下  记忆 "))
    assert len(texts) == 1
    assert client.search_calls[0][0] == "帮我 查一下 记忆"
    assert client.search_calls[0][1] == {"limit": 4, "space_id": "space", "mode": "fast"}
