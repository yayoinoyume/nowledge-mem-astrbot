"""插件级注册与权限：8 个 llm_tool、4 个斜杠命令、save_admin_only 与空参数。

需要 ASTRBOT_SOURCE 指向 AstrBot 源码根（其下有 astrbot/ 包）；不可用时整模块跳过。
"""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import importlib
import json
import logging
import os
import shutil
import sys
import tempfile
from operator import attrgetter
from pathlib import Path
from types import SimpleNamespace

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent
source = os.environ.get("ASTRBOT_SOURCE", "").strip()
if not source:
    pytest.skip("ASTRBOT_SOURCE 未设置", allow_module_level=True)
if not (Path(source) / "astrbot").is_dir():
    pytest.skip(f"找不到 {source}/astrbot", allow_module_level=True)

sys.path.insert(0, str(Path(source)))
sys.path.insert(0, str(PLUGIN_DIR.parent))

# 导入 AstrBot 会在 ASTRBOT_ROOT 下建 data/；指到临时目录，避免污染插件源码目录。
_TMP_ROOT = tempfile.mkdtemp(prefix="astrbot-test-root-")
atexit.register(shutil.rmtree, _TMP_ROOT, ignore_errors=True)
os.environ["ASTRBOT_ROOT"] = _TMP_ROOT

llm_tools_module = pytest.importorskip(
    "astrbot.core.provider.register", reason="ASTRBOT_SOURCE 下无法导入 AstrBot 核心"
)
llm_tools = llm_tools_module.llm_tools

from astrbot.core.star.filter.command import CommandFilter, GreedyStr
from astrbot.core.star.star_handler import EventType, star_handlers_registry

from tests.helpers import FakeStore

try:
    plugin_main = importlib.import_module(f"{PLUGIN_DIR.name}.main")
except ImportError as exc:
    pytest.skip(f"无法导入插件入口：{exc}", allow_module_level=True)

EXPECTED_TOOLS = {
    "nowledge_mem_search",
    "nowledge_mem_save",
    "nowledge_mem_update",
    "nowledge_mem_status",
    "nowledge_mem_get",
    "nowledge_mem_thread_show",
    "nowledge_mem_graph",
    "nowledge_mem_skill",
}
EXPECTED_COMMANDS = {"mem_search", "mem_save", "mem_status", "mem_sum"}


class _Event:
    """只实现 save/search 包装器会碰到的字段。"""

    def __init__(
        self,
        *,
        admin: bool,
        umo: str = "qq:group:1",
        sender_id: str = "123",
    ) -> None:
        self.unified_msg_origin = umo
        self._sender_id = sender_id
        self._admin = admin

    def is_admin(self) -> bool:
        return self._admin

    def get_sender_id(self) -> str:
        return self._sender_id

    def plain_result(self, text: str):
        return ("plain", text)


class _FakeClient:
    configured = True
    api_url = "http://mem.local:14242"

    def __init__(self) -> None:
        self.save_calls: list[dict] = []
        self.update_calls: list[dict] = []
        self.search_calls: list[dict] = []

    async def search(self, query: str, **kwargs) -> list[dict]:
        self.search_calls.append({"query": query, **kwargs})
        return []

    async def search_threads(self, query: str, **kwargs) -> dict:
        return {"threads": []}

    async def save_memory(self, content: str, **kwargs) -> dict:
        self.save_calls.append({"content": content, **kwargs})
        return {"id": "m-new", "title": "标题"}

    async def update_memory(self, memory_id: str, **kwargs) -> dict:
        self.update_calls.append({"memory_id": memory_id, **kwargs})
        return {"id": memory_id, **{k: v for k, v in kwargs.items() if v is not None}}

    async def get_memory(self, memory_id: str, **kwargs) -> dict:
        return {
            "id": memory_id,
            "title": "标题",
            "content": "正文",
            "unit_type": "fact",
            "importance": 0.6,
            "labels": [],
            "metadata": {},
        }

    async def get_thread(self, thread_id: str, **kwargs) -> dict:
        return {
            "thread": {"thread_id": thread_id, "title": "对话"},
            "messages": [{"role": "user", "content": "hi"}],
            "total_messages": 1,
        }

    async def expand_graph(self, node_id: str, **kwargs) -> dict:
        return {
            "center_node": node_id,
            "depth": kwargs.get("depth", 1),
            "neighbors": [],
            "edges": [],
            "metadata": {},
        }


def _registered() -> dict[str, object]:
    return {tool.name: tool for tool in llm_tools.func_list}


def _plugin(*, save_admin_only: bool = False):
    client = _FakeClient()
    stub = SimpleNamespace(
        _tools=plugin_main.NmemTools(
            client,
            FakeStore(),
            space_id="space",
            agent_id="astrbot",
            source_app="astrbot",
            identity="identity123",
            plugin_version="0.18.1",
            save_admin_only=save_admin_only,
            timeout_seconds=3,
        )
    )
    return stub, client


def test_all_tools_are_registered() -> None:
    assert EXPECTED_TOOLS <= set(_registered())


def test_all_commands_are_registered() -> None:
    names = set()
    for handler in star_handlers_registry.get_handlers_by_event_type(
        EventType.AdapterMessageEvent, only_activated=False
    ):
        for event_filter in handler.event_filters:
            if isinstance(event_filter, CommandFilter):
                names.add(event_filter.command_name)
    assert EXPECTED_COMMANDS <= names


@pytest.mark.parametrize(
    ("config", "path", "expected"),
    [
        ({"timeout_seconds": float("inf")}, "_timeout", 3),
        (
            {"mem_sum_rounds": float("inf")},
            "_mem_sum_rounds",
            plugin_main.MEM_SUM_ROUNDS_DEFAULT,
        ),
        ({"recall_limit": float("inf")}, "_injector._recall_limit", 5),
    ],
)
def test_plugin_loading_survives_infinite_config_numbers(
    config: dict, path: str, expected: int
) -> None:
    plugin = plugin_main.NowledgeMemPlugin(SimpleNamespace(), config)
    assert attrgetter(path)(plugin) == expected


def test_greedy_commands_accept_empty_and_join_remaining_words() -> None:
    filters = []
    for handler in star_handlers_registry.get_handlers_by_event_type(
        EventType.AdapterMessageEvent, only_activated=False
    ):
        for event_filter in handler.event_filters:
            if (
                isinstance(event_filter, CommandFilter)
                and event_filter.command_name in {"mem_search", "mem_save"}
            ):
                filters.append(event_filter)
    assert len(filters) == 2
    for event_filter in filters:
        assert event_filter.handler_params["query" if event_filter.command_name == "mem_search" else "text"] is GreedyStr
        param = "query" if event_filter.command_name == "mem_search" else "text"
        assert event_filter.validate_and_convert_params([], event_filter.handler_params) == {param: ""}
        assert event_filter.validate_and_convert_params(
            ["a", "b"], event_filter.handler_params
        ) == {param: "a b"}


def test_tool_parameter_schema() -> None:
    tools = _registered()
    search = tools["nowledge_mem_search"].parameters["properties"]
    assert {name: search[name]["type"] for name in search} == {
        "query": "string",
        "scope": "string",
        "limit": "number",
        "mode": "string",
        "recorded_date_from": "string",
        "recorded_date_to": "string",
    }
    save = tools["nowledge_mem_save"].parameters["properties"]
    assert {name: save[name]["type"] for name in save} == {
        "content": "string",
        "title": "string",
        "unit_type": "string",
        "importance": "number",
        "labels": "array",
    }
    assert save["labels"]["items"] == {"type": "string"}
    update = tools["nowledge_mem_update"].parameters["properties"]
    assert {name: update[name]["type"] for name in update} == {
        "memory_id": "string",
        "content": "string",
        "title": "string",
        "unit_type": "string",
        "importance": "number",
        "labels": "array",
    }
    assert update["labels"]["items"] == {"type": "string"}
    assert tools["nowledge_mem_status"].parameters["properties"] == {}
    assert {
        name: tools["nowledge_mem_get"].parameters["properties"][name]["type"]
        for name in tools["nowledge_mem_get"].parameters["properties"]
    } == {"memory_id": "string"}
    assert {
        name: tools["nowledge_mem_thread_show"].parameters["properties"][name]["type"]
        for name in tools["nowledge_mem_thread_show"].parameters["properties"]
    } == {"thread_id": "string", "offset": "number", "limit": "number"}
    assert {
        name: tools["nowledge_mem_graph"].parameters["properties"][name]["type"]
        for name in tools["nowledge_mem_graph"].parameters["properties"]
    } == {"query": "string", "memory_id": "string", "depth": "number"}
    assert {
        name: tools["nowledge_mem_skill"].parameters["properties"][name]["type"]
        for name in tools["nowledge_mem_skill"].parameters["properties"]
    } == {"name": "string"}


def test_save_tool_description_warns_about_secrets() -> None:
    description = _registered()["nowledge_mem_save"].description
    for word in ("密钥", "密码", "令牌", "身份证号"):
        assert word in description


def test_search_wrapper_without_query_returns_error() -> None:
    stub, _ = _plugin()
    result = asyncio.run(
        plugin_main.NowledgeMemPlugin.nowledge_mem_search(stub, _Event(admin=True))
    )
    assert result == "error: query 不能为空"


def test_save_wrapper_without_content_returns_error() -> None:
    stub, client = _plugin()
    result = asyncio.run(
        plugin_main.NowledgeMemPlugin.nowledge_mem_save(stub, _Event(admin=True))
    )
    assert result == "error: content 不能为空"
    assert client.save_calls == []


def test_save_admin_only_blocks_member_through_wrapper() -> None:
    stub, client = _plugin(save_admin_only=True)
    result = asyncio.run(
        plugin_main.NowledgeMemPlugin.nowledge_mem_save(
            stub, _Event(admin=False), "内容"
        )
    )
    assert result == "error: 仅管理员可保存"
    assert client.save_calls == []


def test_save_admin_only_allows_admin_through_wrapper() -> None:
    stub, client = _plugin(save_admin_only=True)
    result = asyncio.run(
        plugin_main.NowledgeMemPlugin.nowledge_mem_save(
            stub, _Event(admin=True), "内容"
        )
    )
    assert result.startswith("saved: id=m-new")
    call = client.save_calls[0]
    assert call["source_umo"] == "qq:group:1"
    assert call["sender_hash"] == hashlib.sha1(b"123").hexdigest()[:16]


def test_update_tool_description_mentions_search_first() -> None:
    description = _registered()["nowledge_mem_update"].description
    assert "search" in description


def test_update_wrapper_without_memory_id_returns_error() -> None:
    stub, client = _plugin()
    result = asyncio.run(
        plugin_main.NowledgeMemPlugin.nowledge_mem_update(
            stub, _Event(admin=True), content="新正文"
        )
    )
    assert result == "error: memory_id 不能为空"
    assert client.update_calls == []


def test_update_admin_only_blocks_member_through_wrapper() -> None:
    stub, client = _plugin(save_admin_only=True)
    result = asyncio.run(
        plugin_main.NowledgeMemPlugin.nowledge_mem_update(
            stub, _Event(admin=False), "m1", content="新正文"
        )
    )
    assert result == "error: 仅管理员可更新记忆"
    assert client.update_calls == []


def test_update_wrapper_allows_admin_and_passes_fields() -> None:
    stub, client = _plugin(save_admin_only=True)
    result = asyncio.run(
        plugin_main.NowledgeMemPlugin.nowledge_mem_update(
            stub, _Event(admin=True), "m1", content="新正文", unit_type="decision"
        )
    )
    assert result.startswith("updated: id=m1")
    call = client.update_calls[0]
    assert call["memory_id"] == "m1"
    assert call["content"] == "新正文"
    assert call["unit_type"] == "decision"
    assert call["title"] is None


def _collect(agen):
    async def run():
        return [item[1] async for item in agen]

    return asyncio.run(run())


class _CommandTools:
    def __init__(
        self,
        *,
        search_result: str = "matches: 1",
        save_result: str = (
            "saved: id=m-new title=标题 unit_type=fact importance=0.60"
        ),
        status_result: str = "connected: true",
    ) -> None:
        self.search_result = search_result
        self.save_result = save_result
        self.status_result = status_result
        self.search_calls: list[dict] = []
        self.save_calls: list[dict] = []
        self.status_calls: list[str] = []

    async def search(self, query: str, **kwargs) -> str:
        self.search_calls.append({"query": query, **kwargs})
        return self.search_result

    async def save(self, content: str, **kwargs) -> str:
        self.save_calls.append({"content": content, **kwargs})
        return self.save_result

    async def status(self, umo: str = "") -> str:
        self.status_calls.append(umo)
        return self.status_result


def test_search_wrapper_passes_date_window() -> None:
    stub, client = _plugin()
    asyncio.run(
        plugin_main.NowledgeMemPlugin.nowledge_mem_search(
            stub,
            _Event(admin=False),
            query="查询",
            recorded_date_from="2026-11-01",
            recorded_date_to="2026-11-30",
        )
    )
    call = client.search_calls[0]
    assert call["recorded_date_from"] == "2026-11-01"
    assert call["recorded_date_to"] == "2026-11-30"


def test_get_wrapper_reads_memory() -> None:
    stub, _ = _plugin()
    result = asyncio.run(
        plugin_main.NowledgeMemPlugin.nowledge_mem_get(
            stub, _Event(admin=True), "m1"
        )
    )
    assert "memory: id=m1" in result
    assert "content:" in result


def test_thread_show_wrapper_reads_thread() -> None:
    stub, _ = _plugin()
    result = asyncio.run(
        plugin_main.NowledgeMemPlugin.nowledge_mem_thread_show(
            stub, _Event(admin=True), "t1", 0, 5
        )
    )
    assert "thread: id=t1" in result
    assert "total_messages: 1" in result
    assert "1. [user] hi" in result


def test_graph_wrapper_expands_memory() -> None:
    stub, _ = _plugin()
    result = asyncio.run(
        plugin_main.NowledgeMemPlugin.nowledge_mem_graph(
            stub, _Event(admin=True), memory_id="m1", depth=2
        )
    )
    assert "graph: center=m1 depth=2" in result


def test_skill_wrapper_reads_skill() -> None:
    stub, _ = _plugin()
    result = asyncio.run(
        plugin_main.NowledgeMemPlugin.nowledge_mem_skill(
            stub, _Event(admin=True), "status"
        )
    )
    assert result.startswith("skill: status")
    assert "## 何时使用" in result


def test_mem_search_command_uses_limit_5() -> None:
    tools = _CommandTools()
    stub = SimpleNamespace(_tools=tools)
    outputs = _collect(
        plugin_main.NowledgeMemPlugin.mem_search(
            stub, _Event(admin=False), "旧决策"
        )
    )
    assert outputs == ["matches: 1"]
    assert tools.search_calls == [
        {"query": "旧决策", "scope": "memories", "limit": 5}
    ]


def test_mem_search_command_empty_argument() -> None:
    tools = _CommandTools()
    stub = SimpleNamespace(_tools=tools)
    outputs = _collect(
        plugin_main.NowledgeMemPlugin.mem_search(stub, _Event(admin=False), "")
    )
    assert outputs == ["用法：/mem_search <查询内容>"]
    assert tools.search_calls == []


def test_mem_save_command_defaults_to_fact() -> None:
    tools = _CommandTools()
    stub = SimpleNamespace(_tools=tools)
    outputs = _collect(
        plugin_main.NowledgeMemPlugin.mem_save(
            stub, _Event(admin=False), "普通内容"
        )
    )
    assert outputs[0].startswith("saved:")
    call = tools.save_calls[0]
    assert call["content"] == "普通内容"
    assert call["unit_type"] == "fact"
    assert call["umo"] == "qq:group:1"
    assert call["sender_id"] == "123"
    assert call["is_admin"] is False


def test_mem_save_command_context_prefix() -> None:
    tools = _CommandTools()
    stub = SimpleNamespace(_tools=tools)
    _collect(
        plugin_main.NowledgeMemPlugin.mem_save(
            stub, _Event(admin=True), "context 一段背景"
        )
    )
    call = tools.save_calls[0]
    assert call["content"] == "一段背景"
    assert call["unit_type"] == "context"


def test_mem_save_command_empty_argument() -> None:
    tools = _CommandTools()
    stub = SimpleNamespace(_tools=tools)
    outputs = _collect(
        plugin_main.NowledgeMemPlugin.mem_save(stub, _Event(admin=True), "")
    )
    assert outputs == ["用法：/mem_save [fact|context] <要保存的内容>"]
    assert tools.save_calls == []


def test_mem_save_command_respects_admin_only() -> None:
    stub, client = _plugin(save_admin_only=True)
    outputs = _collect(
        plugin_main.NowledgeMemPlugin.mem_save(
            stub, _Event(admin=False), "内容"
        )
    )
    assert outputs == ["error: 仅管理员可保存"]
    assert client.save_calls == []


def test_mem_status_command_outputs_status() -> None:
    tools = _CommandTools(status_result="connected: true\nspace: space")
    stub = SimpleNamespace(_tools=tools)
    outputs = _collect(
        plugin_main.NowledgeMemPlugin.mem_status(stub, _Event(admin=False))
    )
    assert outputs == ["connected: true\nspace: space"]
    assert tools.status_calls == ["qq:group:1"]


class _FakeProvider:
    def __init__(self, text: str = "", error: BaseException | None = None) -> None:
        self.text = text
        self.error = error
        self.calls: list[dict] = []

    async def text_chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(completion_text=self.text)


class _FakeContext:
    def __init__(self, provider) -> None:
        self._provider = provider
        self.umos: list[str | None] = []

    async def get_using_provider_async(self, umo=None):
        self.umos.append(umo)
        return self._provider


class _SyncOnlyContext:
    """模拟 4.24.2-4.27.2：只有同步 get_using_provider，且不接受 umo。"""

    def __init__(self, provider) -> None:
        self._provider = provider
        self.calls = 0

    def get_using_provider(self):
        self.calls += 1
        return self._provider


def _mem_sum_stub(
    tools,
    provider,
    *,
    material: str = "会话总结（最近 2 轮）\nuser: 我们用 PATCH 更新记忆",
    count: int = 4,
    rounds: int = 2,
    round_config: int = 10,
    context=None,
):
    async def fake_history(umo, max_rounds):
        assert umo == "qq:group:1"
        assert max_rounds == round_config
        return material, count, rounds, "会话总结"

    stub = SimpleNamespace(
        _tools=tools,
        _mem_sum_rounds=round_config,
        context=context if context is not None else _FakeContext(provider),
        logger=logging.getLogger("test"),
    )
    stub._recent_history_text = fake_history
    return stub


DISTILLED = [
    {
        "content": "更新记忆用 PATCH /memories/{id}",
        "title": "更新记忆用 PATCH",
        "unit_type": "decision",
        "importance": 0.9,
        "labels": ["mem"],
    },
    {
        "content": "服务端不校验 PATCH body",
        "title": "PATCH body 不校验",
        "unit_type": "learning",
        "importance": 0.7,
        "labels": ["mem"],
    },
]


def test_mem_sum_command_member_denied() -> None:
    tools = _CommandTools()
    provider = _FakeProvider("[]")
    stub = _mem_sum_stub(tools, provider)
    outputs = _collect(
        plugin_main.NowledgeMemPlugin.mem_sum(stub, _Event(admin=False))
    )
    assert outputs == ["error: 仅管理员可使用 /mem_sum"]
    assert tools.save_calls == []
    assert provider.calls == []


def test_mem_sum_command_without_history() -> None:
    tools = _CommandTools()
    provider = _FakeProvider("[]")
    stub = _mem_sum_stub(tools, provider, material="")
    outputs = _collect(
        plugin_main.NowledgeMemPlugin.mem_sum(stub, _Event(admin=True))
    )
    assert outputs == ["error: 当前会话没有可总结的对话"]
    assert tools.save_calls == []
    assert provider.calls == []


def test_mem_sum_command_without_provider() -> None:
    tools = _CommandTools()
    stub = _mem_sum_stub(tools, None)
    outputs = _collect(
        plugin_main.NowledgeMemPlugin.mem_sum(stub, _Event(admin=True))
    )
    assert outputs == [
        "error: 当前没有可用的模型，无法蒸馏（可改用 /mem_search 手动检索已有记忆）"
    ]
    assert tools.save_calls == []


def test_mem_sum_falls_back_to_sync_provider() -> None:
    tools = _CommandTools()
    provider = _FakeProvider(json.dumps(DISTILLED[:1], ensure_ascii=False))
    context = _SyncOnlyContext(provider)
    stub = _mem_sum_stub(tools, provider, context=context)
    outputs = _collect(
        plugin_main.NowledgeMemPlugin.mem_sum(stub, _Event(admin=True))
    )
    assert outputs[0].startswith("distilled: 保存 1/1 条记忆")
    assert context.calls == 1
    assert len(provider.calls) == 1


def test_using_provider_awaits_coroutine_from_legacy_async_getter() -> None:
    class _CoroContext:
        async def get_using_provider(self) -> str:
            return "coro-provider"

    provider = asyncio.run(
        plugin_main._using_provider(_CoroContext(), "umo")  # type: ignore[arg-type]
    )
    assert provider == "coro-provider"


def test_using_provider_returns_none_without_getter() -> None:
    class _EmptyContext:
        pass

    assert asyncio.run(
        plugin_main._using_provider(_EmptyContext(), "umo")  # type: ignore[arg-type]
    ) is None


def test_mem_sum_command_saves_distilled_items() -> None:
    tools = _CommandTools()
    provider = _FakeProvider(json.dumps(DISTILLED, ensure_ascii=False))
    stub = _mem_sum_stub(tools, provider)
    outputs = _collect(
        plugin_main.NowledgeMemPlugin.mem_sum(stub, _Event(admin=True))
    )
    assert outputs == [
        "distilled: 保存 2/2 条记忆\n"
        "- [decision] 更新记忆用 PATCH (0.90)\n"
        "- [learning] PATCH body 不校验 (0.70)\n"
        "note: 已扫描最近 2 轮（4 条消息）"
    ]
    call = provider.calls[0]
    assert call["prompt"].startswith("会话总结（最近 2 轮）")
    assert "system_prompt" in call
    saves = tools.save_calls
    assert [item["unit_type"] for item in saves] == ["decision", "learning"]
    assert [item["importance"] for item in saves] == [0.9, 0.7]
    assert saves[0]["labels"] == ["mem"]
    assert saves[0]["umo"] == "qq:group:1"
    assert saves[0]["sender_id"] == "123"
    assert saves[0]["is_admin"] is True
    assert stub.context.umos == ["qq:group:1"]


def test_mem_sum_command_strips_json_fence() -> None:
    tools = _CommandTools()
    fenced = "\x60\x60\x60json\n" + json.dumps(
        DISTILLED[:1], ensure_ascii=False
    ) + "\n\x60\x60\x60"
    stub = _mem_sum_stub(tools, _FakeProvider(fenced))
    outputs = _collect(
        plugin_main.NowledgeMemPlugin.mem_sum(stub, _Event(admin=True))
    )
    assert outputs[0].startswith("distilled: 保存 1/1 条记忆")


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(
            json.dumps(DISTILLED[:1], ensure_ascii=False) + "\n以上是结果。",
            id="trailing-prose",
        ),
        pytest.param(
            "以下是提炼结果：\n" + json.dumps(DISTILLED[:1], ensure_ascii=False),
            id="leading-prose",
        ),
        pytest.param(
            "\x60\x60\x60json\n"
            + json.dumps(DISTILLED[:1], ensure_ascii=False)
            + "\n\x60\x60\x60\n以上是结果。",
            id="fenced-with-trailing-prose",
        ),
    ],
)
def test_mem_sum_command_parses_array_surrounded_by_prose(raw: str) -> None:
    tools = _CommandTools()
    stub = _mem_sum_stub(tools, _FakeProvider(raw))
    outputs = _collect(
        plugin_main.NowledgeMemPlugin.mem_sum(stub, _Event(admin=True))
    )
    assert outputs[0].startswith("distilled: 保存 1/1 条记忆")
    assert len(tools.save_calls) == 1


def test_mem_sum_command_keeps_brackets_inside_strings() -> None:
    entry = dict(
        DISTILLED[0],
        content='方括号 ] 与引号 " 都在字符串里 [1]',
        title="带括号的标题",
    )
    raw = "说明：\n" + json.dumps([entry], ensure_ascii=False) + "\n结束"
    tools = _CommandTools()
    stub = _mem_sum_stub(tools, _FakeProvider(raw))
    outputs = _collect(
        plugin_main.NowledgeMemPlugin.mem_sum(stub, _Event(admin=True))
    )
    assert outputs[0].startswith("distilled: 保存 1/1 条记忆")
    assert tools.save_calls[0]["content"] == entry["content"]


def test_mem_sum_command_rejects_unterminated_array() -> None:
    raw = "\x60\x60\x60json\n" + '[{"content": "缺右括号", "unit_type": "fact"}'
    tools = _CommandTools()
    stub = _mem_sum_stub(tools, _FakeProvider(raw))
    outputs = _collect(
        plugin_main.NowledgeMemPlugin.mem_sum(stub, _Event(admin=True))
    )
    assert outputs == ["error: 提炼结果不是合法 JSON"]
    assert tools.save_calls == []


def test_mem_sum_command_invalid_json() -> None:
    tools = _CommandTools()
    provider = _FakeProvider("这不是 JSON")
    stub = _mem_sum_stub(tools, provider)
    outputs = _collect(
        plugin_main.NowledgeMemPlugin.mem_sum(stub, _Event(admin=True))
    )
    assert outputs == ["error: 提炼结果不是合法 JSON"]
    assert tools.save_calls == []


def test_mem_sum_command_empty_array() -> None:
    tools = _CommandTools()
    provider = _FakeProvider("[]")
    stub = _mem_sum_stub(tools, provider)
    outputs = _collect(
        plugin_main.NowledgeMemPlugin.mem_sum(stub, _Event(admin=True))
    )
    assert outputs == ["no_insight: 最近 2 轮没有值得沉淀的内容"]
    assert tools.save_calls == []


def test_mem_sum_command_drops_invalid_items() -> None:
    tools = _CommandTools()
    provider = _FakeProvider(
        json.dumps(
            [
                DISTILLED[0],
                {"content": "类型非法", "unit_type": "bogus", "importance": 0.5},
                {"content": "重要度越界", "unit_type": "fact", "importance": 1.7},
                {"content": "", "unit_type": "fact", "importance": 0.5},
            ],
            ensure_ascii=False,
        )
    )
    stub = _mem_sum_stub(tools, provider)
    outputs = _collect(
        plugin_main.NowledgeMemPlugin.mem_sum(stub, _Event(admin=True))
    )
    assert outputs[0].startswith("distilled: 保存 1/1 条记忆")
    assert "note: 丢弃 3 条（非法或超出上限）" in outputs[0]
    assert len(tools.save_calls) == 1


def test_distill_items_drops_only_the_overflowing_importance() -> None:
    raw = json.dumps(
        [
            {"content": "合法条目", "unit_type": "fact", "importance": 0.5},
            {"content": "超大重要度", "unit_type": "fact", "importance": 10**400},
        ],
        ensure_ascii=False,
    )
    items, dropped = plugin_main._distill_items(raw)
    assert [item["content"] for item in items] == ["合法条目"]
    assert dropped == 1


def test_mem_sum_command_keeps_valid_items_when_importance_overflows() -> None:
    entries = [
        DISTILLED[0],
        {
            "content": "超大重要度",
            "title": "超大重要度",
            "unit_type": "fact",
            "importance": 10**400,
        },
    ]
    tools = _CommandTools()
    provider = _FakeProvider(json.dumps(entries, ensure_ascii=False))
    stub = _mem_sum_stub(tools, provider)
    outputs = _collect(
        plugin_main.NowledgeMemPlugin.mem_sum(stub, _Event(admin=True))
    )
    assert outputs[0].startswith("distilled: 保存 1/1 条记忆")
    assert "note: 丢弃 1 条（非法或超出上限）" in outputs[0]
    assert len(tools.save_calls) == 1


def test_mem_sum_command_reports_items_dropped_by_limit() -> None:
    entries = [
        {
            "content": f"第 {index} 条",
            "title": f"第 {index} 条",
            "unit_type": "fact",
            "importance": 0.5,
        }
        for index in range(10)
    ]
    tools = _CommandTools()
    provider = _FakeProvider(json.dumps(entries, ensure_ascii=False))
    stub = _mem_sum_stub(tools, provider)
    outputs = _collect(
        plugin_main.NowledgeMemPlugin.mem_sum(stub, _Event(admin=True))
    )
    assert outputs[0].startswith("distilled: 保存 8/8 条记忆")
    assert "note: 丢弃 2 条（非法或超出上限）" in outputs[0]
    assert len(tools.save_calls) == 8


def test_mem_sum_command_provider_error() -> None:
    tools = _CommandTools()
    provider = _FakeProvider(error=RuntimeError("上游 500"))
    stub = _mem_sum_stub(tools, provider)
    outputs = _collect(
        plugin_main.NowledgeMemPlugin.mem_sum(stub, _Event(admin=True))
    )
    assert outputs == ["error: 蒸馏失败：RuntimeError"]
    assert tools.save_calls == []


def test_mem_sum_command_marks_failed_saves() -> None:
    tools = _CommandTools(save_result="error: Nowledge Mem 保存失败：HTTP 500")
    provider = _FakeProvider(json.dumps(DISTILLED[:1], ensure_ascii=False))
    stub = _mem_sum_stub(tools, provider)
    outputs = _collect(
        plugin_main.NowledgeMemPlugin.mem_sum(stub, _Event(admin=True))
    )
    assert outputs[0].startswith("distilled: 保存 0/1 条记忆")
    assert "(保存失败)" in outputs[0]


def test_mem_sum_command_marks_truncated_saves() -> None:
    tools = _CommandTools(
        save_result=(
            "saved: id=m-new title=标题 unit_type=fact importance=0.60"
            "\nnote: 标题或标签超长已截断"
        )
    )
    provider = _FakeProvider(json.dumps(DISTILLED[:1], ensure_ascii=False))
    stub = _mem_sum_stub(tools, provider)
    outputs = _collect(
        plugin_main.NowledgeMemPlugin.mem_sum(stub, _Event(admin=True))
    )
    assert outputs[0].startswith("distilled: 保存 1/1 条记忆")
    assert "（已截断）" in outputs[0]


class _HistoryManager:
    def __init__(self, history: list[dict]) -> None:
        self._history = json.dumps(history)

    async def get_curr_conversation_id(self, umo: str) -> str:
        return "conversation-1"

    async def get_conversation(self, umo: str, cid: str) -> SimpleNamespace:
        return SimpleNamespace(history=self._history)


def _history_messages(count: int) -> list[dict]:
    return [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"message-{index}",
        }
        for index in range(count)
    ]


@pytest.mark.parametrize(
    ("message_count", "expected_rounds"), [(0, 0), (3, 2), (10, 3)]
)
def test_mem_sum_scans_configured_rounds(
    message_count: int, expected_rounds: int
) -> None:
    tools = _CommandTools()
    provider = _FakeProvider("[]")
    context = _FakeContext(provider)
    context.conversation_manager = _HistoryManager(_history_messages(message_count))
    stub = SimpleNamespace(
        _tools=tools,
        _mem_sum_rounds=3,
        context=context,
        logger=logging.getLogger("test"),
    )
    stub._recent_history_text = (
        lambda umo, max_rounds: plugin_main.NowledgeMemPlugin._recent_history_text(
            stub, umo, max_rounds
        )
    )

    outputs = _collect(
        plugin_main.NowledgeMemPlugin.mem_sum(stub, _Event(admin=True))
    )
    if message_count == 0:
        assert outputs == ["error: 当前会话没有可总结的对话"]
        assert provider.calls == []
        return
    assert outputs == [f"no_insight: 最近 {expected_rounds} 轮没有值得沉淀的内容"]
    prompt = provider.calls[0]["prompt"]
    assert prompt.startswith(f"会话总结（最近 {expected_rounds} 轮）")
    assert prompt.count("message-") == min(message_count, expected_rounds * 2)


def test_recent_history_keeps_newest_messages_when_over_budget() -> None:
    history = [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"message-{index} " + "x" * 500,
        }
        for index in range(20)
    ]
    context = SimpleNamespace(conversation_manager=_HistoryManager(history))
    stub = SimpleNamespace(context=context)

    text, count, rounds, _title = asyncio.run(
        plugin_main.NowledgeMemPlugin._recent_history_text(stub, "qq:group:1", 10)
    )

    assert "message-19" in text
    assert "message-0" not in text
    assert len(text) <= plugin_main.MAX_SUMMARY_CHARS
    assert count < 20
    # 顺序必须是从旧到新：第一条的序号小于最后一条
    lines = [ln for ln in text.splitlines() if ln.startswith(("user:", "assistant:"))]
    first = int(lines[0].split("message-", 1)[1].split()[0])
    last = int(lines[-1].split("message-", 1)[1].split()[0])
    assert first < last
    assert len(text.splitlines()) - 1 == count
    assert rounds == (count + 1) // 2
    assert text.splitlines()[0] == f"会话总结（最近 {rounds} 轮）"
