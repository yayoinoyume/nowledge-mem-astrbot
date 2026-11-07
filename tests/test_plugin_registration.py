"""插件级注册与权限：7 个 llm_tool、4 个斜杠命令、save_admin_only 与空参数。

需要 ASTRBOT_SOURCE 指向 AstrBot 源码根（其下有 astrbot/ 包）；不可用时整模块跳过。
"""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import importlib
import json
import os
import shutil
import sys
import tempfile
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
        self.search_calls: list[dict] = []

    async def search(self, query: str, **kwargs) -> list[dict]:
        self.search_calls.append({"query": query, **kwargs})
        return []

    async def search_threads(self, query: str, **kwargs) -> dict:
        return {"threads": []}

    async def save_memory(self, content: str, **kwargs) -> dict:
        self.save_calls.append({"content": content, **kwargs})
        return {"id": "m-new", "title": "标题"}

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
            plugin_version="0.17.0",
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


def test_mem_sum_command_member_denied() -> None:
    tools = _CommandTools()
    stub = SimpleNamespace(_tools=tools)
    outputs = _collect(
        plugin_main.NowledgeMemPlugin.mem_sum(stub, _Event(admin=False))
    )
    assert outputs == ["error: 仅管理员可使用 /mem_sum"]
    assert tools.save_calls == []


def test_mem_sum_command_admin_saves_context() -> None:
    tools = _CommandTools()
    stub = SimpleNamespace(_tools=tools)

    async def fake_history(umo, rounds):
        assert umo == "qq:group:1"
        assert rounds == 5
        return "会话总结正文", 4, 2, "会话总结 - 标题"

    stub._recent_history_text = fake_history
    outputs = _collect(
        plugin_main.NowledgeMemPlugin.mem_sum(stub, _Event(admin=True))
    )
    assert outputs[0].startswith("saved:")
    assert "已汇总最近 2 轮（4 条消息）" in outputs[0]
    call = tools.save_calls[0]
    assert call["content"] == "会话总结正文"
    assert call["title"] == "会话总结 - 标题"
    assert call["unit_type"] == "context"
    assert call["importance"] == 0.5
    assert call["labels"] == ["会话总结"]
    assert call["is_admin"] is True


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
    ("message_count", "rounds"), [(0, 0), (3, 2), (10, 5)]
)
def test_mem_sum_uses_actual_rounds(message_count: int, rounds: int) -> None:
    tools = _CommandTools()
    manager = _HistoryManager(_history_messages(message_count))
    stub = SimpleNamespace(
        _tools=tools,
        context=SimpleNamespace(conversation_manager=manager),
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
        assert tools.save_calls == []
        return
    assert outputs[0].startswith("saved:")
    assert f"已汇总最近 {rounds} 轮（{message_count} 条消息）" in outputs[0]
    saved = tools.save_calls[0]["content"]
    assert saved.startswith(f"会话总结（最近 {rounds} 轮）")
    assert saved.count("message-") == message_count


def test_mem_sum_command_without_history() -> None:
    tools = _CommandTools()
    stub = SimpleNamespace(_tools=tools)

    async def empty_history(umo, rounds):
        return "", 0, 0, ""

    stub._recent_history_text = empty_history
    outputs = _collect(
        plugin_main.NowledgeMemPlugin.mem_sum(stub, _Event(admin=True))
    )
    assert outputs == ["error: 当前会话没有可总结的对话"]
    assert tools.save_calls == []
