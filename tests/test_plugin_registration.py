"""插件级注册与权限：三个 llm_tool 的 schema、save_admin_only 两条路径、空参数不抛异常。

需要 ASTRBOT_SOURCE 指向 AstrBot 源码根（其下有 astrbot/ 包）；不可用时整模块跳过。
"""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import importlib
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

from tests.helpers import FakeStore

try:
    plugin_main = importlib.import_module(f"{PLUGIN_DIR.name}.main")
except ImportError as exc:
    pytest.skip(f"无法导入插件入口：{exc}", allow_module_level=True)

EXPECTED_TOOLS = {"nowledge_mem_search", "nowledge_mem_save", "nowledge_mem_status"}


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


class _FakeClient:
    configured = True
    api_url = "http://mem.local:14242"

    def __init__(self) -> None:
        self.save_calls: list[dict] = []

    async def search(self, query: str, **kwargs) -> list[dict]:
        return []

    async def search_threads(self, query: str, **kwargs) -> dict:
        return {"threads": []}

    async def save_memory(self, content: str, **kwargs) -> dict:
        self.save_calls.append({"content": content, **kwargs})
        return {"id": "m-new", "title": "标题"}


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
            plugin_version="0.16.0",
            save_admin_only=save_admin_only,
            timeout_seconds=3,
        )
    )
    return stub, client


def test_three_tools_are_registered() -> None:
    assert EXPECTED_TOOLS <= set(_registered())


def test_tool_parameter_schema() -> None:
    tools = _registered()
    search = tools["nowledge_mem_search"].parameters["properties"]
    assert {name: search[name]["type"] for name in search} == {
        "query": "string",
        "scope": "string",
        "limit": "number",
        "mode": "string",
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
