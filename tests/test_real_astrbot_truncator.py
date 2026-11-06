"""使用 AstrBot 真实 Message/ContextTruncator 验证 102→84 截断。"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

source = os.environ.get("ASTRBOT_SOURCE", "").strip()
if not source:
    pytest.skip("ASTRBOT_SOURCE 未设置", allow_module_level=True)

agent_dir = Path(source) / "astrbot" / "core" / "agent"
message_path = agent_dir / "message.py"
truncator_path = agent_dir / "context" / "truncator.py"
if not message_path.is_file() or not truncator_path.is_file():
    pytest.skip("找不到 AstrBot message.py 或 truncator.py", allow_module_level=True)

package = types.ModuleType("nmem_real_astrbot")
package.__path__ = []
sys.modules["nmem_real_astrbot"] = package
context_package = types.ModuleType("nmem_real_astrbot.context")
context_package.__path__ = []
sys.modules["nmem_real_astrbot.context"] = context_package


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        pytest.skip(f"无法加载 {path}", allow_module_level=True)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


message_module = load_module("nmem_real_astrbot.message", message_path)
truncator_module = load_module(
    "nmem_real_astrbot.context.truncator", truncator_path
)
Message = message_module.Message
ContextTruncator = truncator_module.ContextTruncator

from core.delta import compute_delta, extract_messages, message_hash, prefix_fingerprint


def test_real_context_truncator_102_to_84_resets_and_append_recovers() -> None:
    full = [
        Message(
            role="user" if index % 2 == 0 else "assistant",
            content=f"m{index}",
        )
        for index in range(102)
    ]
    truncated = ContextTruncator().truncate_by_turns(
        full, keep_most_recent_turns=42
    )

    assert len(truncated) == 84
    full_messages = extract_messages(full)
    window = extract_messages(truncated)
    assert len(full_messages) == 102
    assert len(window) == 84
    assert window[0]["content"] == "m18"

    full_hashes = [
        message_hash(item["role"], item["content"]) for item in full_messages
    ]
    cursor = {
        "thread_id": "astrbot-test",
        "last_confirmed_hash": full_hashes[-1],
        "confirmed_count": 102,
        "prefix_fingerprint": prefix_fingerprint(full_messages, 102),
        "remote_count": 102,
        "title": "SELFTEST",
    }

    plan = compute_delta(window, cursor)
    assert plan.mode == "reset"
    assert plan.messages == window

    window_hashes = [
        message_hash(item["role"], item["content"]) for item in window
    ]
    reset_cursor = {
        "thread_id": "astrbot-test",
        "last_confirmed_hash": window_hashes[-1],
        "confirmed_count": 84,
        "prefix_fingerprint": prefix_fingerprint(window, 84),
        "remote_count": 102,
        "title": "SELFTEST",
    }
    next_window = window + [
        {"role": "user", "content": "new-user"},
        {"role": "assistant", "content": "new-assistant"},
    ]
    next_plan = compute_delta(next_window, reset_cursor)
    assert next_plan.mode == "append"
    assert [item["content"] for item in next_plan.messages] == [
        "new-user",
        "new-assistant",
    ]
