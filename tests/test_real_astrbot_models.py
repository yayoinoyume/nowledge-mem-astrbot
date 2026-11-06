"""使用 AstrBot 真实 pydantic Message/TextPart 的提取测试。"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

source = os.environ.get("ASTRBOT_SOURCE", "").strip()
if not source:
    pytest.skip("ASTRBOT_SOURCE 未设置", allow_module_level=True)

message_path = Path(source) / "astrbot" / "core" / "agent" / "message.py"
if not message_path.is_file():
    pytest.skip(f"找不到 {message_path}", allow_module_level=True)

spec = importlib.util.spec_from_file_location("astrbot_real_message", message_path)
if spec is None or spec.loader is None:
    pytest.skip("无法加载 AstrBot message.py", allow_module_level=True)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Message = module.Message
TextPart = module.TextPart

from core.delta import extract_messages, message_hash


def test_real_textpart_marked_temp_is_filtered() -> None:
    message = Message(
        role="user",
        content=[
            TextPart(text="真实用户问题"),
            TextPart(text="<nowledge_mem_context>注入</nowledge_mem_context>").mark_as_temp(),
        ],
    )

    assert extract_messages([message]) == [
        {"role": "user", "content": "真实用户问题"}
    ]


def test_real_message_string_with_marker_is_kept() -> None:
    message = Message(
        role="user",
        content="用户正文 <nowledge_mem_context> 标签",
    )

    assert extract_messages([message]) == [
        {"role": "user", "content": "用户正文 <nowledge_mem_context> 标签"}
    ]


def test_real_message_system_reminder_timestamp_is_stripped() -> None:
    reminder_1147 = (
        "<system_reminder>Current datetime: 2026-11-06 11:47 (CST)\n"
        "The current date is 2026-11-06.\n</system_reminder>"
    )
    reminder_1148 = (
        "<system_reminder>Current datetime: 2026-11-06 11:48 (CST)\n"
        "The current date is 2026-11-06.\n</system_reminder>"
    )
    first = Message(role="user", content=f"真实问题\n{reminder_1147}")
    second = Message(role="user", content=f"真实问题\n{reminder_1148}")

    first_messages = extract_messages([first])
    second_messages = extract_messages([second])

    assert first_messages == [{"role": "user", "content": "真实问题"}]
    assert second_messages == first_messages
    assert message_hash(
        first_messages[0]["role"], first_messages[0]["content"]
    ) == message_hash(
        second_messages[0]["role"], second_messages[0]["content"]
    )
