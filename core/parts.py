"""AstrBot ContentPart 构造。"""

from __future__ import annotations

from typing import Any


def make_temp_parts(texts: list[str], part_cls: Any = None) -> list[Any]:
    """把文本包装成仅本轮生效、不写入会话历史的 TextPart。"""
    if part_cls is None:
        from astrbot.core.agent.message import TextPart as part_cls
    return [part_cls(text=text).mark_as_temp() for text in texts if text]
