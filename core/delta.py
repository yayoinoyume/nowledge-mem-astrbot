"""消息提取与 append/reset 决策。"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Any, NamedTuple

APPEND = "append"
RESET = "reset"
MAX_TOOL_RESULT_CHARS = 20_000
SYSTEM_REMINDER_PART_RE = re.compile(
    r"\s*<system_reminder>.*?</system_reminder>\s*$", re.DOTALL
)
TOOL_TRANSCRIPT_RE = re.compile(r"^\[Tool(?:\s[^\]]*)?\]")
HEARTBEAT_SILENT = "<HEARTBEAT_SILENT>"

class DeltaPlan(NamedTuple):
    mode: str
    start: int
    messages: list[dict[str, str]]
    hashes: list[str]

def message_hash(role: str, content: str) -> str:
    return hashlib.sha1(f"{role}\n{content}".encode("utf-8")).hexdigest()

def _field(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)

def _part_text(part: Any) -> str | None:
    text = _field(part, "text")
    return text if isinstance(text, str) else None

def _clean_text(text: str) -> str:
    stripped = text.rstrip()
    start = stripped.rfind("<system_reminder>")
    if start >= 0:
        block = stripped[start:]
        if (
            block.endswith("</system_reminder>")
            and "Current datetime:" in block
            and block.count("<system_reminder>") == 1
        ):
            return stripped[:start].strip()
    return text.strip()

def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return _clean_text(content)
    if isinstance(content, list):
        parts = []
        for part in content:
            text = _part_text(part)
            if text is None or _field(part, "_no_save") is True:
                continue
            if SYSTEM_REMINDER_PART_RE.fullmatch(text):
                continue
            cleaned = text.strip()
            if cleaned:
                parts.append(cleaned)
        return "\n".join(parts).strip()
    return ""

def is_internal_heartbeat_message(
    role: str, content: str, markers: Sequence[str] = ()
) -> bool:
    """心跳等内部消息：用户侧按配置前缀匹配，助手侧匹配静默标记。"""
    text = content.strip()
    if role == "user":
        return any(text.startswith(marker) for marker in markers if marker)
    return role == "assistant" and text == HEARTBEAT_SILENT

def _tool_names(raw_messages: Any) -> dict[str, str]:
    names: dict[str, str] = {}
    for raw in raw_messages or []:
        for call in _field(raw, "tool_calls") or []:
            call_id = _field(call, "id")
            name = _field(_field(call, "function"), "name")
            if call_id is not None and isinstance(name, str) and name:
                names[str(call_id)] = name
    return names

def _extract_messages(
    raw_messages: Any, *, drop_internal: bool, heartbeat_markers: Sequence[str] = ()
) -> list[dict[str, str]]:
    names = _tool_names(raw_messages)
    messages: list[dict[str, str]] = []
    for raw in raw_messages or []:
        role = _field(raw, "role")
        content = _content_text(_field(raw, "content")).strip()
        if role == "tool":
            name = names.get(str(_field(raw, "tool_call_id")), "tool")
            content = f"[Tool {name}] {content[:MAX_TOOL_RESULT_CHARS]}".strip()
            role = "assistant"
        if role not in ("user", "assistant") or not content:
            continue
        # 工具结果与工具正文不进对话线程；只匹配行首的 "[Tool xxx]"，
        # 正常正文里提到 Tool 不受影响。
        if role == "assistant" and TOOL_TRANSCRIPT_RE.match(content):
            continue
        if drop_internal and is_internal_heartbeat_message(
            role, content, heartbeat_markers
        ):
            continue
        messages.append({"role": role, "content": content})
    return messages

def extract_messages(
    raw_messages: Any, heartbeat_markers: Sequence[str] = ()
) -> list[dict[str, str]]:
    return _extract_messages(
        raw_messages, drop_internal=True, heartbeat_markers=heartbeat_markers
    )

def digest_api_key(api_key: str) -> str:
    """api_key 的 8 位指纹；只进 identity，不写日志。"""
    return hashlib.sha1((api_key or "").encode("utf-8")).hexdigest()[:8]

def identity_digest(
    umo: str,
    space_id: str,
    agent_id: str,
    api_url: str,
    api_key_digest: str = "",
) -> str:
    raw = f"{umo}|{space_id}|{agent_id}|{api_url}|{api_key_digest}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

def thread_id_for(
    umo: str,
    space_id: str = "",
    agent_id: str = "",
    api_url: str = "",
    api_key_digest: str = "",
) -> str:
    return f"astrbot-{identity_digest(umo, space_id, agent_id, api_url, api_key_digest)}"

def thread_title(messages: list[dict[str, str]], previous: Any = None) -> str:
    if isinstance(previous, str) and previous.strip():
        return previous.strip()[:80]
    for message in messages:
        if message["role"] != "user":
            continue
        for line in message["content"].splitlines():
            if line.strip():
                return line.strip()[:80]
    return "AstrBot 对话"

def prefix_fingerprint(messages: list[dict[str, str]], count: int) -> str:
    """已确认前缀的 sha256(role+content 拼接)。"""
    digest = hashlib.sha256()
    for item in messages[:count]:
        digest.update(item["role"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(item["content"].encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def compute_delta(
    messages: list[dict[str, str]], cursor: dict[str, Any] | None
) -> DeltaPlan:
    hashes = [message_hash(item["role"], item["content"]) for item in messages]
    if not cursor:
        return DeltaPlan(RESET, 0, list(messages), hashes)
    last = cursor.get("last_confirmed_hash")
    confirmed = cursor.get("confirmed_count")
    remote = cursor.get("remote_count")
    prefix = cursor.get("prefix_fingerprint")
    if (
        not isinstance(last, str)
        or not last
        or type(confirmed) is not int
        or confirmed <= 0
        or type(remote) is not int
        or remote < 0
        or not isinstance(prefix, str)
        or not prefix
        or confirmed > len(hashes)
        or prefix_fingerprint(messages, confirmed) != prefix
    ):
        return DeltaPlan(RESET, 0, list(messages), hashes)
    if hashes[confirmed - 1] == last:
        return DeltaPlan(APPEND, confirmed, list(messages[confirmed:]), hashes)
    positions = [i for i, item in enumerate(hashes) if item == last]
    if len(positions) != 1:
        return DeltaPlan(RESET, 0, list(messages), hashes)
    index = positions[0]
    return DeltaPlan(APPEND, index + 1, list(messages[index + 1 :]), hashes)
