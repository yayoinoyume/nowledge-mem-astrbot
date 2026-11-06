"""服务端导入 ack 的解析与校验。"""

from __future__ import annotations

from typing import Any, NamedTuple

RECOVERABLE = frozenset({"checkpoint_conflict", "thread_not_found"})


class ImportAck(NamedTuple):
    status: str
    message_count: int | None = None
    append_mode: str | None = None
    error: str | None = None
    messages_added: int | None = None
    already_existed: bool | None = None


def _first_error(results: Any) -> str | None:
    if isinstance(results, list):
        for item in results:
            if isinstance(item, dict) and isinstance(item.get("error_code"), str):
                return item["error_code"]
    return None


def parse_import_ack(data: dict[str, Any] | None) -> ImportAck:
    if not isinstance(data, dict):
        return ImportAck("failed", error="invalid_ack")
    results = data.get("results")
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            code = item.get("error_code")
            if isinstance(code, str) and code:
                return ImportAck("conflict" if code in RECOVERABLE else "failed", error=code)
    if data.get("success") is not True:
        return ImportAck("failed", error="ack_not_success")
    try:
        failed = int(data.get("failed_count") or 0)
    except (TypeError, ValueError):
        return ImportAck("failed", error="invalid_failed_count")
    if failed > 0:
        return ImportAck("failed", error=_first_error(results) or "ack_failed")
    if not isinstance(results, list) or not results:
        return ImportAck("failed", error="empty_results")
    if any(not isinstance(item, dict) or item.get("success") is not True for item in results):
        return ImportAck("failed", error=_first_error(results) or "ack_failed")
    first = results[0]
    count = first.get("message_count")
    if type(count) is not int or count < 0:
        return ImportAck("failed", error="missing_message_count")
    mode = first.get("append_mode")
    added = first.get("messages_added")
    existed = first.get("already_existed")
    return ImportAck(
        "acknowledged",
        count,
        mode if isinstance(mode, str) else None,
        None,
        added if type(added) is int else None,
        existed if isinstance(existed, bool) else None,
    )
