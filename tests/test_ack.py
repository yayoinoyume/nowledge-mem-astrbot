"""ack 严格校验回归。"""

from __future__ import annotations

from core.ack import parse_import_ack


def _ack(**overrides):
    data = {
        "success": True,
        "failed_count": 0,
        "results": [
            {
                "success": True,
                "message_count": 2,
                "append_mode": "checkpointed",
            }
        ],
    }
    data.update(overrides)
    return data


def test_ack_missing_message_count_fails() -> None:
    data = _ack(results=[{"success": True, "append_mode": "checkpointed"}])
    ack = parse_import_ack(data)
    assert ack.status == "failed"
    assert ack.error == "missing_message_count"


def test_ack_thread_not_found_is_conflict() -> None:
    data = _ack(results=[{"success": False, "error_code": "thread_not_found"}])
    ack = parse_import_ack(data)
    assert ack.status == "conflict"
    assert ack.error == "thread_not_found"


def test_ack_checkpoint_conflict_is_conflict() -> None:
    data = _ack(results=[{"success": False, "error_code": "checkpoint_conflict"}])
    assert parse_import_ack(data).status == "conflict"


def test_ack_negative_count_fails() -> None:
    data = _ack(results=[{"success": True, "message_count": -1}])
    assert parse_import_ack(data).error == "missing_message_count"
