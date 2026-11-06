"""消息提取、身份、tool、reminder 与锚点测试。"""

from __future__ import annotations

from core.delta import (
    APPEND,
    RESET,
    compute_delta,
    extract_messages,
    identity_digest,
    message_hash,
    prefix_fingerprint,
    thread_id_for,
)
from tests.helpers import make_cursor


def test_append_same_prefix_extension() -> None:
    old = [{"role": "user", "content": f"m{i}"} for i in range(4)]
    current = old + [{"role": "assistant", "content": "new"}]
    plan = compute_delta(current, make_cursor(old, remote_count=4))
    assert plan.mode == APPEND
    assert plan.messages == [current[-1]]


def test_append_empty_when_anchor_is_tail() -> None:
    messages = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    plan = compute_delta(messages, make_cursor(messages, remote_count=2))
    assert plan.mode == APPEND
    assert plan.messages == []


def test_duplicate_assistant_anchor_uses_confirmed_count() -> None:
    old = [{"role": "user", "content": "在吗"}, {"role": "assistant", "content": "在的"}]
    current = old + [
        {"role": "user", "content": "好的"},
        {"role": "assistant", "content": "在的"},
    ]
    plan = compute_delta(current, make_cursor(old, remote_count=2))
    assert plan.mode == APPEND
    assert [m["content"] for m in plan.messages] == ["好的", "在的"]


def test_duplicate_assistant_three_times_keeps_all_new() -> None:
    old = [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "收到"}]
    current = old + [
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "收到"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "收到"},
    ]
    plan = compute_delta(current, make_cursor(old, remote_count=2))
    assert plan.mode == APPEND
    assert [m["content"] for m in plan.messages] == ["u2", "收到", "u3", "收到"]


def test_duplicate_anchor_without_confirmed_count_resets() -> None:
    old = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "好"}]
    current = old + [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "好"},
    ]
    cursor = make_cursor(old, remote_count=2)
    cursor["confirmed_count"] = None
    plan = compute_delta(current, cursor)
    assert plan.mode == RESET


def test_window_shrink_resets() -> None:
    full = [{"role": "user", "content": f"m{i}"} for i in range(102)]
    window = full[-84:]
    plan = compute_delta(window, make_cursor(full, remote_count=102))
    assert plan.mode == RESET


def test_rewrite_before_confirmed_prefix_resets() -> None:
    full = [{"role": "user", "content": f"m{i}"} for i in range(102)]
    window = list(full)
    window[30] = {"role": "user", "content": "rewritten"}
    plan = compute_delta(window, make_cursor(full, remote_count=102))
    assert plan.mode == RESET


def test_legacy_cursor_resets() -> None:
    messages = [{"role": "user", "content": "u"}]
    plan = compute_delta(messages, {"count": 1, "tail_hashes": ["old"]})
    assert plan.mode == RESET


def test_tool_message_keeps_full_result() -> None:
    raw = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "web_search", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "x" * 300},
    ]
    messages = extract_messages(raw)
    assert messages == [{"role": "assistant", "content": "[Tool web_search] " + "x" * 300}]


def test_tool_message_truncates_at_20000_chars() -> None:
    raw = [{"role": "tool", "tool_call_id": "missing", "content": "y" * 20_001}]
    assert extract_messages(raw) == [
        {"role": "assistant", "content": "[Tool tool] " + "y" * 20_000}
    ]


def test_tool_message_without_name_uses_tool() -> None:
    raw = [{"role": "tool", "tool_call_id": "missing", "content": "结果"}]
    assert extract_messages(raw) == [{"role": "assistant", "content": "[Tool tool] 结果"}]


def test_system_reminder_tail_with_datetime_is_removed() -> None:
    raw = [
        {
            "role": "user",
            "content": "问题<system_reminder>Current datetime: 2026-11-06</system_reminder>",
        }
    ]
    assert extract_messages(raw) == [{"role": "user", "content": "问题"}]


def test_system_reminder_inline_user_text_is_kept() -> None:
    raw = [
        {"role": "user", "content": "正文<system_reminder>foo</system_reminder>结尾"},
    ]
    assert extract_messages(raw) == [{"role": "user", "content": raw[0]["content"]}]


def test_system_reminder_tail_without_datetime_is_kept() -> None:
    raw = [{"role": "user", "content": "正文<system_reminder>foo</system_reminder>"}]
    assert extract_messages(raw) == [{"role": "user", "content": raw[0]["content"]}]


def test_system_reminder_independent_part_is_removed() -> None:
    raw = [
        {
            "role": "user",
            "content": [
                {"text": "问题"},
                {"text": "<system_reminder>foo</system_reminder>"},
                {"text": "后续"},
            ],
        }
    ]
    assert extract_messages(raw) == [{"role": "user", "content": "问题\n后续"}]


def test_system_reminder_inline_part_is_kept() -> None:
    raw = [
        {
            "role": "user",
            "content": [
                {"text": "问题"},
                {"text": "正文<system_reminder>foo</system_reminder>结尾"},
            ],
        }
    ]
    assert extract_messages(raw) == [
        {"role": "user", "content": "问题\n正文<system_reminder>foo</system_reminder>结尾"}
    ]


def test_identity_changes_thread_id() -> None:
    a = thread_id_for("umo", "space-a", "agent", "http://mem")
    b = thread_id_for("umo", "space-b", "agent", "http://mem")
    assert a != b
    assert a == thread_id_for("umo", "space-a", "agent", "http://mem")
    assert a == "astrbot-" + identity_digest("umo", "space-a", "agent", "http://mem")


def test_api_key_digest_changes_thread_id() -> None:
    from core.delta import digest_api_key

    digest = digest_api_key("secret-key")
    assert len(digest) == 8
    assert digest == digest_api_key("secret-key")
    assert digest != digest_api_key("other-key")
    a = thread_id_for("umo", "space", "agent", "http://mem", digest)
    b = thread_id_for("umo", "space", "agent", "http://mem", digest_api_key("other-key"))
    assert a != b
    assert a == thread_id_for("umo", "space", "agent", "http://mem", digest)
    assert "secret-key" not in a


def test_prefix_fingerprint_is_stable() -> None:
    messages = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    assert prefix_fingerprint(messages, 2) == prefix_fingerprint(messages, 2)
    assert prefix_fingerprint(messages, 2) != prefix_fingerprint(messages, 1)


def test_heartbeat_internal_markers_are_recognized() -> None:
    from core.delta import is_internal_heartbeat_message

    markers = ["这是一次自主心跳", "【PERSONA_LOAD】"]
    assert is_internal_heartbeat_message("user", "这是一次自主心跳，不是用户", markers)
    assert is_internal_heartbeat_message("user", "【PERSONA_LOAD】\n内容", markers)
    assert is_internal_heartbeat_message("assistant", "<HEARTBEAT_SILENT>", markers)
    assert not is_internal_heartbeat_message("user", "普通引用：这是一次自主心跳", markers)
    assert not is_internal_heartbeat_message("assistant", "普通回复", markers)


def test_heartbeat_markers_default_to_empty() -> None:
    from core.delta import is_internal_heartbeat_message

    assert not is_internal_heartbeat_message("user", "这是一次自主心跳，不是用户")
    assert is_internal_heartbeat_message("assistant", "<HEARTBEAT_SILENT>")


def test_heartbeat_internal_messages_are_filtered_from_payload() -> None:
    raw = [
        {"role": "user", "content": "这是一次自主心跳，不是用户"},
        {"role": "assistant", "content": "<HEARTBEAT_SILENT>"},
        {"role": "user", "content": "正常问题"},
    ]
    assert extract_messages(raw, ["这是一次自主心跳"]) == [
        {"role": "user", "content": "正常问题"}
    ]
    assert extract_messages(raw) == [
        {"role": "user", "content": "这是一次自主心跳，不是用户"},
        {"role": "user", "content": "正常问题"},
    ]


def test_reminder_inline_block_plus_tail_only_removes_tail() -> None:
    raw = [
        {
            "role": "user",
            "content": "内联 <system_reminder>foo</system_reminder> 正文 "
            "<system_reminder>Current datetime: 2026-11-06</system_reminder>",
        }
    ]
    assert extract_messages(raw) == [
        {
            "role": "user",
            "content": "内联 <system_reminder>foo</system_reminder> 正文",
        }
    ]


def test_prefix_fingerprint_separates_role_and_content() -> None:
    a = [{"role": "ab", "content": "c"}]
    b = [{"role": "a", "content": "bc"}]
    assert prefix_fingerprint(a, 1) != prefix_fingerprint(b, 1)


def test_prefix_fingerprint_separates_message_boundaries() -> None:
    a = [
        {"role": "user", "content": "ab"},
        {"role": "assistant", "content": "c"},
    ]
    b = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "bc"},
    ]
    assert prefix_fingerprint(a, 2) != prefix_fingerprint(b, 2)
