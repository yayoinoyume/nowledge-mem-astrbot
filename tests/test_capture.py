"""捕获协调器：锚点、pending、ack、身份隔离。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from core.capture import CaptureCoordinator
from core.delta import message_hash, thread_id_for
from core.store import NmemStore
from tests.helpers import (
    TEST_KEY_DIGEST,
    FakeCaptureClient,
    FakeStore,
    failure_ack,
    make_coordinator,
    make_cursor,
    recording_logger,
    success_ack,
)


class GatedCaptureClient(FakeCaptureClient):
    """第一次 import 卡在闸门，用来验证 retry 与 sync 的锁顺序。"""

    def __init__(self, responses, gate: asyncio.Event) -> None:
        super().__init__(responses)
        self.gate = gate
        self.started = asyncio.Event()

    async def import_thread(self, payload):
        self.calls.append(payload)
        if len(self.calls) == 1:
            self.started.set()
            await self.gate.wait()
        if not self.responses:
            raise AssertionError("FakeCaptureClient 没有剩余响应")
        return self.responses.pop(0)


def test_first_reset_then_append() -> None:
    store, client = FakeStore(), FakeCaptureClient([success_ack(2), success_ack(4)])
    coordinator = make_coordinator(client, store)
    first = [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]
    second = first + [{"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"}]

    async def scenario() -> None:
        await coordinator.sync("umo", first)
        await coordinator.sync("umo", second)

    asyncio.run(scenario())
    assert "expected_message_count" not in client.calls[0]
    assert client.calls[1]["expected_message_count"] == 2
    assert [m["content"] for m in client.calls[1]["messages"]] == ["u2", "a2"]
    cursor = store.cursor("umo")
    assert cursor["last_confirmed_hash"] == message_hash("assistant", "a2")
    assert cursor["confirmed_count"] == 4
    assert cursor["remote_count"] == 4


def test_window_shift_with_changed_prefix_resets() -> None:
    store = FakeStore()
    client = FakeCaptureClient([success_ack(4)])
    coordinator = make_coordinator(client, store)
    old = [{"role": "user", "content": f"m{i}"} for i in range(4)]
    current = old[2:] + [{"role": "assistant", "content": "new1"}, {"role": "user", "content": "new2"}]

    async def scenario() -> None:
        await store.set_cursor("umo", make_cursor(old, remote_count=4))
        await coordinator.sync("umo", current)

    asyncio.run(scenario())
    assert "expected_message_count" not in client.calls[0]
    assert len(client.calls[0]["messages"]) == 4


def test_missing_anchor_resets_without_expected() -> None:
    store, client = FakeStore(), FakeCaptureClient([success_ack(2)])
    coordinator = make_coordinator(client, store)
    current = [{"role": "user", "content": "new"}, {"role": "assistant", "content": "ok"}]

    async def scenario() -> None:
        await store.set_cursor("umo", make_cursor([{"role": "user", "content": "old"}], remote_count=1))
        await coordinator.sync("umo", current)

    asyncio.run(scenario())
    assert "expected_message_count" not in client.calls[0]
    assert len(client.calls[0]["messages"]) == 2


def test_append_requires_checkpointed() -> None:
    store = FakeStore()
    client = FakeCaptureClient([success_ack(2), success_ack(4, "deduplicated")])
    coordinator = make_coordinator(client, store)
    first = [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]
    second = first + [{"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"}]

    async def scenario() -> None:
        await coordinator.sync("umo", first)
        await coordinator.sync("umo", second)

    asyncio.run(scenario())
    pending = store.pending("umo")
    assert pending["messages"] == second[2:]
    assert pending["expected_count"] == 2
    assert pending["attempts"] == 1
    assert store.cursor("umo")["remote_count"] == 2


def test_reset_accepts_any_append_mode() -> None:
    store, client = FakeStore(), FakeCaptureClient([success_ack(2, "weird")])
    coordinator = make_coordinator(client, store)
    messages = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]

    asyncio.run(coordinator.sync("umo", messages))
    assert store.cursor("umo")["remote_count"] == 2
    assert store.pending("umo") is None


def test_pending_stores_messages_and_retry_clears() -> None:
    store = FakeStore()
    client = FakeCaptureClient([failure_ack(), success_ack(2)])
    coordinator = make_coordinator(client, store)
    messages = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]

    async def scenario() -> None:
        await coordinator.sync("umo", messages)
        pending = store.pending("umo")
        assert pending["messages"] == messages
        assert pending["thread_id"]
        await coordinator.retry_pending("umo")

    asyncio.run(scenario())
    assert store.pending("umo") is None
    assert store.cursor("umo")["remote_count"] == 2


def test_pending_discarded_after_max_attempts() -> None:
    store = FakeStore()
    client = FakeCaptureClient([failure_ack(), failure_ack()])
    logger, messages = recording_logger("pending-discard")
    coordinator = make_coordinator(client, store, logger=logger, max_attempts=2)
    batch = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]

    async def scenario() -> None:
        await coordinator.sync("umo", batch)
        await coordinator.retry_pending("umo")

    asyncio.run(scenario())
    assert store.pending("umo") is None
    assert any("丢弃" in item for item in messages)


def test_payload_has_no_timestamp() -> None:
    store, client = FakeStore(), FakeCaptureClient([success_ack(2)])
    coordinator = make_coordinator(client, store)
    messages = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]

    asyncio.run(coordinator.sync("umo", messages))
    assert all("timestamp" not in item for item in client.calls[0]["messages"])


def test_conflict_resets_full_without_expected() -> None:
    store = FakeStore()
    client = FakeCaptureClient(
        [success_ack(2), failure_ack("checkpoint_conflict"), success_ack(4, "deduplicated")]
    )
    coordinator = make_coordinator(client, store)
    first = [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]
    second = first + [{"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"}]

    async def scenario() -> None:
        await coordinator.sync("umo", first)
        await coordinator.sync("umo", second)

    asyncio.run(scenario())
    assert client.calls[1]["expected_message_count"] == 2
    assert "expected_message_count" not in client.calls[2]
    assert len(client.calls[2]["messages"]) == 4


def test_identity_isolates_thread_and_cursor() -> None:
    store_a, store_b = FakeStore("a"), FakeStore("b")
    client_a, client_b = FakeCaptureClient([success_ack(2)]), FakeCaptureClient([success_ack(2)])
    a = make_coordinator(client_a, store_a, space_id="space-a")
    b = make_coordinator(client_b, store_b, space_id="space-b")
    messages = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]

    async def scenario() -> None:
        await a.sync("umo", messages)
        await b.sync("umo", messages)

    asyncio.run(scenario())
    assert store_a.cursor("umo")["thread_id"] != store_b.cursor("umo")["thread_id"]


def test_store_identity_does_not_hit_old_key() -> None:
    class KV:
        def __init__(self) -> None:
            self.data = {"cursor:umo": {"old": True}}

        async def get_kv_data(self, key: str, default=None):
            return self.data.get(key, default)

        async def put_kv_data(self, key: str, value) -> None:
            self.data[key] = value

        async def delete_kv_data(self, key: str) -> None:
            self.data.pop(key, None)

    store = NmemStore(KV(), "new-identity")
    assert asyncio.run(store.get_cursor("umo")) is None


def test_append_log_has_expected_ack_local_remote() -> None:
    store = FakeStore()
    client = FakeCaptureClient([success_ack(2), success_ack(4)])
    logger, messages = recording_logger("append-log", level=20)
    coordinator = make_coordinator(client, store, logger=logger)
    first = [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]
    second = first + [{"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"}]

    async def scenario() -> None:
        await coordinator.sync("umo", first)
        await coordinator.sync("umo", second)

    asyncio.run(scenario())
    assert any(
        "expected=2" in item and "ack_message_count=4" in item and "local_count=4" in item
        for item in messages
    )


def test_schedule_pending_retry_runs_in_background() -> None:
    store = FakeStore()
    client = FakeCaptureClient([failure_ack(), success_ack(2)])
    coordinator = make_coordinator(client, store)
    messages = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]

    async def scenario() -> None:
        await coordinator.sync("umo", messages)
        coordinator.schedule_pending_retry("umo")
        await coordinator.wait()

    asyncio.run(scenario())
    assert store.pending("umo") is None
    assert store.cursor("umo")["remote_count"] == 2


def test_append_pending_preserves_mode_expected_and_key() -> None:
    store = FakeStore()
    client = FakeCaptureClient([success_ack(2), failure_ack(), success_ack(4)])
    coordinator = make_coordinator(client, store)
    first = [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]
    second = first + [{"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"}]

    expected_key = ""

    async def scenario() -> None:
        nonlocal expected_key
        await coordinator.sync("umo", first)
        await coordinator.sync("umo", second)
        pending = store.pending("umo")
        assert pending["mode"] == "append"
        assert pending["expected_count"] == 2
        assert pending["idempotency_key"]
        expected_key = pending["idempotency_key"]
        assert pending["messages"] == second[2:]
        await coordinator.retry_pending("umo")

    asyncio.run(scenario())
    retry = client.calls[2]
    assert retry["expected_message_count"] == 2
    assert retry["idempotency_key"] == expected_key
    assert [m["content"] for m in retry["messages"]] == ["u2", "a2"]
    assert store.pending("umo") is None
    assert store.cursor("umo")["remote_count"] == 4


def test_reset_pending_has_no_expected_or_key() -> None:
    store = FakeStore()
    client = FakeCaptureClient([failure_ack(), success_ack(2)])
    coordinator = make_coordinator(client, store)
    messages = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]

    async def scenario() -> None:
        await coordinator.sync("umo", messages)
        pending = store.pending("umo")
        assert pending["mode"] == "reset"
        assert pending["expected_count"] is None
        assert pending["idempotency_key"] is None
        await coordinator.retry_pending("umo")

    asyncio.run(scenario())
    assert "expected_message_count" not in client.calls[1]
    assert "idempotency_key" not in client.calls[1]


def test_lock_limit_evicts_idle_entries() -> None:
    store = FakeStore()
    client = FakeCaptureClient([success_ack(1), success_ack(1), success_ack(1)])
    coordinator = make_coordinator(client, store, max_locks=2)
    messages = [{"role": "user", "content": "u"}]

    async def scenario() -> None:
        for index in range(3):
            await coordinator.sync(f"umo-{index}", messages)
        assert len(coordinator._locks) <= 2

    asyncio.run(scenario())


def test_source_and_source_id_filtering() -> None:
    from core.capture import should_capture_turn

    assert should_capture_turn(
        source="plugin", source_id="x", capture_plugin_turns=False, exclude_source_ids=[]
    ) == (False, "plugin_turn")
    assert should_capture_turn(
        source="plugin", source_id="x", capture_plugin_turns=True, exclude_source_ids=[]
    ) == (True, "")
    assert should_capture_turn(
        source="other", source_id="astrbot_heartbeat", capture_plugin_turns=True, exclude_source_ids=["astrbot_heartbeat"]
    ) == (False, "excluded_source_id")
    assert should_capture_turn(
        source="other", source_id="ok", capture_plugin_turns=False, exclude_source_ids=[]
    ) == (True, "")


def test_retry_and_sync_share_lock_no_cursor_rollback() -> None:
    store = FakeStore()
    gate = asyncio.Event()
    client = GatedCaptureClient([success_ack(4), success_ack(6)], gate)
    coordinator = make_coordinator(client, store)
    thread_id = thread_id_for(
        "umo", "space", "astrbot", "http://mem.local:14242", TEST_KEY_DIGEST
    )
    old = [{"role": "user", "content": f"old{i}"} for i in range(4)]
    new = old + [{"role": "user", "content": "new4"}, {"role": "user", "content": "new5"}]

    async def scenario() -> None:
        await store.set_cursor("umo", make_cursor(old, remote_count=4))
        await store.set_pending(
            "umo",
            {
                "mode": "reset",
                "thread_id": thread_id,
                "messages": old,
                "expected_count": None,
                "idempotency_key": None,
                "attempts": 0,
                "last_error": "",
                "at": "x",
                "window": old,
                "title": "old",
            },
        )
        retry_task = asyncio.create_task(coordinator.retry_pending("umo"))
        await client.started.wait()
        sync_task = asyncio.create_task(coordinator.sync("umo", new))
        await asyncio.sleep(0)
        assert not sync_task.done()
        assert len(client.calls) == 1
        gate.set()
        await asyncio.gather(retry_task, sync_task)

    asyncio.run(scenario())
    assert len(client.calls) == 2
    assert client.calls[1]["expected_message_count"] == 4
    assert store.cursor("umo")["confirmed_count"] == 6
    assert store.cursor("umo")["remote_count"] == 6


def test_retry_conflict_falls_back_to_window_reset() -> None:
    store = FakeStore()
    client = FakeCaptureClient(
        [success_ack(2), failure_ack(), failure_ack("checkpoint_conflict"), success_ack(4, "deduplicated")]
    )
    logger, messages = recording_logger("retry-conflict", level=20)
    coordinator = make_coordinator(client, store, logger=logger)
    first = [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]
    second = first + [{"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"}]

    async def scenario() -> None:
        await coordinator.sync("umo", first)
        await coordinator.sync("umo", second)
        await coordinator.retry_pending("umo")

    asyncio.run(scenario())
    assert any("pending conflict" in item for item in messages)
    assert client.calls[2]["expected_message_count"] == 2
    assert "expected_message_count" not in client.calls[3]
    assert len(client.calls[3]["messages"]) == 4
    assert store.pending("umo") is None
    assert store.cursor("umo")["confirmed_count"] == 4
    assert store.cursor("umo")["remote_count"] == 4


def test_retry_conflict_reset_failure_increments_once_and_persists_reset_mode() -> None:
    store = FakeStore()
    client = FakeCaptureClient(
        [
            success_ack(2),
            failure_ack(),
            failure_ack("checkpoint_conflict"),
            failure_ack("server_error"),
        ]
    )
    logger, messages = recording_logger("retry-conflict-fail", level=20)
    coordinator = make_coordinator(client, store, logger=logger)
    first = [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]
    second = first + [{"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"}]

    async def scenario() -> None:
        await coordinator.sync("umo", first)
        await coordinator.sync("umo", second)
        assert store.pending("umo")["mode"] == "append"
        await coordinator.retry_pending("umo")

    asyncio.run(scenario())
    assert any("pending conflict" in item for item in messages)
    assert len(client.calls) == 4
    pending = store.pending("umo")
    assert pending["mode"] == "reset"
    assert pending["messages"] == second
    assert pending["expected_count"] is None
    assert pending["idempotency_key"] is None
    assert pending["attempts"] == 2


def test_deduplicated_append_with_added_messages_is_rejected() -> None:
    store = FakeStore()
    client = FakeCaptureClient(
        [success_ack(2), success_ack(4, "deduplicated", messages_added=1, already_existed=False)]
    )
    coordinator = make_coordinator(client, store)
    first = [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]
    second = first + [{"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"}]

    async def scenario() -> None:
        await coordinator.sync("umo", first)
        await coordinator.sync("umo", second)

    asyncio.run(scenario())
    assert len(client.calls) == 2
    assert store.cursor("umo")["confirmed_count"] == 2
    pending = store.pending("umo")
    assert pending["mode"] == "append"
    assert pending["attempts"] == 1


def test_retry_pending_get_error_is_logged_not_raised() -> None:
    class BrokenGetPendingStore(FakeStore):
        async def get_pending(self, umo):
            raise RuntimeError("kv down")

    logger, messages = recording_logger("retry-get-error", level=20)
    coordinator = make_coordinator(FakeCaptureClient([]), BrokenGetPendingStore(), logger=logger)

    asyncio.run(coordinator.retry_pending("umo"))
    assert any("pending 重试异常" in item for item in messages)


def test_retry_pending_kv_error_is_logged_not_raised() -> None:
    class BrokenSetPendingStore(FakeStore):
        async def set_pending(self, umo, state):
            raise RuntimeError("kv down")

    store = BrokenSetPendingStore()
    thread_id = thread_id_for(
        "umo", "space", "astrbot", "http://mem.local:14242", TEST_KEY_DIGEST
    )
    pending = {
        "mode": "reset",
        "thread_id": thread_id,
        "messages": [{"role": "user", "content": "u"}],
        "expected_count": None,
        "idempotency_key": None,
        "attempts": 0,
        "last_error": "",
        "at": "x",
        "window": [{"role": "user", "content": "u"}],
        "title": "t",
    }
    store.data[store._key("pending", "umo")] = pending
    client = FakeCaptureClient([failure_ack()])
    logger, messages = recording_logger("retry-kv-error", level=20)
    coordinator = make_coordinator(client, store, logger=logger)

    asyncio.run(coordinator.retry_pending("umo"))
    assert len(client.calls) == 1
    assert any("pending 重试异常" in item for item in messages)


def test_pending_without_thread_id_is_cleared_on_retry() -> None:
    store = FakeStore()
    window = [{"role": "user", "content": "u"}]
    store.data[store._key("pending", "umo")] = {
        "mode": "reset",
        "messages": window,
        "expected_count": None,
        "idempotency_key": None,
        "attempts": 0,
        "at": "x",
        "window": window,
        "title": "t",
    }
    client = FakeCaptureClient([])
    logger, messages = recording_logger("pending-no-thread", level=20)
    coordinator = make_coordinator(client, store, logger=logger)

    asyncio.run(coordinator.retry_pending("umo"))
    assert store.pending("umo") is None
    assert client.calls == []
    assert any("缺少 thread_id" in item for item in messages)


def test_deduplicated_append_ack_advances_without_retry() -> None:
    store = FakeStore()
    client = FakeCaptureClient(
        [success_ack(2), success_ack(4, "deduplicated", messages_added=0, already_existed=True)]
    )
    coordinator = make_coordinator(client, store)
    first = [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]
    second = first + [{"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"}]

    async def scenario() -> None:
        await coordinator.sync("umo", first)
        await coordinator.sync("umo", second)

    asyncio.run(scenario())
    assert len(client.calls) == 2
    assert store.pending("umo") is None
    assert store.cursor("umo")["confirmed_count"] == 4
    assert store.cursor("umo")["remote_count"] == 4


def test_retry_deduplicated_append_advances() -> None:
    store = FakeStore()
    client = FakeCaptureClient(
        [
            success_ack(2),
            failure_ack(),
            success_ack(4, "deduplicated", messages_added=0, already_existed=True),
        ]
    )
    coordinator = make_coordinator(client, store)
    first = [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]
    second = first + [{"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"}]

    async def scenario() -> None:
        await coordinator.sync("umo", first)
        await coordinator.sync("umo", second)
        await coordinator.retry_pending("umo")

    asyncio.run(scenario())
    assert store.pending("umo") is None
    assert store.cursor("umo")["confirmed_count"] == 4


def test_append_empty_does_not_call_client_or_change_cursor() -> None:
    store = FakeStore()
    client = FakeCaptureClient([])
    coordinator = make_coordinator(client, store)
    messages = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]

    async def scenario() -> None:
        await store.set_cursor("umo", make_cursor(messages, remote_count=2))
        before = dict(store.cursor("umo"))
        await coordinator.sync("umo", messages)
        assert store.cursor("umo") == before

    asyncio.run(scenario())
    assert client.calls == []


def test_schedule_skips_turn_without_assistant_reply() -> None:
    store, client = FakeStore(), FakeCaptureClient([])
    coordinator = make_coordinator(client, store)

    async def scenario() -> None:
        coordinator.schedule("umo", [{"role": "user", "content": "u"}])
        await coordinator.wait()

    asyncio.run(scenario())
    assert client.calls == []
    assert store.cursor("umo") is None


def test_schedule_syncs_turn_with_assistant_reply() -> None:
    store, client = FakeStore(), FakeCaptureClient([success_ack(2)])
    coordinator = make_coordinator(client, store)

    async def scenario() -> None:
        coordinator.schedule(
            "umo",
            [
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": "a"},
            ],
        )
        await coordinator.wait()

    asyncio.run(scenario())
    assert len(client.calls) == 1
    assert store.cursor("umo")["confirmed_count"] == 2


def test_pending_survives_more_than_five_failures() -> None:
    store = FakeStore()
    client = FakeCaptureClient([failure_ack() for _ in range(6)])
    coordinator = make_coordinator(client, store, max_attempts=50)
    messages = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]

    async def scenario() -> None:
        await coordinator.sync("umo", messages)
        for _ in range(5):
            await coordinator.retry_pending("umo")

    asyncio.run(scenario())
    pending = store.pending("umo")
    assert pending is not None
    assert pending["attempts"] == 6
    assert len(client.calls) == 6


def test_pending_expired_by_age_is_discarded_without_import() -> None:
    store = FakeStore()
    client = FakeCaptureClient([])
    thread_id = thread_id_for(
        "umo", "space", "astrbot", "http://mem.local:14242", TEST_KEY_DIGEST
    )
    window = [{"role": "user", "content": "u"}]
    store.data[store._key("pending", "umo")] = {
        "mode": "reset",
        "thread_id": thread_id,
        "messages": window,
        "expected_count": None,
        "idempotency_key": None,
        "attempts": 1,
        "created_at": (
            datetime.now(timezone.utc) - timedelta(hours=25)
        ).isoformat(),
        "last_error": "",
        "at": "x",
        "window": window,
        "title": "t",
    }
    logger, messages = recording_logger("pending-age")
    coordinator = make_coordinator(client, store, logger=logger)

    asyncio.run(coordinator.retry_pending("umo"))
    assert client.calls == []
    assert store.pending("umo") is None
    assert any("丢弃" in item for item in messages)


def test_window_shrink_keeps_pending_created_at() -> None:
    store = FakeStore()
    client = FakeCaptureClient([success_ack(4), failure_ack(), failure_ack()])
    coordinator = make_coordinator(client, store)
    old = [{"role": "user", "content": f"m{i}"} for i in range(4)]
    grown = old + [
        {"role": "user", "content": "u4"},
        {"role": "assistant", "content": "a4"},
    ]
    shrunk = grown[2:]

    async def scenario() -> str:
        await coordinator.sync("umo", old)
        await coordinator.sync("umo", grown)
        created = store.pending("umo")["created_at"]
        await coordinator.sync("umo", shrunk)
        return created

    created = asyncio.run(scenario())
    pending = store.pending("umo")
    assert pending["mode"] == "reset"
    assert pending["created_at"] == created
    assert pending["attempts"] == 1
    assert [m["content"] for m in client.calls[2]["messages"]] == [
        m["content"] for m in shrunk
    ]


def test_api_key_isolates_thread_and_cursor() -> None:
    store_a, store_b = FakeStore("a"), FakeStore("b")
    client_a = FakeCaptureClient([success_ack(2)])
    client_b = FakeCaptureClient([success_ack(2)])
    a = make_coordinator(client_a, store_a, api_key_digest="aaaaaaaa")
    b = make_coordinator(client_b, store_b, api_key_digest="bbbbbbbb")
    messages = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]

    async def scenario() -> None:
        await a.sync("umo", messages)
        await b.sync("umo", messages)

    asyncio.run(scenario())
    assert store_a.cursor("umo")["thread_id"] != store_b.cursor("umo")["thread_id"]
