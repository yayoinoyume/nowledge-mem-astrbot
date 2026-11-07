"""离线 outbox：原子写、权限、加载、上限、迁移与重试成功后删除。"""

from __future__ import annotations

import asyncio
import json
import os
import stat
from pathlib import Path

import pytest

from core.outbox import MAX_OUTBOX_ENTRIES, OutboxStore
from core.store import NmemStore
from tests.helpers import (
    FakeCaptureClient,
    FakeKV,
    failure_ack,
    make_coordinator,
    quiet_logger,
    success_ack,
)


def _outbox(tmp_path: Path, *, max_entries: int = MAX_OUTBOX_ENTRIES) -> OutboxStore:
    return OutboxStore(
        tmp_path / "outbox",
        "identity123",
        logger=quiet_logger(),
        max_entries=max_entries,
    )


def _pending(**overrides):
    item = {
        "mode": "append",
        "thread_id": "astrbot-test",
        "messages": [{"role": "user", "content": "u1"}],
        "expected_count": 0,
        "idempotency_key": "key-1",
        "attempts": 1,
        "created_at": "2026-11-07T00:00:00+00:00",
        "last_error": "HTTP 500",
        "at": "2026-11-07T00:01:00+00:00",
        "window": [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ],
        "title": "标题",
    }
    item.update(overrides)
    return item


def test_set_get_clear_roundtrip_and_permissions(tmp_path: Path) -> None:
    outbox = _outbox(tmp_path)
    pending = _pending()

    async def scenario() -> None:
        await outbox.set_pending("qq:group:1", pending)
        loaded = await outbox.get_pending("qq:group:1")
        assert loaded == pending
        assert await outbox.get_pending("missing") is None
        await outbox.clear_pending("qq:group:1")
        assert await outbox.get_pending("qq:group:1") is None

    asyncio.run(scenario())
    files = list(outbox.directory.glob("*.json"))
    assert files == []
    assert not list(outbox.directory.glob("*.tmp"))


def test_file_is_0600_and_contains_umo(tmp_path: Path) -> None:
    outbox = _outbox(tmp_path)

    async def scenario() -> None:
        await outbox.set_pending("qq:group:1", _pending())

    asyncio.run(scenario())
    path = outbox.directory / "qq%3Agroup%3A1.json"
    assert path.is_file()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["umo"] == "qq:group:1"
    assert payload["thread_id"] == "astrbot-test"
    assert "unexpected" not in payload
    assert not list(outbox.directory.glob("*.tmp"))


def test_only_allowed_fields_are_persisted(tmp_path: Path) -> None:
    outbox = _outbox(tmp_path)
    pending = _pending()
    pending["unexpected"] = "drop-me"
    pending["api_key"] = "secret"

    async def scenario() -> None:
        await outbox.set_pending("umo", pending)

    asyncio.run(scenario())
    path = outbox.directory / "umo.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "unexpected" not in payload
    assert "api_key" not in payload
    assert "secret" not in path.read_text(encoding="utf-8")


def test_load_reads_existing_entries(tmp_path: Path) -> None:
    first = _outbox(tmp_path)

    async def write() -> None:
        await first.set_pending("umo-a", _pending(thread_id="t-a"))
        await first.set_pending("umo-b", _pending(thread_id="t-b"))

    asyncio.run(write())
    second = _outbox(tmp_path)

    async def read() -> None:
        assert await second.load() == 2
        assert (await second.get_pending("umo-a"))["thread_id"] == "t-a"
        assert (await second.get_pending("umo-b"))["thread_id"] == "t-b"

    asyncio.run(read())


def test_load_skips_corrupt_and_missing_umo_files(tmp_path: Path) -> None:
    outbox = _outbox(tmp_path)
    outbox.directory.mkdir(parents=True)
    (outbox.directory / "bad.json").write_text("{not-json", encoding="utf-8")
    (outbox.directory / "no-umo.json").write_text(
        json.dumps({"thread_id": "t"}), encoding="utf-8"
    )

    async def scenario() -> None:
        assert await outbox.load() == 0
        assert await outbox.get_pending("anything") is None

    asyncio.run(scenario())
    assert (outbox.directory / "bad.json").exists()
    assert (outbox.directory / "no-umo.json").exists()


def test_failed_replace_keeps_old_file_and_removes_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outbox = _outbox(tmp_path)

    async def seed() -> None:
        await outbox.set_pending("umo", _pending(thread_id="old"))

    asyncio.run(seed())
    path = outbox.directory / "umo.json"
    before = path.read_text(encoding="utf-8")

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("core.outbox.os.replace", boom)

    async def attempt() -> None:
        with pytest.raises(OSError):
            await outbox.set_pending("umo", _pending(thread_id="new"))

    asyncio.run(attempt())
    assert path.read_text(encoding="utf-8") == before
    assert not list(outbox.directory.glob("*.tmp"))


def test_max_entries_evicts_oldest(tmp_path: Path) -> None:
    outbox = _outbox(tmp_path, max_entries=2)

    async def scenario() -> None:
        await outbox.set_pending("oldest", _pending())
        await outbox.set_pending("middle", _pending())
        os.utime(outbox.directory / "oldest.json", (1, 1))
        os.utime(outbox.directory / "middle.json", (2, 2))
        await outbox.set_pending("newest", _pending())

    asyncio.run(scenario())
    names = sorted(path.name for path in outbox.directory.glob("*.json"))
    assert names == ["middle.json", "newest.json"]


def test_dot_umo_is_not_hidden(tmp_path: Path) -> None:
    outbox = _outbox(tmp_path)

    async def scenario() -> None:
        await outbox.set_pending("..", _pending())
        assert await outbox.get_pending("..") is not None

    asyncio.run(scenario())
    files = list(outbox.directory.glob("*.json"))
    assert len(files) == 1
    assert not files[0].name.startswith(".")


def test_dot_and_underscore_dot_do_not_collide(tmp_path: Path) -> None:
    outbox = _outbox(tmp_path)

    async def scenario() -> None:
        await outbox.set_pending(".", _pending(thread_id="dot"))
        await outbox.set_pending("_.", _pending(thread_id="underscore-dot"))
        assert (await outbox.get_pending("."))["thread_id"] == "dot"
        assert (await outbox.get_pending("_."))["thread_id"] == "underscore-dot"

    asyncio.run(scenario())
    files = sorted(path.name for path in outbox.directory.glob("*.json"))
    assert len(files) == 2
    assert files[0] != files[1]

    second = _outbox(tmp_path)

    async def reload() -> None:
        assert await second.load() == 2
        assert (await second.get_pending("."))["thread_id"] == "dot"
        assert (await second.get_pending("_."))["thread_id"] == "underscore-dot"

    asyncio.run(reload())


def test_long_umo_uses_hash_filename(tmp_path: Path) -> None:
    outbox = _outbox(tmp_path)
    umo = "x" * 300

    async def scenario() -> None:
        await outbox.set_pending(umo, _pending())
        assert await outbox.get_pending(umo) is not None

    asyncio.run(scenario())
    files = list(outbox.directory.glob("*.json"))
    assert len(files) == 1
    assert files[0].name.startswith("sha1-")
    assert len(files[0].name) < 80


def test_nmem_store_migrates_legacy_kv_pending(tmp_path: Path) -> None:
    kv = FakeKV()
    outbox = _outbox(tmp_path)
    store = NmemStore(kv, "identity123", outbox=outbox, logger=quiet_logger())
    key = "pending:identity123:umo"
    kv.data[key] = _pending()

    async def scenario() -> None:
        assert (await store.get_pending("umo"))["thread_id"] == "astrbot-test"
        assert key not in kv.data
        assert len(list(outbox.directory.glob("*.json"))) == 1
        assert (await store.get_pending("umo"))["thread_id"] == "astrbot-test"

    asyncio.run(scenario())


def test_nmem_store_set_pending_clears_legacy_kv_key(tmp_path: Path) -> None:
    kv = FakeKV()
    outbox = _outbox(tmp_path)
    store = NmemStore(kv, "identity123", outbox=outbox, logger=quiet_logger())
    key = "pending:identity123:umo"
    kv.data[key] = _pending(thread_id="legacy")

    async def scenario() -> None:
        await store.set_pending("umo", _pending(thread_id="new"))
        assert key not in kv.data
        assert (await store.get_pending("umo"))["thread_id"] == "new"

    asyncio.run(scenario())


def test_nmem_store_falls_back_to_kv_when_outbox_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kv = FakeKV()
    outbox = _outbox(tmp_path)
    store = NmemStore(kv, "identity123", outbox=outbox, logger=quiet_logger())

    async def boom(*args, **kwargs):
        raise OSError("read-only")

    monkeypatch.setattr(outbox, "set_pending", boom)

    async def scenario() -> None:
        await store.set_pending("umo", _pending())
        assert kv.data["pending:identity123:umo"]["thread_id"] == "astrbot-test"

    asyncio.run(scenario())


def test_capture_retry_success_deletes_outbox_file(tmp_path: Path) -> None:
    kv = FakeKV()
    outbox = _outbox(tmp_path)
    store = NmemStore(kv, "identity123", outbox=outbox, logger=quiet_logger())
    client = FakeCaptureClient([failure_ack("server_error"), success_ack(2)])
    coordinator = make_coordinator(client, store)
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]

    async def scenario() -> None:
        await coordinator.sync("umo", messages)
        assert len(list(outbox.directory.glob("*.json"))) == 1
        await coordinator.retry_pending("umo")
        assert list(outbox.directory.glob("*.json")) == []

    asyncio.run(scenario())
    assert len(client.calls) == 2

def test_concurrent_threaded_writes_keep_single_valid_file(tmp_path: Path) -> None:
    """多线程同时写同一 umo：只留一个文件、合法 JSON、无 tmp 残留。"""
    from concurrent.futures import ThreadPoolExecutor

    outbox = _outbox(tmp_path)

    def write(index: int) -> None:
        asyncio.run(outbox.set_pending("umo", _pending(attempts=index)))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(20)))

    files = list(outbox.directory.glob("*.json"))
    assert len(files) == 1
    assert list(outbox.directory.glob("*.tmp")) == []
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert 0 <= data["attempts"] < 20

    loaded = asyncio.run(outbox.get_pending("umo"))
    assert loaded is not None
    assert loaded["attempts"] == data["attempts"]
