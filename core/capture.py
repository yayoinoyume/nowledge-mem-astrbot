"""对话捕获协调器：后台串行导入 Mem thread。"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from .ack import ImportAck, parse_import_ack
from .delta import (
    APPEND,
    RESET,
    DeltaPlan,
    compute_delta,
    message_hash,
    prefix_fingerprint,
    thread_id_for,
    thread_title,
)

MAX_ATTEMPTS = 50
MAX_PENDING_AGE_SECONDS = 24 * 60 * 60
MAX_LOCKS = 256


def should_capture_turn(
    *,
    source: str | None,
    source_id: str | None,
    capture_plugin_turns: bool,
    exclude_source_ids: list[str],
) -> tuple[bool, str]:
    excluded = {str(item).strip() for item in exclude_source_ids if str(item).strip()}
    source_id_text = str(source_id).strip() if source_id else ""
    if source_id_text and source_id_text in excluded:
        return False, "excluded_source_id"
    if source == "plugin" and not capture_plugin_turns:
        return False, "plugin_turn"
    return True, ""


def has_assistant_reply(messages: list[dict[str, str]]) -> bool:
    """当前回合必须已有 assistant 回复；LLM 报错轮只有 user。"""
    return bool(messages) and messages[-1]["role"] == "assistant"


class _LockEntry:
    __slots__ = ("active", "lock")

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.active = 0


class CaptureCoordinator:
    """每 thread 一把锁，后台串行地把增量导入同一个 Mem thread。"""

    def __init__(
        self,
        client: Any,
        store: Any,
        *,
        source_app: str,
        space_id: str,
        agent_id: str,
        api_url: str,
        api_key_digest: str = "",
        logger: logging.Logger | None = None,
        max_attempts: int = MAX_ATTEMPTS,
        max_pending_age: float = MAX_PENDING_AGE_SECONDS,
        max_locks: int = MAX_LOCKS,
    ) -> None:
        self._client = client
        self._store = store
        self._source_app = source_app
        self._space_id = space_id
        self._agent_id = agent_id
        self._api_url = api_url
        self._api_key_digest = api_key_digest
        self._logger = logger or logging.getLogger(__name__)
        self._max_attempts = max_attempts
        self._max_pending_age = max_pending_age
        self._max_locks = max_locks
        self._locks: dict[str, _LockEntry] = {}
        self._tasks: set[asyncio.Task] = set()

    def schedule(self, umo: str, messages: list[dict[str, str]]) -> None:
        if not messages:
            return
        if not has_assistant_reply(messages):
            self._logger.debug("跳过记忆捕获：reason=no_assistant_reply")
            return
        self._spawn(self.sync(umo, list(messages)))

    def schedule_pending_retry(self, umo: str) -> None:
        self._spawn(self.retry_pending(umo))

    def _spawn(self, coro: Any) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def wait(self) -> None:
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    async def shutdown(self) -> None:
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    async def retry_pending(self, umo: str) -> None:
        try:
            await self._retry_pending(umo)
        except Exception as exc:
            self._logger.warning("pending 重试异常：%s", type(exc).__name__)

    async def _retry_pending(self, umo: str) -> None:
        snapshot = await self._store.get_pending(umo)
        if not isinstance(snapshot, dict):
            return
        thread_id = snapshot.get("thread_id")
        if not isinstance(thread_id, str) or not thread_id:
            thread_id = thread_id_for(
                umo, self._space_id, self._agent_id, self._api_url, self._api_key_digest
            )
        async with self._thread_lock(thread_id):
            pending = await self._store.get_pending(umo)
            if not isinstance(pending, dict):
                return
            if pending.get("thread_id") != thread_id:
                return
            await self._retry_locked(umo, pending)

    @asynccontextmanager
    async def _thread_lock(self, thread_id: str):
        entry = self._locks.pop(thread_id, None)
        if entry is None:
            entry = _LockEntry()
        entry.active += 1
        self._locks[thread_id] = entry
        try:
            async with entry.lock:
                yield
        finally:
            entry.active -= 1
            self._evict_locks()

    async def _retry_locked(self, umo: str, pending: dict[str, Any]) -> None:
        attempts = _int(pending.get("attempts"))
        thread_id = pending.get("thread_id")
        if self._exhausted(pending.get("created_at"), attempts):
            await self._store.clear_pending(umo)
            self._logger.warning(
                "丢弃 Nowledge Mem pending：thread=%s messages=%s attempts=%s",
                thread_id,
                len(pending.get("messages") or []),
                attempts,
            )
            return
        mode = pending.get("mode")
        messages = pending.get("messages")
        expected = pending.get("expected_count")
        key = pending.get("idempotency_key")
        window = pending.get("window")
        if (
            mode not in (APPEND, RESET)
            or not isinstance(thread_id, str)
            or not isinstance(messages, list)
            or not messages
            or not isinstance(window, list)
            or not window
            or (mode == APPEND and (type(expected) is not int or not isinstance(key, str)))
        ):
            await self._store.clear_pending(umo)
            return
        title = str(pending.get("title") or thread_title(window))
        data, request_error = await self._import(
            self._payload(
                thread_id,
                title,
                messages,
                expected if mode == APPEND else None,
                key if mode == APPEND else None,
            )
        )
        ack = parse_import_ack(data)
        if ack.status == "conflict":
            self._logger.info(
                "pending conflict，使用 window 全量 reset：thread=%s error=%s",
                thread_id,
                ack.error,
            )
            mode = RESET
            pending["mode"] = RESET
            pending["messages"] = list(window)
            pending["expected_count"] = None
            pending["idempotency_key"] = None
            data, request_error = await self._import(
                self._payload(thread_id, title, window, None, None)
            )
            ack = parse_import_ack(data)
        if ack.status == "acknowledged":
            if mode == APPEND and not self._append_ack_ok(ack):
                ack = ImportAck(
                    "failed",
                    message_count=ack.message_count,
                    append_mode=ack.append_mode,
                    error="append_not_checkpointed",
                    messages_added=ack.messages_added,
                    already_existed=ack.already_existed,
                )
            else:
                await self._commit_cursor(umo, thread_id, title, window, ack.message_count)
                await self._store.clear_pending(umo)
                self._logger.info(
                    "pending 重试成功：mode=%s thread=%s ack=%s append_mode=%s",
                    mode,
                    thread_id,
                    ack.message_count,
                    ack.append_mode,
                )
                return
        error_label = request_error or ack.error or ack.status
        next_attempts = attempts + 1
        if self._exhausted(pending.get("created_at"), next_attempts):
            await self._store.clear_pending(umo)
            self._logger.warning(
                "Nowledge Mem thread %s pending 重试失败 %s 次（最后错误：%s），"
                "超过 %s 次或 %s 小时上限，丢弃",
                thread_id,
                next_attempts,
                error_label,
                self._max_attempts,
                int(self._max_pending_age // 3600),
            )
            return
        pending["attempts"] = next_attempts
        pending["last_error"] = error_label
        pending["at"] = _now_iso()
        await self._store.set_pending(umo, pending)
        self._logger.warning(
            "pending 重试失败：mode=%s thread=%s 尝试 %s/%s 错误=%s",
            mode,
            thread_id,
            next_attempts,
            self._max_attempts,
            error_label,
        )

    async def sync(self, umo: str, messages: list[dict[str, str]]) -> None:
        if not messages:
            return
        thread_id = thread_id_for(
            umo, self._space_id, self._agent_id, self._api_url, self._api_key_digest
        )
        async with self._thread_lock(thread_id):
            try:
                await self._sync_locked(umo, thread_id, messages)
            except Exception as exc:
                self._logger.warning("对话同步失败：%s", type(exc).__name__)

    def _evict_locks(self) -> None:
        while len(self._locks) > self._max_locks:
            for key, entry in self._locks.items():
                if entry.active > 0:
                    continue
                del self._locks[key]
                break
            else:
                return

    def _exhausted(self, created_at: Any, attempts: int) -> bool:
        if attempts >= self._max_attempts:
            return True
        created = _parse_time(created_at)
        if created is None:
            return False
        age = (datetime.now(timezone.utc) - created).total_seconds()
        return age >= self._max_pending_age

    @staticmethod
    def _append_ack_ok(ack: ImportAck) -> bool:
        if ack.append_mode == "checkpointed":
            return True
        return (
            ack.append_mode == "deduplicated"
            and ack.messages_added == 0
            and ack.already_existed is True
        )

    async def _sync_locked(
        self, umo: str, thread_id: str, messages: list[dict[str, str]]
    ) -> None:
        cursor = await self._store.get_cursor(umo)
        plan = compute_delta(messages, cursor)
        if plan.mode == APPEND and not plan.messages:
            return
        title = thread_title(messages, cursor.get("title") if isinstance(cursor, dict) else None)
        is_append = plan.mode == APPEND
        expected = cursor.get("remote_count") if is_append and isinstance(cursor, dict) else None
        key = (
            self._idempotency_key(thread_id, expected, plan.messages)
            if is_append
            else None
        )
        pending = await self._store.get_pending(umo)
        attempts = 0
        if (
            isinstance(pending, dict)
            and pending.get("mode") == plan.mode
            and pending.get("messages") == plan.messages
            and pending.get("expected_count") == expected
            and pending.get("idempotency_key") == key
        ):
            attempts = _int(pending.get("attempts"))
        # 窗口缩水/批次变化时保留 pending 的创建时间，24 小时预算按会话失败 streak 计算。
        created_at = _pending_created_at(pending)
        data, request_error = await self._import(
            self._payload(thread_id, title, plan.messages, expected, key)
        )
        ack = parse_import_ack(data)
        if ack.status == "conflict":
            self._logger.debug("thread %s 需要 reset 全量重导：%s", thread_id, ack.error)
            is_append, expected, key = False, None, None
            plan = DeltaPlan(RESET, 0, list(messages), plan.hashes)
            data, request_error = await self._import(
                self._payload(thread_id, title, plan.messages, None, None)
            )
            ack = parse_import_ack(data)
        if ack.status == "acknowledged":
            if is_append and not self._append_ack_ok(ack):
                ack = ImportAck(
                    "failed",
                    message_count=ack.message_count,
                    append_mode=ack.append_mode,
                    error="append_not_checkpointed",
                    messages_added=ack.messages_added,
                    already_existed=ack.already_existed,
                )
            else:
                await self._commit_cursor(umo, thread_id, title, messages, ack.message_count)
                await self._store.clear_pending(umo)
                self._logger.info(
                    "已导入 %s 条消息到 Nowledge Mem thread %s",
                    len(plan.messages),
                    thread_id,
                )
                self._logger.info(
                    "Nowledge Mem %s 成功：expected=%s ack_message_count=%s "
                    "append_mode=%s local_count=%s remote_count=%s",
                    plan.mode,
                    expected,
                    ack.message_count,
                    ack.append_mode,
                    len(plan.hashes),
                    ack.message_count,
                )
                return
        error_label = request_error or ack.error or ack.status
        next_attempts = attempts + 1
        if self._exhausted(created_at, next_attempts):
            await self._store.clear_pending(umo)
            self._logger.warning(
                "Nowledge Mem thread %s 同步失败 %s 次（最后错误：%s），"
                "超过 %s 次或 %s 小时上限，丢弃 pending（messages=%s）",
                thread_id,
                next_attempts,
                error_label,
                self._max_attempts,
                int(self._max_pending_age // 3600),
                len(plan.messages),
            )
            return
        await self._store.set_pending(
            umo,
            {
                "mode": plan.mode,
                "thread_id": thread_id,
                "messages": plan.messages,
                "expected_count": expected,
                "idempotency_key": key,
                "attempts": next_attempts,
                "created_at": created_at,
                "last_error": error_label,
                "at": _now_iso(),
                "window": list(messages),
                "title": title,
            },
        )
        self._logger.warning(
            "导入 Nowledge Mem thread %s 失败（尝试 %s/%s，错误：%s）",
            thread_id,
            next_attempts,
            self._max_attempts,
            error_label,
        )

    async def _commit_cursor(
        self,
        umo: str,
        thread_id: str,
        title: str,
        window: list[dict[str, str]],
        remote_count: int,
    ) -> None:
        last = message_hash(window[-1]["role"], window[-1]["content"]) if window else None
        await self._store.set_cursor(
            umo,
            {
                "thread_id": thread_id,
                "last_confirmed_hash": last,
                "confirmed_count": len(window),
                "prefix_fingerprint": prefix_fingerprint(window, len(window)),
                "remote_count": remote_count,
                "title": title,
            },
        )

    async def _import(
        self, payload: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, str | None]:
        try:
            return await self._client.import_thread(payload), None
        except Exception as exc:
            self._logger.warning("导入请求异常：%s", type(exc).__name__)
            return None, type(exc).__name__

    @staticmethod
    def _idempotency_key(
        thread_id: str, expected_count: int | None, messages: list[dict[str, str]]
    ) -> str:
        batch_hash = hashlib.sha1(
            "\n".join(
                message_hash(item["role"], item["content"]) for item in messages
            ).encode("utf-8")
        ).hexdigest()
        return f"astrbot:{thread_id}:{expected_count}:{batch_hash[:16]}"

    def _payload(
        self,
        thread_id: str,
        title: str,
        messages: list[dict[str, str]],
        expected_count: int | None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "thread_id": thread_id,
            "title": title,
            "messages": [
                {
                    "role": item["role"],
                    "content": item["content"],
                    "metadata": {"source_app": self._source_app},
                }
                for item in messages
            ],
            "source": self._source_app,
        }
        if self._space_id:
            payload["space_id"] = self._space_id
        if type(expected_count) is int and expected_count >= 0:
            payload["expected_message_count"] = expected_count
            payload["idempotency_key"] = idempotency_key or self._idempotency_key(
                thread_id, expected_count, messages
            )
        return payload


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _pending_created_at(pending: Any) -> str:
    if isinstance(pending, dict):
        created = pending.get("created_at")
        if isinstance(created, str) and created:
            return created
    return _now_iso()


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
