"""测试用的假 HTTP 会话、假 KV 与共享构造器。"""

from __future__ import annotations

import logging
from typing import Any

from core.capture import CaptureCoordinator
from core.delta import message_hash, prefix_fingerprint

TEST_KEY_DIGEST = "deadbeef"


def make_cursor(
    messages: list[dict[str, str]],
    remote_count: int | None = None,
    *,
    last_confirmed_hash: str | None = None,
    confirmed_count: int | None = None,
    prefix: str | None = None,
    title: str = "标题",
) -> dict[str, Any]:
    hashes = [message_hash(item["role"], item["content"]) for item in messages]
    count = len(messages) if confirmed_count is None else confirmed_count
    return {
        "thread_id": "astrbot-test",
        "last_confirmed_hash": (
            hashes[-1] if last_confirmed_hash is None and hashes else last_confirmed_hash
        ),
        "confirmed_count": count,
        "prefix_fingerprint": (
            prefix_fingerprint(messages, count) if prefix is None else prefix
        ),
        "remote_count": len(messages) if remote_count is None else remote_count,
        "title": title,
    }


class FakeResponse:
    def __init__(self, status: int = 200, payload: Any = None, error: BaseException | None = None) -> None:
        self.status = status
        self._payload = payload
        self._error = error

    async def __aenter__(self) -> "FakeResponse":
        return self

    async def __aexit__(self, *args: Any) -> bool:
        return False

    async def json(self, content_type: str | None = None) -> Any:
        if self._error is not None:
            raise self._error
        return self._payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    def request(self, method: str, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"method": method, "url": url, "kwargs": kwargs})
        if not self.responses:
            raise AssertionError("FakeSession 没有剩余响应")
        return self.responses.pop(0)


class FakeKV:
    """AstrBot 插件级 KV 的最小替身。"""

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}

    async def get_kv_data(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    async def put_kv_data(self, key: str, value: Any) -> None:
        self.data[key] = value

    async def delete_kv_data(self, key: str) -> None:
        self.data.pop(key, None)


class FakeCaptureClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def import_thread(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        self.calls.append(payload)
        if not self.responses:
            raise AssertionError("FakeCaptureClient 没有剩余响应")
        return self.responses.pop(0)


class FakeStore:
    def __init__(self, identity: str = "test") -> None:
        self.identity = identity
        self.data: dict[tuple[str, str, str], Any] = {}

    def _key(self, kind: str, umo: str) -> tuple[str, str, str]:
        return (kind, self.identity, umo)

    async def get_bundle(self, umo: str) -> Any:
        return self.data.get(self._key("bundle", umo))

    async def set_bundle(self, umo: str, state: Any) -> None:
        self.data[self._key("bundle", umo)] = state

    async def get_cursor(self, umo: str) -> Any:
        return self.data.get(self._key("cursor", umo))

    async def set_cursor(self, umo: str, state: Any) -> None:
        self.data[self._key("cursor", umo)] = state

    async def get_pending(self, umo: str) -> Any:
        return self.data.get(self._key("pending", umo))

    async def set_pending(self, umo: str, state: Any) -> None:
        self.data[self._key("pending", umo)] = state

    async def clear_pending(self, umo: str) -> None:
        self.data.pop(self._key("pending", umo), None)

    def cursor(self, umo: str) -> Any:
        return self.data.get(self._key("cursor", umo))

    def pending(self, umo: str) -> Any:
        return self.data.get(self._key("pending", umo))


def quiet_logger() -> logging.Logger:
    logger = logging.getLogger("nowledge-mem-tests")
    logger.setLevel(logging.CRITICAL)
    return logger


def make_coordinator(
    client: Any,
    store: FakeStore,
    *,
    space_id: str = "space",
    agent_id: str = "astrbot",
    api_url: str = "http://mem.local:14242",
    api_key_digest: str = TEST_KEY_DIGEST,
    logger: logging.Logger | None = None,
    max_attempts: int = 50,
    max_pending_age: float = 24 * 60 * 60,
    max_locks: int = 256,
) -> CaptureCoordinator:
    return CaptureCoordinator(
        client,
        store,
        source_app="astrbot",
        space_id=space_id,
        agent_id=agent_id,
        api_url=api_url,
        api_key_digest=api_key_digest,
        logger=logger or quiet_logger(),
        max_attempts=max_attempts,
        max_pending_age=max_pending_age,
        max_locks=max_locks,
    )


def recording_logger(name: str, level: int = logging.WARNING) -> tuple[logging.Logger, list[str]]:
    messages: list[str] = []

    class Handler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    logger = logging.getLogger(name)
    logger.handlers = [Handler()]
    logger.setLevel(level)
    logger.propagate = False
    return logger, messages


def success_ack(
    message_count: int = 2,
    append_mode: str | None = "checkpointed",
    *,
    messages_added: int = 1,
    already_existed: bool = False,
) -> dict[str, Any]:
    return {
        "success": True,
        "imported_count": 1,
        "skipped_count": 0,
        "failed_count": 0,
        "results": [
            {
                "success": True,
                "thread_id": "astrbot-test",
                "message_count": message_count,
                "messages_added": messages_added,
                "already_existed": already_existed,
                "append_mode": append_mode,
            }
        ],
    }


def failure_ack(error_code: str = "bad_request") -> dict[str, Any]:
    return {
        "success": False,
        "imported_count": 0,
        "skipped_count": 0,
        "failed_count": 1,
        "results": [
            {
                "success": False,
                "thread_id": "astrbot-test",
                "message_count": 0,
                "messages_added": 0,
                "already_existed": False,
                "append_mode": None,
                "error": "failed",
                "error_code": error_code,
            }
        ],
    }
