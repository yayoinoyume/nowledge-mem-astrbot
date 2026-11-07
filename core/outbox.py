"""离线 outbox：把待重试的 capture 批次原子落盘。

一个会话（umo）一个文件，路径为
`<plugin_data>/nowledge_mem_astrbot/outbox/<identity>/<umo>.json`。
文件只保留重试真正用到的字段；写入走同目录临时文件 + rename，
权限固定 0600，单个身份最多保留 50 个会话文件。
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

MAX_OUTBOX_ENTRIES = 50
MAX_NAME_CHARS = 180

# 只落盘重试路径真正读取的字段；umo 用于启动加载时还原键。
ALLOWED_FIELDS = (
    "mode",
    "thread_id",
    "messages",
    "expected_count",
    "idempotency_key",
    "attempts",
    "created_at",
    "last_error",
    "at",
    "window",
    "title",
)


def _safe_name(value: Any) -> str:
    """把 umo/identity 变成单层文件名；超长时改用哈希，避免路径穿越。

    URL 编码不会转义 "." 与 "_"，只给这两种首字符加前缀会让 "." 与 "_." 撞成同一
    文件名（pending 被静默覆盖）。因此在加前缀时附加 8 位文本哈希，保持映射单射；
    其余值仍用原编码，日常 umo 的文件名不变。
    """
    text = str(value or "").strip() or "_"
    encoded = quote(text, safe="")
    if len(encoded) > MAX_NAME_CHARS:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        return f"sha1-{digest}"
    if encoded.startswith((".", "_")):
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        return f"_.{digest}.{encoded}"
    return encoded


def _allowed(state: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(state[key])
        for key in ALLOWED_FIELDS
        if key in state
    }


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    """同目录临时文件 + rename；失败时清理临时文件，旧文件保持不变。"""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        os.chmod(path, 0o600)
        _fsync_dir(path.parent)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


class OutboxStore:
    """磁盘版 pending 存储；接口与 NmemStore 的 pending 三件套一致。"""

    def __init__(
        self,
        root: Path,
        identity: str,
        *,
        logger: logging.Logger | None = None,
        max_entries: int = MAX_OUTBOX_ENTRIES,
    ) -> None:
        self._dir = Path(root) / _safe_name(identity or "default")
        self._logger = logger or logging.getLogger(__name__)
        self._max_entries = max(1, int(max_entries))
        self._lock = threading.RLock()
        self._cache: dict[str, dict[str, Any]] = {}

    @property
    def directory(self) -> Path:
        return self._dir

    async def load(self) -> int:
        """启动时扫描目录；损坏文件只跳过并记 warning，不删除。"""
        return await asyncio.to_thread(self._load_sync)

    async def get_pending(self, umo: str) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._get_sync, umo)

    async def set_pending(self, umo: str, state: dict[str, Any]) -> None:
        if not str(umo or "").strip():
            raise ValueError("umo 不能为空")
        if not isinstance(state, dict):
            raise ValueError("pending 必须是对象")
        payload = _allowed(state)
        payload["umo"] = umo
        await asyncio.to_thread(self._set_sync, umo, payload)

    async def clear_pending(self, umo: str) -> None:
        await asyncio.to_thread(self._clear_sync, umo)

    def _load_sync(self) -> int:
        with self._lock:
            self._cache = {}
            if not self._dir.is_dir():
                return 0
            for path in sorted(self._dir.glob("*.json")):
                entry = self._read_entry(path)
                if entry is None:
                    continue
                umo = entry.pop("umo")
                self._cache[umo] = entry
            self._enforce_cap_sync()
            return len(self._cache)

    def _get_sync(self, umo: str) -> dict[str, Any] | None:
        with self._lock:
            cached = self._cache.get(umo)
            if cached is not None:
                return copy.deepcopy(cached)
            path = self._path(umo)
            if not path.is_file():
                return None
            entry = self._read_entry(path)
            if entry is None:
                return None
            if entry.pop("umo") != umo:
                self._logger.warning(
                    "Nowledge Mem outbox 文件名与 umo 不一致：%s", path.name
                )
                return None
            self._cache[umo] = entry
            return copy.deepcopy(entry)

    def _set_sync(self, umo: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                os.chmod(self._dir, 0o700)
            except OSError:
                pass
            path = self._path(umo)
            _atomic_write(path, payload)
            self._cache[umo] = _allowed(payload)
            self._enforce_cap_sync(keep=path.name)

    def _clear_sync(self, umo: str) -> None:
        with self._lock:
            self._cache.pop(umo, None)
            try:
                self._path(umo).unlink()
            except FileNotFoundError:
                pass

    def _path(self, umo: str) -> Path:
        return self._dir / f"{_safe_name(umo)}.json"

    def _read_entry(self, path: Path) -> dict[str, Any] | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._logger.warning("Nowledge Mem outbox 跳过损坏文件：%s", path.name)
            return None
        if not isinstance(data, dict):
            self._logger.warning("Nowledge Mem outbox 跳过非对象文件：%s", path.name)
            return None
        umo = data.get("umo")
        if not isinstance(umo, str) or not umo:
            self._logger.warning("Nowledge Mem outbox 跳过缺少 umo 的文件：%s", path.name)
            return None
        entry = _allowed(data)
        entry["umo"] = umo
        return entry

    def _enforce_cap_sync(self, keep: str | None = None) -> None:
        try:
            entries = list(self._dir.glob("*.json"))
        except OSError:
            return
        if len(entries) <= self._max_entries:
            return
        try:
            entries.sort(key=lambda item: (item.stat().st_mtime, item.name))
        except OSError:
            return
        removed = 0
        for path in entries:
            if len(entries) - removed <= self._max_entries:
                break
            if keep is not None and path.name == keep:
                continue
            try:
                path.unlink()
            except OSError:
                continue
            removed += 1
            for umo, _ in list(self._cache.items()):
                if self._path(umo) == path:
                    self._cache.pop(umo, None)
        if removed:
            self._logger.warning(
                "Nowledge Mem outbox 超过 %s 个会话文件，淘汰最旧 %s 个",
                self._max_entries,
                removed,
            )
