"""AstrBot 插件 KV 封装。"""

from __future__ import annotations

import logging
from typing import Any


class NmemStore:
    """Bundle/游标放 AstrBot 插件级 KV；pending 优先落盘 outbox。"""

    def __init__(
        self,
        kv: Any,
        identity: str,
        *,
        outbox: Any = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._kv = kv
        self._identity = identity
        self._outbox = outbox
        self._logger = logger or logging.getLogger(__name__)

    def _key(self, kind: str, umo: str) -> str:
        return f"{kind}:{self._identity}:{umo}"

    async def get_bundle(self, umo: str) -> dict[str, Any] | None:
        value = await self._kv.get_kv_data(self._key("bundle", umo), None)
        return value if isinstance(value, dict) else None

    async def set_bundle(self, umo: str, state: dict[str, Any]) -> None:
        await self._kv.put_kv_data(self._key("bundle", umo), state)

    async def get_cursor(self, umo: str) -> dict[str, Any] | None:
        value = await self._kv.get_kv_data(self._key("cursor", umo), None)
        return value if isinstance(value, dict) else None

    async def set_cursor(self, umo: str, state: dict[str, Any]) -> None:
        await self._kv.put_kv_data(self._key("cursor", umo), state)

    async def get_pending(self, umo: str) -> dict[str, Any] | None:
        if self._outbox is None:
            return await self._get_pending_kv(umo)
        try:
            pending = await self._outbox.get_pending(umo)
        except Exception as exc:  # noqa: BLE001 - 磁盘失败必须回退，不能丢 pending
            self._logger.warning("Nowledge Mem outbox 读取失败：%s", type(exc).__name__)
            return await self._get_pending_kv(umo)
        if isinstance(pending, dict):
            return pending
        # 旧版本把 pending 存在 KV；首次读到就迁到磁盘并删掉旧键。
        legacy = await self._get_pending_kv(umo)
        if not isinstance(legacy, dict):
            return None
        try:
            await self._outbox.set_pending(umo, legacy)
        except Exception as exc:  # noqa: BLE001 - 迁移失败仍返回旧值，重试语义不变
            self._logger.warning("Nowledge Mem outbox 迁移失败：%s", type(exc).__name__)
            return legacy
        await self._kv.delete_kv_data(self._key("pending", umo))
        return legacy

    async def _get_pending_kv(self, umo: str) -> dict[str, Any] | None:
        value = await self._kv.get_kv_data(self._key("pending", umo), None)
        return value if isinstance(value, dict) else None

    async def set_pending(self, umo: str, state: dict[str, Any]) -> None:
        if self._outbox is not None:
            try:
                await self._outbox.set_pending(umo, state)
            except Exception as exc:  # noqa: BLE001 - 落盘失败回退 KV，保留重试
                self._logger.warning("Nowledge Mem outbox 写入失败：%s", type(exc).__name__)
            else:
                # 迁移后清掉旧 KV 键，避免磁盘与 KV 各留一份。
                try:
                    await self._kv.delete_kv_data(self._key("pending", umo))
                except Exception as exc:  # noqa: BLE001 - 清理失败不影响已落盘的 pending
                    self._logger.debug(
                        "Nowledge Mem 旧 pending KV 清理失败：%s",
                        type(exc).__name__,
                    )
                return
        await self._kv.put_kv_data(self._key("pending", umo), state)

    async def clear_pending(self, umo: str) -> None:
        if self._outbox is not None:
            try:
                await self._outbox.clear_pending(umo)
            except Exception as exc:  # noqa: BLE001 - 清理失败也要继续删旧 KV
                self._logger.warning("Nowledge Mem outbox 清理失败：%s", type(exc).__name__)
        await self._kv.delete_kv_data(self._key("pending", umo))
