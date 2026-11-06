"""AstrBot 插件 KV 封装。"""

from __future__ import annotations

from typing import Any

class NmemStore:
    """所有状态都保存在 AstrBot 插件级 KV，不写本地文件。"""

    def __init__(self, kv: Any, identity: str) -> None:
        self._kv = kv
        self._identity = identity

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
        value = await self._kv.get_kv_data(self._key("pending", umo), None)
        return value if isinstance(value, dict) else None

    async def set_pending(self, umo: str, state: dict[str, Any]) -> None:
        await self._kv.put_kv_data(self._key("pending", umo), state)

    async def clear_pending(self, umo: str) -> None:
        await self._kv.delete_kv_data(self._key("pending", umo))
