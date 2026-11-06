"""Nowledge Mem HTTP 客户端。"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urlparse

import aiohttp

_IMPORT_TIMEOUT = 15.0

class NmemClient:
    """带超时与 Bearer 认证的异步客户端。"""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        *,
        timeout: float = 3.0,
        import_timeout: float = _IMPORT_TIMEOUT,
        session: aiohttp.ClientSession | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._api_url = (api_url or "").strip().rstrip("/")
        self._api_key = (api_key or "").strip()
        self._timeout = float(timeout)
        self._import_timeout = float(import_timeout)
        self._session = session
        self._owns_session = session is None
        self._logger = logger or logging.getLogger(__name__)
        parsed = urlparse(self._api_url)
        if parsed.scheme == "http" and parsed.hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            self._logger.warning(
                "Nowledge Mem 使用非回环明文 HTTP，Bearer 会明文传输；建议 HTTPS 或本机代理"
            )

    @property
    def configured(self) -> bool:
        return bool(self._api_url and self._api_key)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        if self._owns_session and self._session is not None and not self._session.closed:
            await self._session.close()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: float,
        **kwargs: Any,
    ) -> Any | None:
        if not self.configured:
            return None
        session = await self._get_session()
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
        }
        if "json" in kwargs:
            headers["Content-Type"] = "application/json"
        try:
            async with session.request(
                method,
                f"{self._api_url}{path}",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
                **kwargs,
            ) as response:
                if response.status >= 400:
                    self._logger.debug(
                        "Nowledge Mem %s %s 返回 HTTP %s",
                        method,
                        path,
                        response.status,
                    )
                    return None
                return await response.json(content_type=None)
        except (TimeoutError, aiohttp.ClientError, OSError, ValueError) as exc:
            self._logger.debug(
                "Nowledge Mem %s %s 请求失败：%s", method, path, type(exc).__name__
            )
            return None

    async def context_bundle(
        self,
        *,
        agent_id: str,
        source_app: str,
        space_id: str,
    ) -> dict[str, Any] | None:
        data = await self._request(
            "GET",
            "/context/bundle",
            timeout=self._timeout,
            params={
                "agent_id": agent_id,
                "source_app": source_app,
                "space_id": space_id,
                "include_working_memory": "true",
            },
        )
        return data if isinstance(data, dict) else None

    async def search(
        self,
        query: str,
        *,
        limit: int,
        space_id: str,
        mode: str = "fast",
    ) -> list[dict[str, Any]]:
        body: dict[str, Any] = {"query": query, "limit": limit, "mode": mode}
        if space_id:
            body["space_id"] = space_id
        data = await self._request(
            "POST",
            "/memories/search",
            timeout=self._timeout,
            json=body,
        )
        return data if isinstance(data, list) else []

    async def import_thread(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        data = await self._request(
            "POST",
            "/threads/import",
            timeout=self._import_timeout,
            json=payload,
        )
        return data if isinstance(data, dict) else None
