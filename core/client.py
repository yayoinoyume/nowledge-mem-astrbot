"""Nowledge Mem HTTP 客户端。"""

from __future__ import annotations

import json
import logging
import re
from typing import Any
from urllib.parse import quote, urlparse

import aiohttp

_IMPORT_TIMEOUT = 15.0
_WRITE_TIMEOUT_MIN = 5.0
_WRITE_TIMEOUT_MAX = 15.0
_ERROR_DETAIL_CHARS = 200


class NmemError(Exception):
    """Nowledge Mem 请求失败；status 是 HTTP 状态码（如果拿得到）。"""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _as_text(value: Any) -> str:
    return str(value or "").strip()


_DETAIL_KEYS = ("detail", "message", "error", "error_description")


def _redact(text: str) -> str:
    """去掉可能是凭据或内部路径的片段：长串 token、Bearer 值、绝对路径。"""
    cleaned = re.sub(r"(?i)bearer\s+[A-Za-z0-9._\-]+", "Bearer <redacted>", text)
    cleaned = re.sub(r"\bnmem_[A-Za-z0-9._\-]{8,}", "<redacted-key>", cleaned)
    cleaned = re.sub(r"\bsk-[A-Za-z0-9._\-]{8,}", "<redacted-key>", cleaned)
    cleaned = re.sub(r"(/[A-Za-z0-9._\-]+){3,}", "<redacted-path>", cleaned)
    return " ".join(cleaned.split())[:_ERROR_DETAIL_CHARS]


async def _response_detail(response: Any) -> str:
    """取错误响应里可安全展示的说明；非结构化响应一律不原文回显。

    只接受 JSON 对象里的 detail/message/error 字段并做脱敏，避免把服务端
    响应中的凭据、内部路径或调试信息带进模型上下文。
    """
    try:
        body = await response.text()
    except Exception:
        return ""
    if not body:
        return ""
    try:
        data = json.loads(body)
    except ValueError:
        return ""
    if not isinstance(data, dict):
        return ""
    for key in _DETAIL_KEYS:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return _redact(value)
    return ""


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return 0.0
    return 0.0


def _first_score(*sources: Any) -> float | None:
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in ("similarity_score", "relevance_score", "score", "confidence"):
            if key in source:
                return _as_float(source.get(key))
    return None


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        raw = [value]
    elif isinstance(value, (list, tuple)):
        raw = value
    else:
        return []
    labels: list[str] = []
    for item in raw:
        text = _as_text(item)
        if text and text not in labels:
            labels.append(text)
    return labels


def _thread_ref(value: Any) -> str:
    if isinstance(value, dict):
        return _as_text(value.get("id") or value.get("thread_id"))
    return _as_text(value)


def _source_thread(memory: dict[str, Any], entry: dict[str, Any]) -> str:
    metadata = memory.get("metadata") if isinstance(memory.get("metadata"), dict) else {}
    candidates = (
        memory.get("source_thread"),
        memory.get("source_thread_id"),
        entry.get("source_thread"),
        entry.get("source_thread_id"),
        metadata.get("source_thread_id"),
        metadata.get("source_thread"),
    )
    for candidate in candidates:
        thread_id = _thread_ref(candidate)
        if thread_id:
            return thread_id
    return ""


def normalize_search_response(data: Any) -> list[dict[str, Any]]:
    """把 GET 的 {memories:[...]} 与 POST 的裸数组统一成内部结构。"""
    if isinstance(data, dict):
        if "memories" in data:
            raw = data.get("memories")
        elif "results" in data:
            raw = data.get("results")
        else:
            raise NmemError("记忆搜索响应缺少 memories 列表")
        if raw is None:
            raw = []
        if not isinstance(raw, list):
            raise NmemError("记忆搜索响应 memories 不是列表")
    elif isinstance(data, list):
        raw = data
    else:
        raise NmemError("记忆搜索响应格式错误")

    normalized: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        memory = entry.get("memory") if isinstance(entry.get("memory"), dict) else entry
        item_id = _as_text(memory.get("id") or entry.get("id"))
        title = _as_text(memory.get("title") or entry.get("title"))
        content = _as_text(memory.get("content") or entry.get("content"))
        if not item_id and not title and not content:
            continue
        item: dict[str, Any] = {
            "id": item_id,
            "title": title,
            "content": content,
            "unit_type": _as_text(
                memory.get("unit_type") or entry.get("unit_type")
            )
            or "memory",
            "importance": _as_float(
                memory.get("importance")
                if "importance" in memory
                else entry.get("importance")
            ),
            "labels": _string_list(
                memory.get("labels")
                or memory.get("label_ids")
                or entry.get("labels")
                or entry.get("label_ids")
            ),
            "source_thread": _source_thread(memory, entry),
        }
        score = _first_score(entry, memory)
        if score is not None:
            item["score"] = score
        normalized.append(item)
    if raw and not normalized:
        raise NmemError("记忆搜索结果无法解析")
    return normalized


def normalize_threads_response(data: Any) -> dict[str, Any]:
    """把 /threads/search 的裸数组、{threads:[]}、{results:[]} 统一成 {threads:[]}。

    其他形状（含 {"error": ...}）一律抛错：调用方不能把错形状当成空结果。
    """
    if isinstance(data, list):
        return {"threads": data}
    if not isinstance(data, dict):
        raise NmemError("thread 搜索响应格式错误")
    for key in ("threads", "results"):
        if key not in data:
            continue
        raw = data.get(key)
        if raw is None:
            # 服务端无结果时可能返回 {"threads": null}：视为空列表，不当作错误。
            return {**data, "threads": []}
        if not isinstance(raw, list):
            raise NmemError(f"thread 搜索响应 {key} 不是列表")
        return {**data, "threads": raw}
    raise NmemError("thread 搜索响应缺少 threads 列表")


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

    @property
    def api_url(self) -> str:
        return self._api_url

    @property
    def _write_timeout(self) -> float:
        """写路径可能触发 embedding/去重：在读取超时上放大，但封顶 15 秒。"""
        return min(
            max(self._timeout * 2, _WRITE_TIMEOUT_MIN), _WRITE_TIMEOUT_MAX
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        if self._owns_session and self._session is not None and not self._session.closed:
            await self._session.close()

    async def _request_strict(
        self,
        method: str,
        path: str,
        *,
        timeout: float,
        **kwargs: Any,
    ) -> Any:
        """发请求；任何失败都抛 NmemError，调用方无法把失败当成空结果。"""
        if not self.configured:
            raise NmemError("未配置 api_url 或 api_key")
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
                    detail = await _response_detail(response)
                    message = f"HTTP {response.status}"
                    if detail:
                        message = f"{message}: {detail}"
                    raise NmemError(message, status=response.status)
                try:
                    return await response.json(content_type=None)
                except ValueError as exc:
                    raise NmemError("响应不是合法 JSON") from exc
        except NmemError:
            raise
        except (TimeoutError, aiohttp.ClientError, OSError, ValueError) as exc:
            raise NmemError(f"请求失败：{type(exc).__name__}") from exc

    async def _request(
        self,
        method: str,
        path: str,
        *,
        timeout: float,
        **kwargs: Any,
    ) -> Any | None:
        """旧接口：失败返回 None，供导入等“失败可重试”的路径使用。"""
        try:
            return await self._request_strict(method, path, timeout=timeout, **kwargs)
        except NmemError as exc:
            self._logger.debug(
                "Nowledge Mem %s %s 请求失败：%s", method, path, exc
            )
            return None

    async def context_bundle(
        self,
        *,
        agent_id: str,
        source_app: str,
        space_id: str,
    ) -> dict[str, Any]:
        data = await self._request_strict(
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
        if not isinstance(data, dict):
            raise NmemError("Context Bundle 响应不是对象")
        return data

    async def search(
        self,
        query: str,
        *,
        limit: int,
        space_id: str,
        mode: str = "normal",
        recorded_date_from: str = "",
        recorded_date_to: str = "",
        timeout: float | None = None,
    ) -> list[dict[str, Any]]:
        body: dict[str, Any] = {"query": query, "limit": limit, "mode": mode}
        if space_id:
            body["space_id"] = space_id
        for key, value in (
            ("recorded_date_from", recorded_date_from),
            ("recorded_date_to", recorded_date_to),
        ):
            if value:
                body[key] = value
        data = await self._request_strict(
            "POST",
            "/memories/search",
            timeout=self._timeout if timeout is None else timeout,
            json=body,
        )
        return normalize_search_response(data)

    async def search_threads(
        self,
        query: str,
        *,
        limit: int,
        space_id: str = "",
        timeout: float | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"query": query, "limit": limit}
        if space_id:
            params["space_id"] = space_id
        data = await self._request_strict(
            "GET",
            "/threads/search",
            timeout=self._timeout if timeout is None else timeout,
            params=params,
        )
        return normalize_threads_response(data)

    async def get_memory(
        self,
        memory_id: str,
        *,
        space_id: str = "",
        timeout: float | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if space_id:
            params["space_id"] = space_id
        data = await self._request_strict(
            "GET",
            f"/memories/{quote(memory_id, safe='')}",
            timeout=self._timeout if timeout is None else timeout,
            params=params or None,
        )
        if not isinstance(data, dict):
            raise NmemError("记忆详情响应不是对象")
        return data

    async def get_thread(
        self,
        thread_id: str,
        *,
        offset: int = 0,
        limit: int = 20,
        space_id: str = "",
        timeout: float | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"offset": offset, "limit": limit}
        if space_id:
            params["space_id"] = space_id
        data = await self._request_strict(
            "GET",
            f"/threads/{quote(thread_id, safe='')}",
            timeout=self._timeout if timeout is None else timeout,
            params=params,
        )
        if not isinstance(data, dict):
            raise NmemError("thread 详情响应不是对象")
        messages = data.get("messages")
        if not isinstance(messages, list):
            raise NmemError("thread 详情响应缺少 messages 列表")
        # related_memories / entities 不随分页变化且体积很大，直接丢弃。
        thread = data.get("thread")
        total = data.get("total_messages")
        return {
            "thread": thread if isinstance(thread, dict) else {},
            "messages": messages,
            "total_messages": total if isinstance(total, int) else len(messages),
        }

    async def expand_graph(
        self,
        node_id: str,
        *,
        depth: int = 1,
        limit: int = 50,
        space_id: str = "",
        timeout: float | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "depth": depth,
            "limit": limit,
            "include_metadata": "true",
        }
        if space_id:
            params["space_id"] = space_id
        data = await self._request_strict(
            "GET",
            f"/graph/expand/{quote(node_id, safe='')}",
            timeout=self._timeout if timeout is None else timeout,
            params=params,
        )
        if not isinstance(data, dict):
            raise NmemError("图扩展响应不是对象")
        return data

    async def evolves(
        self,
        memory_id: str,
        *,
        limit: int = 20,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """GET /agent/evolves。

        注意：该端点没有 space_id，实测会跨空间返回；插件工具不暴露它，
        只在需要版本链时由调用方显式使用并自行标注风险。
        """
        data = await self._request_strict(
            "GET",
            "/agent/evolves",
            timeout=self._timeout if timeout is None else timeout,
            params={"memory_id": memory_id, "limit": limit},
        )
        if not isinstance(data, dict):
            raise NmemError("EVOLVES 响应不是对象")
        return data

    async def working_memory(
        self,
        *,
        space_id: str = "",
        timeout: float | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if space_id:
            params["space_id"] = space_id
        data = await self._request_strict(
            "GET",
            "/agent/working-memory",
            timeout=self._timeout if timeout is None else timeout,
            params=params or None,
        )
        if not isinstance(data, dict):
            raise NmemError("Working Memory 响应不是对象")
        # 服务端会带绝对路径；插件不展示 WM 正文，也不保留这个字段。
        data.pop("file_path", None)
        return data

    async def save_memory(
        self,
        content: str,
        *,
        title: str,
        unit_type: str,
        importance: float,
        labels: list[str],
        space_id: str,
        agent_id: str,
        source_app: str,
        source_umo: str = "",
        sender_hash: str = "",
    ) -> dict[str, Any]:
        # 服务端 MemoryCreateRequest 没有 source_umo / sender_hash / import_origin
        # 这三个顶层字段：传了会 200 但被静默忽略（2026-11-07 实测，GET 查不到）。
        # metadata 是自由对象，会原样落库并在 GET /memories/{id} 返回，所以溯源放这里。
        metadata: dict[str, Any] = {"import_origin": source_app}
        if source_umo:
            metadata["source_umo"] = source_umo
        if sender_hash:
            metadata["sender_hash"] = sender_hash
        body: dict[str, Any] = {
            "content": content,
            "importance": importance,
            "unit_type": unit_type,
            "source": source_app,
            "source_app": source_app,
            "metadata": metadata,
        }
        if title:
            body["title"] = title
        if labels:
            body["labels"] = labels
        if space_id:
            body["space_id"] = space_id
        if agent_id:
            body["agent_id"] = agent_id
        data = await self._request_strict(
            "POST",
            "/memories",
            timeout=self._write_timeout,
            json=body,
        )
        if not isinstance(data, dict):
            raise NmemError("保存记忆响应不是对象")
        return data

    async def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        title: str | None = None,
        unit_type: str | None = None,
        importance: float | None = None,
        labels: list[str] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """PATCH /memories/{id}：只发白名单字段。

        服务端不校验 body：未知字段会 200 但被静默丢弃，所以白名单只能由客户端维护。
        该端点没有任何 query 参数，也不需要 space_id。响应里的 labels 会变成
        label_ids（存的是 id 不是名字），调用方不要拿返回值回写标签。
        """
        body: dict[str, Any] = {}
        for key, value in (
            ("content", content),
            ("title", title),
            ("unit_type", unit_type),
            ("importance", importance),
            ("labels", labels),
        ):
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                # 空字符串视为“不改”，避免误清空服务端已有字段。
                continue
            body[key] = value
        data = await self._request_strict(
            "PATCH",
            f"/memories/{quote(memory_id, safe='')}",
            timeout=self._write_timeout if timeout is None else timeout,
            json=body,
        )
        if not isinstance(data, dict):
            raise NmemError("更新记忆响应不是对象")
        return data

    async def health_details(self) -> dict[str, Any]:
        data = await self._request_strict(
            "GET", "/health/details", timeout=min(self._timeout, 3.0)
        )
        if not isinstance(data, dict):
            raise NmemError("健康检查响应不是对象")
        return data

    async def has_pending_reviews(self) -> bool | None:
        data = await self._request_strict(
            "GET",
            "/agent/feed/events",
            timeout=min(self._timeout, 3.0),
            params={
                "review_only": "true",
                "unresolved_only": "true",
                "last_n_days": 365,
                "limit": 1,
                "include_total": "false",
            },
        )
        if isinstance(data, list):
            return bool(data)
        if isinstance(data, dict):
            events = data.get("events")
            if isinstance(events, list):
                return bool(events)
            total = data.get("total")
            if isinstance(total, int):
                return total > 0
            pending = data.get("pending_reviews")
            if isinstance(pending, int):
                return pending > 0
        return None

    async def health(self) -> dict[str, Any]:
        data = await self._request_strict(
            "GET", "/health", timeout=min(self._timeout, 3.0)
        )
        if isinstance(data, dict):
            return data
        return {"status": str(data)}

    async def import_thread(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        data = await self._request(
            "POST",
            "/threads/import",
            timeout=self._import_timeout,
            json=payload,
        )
        return data if isinstance(data, dict) else None
