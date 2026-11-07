"""Agent 可调用的 Nowledge Mem 工具实现。"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any
from urllib.parse import urlparse

from .client import NmemError

MAX_SAVE_CHARS = 8000
MAX_PREVIEW_CHARS = 300
MAX_TITLE_CHARS = 200
MAX_LABEL_CHARS = 64
MAX_LABELS = 20
MAX_SEARCH_TIMEOUT_SECONDS = 15.0
VALID_SCOPES = ("memories", "threads", "both")
VALID_MODES = ("normal", "deep")
VALID_UNIT_TYPES = (
    "fact",
    "preference",
    "decision",
    "plan",
    "procedure",
    "learning",
    "context",
    "event",
)


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _preview(value: Any, limit: int = MAX_PREVIEW_CHARS) -> str:
    text = _text(value)
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _clip(value: Any, limit: int) -> str:
    """出站字段硬截断：服务端有长度上限，超出的部分直接丢弃。"""
    return _text(value)[:limit]


def _sender_hash(sender_id: Any) -> str:
    """共享 space 里只留哈希，不存明文发送者 ID。"""
    text = str(sender_id or "").strip()
    if not text:
        return ""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _safe_url(value: Any) -> str:
    """只保留 scheme://host[:port]，丢掉 userinfo、路径与 query。"""
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = urlparse(text)
        port = parsed.port
    except ValueError:
        return "(无法解析)"
    if not parsed.scheme or not parsed.hostname:
        return "(无法解析)"
    host = parsed.hostname
    if ":" in host:
        host = f"[{host}]"
    return f"{parsed.scheme}://{host}" + (f":{port}" if port else "")


def _search_timeout(mode: str, base: float) -> float:
    """deep 更慢：最多给 base 的两倍，且不超过 15 秒。"""
    if mode == "deep":
        return min(MAX_SEARCH_TIMEOUT_SECONDS, base * 2)
    return base


def _number(value: Any, default: float = 0.0) -> float:
    """非有限值（nan/inf）回落默认值，避免把非法数字发出去。"""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return default
    else:
        return default
    return number if math.isfinite(number) else default


def _score(item: dict[str, Any], memory: dict[str, Any] | None = None) -> float:
    for source in (item, memory or {}):
        for key in ("similarity_score", "relevance_score", "score", "confidence"):
            if key in source:
                return _number(source.get(key))
    return 0.0


def _thread_id(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("id") or value.get("thread_id") or "").strip()
    if isinstance(value, str):
        return value.strip()
    return ""


def _clamp_limit(value: Any, default: int = 10) -> int:
    """inf/nan 与非法字符串都回落默认值；int(inf) 抛 OverflowError。"""
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(1, min(20, number))


def _labels(value: Any) -> list[str]:
    if isinstance(value, str):
        raw = [value]
    elif isinstance(value, list):
        raw = value
    else:
        raw = []
    labels: list[str] = []
    for item in raw:
        label = _clip(item, MAX_LABEL_CHARS)
        if label and label not in labels:
            labels.append(label)
        if len(labels) >= MAX_LABELS:
            break
    return labels


def _dedupe(items: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, dict[str, Any]]]:
    merged: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    for kind, item in items:
        if not isinstance(item, dict):
            continue
        if kind == "memory":
            memory = item.get("memory") if isinstance(item.get("memory"), dict) else item
            item_id = str(memory.get("id") or item.get("id") or "").strip()
        else:
            item_id = _thread_id(item)
        key = (kind, item_id or f"{kind}:{len(merged)}")
        if key not in merged:
            merged[key] = (kind, item)
    return list(merged.values())


def _format_memory(index: int, item: dict[str, Any]) -> list[str]:
    memory = item.get("memory") if isinstance(item.get("memory"), dict) else item
    memory_id = str(memory.get("id") or item.get("id") or "?").strip()
    title = _preview(
        memory.get("title") or item.get("title"), MAX_TITLE_CHARS
    ) or "(无标题)"
    content = _preview(memory.get("content") or item.get("content"))
    score = _score(item, memory)
    source_thread = (
        _thread_id(memory.get("source_thread"))
        or _thread_id(memory.get("source_thread_id"))
        or _thread_id(item.get("source_thread"))
    )
    lines = [
        f"{index}. [memory] id={memory_id} score={score:.2f}",
        f"title: {title}",
    ]
    if content:
        lines.append(f"content: {content}")
    if source_thread:
        lines.append(f"source_thread: {source_thread}")
    return lines


def _format_thread(index: int, item: dict[str, Any]) -> list[str]:
    thread_id = _thread_id(item) or "?"
    title = _preview(item.get("title"), MAX_TITLE_CHARS) or "(无标题)"
    score = _score(item)
    snippets: list[str] = []
    matched = item.get("matched_messages")
    if isinstance(matched, list):
        for message in matched:
            if not isinstance(message, dict):
                continue
            snippet = _preview(message.get("snippet"))
            if snippet:
                snippets.append(snippet)
            if len(snippets) >= 2:
                break
    content = " / ".join(snippets) or _preview(
        item.get("content") or item.get("summary") or item.get("snippet")
    )
    source = _preview(item.get("source"), MAX_LABEL_CHARS)
    lines = [
        f"{index}. [thread] id={thread_id} score={score:.2f}",
        f"title: {title}",
    ]
    if content:
        lines.append(f"content: {content}")
    lines.append(f"source_thread: {thread_id}")
    if source:
        lines.append(f"source: {source}")
    return lines


class NmemTools:
    """三个 llm_tool 的纯逻辑实现；所有失败都转成可读字符串。"""

    def __init__(
        self,
        client: Any,
        store: Any,
        *,
        space_id: str,
        agent_id: str,
        source_app: str,
        identity: str,
        plugin_version: str,
        save_admin_only: bool = False,
        timeout_seconds: float = 3.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._client = client
        self._store = store
        self._space_id = space_id
        self._agent_id = agent_id
        self._source_app = source_app
        self._identity = identity
        self._plugin_version = plugin_version
        self._save_admin_only = save_admin_only
        self._timeout_seconds = float(timeout_seconds)
        self._logger = logger or logging.getLogger(__name__)

    async def search(
        self,
        query: str,
        *,
        scope: str = "memories",
        limit: int = 10,
        mode: str = "normal",
    ) -> str:
        try:
            text = str(query or "").strip()
            if not text:
                return "error: query 不能为空"
            scope_text = str(scope or "memories").strip().lower()
            if scope_text not in VALID_SCOPES:
                return "error: scope 必须是 memories、threads 或 both"
            mode_text = str(mode or "normal").strip().lower()
            if mode_text not in VALID_MODES:
                return "error: mode 必须是 normal 或 deep"
            count = _clamp_limit(limit)
            timeout = _search_timeout(mode_text, self._timeout_seconds)
            collected: list[tuple[str, dict[str, Any]]] = []
            if scope_text in ("memories", "both"):
                memories = await self._client.search(
                    text,
                    limit=count,
                    space_id=self._space_id,
                    mode=mode_text,
                    timeout=timeout,
                )
                collected.extend(("memory", item) for item in memories)
            if scope_text in ("threads", "both"):
                data = await self._client.search_threads(
                    text,
                    limit=count,
                    space_id=self._space_id,
                    timeout=timeout,
                )
                threads = data.get("threads") if isinstance(data, dict) else None
                if not isinstance(threads, list):
                    return "error: thread 搜索响应缺少 threads 列表"
                collected.extend(("thread", item) for item in threads)
            merged = _dedupe(collected)
            merged.sort(key=lambda pair: _score(pair[1]), reverse=True)
            merged = merged[:count]
            if not merged:
                return f'no matches: query="{text}" scope={scope_text}'
            lines = [
                f'matches: {len(merged)} (query="{text}", scope={scope_text})'
            ]
            for index, (kind, item) in enumerate(merged, start=1):
                if kind == "memory":
                    lines.extend(_format_memory(index, item))
                else:
                    lines.extend(_format_thread(index, item))
            return "\n".join(lines)
        except NmemError as exc:
            return f"error: Nowledge Mem 搜索失败：{exc}"
        except Exception as exc:
            self._logger.warning("nowledge_mem_search 失败：%s", type(exc).__name__)
            return f"error: 搜索失败：{type(exc).__name__}: {exc}"

    async def save(
        self,
        content: str,
        *,
        title: str = "",
        unit_type: str = "fact",
        importance: float = 0.6,
        labels: list[str] | None = None,
        umo: str = "",
        sender_id: str = "",
        is_admin: bool = False,
    ) -> str:
        try:
            if self._save_admin_only and not is_admin:
                return "error: 仅管理员可保存"
            text = str(content or "").strip()
            if not text:
                return "error: content 不能为空"
            truncated = len(text) > MAX_SAVE_CHARS
            if truncated:
                text = text[:MAX_SAVE_CHARS]
            unit = str(unit_type or "fact").strip()
            if unit not in VALID_UNIT_TYPES:
                return (
                    "error: unit_type 必须是 fact、preference、decision、plan、"
                    "procedure、learning、context 或 event"
                )
            score = _number(importance, 0.6)
            score = max(0.0, min(1.0, score))
            result = await self._client.save_memory(
                text,
                title=_clip(title, MAX_TITLE_CHARS),
                unit_type=unit,
                importance=score,
                labels=_labels(labels),
                space_id=self._space_id,
                agent_id=self._agent_id,
                source_app=self._source_app,
                source_umo=str(umo or "").strip(),
                sender_hash=_sender_hash(sender_id),
            )
            memory = result.get("memory") if isinstance(result, dict) else None
            if not isinstance(memory, dict):
                memory = {}
            memory_id = str(
                result.get("id")
                or result.get("memory_id")
                or memory.get("id")
                or ""
            ).strip()
            if not memory_id:
                return "error: Nowledge Mem 保存响应缺少 memory id"
            saved_title = _preview(
                result.get("title") or memory.get("title") or title, MAX_TITLE_CHARS
            ) or "(无标题)"
            note = (
                f"\nnote: 内容超过 {MAX_SAVE_CHARS} 字符，已截断后保存"
                if truncated
                else ""
            )
            return (
                f"saved: id={memory_id} title={saved_title} unit_type={unit} "
                f"importance={score:.2f}{note}"
            )
        except NmemError as exc:
            return f"error: Nowledge Mem 保存失败：{exc}"
        except Exception as exc:
            self._logger.warning("nowledge_mem_save 失败：%s", type(exc).__name__)
            return f"error: 保存失败：{type(exc).__name__}: {exc}"

    async def status(self, umo: str = "") -> str:
        connected = False
        backend = ""
        error = ""
        details: dict[str, Any] | None = None
        if not self._client.configured:
            error = "未配置 api_url 或 api_key"
        else:
            try:
                details = await self._client.health_details()
                connected = True
            except NmemError as exc:
                if exc.status == 404:
                    try:
                        details = await self._client.health()
                        connected = True
                    except NmemError as fallback:
                        if fallback.status == 404:
                            try:
                                await self._client.context_bundle(
                                    agent_id=self._agent_id,
                                    source_app=self._source_app,
                                    space_id=self._space_id,
                                )
                                connected = True
                                backend = "/context/bundle"
                            except NmemError as bundle_error:
                                error = str(bundle_error)
                        else:
                            error = str(fallback)
                else:
                    error = str(exc)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        if isinstance(details, dict):
            backend = backend or _text(
                details.get("status")
                or details.get("version")
                or details.get("service")
            )
        server_version = ""
        cli_version = ""
        attention_pending: bool | None = None
        if isinstance(details, dict):
            server_version = _text(
                details.get("server_version") or details.get("version")
            )
            cli_version = _text(details.get("cli_version"))
            attention = details.get("attention")
            if isinstance(attention, dict):
                value = attention.get("has_pending_reviews")
                if isinstance(value, bool):
                    attention_pending = value
        if attention_pending is None and connected:
            try:
                attention_pending = await self._client.has_pending_reviews()
            except Exception as exc:
                self._logger.debug(
                    "nowledge_mem_status 待审阅探测失败：%s", type(exc).__name__
                )
        pending = 0
        last_sync = "never"
        try:
            if umo:
                pending_state = await self._store.get_pending(umo)
                pending = 1 if isinstance(pending_state, dict) else 0
                cursor = await self._store.get_cursor(umo)
                if isinstance(cursor, dict) and isinstance(cursor.get("synced_at"), str):
                    last_sync = cursor["synced_at"]
        except Exception as exc:
            self._logger.warning("nowledge_mem_status 读取本地状态失败：%s", type(exc).__name__)
        lines = [
            f"connected: {'true' if connected else 'false'}",
            f"api_url: {_safe_url(self._client.api_url) or '(未配置)'}",
            f"space: {self._space_id or '(未配置)'}",
            f"agent: {self._agent_id or '(未配置)'}",
            f"identity: {self._identity or '(未知)'}",
            f"api_key: {'configured' if self._client.configured else 'missing'}",
        ]
        if server_version:
            lines.append(f"server_version: {server_version}")
        lines.append(f"cli_version: {cli_version or 'n/a (插件无 CLI)'}")
        if attention_pending is not None:
            lines.append(
                "attention_has_pending_reviews: "
                + ("true" if attention_pending else "false")
                + " (非精确计数)"
            )
        lines.extend(
            [
                f"pending: {pending}" + (" (本会话)" if umo else ""),
                f"last_sync: {last_sync}",
                f"plugin_version: {self._plugin_version}",
            ]
        )
        if backend:
            lines.append(f"backend: {backend}")
        if error:
            lines.append(f"error: {error}")
        return "\n".join(lines)
