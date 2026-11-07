"""Agent 可调用的 Nowledge Mem 工具实现。"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .client import NmemError

MAX_SAVE_CHARS = 8000
MAX_GET_CHARS = 8000
MAX_PREVIEW_CHARS = 300
MAX_TITLE_CHARS = 200
MAX_LABEL_CHARS = 64
MAX_LABELS = 20
MAX_SEARCH_TIMEOUT_SECONDS = 15.0
MAX_THREAD_LIMIT = 100
MAX_THREAD_CONTENT_CHARS = 500
MAX_GRAPH_DEPTH = 5
GRAPH_NEIGHBOR_LIMIT = 50
MAX_GRAPH_LINES = 50
MAX_SKILL_CHARS = 20_000
SKILL_NAMES = (
    "read-working-memory",
    "search-memory",
    "distill-memory",
    "status",
    "save-thread",
    "check-integration",
    "explore-graph",
)
SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"
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


def _importance(value: Any) -> float | None:
    """显式 importance 必须是 0-1 的有限数字；非法返回 None，不静默改默认值。"""
    if isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        return None
    return score


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


def _limit(value: Any, default: int, low: int, high: int) -> int:
    """数值参数越界时夹到区间内；非法值回落默认值。"""
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(low, min(high, number))


def _normalize_date(value: Any) -> tuple[str, str]:
    """接受 YYYY-MM-DD 或 ISO 日期时间；返回 (YYYY-MM-DD, 错误说明)。"""
    text = str(value or "").strip()
    if not text:
        return "", ""
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return "", "必须是 YYYY-MM-DD 或 ISO 日期"
    return parsed.date().isoformat(), ""


def _one_line(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _strip_frontmatter(text: str) -> str:
    if not text.startswith("---"):
        return text
    parts = text.split("---", 2)
    if len(parts) == 3:
        return parts[2].lstrip("\n")
    return text


def _strip_file_paths(value: Any) -> Any:
    """递归剔除服务端绝对路径字段，避免 file_path 泄漏。"""
    if isinstance(value, dict):
        return {
            key: _strip_file_paths(item)
            for key, item in value.items()
            if key != "file_path"
        }
    if isinstance(value, list):
        return [_strip_file_paths(item) for item in value]
    return value


def _format_memory_detail(data: dict[str, Any]) -> str:
    memory_id = str(data.get("id") or "?").strip()
    title = str(data.get("title") or "").strip() or "(无标题)"
    unit_type = str(data.get("unit_type") or "memory").strip()
    importance = _number(data.get("importance"))
    raw_labels = data.get("labels")
    if not isinstance(raw_labels, list):
        raw_labels = data.get("label_ids")
    if not isinstance(raw_labels, list):
        raw_labels = []
    labels = [str(item).strip() for item in raw_labels if str(item).strip()]
    source_thread = (
        _thread_id(data.get("source_thread"))
        or _thread_id(data.get("source_thread_id"))
    )
    metadata = _strip_file_paths(data.get("metadata"))
    metadata_text = (
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str)
        if isinstance(metadata, dict) and metadata
        else "{}"
    )
    content = str(data.get("content") or "")
    lines = [
        f"memory: id={memory_id}",
        f"title: {title}",
        f"unit_type: {unit_type}",
        f"importance: {importance:.2f}",
        f"labels: {', '.join(labels) if labels else '(none)'}",
        f"metadata: {metadata_text}",
    ]
    if source_thread:
        lines.append(f"source_thread: {source_thread}")
    lines.append("content:")
    lines.append(content if content else "(空)")
    return "\n".join(lines)


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


def _update_unconfirmed(data: dict[str, Any], name: str, expected: Any) -> bool:
    """PATCH 回读校验：服务端没回显或回显值不一致都算未确认。

    只覆盖可按值核对的字段；labels 服务端只回 label_ids，由 update 单独处理。
    """
    if name == "importance":
        return abs(_number(data.get("importance"), -1.0) - float(expected)) > 1e-6
    returned = data.get(name)
    if returned is None:
        return True
    return str(returned).strip() != str(expected).strip()


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
    """八个 llm_tool 的纯逻辑实现；所有失败都转成可读字符串。"""

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
        recorded_date_from: str = "",
        recorded_date_to: str = "",
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
            dates: dict[str, str] = {}
            for name, value in (
                ("recorded_date_from", recorded_date_from),
                ("recorded_date_to", recorded_date_to),
            ):
                normalized, date_error = _normalize_date(value)
                if date_error:
                    return f"error: {name} {date_error}"
                dates[name] = normalized
            if (
                dates["recorded_date_from"]
                and dates["recorded_date_to"]
                and dates["recorded_date_from"] > dates["recorded_date_to"]
            ):
                return "error: recorded_date_from 不能晚于 recorded_date_to"
            collected: list[tuple[str, dict[str, Any]]] = []
            if scope_text in ("memories", "both"):
                memories = await self._client.search(
                    text,
                    limit=count,
                    space_id=self._space_id,
                    mode=mode_text,
                    recorded_date_from=dates["recorded_date_from"],
                    recorded_date_to=dates["recorded_date_to"],
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

    async def update(
        self,
        memory_id: str,
        *,
        content: str = "",
        title: str = "",
        unit_type: str = "",
        importance: float | None = None,
        labels: list[str] | None = None,
        umo: str = "",
        sender_id: str = "",
        is_admin: bool = False,
    ) -> str:
        # umo / sender_id 只为与 save 保持签名一致：PATCH 端点没有溯源字段。
        del umo, sender_id
        try:
            if self._save_admin_only and not is_admin:
                return "error: 仅管理员可更新记忆"
            target = str(memory_id or "").strip()
            if not target:
                return "error: memory_id 不能为空"
            body: dict[str, Any] = {}
            truncated = False
            text = str(content or "").strip()
            if text:
                truncated = len(text) > MAX_SAVE_CHARS
                body["content"] = text[:MAX_SAVE_CHARS]
            heading = str(title or "").strip()
            if heading:
                truncated = truncated or len(heading) > MAX_TITLE_CHARS
                body["title"] = heading[:MAX_TITLE_CHARS]
            unit = str(unit_type or "").strip().lower()
            if unit:
                if unit not in VALID_UNIT_TYPES:
                    return (
                        "error: unit_type 必须是 fact、preference、decision、plan、"
                        "procedure、learning、context 或 event"
                    )
                body["unit_type"] = unit
            if importance is not None:
                score = _importance(importance)
                if score is None:
                    return "error: importance 必须是 0 到 1 之间的数字"
                body["importance"] = score
            normalized_labels = _labels(labels)
            if normalized_labels:
                body["labels"] = normalized_labels
            if not body:
                return (
                    "error: 至少提供 content、title、unit_type、importance 或 "
                    "labels 之一"
                )
            data = await self._client.update_memory(
                target,
                content=body.get("content"),
                title=body.get("title"),
                unit_type=body.get("unit_type"),
                importance=body.get("importance"),
                labels=body.get("labels"),
            )
            resolved_id = str(data.get("id") or target).strip()
            lines = [f"updated: id={resolved_id} fields={','.join(body)}"]
            for name, expected in body.items():
                if name == "labels":
                    continue
                if _update_unconfirmed(data, name, expected):
                    lines.append(f"warn: 服务端未确认字段 {name}")
            if "labels" in body:
                # 服务端只回 label_ids 且当前忽略 labels；提示一次，不算未确认。
                lines.append("note: 服务端当前忽略 labels 字段，本次标签变更未生效")
            if truncated:
                lines.append("note: 内容或标题超长已截断")
            return "\n".join(lines)
        except NmemError as exc:
            return f"error: Nowledge Mem 更新失败：{exc}"
        except Exception as exc:
            self._logger.warning("nowledge_mem_update 失败：%s", type(exc).__name__)
            return f"error: 更新失败：{type(exc).__name__}: {exc}"

    async def get_memory(self, memory_id: str) -> str:
        try:
            text = str(memory_id or "").strip()
            if not text:
                return "error: memory_id 不能为空"
            data = await self._client.get_memory(text, space_id=self._space_id)
            detail = _format_memory_detail(data)
            if len(detail) > MAX_GET_CHARS:
                return (
                    detail[:MAX_GET_CHARS]
                    + f"\nnote: 输出超过 {MAX_GET_CHARS} 字符，已截断"
                )
            return detail
        except NmemError as exc:
            return f"error: Nowledge Mem 读取记忆失败：{exc}"
        except Exception as exc:
            self._logger.warning("nowledge_mem_get 失败：%s", type(exc).__name__)
            return f"error: 读取记忆失败：{type(exc).__name__}: {exc}"

    async def thread_show(
        self,
        thread_id: str,
        *,
        offset: int = 0,
        limit: int = 20,
    ) -> str:
        try:
            text = str(thread_id or "").strip()
            if not text:
                return "error: thread_id 不能为空"
            start = _limit(offset, 0, 0, 1_000_000)
            count = _limit(limit, 20, 1, MAX_THREAD_LIMIT)
            data = await self._client.get_thread(
                text,
                offset=start,
                limit=count,
                space_id=self._space_id,
            )
            thread = data.get("thread")
            if not isinstance(thread, dict):
                thread = {}
            messages = data.get("messages")
            if not isinstance(messages, list):
                messages = []
            total = data.get("total_messages")
            if not isinstance(total, int):
                total = thread.get("message_count")
            if not isinstance(total, int):
                total = len(messages)
            resolved_id = (
                str(thread.get("thread_id") or thread.get("id") or text).strip()
            )
            title = str(thread.get("title") or "").strip() or "(无标题)"
            lines = [
                f"thread: id={resolved_id} title={title}",
                f"total_messages: {total}",
                f"range: offset={start} limit={count} returned={len(messages)}",
            ]
            if not messages:
                lines.append("messages: (empty)")
                return "\n".join(lines)
            for index, message in enumerate(messages, start=1):
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role") or "unknown").strip()
                content = _one_line(
                    message.get("content"), MAX_THREAD_CONTENT_CHARS
                ) or "(空)"
                lines.append(f"{index}. [{role}] {content}")
            return "\n".join(lines)
        except NmemError as exc:
            return f"error: Nowledge Mem 读取对话失败：{exc}"
        except Exception as exc:
            self._logger.warning("nowledge_mem_thread_show 失败：%s", type(exc).__name__)
            return f"error: 读取对话失败：{type(exc).__name__}: {exc}"

    async def graph(
        self,
        *,
        query: str = "",
        memory_id: str = "",
        depth: int = 1,
    ) -> str:
        try:
            text = str(query or "").strip()
            center = str(memory_id or "").strip()
            if bool(text) == bool(center):
                return "error: query 与 memory_id 必须且只能给一个"
            level = _limit(depth, 1, 1, MAX_GRAPH_DEPTH)
            matched = ""
            center_title = ""
            if text:
                results = await self._client.search(
                    text,
                    limit=1,
                    space_id=self._space_id,
                    mode="normal",
                )
                if not results:
                    return f'no matches: query="{text}"'
                first = results[0] if isinstance(results[0], dict) else {}
                center = str(first.get("id") or "").strip()
                if not center:
                    return "error: 搜索结果缺少 memory id"
                center_title = _preview(first.get("title"), MAX_TITLE_CHARS)
                matched = (
                    f"matched: id={center} title={center_title or '(无标题)'}"
                )
            data = await self._client.expand_graph(
                center,
                depth=level,
                limit=GRAPH_NEIGHBOR_LIMIT,
                space_id=self._space_id,
            )
            metadata = data.get("metadata")
            if isinstance(metadata, dict) and metadata.get("error"):
                return f"error: 图扩展失败：{metadata['error']}"
            neighbors = data.get("neighbors")
            if not isinstance(neighbors, list):
                neighbors = []
            edges = data.get("edges")
            if not isinstance(edges, list):
                edges = []
            lines = [
                f"graph: center={center} depth={level} "
                f"neighbors={len(neighbors)} edges={len(edges)}"
            ]
            if matched:
                lines.insert(0, matched)
            if not neighbors and not edges:
                lines.append("no relations: 无邻居或节点不存在")
                return "\n".join(lines)
            if neighbors:
                lines.append("neighbors:")
                for item in neighbors[:MAX_GRAPH_LINES]:
                    if not isinstance(item, dict):
                        continue
                    node_id = str(item.get("id") or "?").strip()
                    label = _preview(
                        item.get("label") or item.get("title"), MAX_TITLE_CHARS
                    ) or "(无标签)"
                    node_type = str(item.get("node_type") or "node").strip()
                    lines.append(f"- [{node_type}] {label} (id={node_id})")
            if edges:
                lines.append("edges:")
                for item in edges[:MAX_GRAPH_LINES]:
                    if not isinstance(item, dict):
                        continue
                    source = str(item.get("source") or "?").strip()
                    target = str(item.get("target") or "?").strip()
                    edge_type = str(item.get("edge_type") or "related").strip()
                    label = _preview(item.get("label"), MAX_TITLE_CHARS)
                    line = f"- {source} -> {target} [{edge_type}]"
                    if label:
                        line += f" {label}"
                    lines.append(line)
            return "\n".join(lines)
        except NmemError as exc:
            return f"error: Nowledge Mem 图探索失败：{exc}"
        except Exception as exc:
            self._logger.warning("nowledge_mem_graph 失败：%s", type(exc).__name__)
            return f"error: 图探索失败：{type(exc).__name__}: {exc}"

    async def skill(self, name: str) -> str:
        try:
            key = str(name or "").strip().lower()
            available = "、".join(SKILL_NAMES)
            if not key:
                return f"error: name 不能为空；可用技能：{available}"
            if key not in SKILL_NAMES:
                return f"error: 未知技能 {key}；可用技能：{available}"
            path = SKILLS_DIR / key / "SKILL.md"
            if not path.is_file():
                return f"error: 技能文件不存在：{key}"
            body = _strip_frontmatter(path.read_text(encoding="utf-8")).strip()
            if not body:
                return f"error: 技能正文为空：{key}"
            if len(body) > MAX_SKILL_CHARS:
                body = body[:MAX_SKILL_CHARS] + "\n…"
            return f"skill: {key}\n\n{body}"
        except OSError as exc:
            self._logger.warning("nowledge_mem_skill 读取失败：%s", type(exc).__name__)
            return f"error: 读取技能失败：{type(exc).__name__}"
        except Exception as exc:
            self._logger.warning("nowledge_mem_skill 失败：%s", type(exc).__name__)
            return f"error: 读取技能失败：{type(exc).__name__}: {exc}"

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
                                try:
                                    await self._client.working_memory(
                                        space_id=self._space_id
                                    )
                                    connected = True
                                    backend = "/agent/working-memory"
                                except NmemError as wm_error:
                                    error = str(wm_error)
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
