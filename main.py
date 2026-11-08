"""Nowledge Mem AstrBot 插件入口。"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from pathlib import Path
from typing import Any

from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core.star.filter.command import GreedyStr
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

from .core.capture import CaptureCoordinator, should_capture_turn
from .core.client import NmemClient
from .core.delta import digest_api_key, identity_digest
from .core.inject import ContextInjector
from .core.outbox import OutboxStore
from .core.parts import make_temp_parts
from .core.store import NmemStore
from .core.tools import NmemTools, VALID_UNIT_TYPES

_DEFAULT_API_URL = "http://127.0.0.1:14242"
_PLUGIN_VERSION = "0.18.3"
MEM_SUM_ROUNDS_DEFAULT = 10
MAX_DISTILL_ITEMS = 8
MAX_SUMMARY_MESSAGE_CHARS = 1000
MAX_SUMMARY_CHARS = 7000
DISTILL_SYSTEM_PROMPT = """你是记忆蒸馏器：把用户给出的对话素材提炼成多条原子记忆。

只输出 JSON 数组本身，不要解释、不要 Markdown 围栏。每条形如：
{"content": "一条原子事实，脱离上下文也能读懂", "title": "不超过 60 字", "unit_type": "decision|learning|fact|procedure|context|preference|plan|event", "importance": 0.0}

要求：
- 一条只讲一件事；不要出现“他”“上面”“刚才”这类脱离上下文就读不懂的指代。
- 只保留决策、结论、偏好、做法、事实、计划、事件；跳过寒暄、重复确认和过程性叙述，不要写流水账。
- 条数 1-8；没有值得沉淀的内容就输出 []。
- unit_type 只能取上面八个值之一；importance 是 0 到 1 之间的数字。
"""


def _value(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    try:
        value = config.get(key, default)
    except AttributeError:
        return default
    return default if value is None else value


def _clamp(value: Any, low: int, high: int, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(low, min(high, number))


def _is_admin(event: AstrMessageEvent) -> bool:
    """事件级权限判定；事件缺 role 字段时按非管理员处理。"""
    try:
        return bool(event.is_admin())
    except Exception:  # noqa: BLE001 - 判定失败必须落到拒绝
        return False


def _message_text(value: Any) -> str:
    """把 AstrBot 历史里的字符串/多段 content 统一成纯文本。"""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        text = value.get("text") or value.get("content")
        return str(text or "").strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = _message_text(item)
            if text:
                parts.append(text)
        return " ".join(parts)
    return str(value or "").strip()


def _completion_text(resp: Any) -> str:
    """取 provider 返回的文本；兼容没有 completion_text 的旧实现。"""
    text = getattr(resp, "completion_text", None)
    if isinstance(text, str):
        return text
    if isinstance(resp, str):
        return resp
    return str(resp or "")


def _extract_json_array(text: str) -> str:
    """从模型输出里取出第一个 JSON 数组，容忍围栏与前后解释文字。

    从第一个 '[' 开始做括号配对，字符串里的括号与转义不算结构；找不到配对的
    ']' 时原样返回，交给 json.loads 报错。
    """
    body = str(text or "")
    start = body.find("[")
    if start == -1:
        return body.strip()
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(body)):
        char = body[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return body[start : index + 1]
    return body.strip()


def _distill_importance(value: Any) -> float | None:
    """importance 必须是 0-1 的有限数字；缺失取默认值，非法返回 None。"""
    if value is None:
        return 0.6
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    # nan / inf 也过不了这个区间判断，直接按非法处理。
    if not 0.0 <= score <= 1.0:
        return None
    return score


def _distill_item(entry: Any) -> dict[str, Any] | None:
    """校验单条蒸馏结果；content 为空、unit_type 或 importance 非法都丢弃。"""
    if not isinstance(entry, dict):
        return None
    content = str(entry.get("content") or "").strip()
    if not content:
        return None
    unit_type = str(entry.get("unit_type") or "fact").strip().lower()
    if unit_type not in VALID_UNIT_TYPES:
        return None
    importance = _distill_importance(entry.get("importance"))
    if importance is None:
        return None
    labels = entry.get("labels")
    return {
        "content": content,
        "title": str(entry.get("title") or "").strip(),
        "unit_type": unit_type,
        "importance": importance,
        "labels": labels if isinstance(labels, list) else [],
    }


def _distill_items(raw: str) -> tuple[list[dict[str, Any]], int]:
    """解析模型输出，返回 (合法条目, 丢弃条数)；JSON 非法时抛 ValueError。"""
    data = json.loads(_extract_json_array(raw))
    if not isinstance(data, list):
        raise ValueError("提炼结果不是 JSON 数组")
    items: list[dict[str, Any]] = []
    dropped = 0
    for index, entry in enumerate(data):
        item = _distill_item(entry)
        if item is None:
            dropped += 1
            continue
        items.append(item)
        if len(items) >= MAX_DISTILL_ITEMS:
            dropped += len(data) - index - 1
            break
    return items, dropped


async def _using_provider(context: Any, umo: str) -> Any:
    """取当前文本 Provider：4.27.3+ 用 async 版并传 umo，旧版回退同步方法。"""
    getter = getattr(context, "get_using_provider_async", None)
    if callable(getter):
        return await getter(umo=umo)
    getter = getattr(context, "get_using_provider", None)
    if callable(getter):
        value = getter()
        # 旧宿主若把同步 getter 写成 async def，这里会拿到 coroutine
        if inspect.isawaitable(value):
            return await value
        return value
    return None


def _flag(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    if value is None:
        return default
    return bool(value)


class NowledgeMemPlugin(Star):
    """把 Nowledge Mem 的上下文和长期记忆接入 AstrBot 对话。"""

    def __init__(self, context: Context, config: Any = None) -> None:
        super().__init__(context, config)
        self.config = config or {}
        self._debug = _flag(_value(self.config, "debug", False), False)
        if self._debug:
            self.logger.setLevel(logging.DEBUG)
        api_url = str(_value(self.config, "api_url", _DEFAULT_API_URL)).strip().rstrip("/")
        api_key = str(_value(self.config, "api_key", "")).strip()
        self._space_id = str(_value(self.config, "space_id", "space")).strip()
        self._agent_id = str(_value(self.config, "agent_id", "astrbot")).strip()
        self._source_app = str(_value(self.config, "source_app", "astrbot")).strip() or "astrbot"
        self._timeout = _clamp(_value(self.config, "timeout_seconds", 3), 1, 15, 3)
        self._enable_capture = _flag(_value(self.config, "enable_capture", True), True)
        self._capture_plugin_turns = _flag(
            _value(self.config, "capture_plugin_turns", False), False
        )
        self._fail_open = _flag(_value(self.config, "fail_open", True), True)
        self._save_admin_only = _flag(
            _value(self.config, "save_admin_only", False), False
        )
        self._mem_sum_rounds = _clamp(
            _value(self.config, "mem_sum_rounds", MEM_SUM_ROUNDS_DEFAULT),
            3,
            30,
            MEM_SUM_ROUNDS_DEFAULT,
        )
        raw_exclude = _value(self.config, "capture_exclude_source_ids", [])
        self._capture_exclude_source_ids = {
            str(item).strip()
            for item in (raw_exclude if isinstance(raw_exclude, list) else [])
            if str(item).strip()
        }
        raw_markers = _value(self.config, "heartbeat_markers", [])
        self._heartbeat_markers = [
            str(item)
            for item in (raw_markers if isinstance(raw_markers, list) else [])
            if str(item).strip()
        ]
        self._client = NmemClient(api_url, api_key, timeout=self._timeout, logger=self.logger)
        api_key_digest = digest_api_key(api_key)
        identity = identity_digest(
            "", self._space_id, self._agent_id, self._client.api_url, api_key_digest
        )
        self._identity = identity
        outbox_root = (
            Path(get_astrbot_plugin_data_path())
            / "nowledge_mem_astrbot"
            / "outbox"
        )
        self._outbox = OutboxStore(outbox_root, identity, logger=self.logger)
        self._store = NmemStore(
            self,
            identity,
            outbox=self._outbox,
            logger=self.logger,
        )
        self._injector = ContextInjector(
            self._client,
            self._store,
            agent_id=self._agent_id,
            source_app=self._source_app,
            space_id=self._space_id,
            identity=identity,
            enable_bundle=_flag(_value(self.config, "enable_bundle", True), True),
            enable_recall=_flag(_value(self.config, "enable_recall", True), True),
            bundle_every_turn=_flag(
                _value(self.config, "bundle_every_turn", True), True
            ),
            bundle_on_change=_flag(
                _value(self.config, "bundle_on_change", False), False
            ),
            recall_limit=_clamp(_value(self.config, "recall_limit", 5), 1, 10, 5),
            fail_open=self._fail_open,
            logger=self.logger,
        )
        self._tools = NmemTools(
            self._client,
            self._store,
            space_id=self._space_id,
            agent_id=self._agent_id,
            source_app=self._source_app,
            identity=identity,
            plugin_version=_PLUGIN_VERSION,
            save_admin_only=self._save_admin_only,
            timeout_seconds=self._timeout,
            logger=self.logger,
        )
        self._capture = CaptureCoordinator(
            self._client,
            self._store,
            source_app=self._source_app,
            space_id=self._space_id,
            agent_id=self._agent_id,
            api_url=self._client.api_url,
            api_key_digest=api_key_digest,
            logger=self.logger,
        )
        if not self._client.configured:
            self.logger.warning("Nowledge Mem 未配置 api_url 或 api_key，插件不会请求记忆服务")

    @filter.on_astrbot_loaded()
    async def on_astrbot_loaded(self) -> None:
        """启动时加载离线 outbox；损坏文件只记 warning。"""
        try:
            count = await self._outbox.load()
            if count:
                self.logger.info("Nowledge Mem outbox 已加载 %s 条待同步", count)
        except Exception as exc:
            self.logger.warning(
                "Nowledge Mem outbox 加载失败：%s", type(exc).__name__
            )

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        """每轮注入 Bundle（默认每轮）与记忆召回，并顺带重试 pending。"""
        if not self._client.configured:
            return
        try:
            self._capture.schedule_pending_retry(event.unified_msg_origin)
            texts = await asyncio.wait_for(
                self._injector.collect(event.unified_msg_origin, event.message_str),
                timeout=self._timeout + 1,
            )
            if not texts:
                return
            parts = getattr(req, "extra_user_content_parts", None)
            if parts is None:
                req.extra_user_content_parts = parts = []
            parts.extend(make_temp_parts(texts))
        except Exception as exc:
            if not self._fail_open:
                raise
            self.logger.debug("上下文注入失败：%s", type(exc).__name__)

    @filter.on_agent_done()
    async def on_agent_done(
        self, event: AstrMessageEvent, run_context: Any, resp: LLMResponse
    ) -> None:
        """一轮对话结束后，把增量消息交给后台导入。"""
        del resp
        if not self._client.configured or not self._enable_capture:
            return
        try:
            from .core.delta import extract_messages

            messages = extract_messages(
                getattr(run_context, "messages", None), self._heartbeat_markers
            )
            if not messages:
                return
            capture, reason = should_capture_turn(
                source=getattr(event, "source", None),
                source_id=getattr(event, "source_id", None),
                capture_plugin_turns=self._capture_plugin_turns,
                exclude_source_ids=list(self._capture_exclude_source_ids),
            )
            if not capture:
                self.logger.debug("跳过记忆捕获：reason=%s", reason)
                return
            self._capture.schedule(event.unified_msg_origin, messages)
        except Exception as exc:
            self.logger.debug("对话捕获失败：%s", type(exc).__name__)

    @filter.llm_tool(name="nowledge_mem_search")
    async def nowledge_mem_search(
        self,
        event: AstrMessageEvent,
        query: str = "",
        scope: str = "memories",
        limit: int = 10,
        mode: str = "normal",
        recorded_date_from: str = "",
        recorded_date_to: str = "",
    ) -> str:
        """搜索 Nowledge Mem 里的记忆或历史对话，返回 id、标题、摘要、分数与来源 thread。

        Args:
            query(string): 要找的内容，尽量具体；不能为空
            scope(string): memories 搜记忆、threads 搜历史对话、both 两者都搜；默认 memories
            limit(number): 最多返回几条，范围 1-20，默认 10
            mode(string): normal 或 deep，默认 normal；deep 更慢更细
            recorded_date_from(string): 只要写入 Mem 的时间不早于此日期；YYYY-MM-DD 或 ISO 日期
            recorded_date_to(string): 只要写入 Mem 的时间不晚于此日期；YYYY-MM-DD 或 ISO 日期
        """
        del event
        return await self._tools.search(
            query,
            scope=scope,
            limit=limit,
            mode=mode,
            recorded_date_from=recorded_date_from,
            recorded_date_to=recorded_date_to,
        )

    @filter.llm_tool(name="nowledge_mem_save")
    async def nowledge_mem_save(
        self,
        event: AstrMessageEvent,
        content: str = "",
        title: str = "",
        unit_type: str = "fact",
        importance: float = 0.6,
        labels: list[str] | None = None,
    ) -> str:
        """把值得长期保留的知识主动保存到 Nowledge Mem，不需要用户先开口要求。

        不要保存密钥、密码、令牌、身份证号等敏感信息。

        Args:
            content(string): 要保存的内容，写清楚前因后果，最多 8000 字符
            title(string): 便于搜索的短标题，可空
            unit_type(string): fact、preference、decision、plan、procedure、learning、context 或 event，默认 fact
            importance(number): 重要度 0-1，默认 0.6
            labels(list[string]): 主题或项目标签，可空
        """
        return await self._tools.save(
            content,
            title=title,
            unit_type=unit_type,
            importance=importance,
            labels=labels,
            umo=event.unified_msg_origin,
            sender_id=event.get_sender_id(),
            is_admin=_is_admin(event),
        )

    @filter.llm_tool(name="nowledge_mem_update")
    async def nowledge_mem_update(
        self,
        event: AstrMessageEvent,
        memory_id: str = "",
        content: str = "",
        title: str = "",
        unit_type: str = "",
        importance: float | None = None,
        labels: list[str] | None = None,
    ) -> str:
        """修改一条已有记忆的正文、标题、类型或重要度，只改传入的字段。

        labels 字段服务端当前不生效。先 search 拿到 id 再 update，避免重复新建。

        Args:
            memory_id(string): 记忆 id，不能为空；来自 nowledge_mem_search 或 nowledge_mem_get
            content(string): 新正文，最多 8000 字符；不传则不改
            title(string): 新标题；不传则不改
            unit_type(string): fact、preference、decision、plan、procedure、learning、context 或 event；不传则不改
            importance(number): 重要度 0-1；不传则不改
            labels(list[string]): 服务端当前忽略该字段，传入不会生效；不传则不改
        """
        return await self._tools.update(
            memory_id,
            content=content,
            title=title,
            unit_type=unit_type,
            importance=importance,
            labels=labels,
            umo=event.unified_msg_origin,
            sender_id=event.get_sender_id(),
            is_admin=_is_admin(event),
        )

    @filter.llm_tool(name="nowledge_mem_status")
    async def nowledge_mem_status(self, event: AstrMessageEvent) -> str:
        """检查 Nowledge Mem 连接、当前身份、待同步数量与插件版本；不返回 API Key。"""
        return await self._tools.status(event.unified_msg_origin)

    @filter.llm_tool(name="nowledge_mem_get")
    async def nowledge_mem_get(
        self,
        event: AstrMessageEvent,
        memory_id: str = "",
    ) -> str:
        """按 id 读取一条记忆的全文、标题、类型、重要度、标签与 metadata。

        Args:
            memory_id(string): 记忆 id，不能为空；通常来自 nowledge_mem_search 的结果
        """
        del event
        return await self._tools.get_memory(memory_id)

    @filter.llm_tool(name="nowledge_mem_thread_show")
    async def nowledge_mem_thread_show(
        self,
        event: AstrMessageEvent,
        thread_id: str = "",
        offset: int = 0,
        limit: int = 20,
    ) -> str:
        """读取一段历史对话的消息列表，单条内容最多返回 500 字。

        Args:
            thread_id(string): 对话 id，不能为空；通常来自 nowledge_mem_search 的 source_thread
            offset(number): 跳过前几条消息，默认 0
            limit(number): 最多返回几条，范围 1-100，默认 20
        """
        del event
        return await self._tools.thread_show(
            thread_id, offset=offset, limit=limit
        )

    @filter.llm_tool(name="nowledge_mem_graph")
    async def nowledge_mem_graph(
        self,
        event: AstrMessageEvent,
        query: str = "",
        memory_id: str = "",
        depth: int = 1,
    ) -> str:
        """从一条记忆出发探索知识图谱，返回相邻实体、关系和方向。

        query 与 memory_id 必须且只能给一个；给 query 时先搜索最相关的一条记忆。

        Args:
            query(string): 要探索的主题；与 memory_id 二选一
            memory_id(string): 已知记忆 id；与 query 二选一
            depth(number): 扩展深度，范围 1-5，默认 1
        """
        del event
        return await self._tools.graph(
            query=query, memory_id=memory_id, depth=depth
        )

    @filter.llm_tool(name="nowledge_mem_skill")
    async def nowledge_mem_skill(
        self,
        event: AstrMessageEvent,
        name: str = "",
    ) -> str:
        """读取插件内置技能正文，按需加载记忆使用规范。

        可用技能：read-working-memory、search-memory、distill-memory、status、save-thread、check-integration、explore-graph。

        Args:
            name(string): 技能名，不能为空
        """
        del event
        return await self._tools.skill(name)

    async def _recent_history_text(
        self, umo: str, max_rounds: int
    ) -> tuple[str, int, int, str]:
        """取当前会话最近最多 N 轮文本；返回 (正文, 消息数, 实际轮数, 标题)。

        超预算时从最老的消息开始丢弃，保证最新一轮留在提示词里。
        """
        manager = getattr(self.context, "conversation_manager", None)
        if manager is None:
            return "", 0, 0, ""
        cid = await manager.get_curr_conversation_id(umo)
        if not cid:
            return "", 0, 0, ""
        conversation = await manager.get_conversation(umo, cid)
        if conversation is None:
            return "", 0, 0, ""
        try:
            items = json.loads(getattr(conversation, "history", "") or "[]")
        except (TypeError, ValueError):
            return "", 0, 0, ""
        if not isinstance(items, list):
            return "", 0, 0, ""
        messages: list[tuple[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip()
            if role not in ("user", "assistant"):
                continue
            content = _message_text(item.get("content"))
            if content:
                messages.append((role, content))
        selected_newest_first: list[tuple[str, str]] = []
        used = 0
        header_reserve = len(f"会话总结（最近 {max(1, max_rounds)} 轮）\n")
        for role, content in reversed(messages[-max(1, max_rounds) * 2 :]):
            line = f"{role}: {content[:MAX_SUMMARY_MESSAGE_CHARS]}"
            extra = len(line) + (1 if selected_newest_first else 0)
            if used + extra > MAX_SUMMARY_CHARS - header_reserve:
                break
            selected_newest_first.append((role, content))
            used += extra
        if not selected_newest_first:
            return "", 0, 0, ""
        selected = list(reversed(selected_newest_first))
        actual_rounds = sum(1 for role, _ in selected if role == "user")
        lines = [
            f"{role}: {content[:MAX_SUMMARY_MESSAGE_CHARS]}"
            for role, content in selected
        ]
        text = f"会话总结（最近 {actual_rounds} 轮）\n" + "\n".join(lines)
        first_user = next(
            (content for role, content in selected if role == "user"), ""
        )
        title = "会话总结"
        if first_user:
            title = "会话总结 - " + " ".join(first_user.split())[:40]
        return text, len(selected), actual_rounds, title

    @filter.command("mem_search")
    async def mem_search(
        self, event: AstrMessageEvent, query: GreedyStr = GreedyStr
    ) -> Any:
        """搜索记忆，返回前 5 条标题与摘要。"""
        try:
            text = str(query or "").strip()
            if not text:
                yield event.plain_result("用法：/mem_search <查询内容>")
                return
            result = await self._tools.search(text, scope="memories", limit=5)
        except Exception as exc:
            result = f"error: 搜索失败：{type(exc).__name__}"
        yield event.plain_result(result)

    @filter.command("mem_save")
    async def mem_save(
        self, event: AstrMessageEvent, text: GreedyStr = GreedyStr
    ) -> Any:
        """保存一条记忆；默认 fact，可用 context 前缀指定类型。"""
        try:
            raw = str(text or "").strip()
            unit_type = "fact"
            lowered = raw.lower()
            if lowered.startswith("context "):
                unit_type = "context"
                raw = raw[len("context ") :].strip()
            elif lowered.startswith("fact "):
                unit_type = "fact"
                raw = raw[len("fact ") :].strip()
            if not raw:
                yield event.plain_result(
                    "用法：/mem_save [fact|context] <要保存的内容>"
                )
                return
            result = await self._tools.save(
                raw,
                unit_type=unit_type,
                umo=event.unified_msg_origin,
                sender_id=event.get_sender_id(),
                is_admin=_is_admin(event),
            )
        except Exception as exc:
            result = f"error: 保存失败：{type(exc).__name__}"
        yield event.plain_result(result)

    @filter.command("mem_status")
    async def mem_status(self, event: AstrMessageEvent) -> Any:
        """输出与 nowledge_mem_status 相同的连接与身份信息。"""
        try:
            result = await self._tools.status(event.unified_msg_origin)
        except Exception as exc:
            result = f"error: 状态检查失败：{type(exc).__name__}"
        yield event.plain_result(result)

    @filter.command("mem_sum")
    async def mem_sum(self, event: AstrMessageEvent) -> Any:
        """仅管理员：用当前模型把最近若干轮对话蒸馏成多条原子记忆。"""
        if not _is_admin(event):
            yield event.plain_result("error: 仅管理员可使用 /mem_sum")
            return
        umo = event.unified_msg_origin
        try:
            material, count, rounds, _title = await self._recent_history_text(
                umo, self._mem_sum_rounds
            )
            if not material:
                yield event.plain_result("error: 当前会话没有可总结的对话")
                return
            provider = await _using_provider(self.context, umo)
            if provider is None:
                yield event.plain_result(
                    "error: 当前没有可用的模型，无法蒸馏"
                    "（可改用 /mem_search 手动检索已有记忆）"
                )
                return
            resp = await provider.text_chat(
                prompt=material, system_prompt=DISTILL_SYSTEM_PROMPT
            )
            try:
                items, dropped = _distill_items(_completion_text(resp))
            except ValueError:
                yield event.plain_result("error: 提炼结果不是合法 JSON")
                return
            if not items:
                lines = [f"no_insight: 最近 {rounds} 轮没有值得沉淀的内容"]
                if dropped:
                    lines.append(f"note: 丢弃 {dropped} 条（非法或超出上限）")
                yield event.plain_result("\n".join(lines))
                return
            saved = 0
            lines = []
            for item in items:
                result = await self._tools.save(
                    item["content"],
                    title=item["title"],
                    unit_type=item["unit_type"],
                    importance=item["importance"],
                    labels=item["labels"],
                    umo=umo,
                    sender_id=event.get_sender_id(),
                    is_admin=True,
                )
                title = item["title"] or item["content"]
                if len(title) > 60:
                    title = title[:59] + "…"
                line = f"- [{item['unit_type']}] {title} ({item['importance']:.2f})"
                if result.startswith("saved:"):
                    saved += 1
                    if "已截断" in result:
                        line += "（已截断）"
                else:
                    line += " (保存失败)"
                lines.append(line)
            report = [f"distilled: 保存 {saved}/{len(items)} 条记忆"]
            report.extend(lines)
            if dropped:
                report.append(f"note: 丢弃 {dropped} 条（非法或超出上限）")
            report.append(f"note: 已扫描最近 {rounds} 轮（{count} 条消息）")
            yield event.plain_result("\n".join(report))
        except Exception as exc:
            self.logger.warning("/mem_sum 蒸馏失败：%s", type(exc).__name__)
            yield event.plain_result(f"error: 蒸馏失败：{type(exc).__name__}")

    async def terminate(self) -> None:
        try:
            await self._capture.shutdown()
        finally:
            await self._client.close()
