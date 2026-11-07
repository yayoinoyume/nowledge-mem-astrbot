"""Nowledge Mem AstrBot 插件入口。"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import LLMResponse, ProviderRequest
from astrbot.api.star import Context, Star

from .core.capture import CaptureCoordinator, should_capture_turn
from .core.client import NmemClient
from .core.delta import digest_api_key, identity_digest
from .core.inject import ContextInjector
from .core.parts import make_temp_parts
from .core.store import NmemStore
from .core.tools import NmemTools

_DEFAULT_API_URL = "http://127.0.0.1:14242"
_PLUGIN_VERSION = "0.16.0"


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
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _is_admin(event: AstrMessageEvent) -> bool:
    """事件级权限判定；事件缺 role 字段时按非管理员处理。"""
    try:
        return bool(event.is_admin())
    except Exception:  # noqa: BLE001 - 判定失败必须落到拒绝
        return False


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
        self._store = NmemStore(self, identity)
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
    ) -> str:
        """搜索 Nowledge Mem 里的记忆或历史对话，返回 id、标题、摘要、分数与来源 thread。

        Args:
            query(string): 要找的内容，尽量具体；不能为空
            scope(string): memories 搜记忆、threads 搜历史对话、both 两者都搜；默认 memories
            limit(number): 最多返回几条，范围 1-20，默认 10
            mode(string): normal 或 deep，默认 normal；deep 更慢更细
        """
        del event
        return await self._tools.search(query, scope=scope, limit=limit, mode=mode)

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

    @filter.llm_tool(name="nowledge_mem_status")
    async def nowledge_mem_status(self, event: AstrMessageEvent) -> str:
        """检查 Nowledge Mem 连接、当前身份、待同步数量与插件版本；不返回 API Key。"""
        return await self._tools.status(event.unified_msg_origin)

    async def terminate(self) -> None:
        try:
            await self._capture.shutdown()
        finally:
            await self._client.close()
