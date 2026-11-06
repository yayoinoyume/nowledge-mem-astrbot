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

_DEFAULT_API_URL = "http://127.0.0.1:14242"

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

class NowledgeMemPlugin(Star):
    """把 Nowledge Mem 的上下文和长期记忆接入 AstrBot 对话。"""

    def __init__(self, context: Context, config: Any = None) -> None:
        super().__init__(context, config)
        self.config = config or {}
        self._debug = bool(_value(self.config, "debug", False))
        if self._debug:
            self.logger.setLevel(logging.DEBUG)
        api_url = str(_value(self.config, "api_url", _DEFAULT_API_URL)).strip().rstrip("/")
        api_key = str(_value(self.config, "api_key", "")).strip()
        self._space_id = str(_value(self.config, "space_id", "space")).strip()
        self._agent_id = str(_value(self.config, "agent_id", "astrbot")).strip()
        self._source_app = str(_value(self.config, "source_app", "astrbot")).strip() or "astrbot"
        self._timeout = _clamp(_value(self.config, "timeout_seconds", 3), 1, 15, 3)
        self._enable_capture = bool(_value(self.config, "enable_capture", True))
        self._capture_plugin_turns = bool(_value(self.config, "capture_plugin_turns", False))
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
            "", self._space_id, self._agent_id, self._client._api_url, api_key_digest
        )
        self._store = NmemStore(self, identity)
        self._injector = ContextInjector(
            self._client,
            self._store,
            agent_id=self._agent_id,
            source_app=self._source_app,
            space_id=self._space_id,
            enable_bundle=bool(_value(self.config, "enable_bundle", True)),
            enable_recall=bool(_value(self.config, "enable_recall", True)),
            bundle_every_turn=bool(_value(self.config, "bundle_every_turn", True)),
            bundle_on_change=bool(_value(self.config, "bundle_on_change", False)),
            recall_limit=_clamp(_value(self.config, "recall_limit", 5), 1, 10, 5),
            logger=self.logger,
        )
        self._capture = CaptureCoordinator(
            self._client,
            self._store,
            source_app=self._source_app,
            space_id=self._space_id,
            agent_id=self._agent_id,
            api_url=self._client._api_url,
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

    async def terminate(self) -> None:
        try:
            await self._capture.shutdown()
        finally:
            await self._client.close()
