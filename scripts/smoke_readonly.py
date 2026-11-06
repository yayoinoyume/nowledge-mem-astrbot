#!/usr/bin/env python3
"""只读 smoke：验证 Nowledge Mem 的 health、bundle、search。"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

import aiohttp

TIMEOUT = 10.0


def _masked_key() -> str:
    key = os.environ.get("NMEM_API_KEY", "")
    return f"<redacted> (len={len(key)})" if key else "(empty)"


async def _get_json(
    session: aiohttp.ClientSession,
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str],
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
) -> Any:
    async with session.request(
        method,
        url,
        headers=headers,
        params=params,
        json=body,
        timeout=aiohttp.ClientTimeout(total=TIMEOUT),
    ) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
        return await response.json(content_type=None)


async def main() -> int:
    api_url = os.environ.get("NMEM_API_URL", "").strip().rstrip("/")
    api_key = os.environ.get("NMEM_API_KEY", "").strip()
    print(f"api_url: {api_url or '(empty)'}")
    print(f"api_key: {_masked_key()}")
    if not api_url or not api_key:
        print("缺少 NMEM_API_URL 或 NMEM_API_KEY")
        return 1
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    space_id = os.environ.get("NMEM_SPACE", "space").strip() or "space"
    agent_id = os.environ.get("NMEM_AGENT_ID", "astrbot").strip() or "astrbot"
    try:
        async with aiohttp.ClientSession() as session:
            health = await _get_json(
                session, f"{api_url}/health", headers=headers
            )
            print(f"health: ok ({json.dumps(health, ensure_ascii=False)})")

            bundle = await _get_json(
                session,
                f"{api_url}/context/bundle",
                headers=headers,
                params={
                    "agent_id": agent_id,
                    "source_app": "astrbot",
                    "space_id": space_id,
                    "include_working_memory": "true",
                },
            )
            markdown = bundle.get("rendered_markdown") if isinstance(bundle, dict) else None
            if not isinstance(markdown, str) or not markdown:
                raise RuntimeError("bundle 缺少 rendered_markdown")
            print(
                "bundle: ok "
                f"rendered_markdown={len(markdown)} 字符 "
                f"compiled_hash={bundle.get('compiled_hash')}"
            )

            results = await _get_json(
                session,
                f"{api_url}/memories/search",
                method="POST",
                headers={**headers, "Content-Type": "application/json"},
                body={"query": "Nowledge Mem 插件", "limit": 3, "mode": "fast"},
            )
            if not isinstance(results, list):
                raise RuntimeError("search 响应不是数组")
            print(f"search: ok {len(results)} 条")
            if not results:
                return 1
            return 0
    except Exception as exc:
        print(f"smoke 失败：{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
