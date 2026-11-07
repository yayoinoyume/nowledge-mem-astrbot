# Nowledge Mem AstrBot 插件

把 [Nowledge Mem](https://mem.nowledge.co) 的上下文与长期记忆接入 AstrBot：每轮对话前注入 Context Bundle 与相关记忆召回，每轮结束后把新增消息增量写回同一个 Mem thread。

- 插件名：`nowledge_mem_astrbot`
- 版本：0.18.0
- 要求：AstrBot >= 4.27.3

## 能力

| 能力 | 说明 |
| --- | --- |
| 每轮注入 Context Bundle | 默认每轮注入，使用临时 part，不写入会话历史 |
| 每轮记忆召回 | 用当前用户消息调 `/memories/search`，在当前 space 内按服务端相关性排序取前 `recall_limit` 条，不设阈值，不做本地重排 |
| 增量回写 | 一轮结束后只把新增消息 append 到同一 thread |
| append / reset | 前缀未变走 append；前缀变化或窗口缩水走整段 reset |
| 失败重试 | 失败批次落到 pending，下一轮在后台按原 mode 重试 |
| 八个工具 | nowledge_mem_search / save / update / status / get / thread_show / graph / skill，模型可主动检索、保存、更新、诊断与读技能 |

## 安装

两种方式都落到同一个目录 `<AstrBot 根目录>/data/plugins/nowledge_mem_astrbot`，选一种即可，不要同时维护两份。这里指包含 `data/` 的那层根目录，不是 `data/` 本身。

**方式一：复制目录**

```bash
cp -r nowledge_mem_astrbot <AstrBot 根目录>/data/plugins/
```

**方式二：WebUI 在线安装**

在 AstrBot WebUI 的插件管理里用仓库地址 `https://github.com/yayoinoyume/nowledge-mem-astrbot` 在线安装/更新。AstrBot 会把仓库名转成合法模块名（`-` → `_`），安装目录同样是 `nowledge_mem_astrbot`。

装好后重启 AstrBot，在 WebUI 的插件配置里填写 `api_url` 与 `api_key`。依赖只有 `aiohttp`（AstrBot 已自带，`requirements.txt` 仅作声明）。

## 配置

| 键 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `api_url` | string | `http://127.0.0.1:14242` | Nowledge Mem 服务地址；非回环明文 HTTP 启动时打 WARNING |
| `api_key` | string | 空 | Bearer 凭据，只保存在 AstrBot 配置里 |
| `space_id` | string | `space` | 记忆空间 ID |
| `agent_id` | string | `astrbot` | Context Bundle 与工具使用的 Agent ID |
| `source_app` | string | `astrbot` | 写入 Mem 时标记的来源应用 |
| `save_admin_only` | bool | `false` | 仅管理员可调用 `nowledge_mem_save` 与 `nowledge_mem_update`；默认 member 也能写共享 space |
| `enable_bundle` | bool | `true` | 是否注入 Context Bundle |
| `bundle_every_turn` | bool | `true` | 每轮注入 Bundle |
| `bundle_on_change` | bool | `false` | 缓存过期后是否重新拉取 Bundle 并刷新缓存；不控制是否注入 |
| `enable_recall` | bool | `true` | 每轮按当前消息召回记忆 |
| `recall_limit` | int | `5` | 每轮最多注入的召回条数（1–10） |
| `mem_sum_rounds` | int | `10` | `/mem_sum` 取最近多少轮对话做蒸馏（3–30） |
| `timeout_seconds` | int | `3` | 读取接口超时，秒（1–15） |
| `fail_open` | bool | `true` | 读取失败时注入降级文本；`false` 时不降级，异常记入 AstrBot 错误日志（不承诺中止回合） |
| `enable_capture` | bool | `true` | 对话结束后同步增量消息到 Mem thread |
| `capture_plugin_turns` | bool | `false` | 是否捕获 `source=plugin` 的回合 |
| `capture_exclude_source_ids` | list | `[]` | 永不捕获的 `source_id` 列表 |
| `heartbeat_markers` | list | `[]` | 心跳等内部消息的前缀标记 |
| `debug` | bool | `false` | 输出调试日志 |

`heartbeat_markers` 用来过滤由定时任务、心跳一类插件触发的回合。用户消息以其中任一前缀开头时，整条消息不写入 Mem；助手回复 `<HEARTBEAT_SILENT>` 同样被丢弃。默认空列表表示不过滤，需要时自行填写，例如：

```json
"heartbeat_markers": ["这是一次自主心跳"]
```

## 行为

### 三个钩子

- `on_llm_request`：每轮注入 Bundle 与召回，并顺带重试 pending。
- `on_agent_done`：一轮结束后提取增量消息，交给后台串行导入。
- `on_astrbot_loaded`：启动时加载离线 outbox 里的待同步批次。

`terminate` 是插件生命周期回调，负责取消后台任务并关闭 HTTP 会话，不属于 `@filter` 钩子。

### Bundle 与召回

- `bundle_every_turn=true`（默认）：每轮都注入，用临时 part，不写入会话历史。同一会话 60 秒内复用 KV 里的上次 Bundle 以减少 GET；超过 60 秒重新拉取，`compiled_hash` 变化时立即刷新缓存。
- `bundle_on_change=true`：只在 `bundle_every_turn=false` 时起作用——缓存超过 60 秒后重新 GET 并刷新缓存。两个开关都不控制是否注入，Bundle 始终每轮注入。
- 注入文本带八条 guidance（387 字）：记忆是参考、主动保存、同一件事的细化或纠正先 search 再 update、不存敏感信息、按需搜索、先说明再搜、Bundle 只是简报、不要引用包装；全文见 `core/inject.py` 的 `GUIDANCE`。
- Bundle 失败回退链：GET `/context/bundle` → KV 缓存的最近一次 Bundle → 注入「记忆服务暂不可用，本轮无上下文」。不会静默空注入。

### 工具

| 工具 | 参数 | 说明 |
| --- | --- | --- |
| nowledge_mem_search | query、scope（memories/threads/both）、limit（1–20，默认 10）、mode（normal/deep）、recorded_date_from/to | 搜记忆或历史对话；空结果返回 no matches，请求失败返回 error:，两者不混淆 |
| nowledge_mem_save | content、title、unit_type、importance（0–1）、labels | 写入记忆；标题截 200 字、标签截 64 字；save_admin_only=true 时仅管理员可用 |
| nowledge_mem_update | memory_id、content、title、unit_type、importance、labels | 改已有记忆，只提交给出的字段并回读校验；labels 服务端当前不生效，会返回一行 note；save_admin_only=true 时仅管理员可用 |
| nowledge_mem_status | 无 | 连接、space、agent、identity、pending、last_sync、服务端版本；不输出 API Key |
| nowledge_mem_get | memory_id | 读一条记忆全文、标题、类型、重要度、标签与 metadata；metadata 里的 file_path 递归剔除 |
| nowledge_mem_thread_show | thread_id、offset（≥0，默认 0）、limit（1–100，默认 20） | 读历史对话消息；单条内容最多 500 字，丢弃 related_memories、entities、total_tokens |
| nowledge_mem_graph | query 或 memory_id（二选一）、depth（1–5，默认 1） | 探索知识图谱，返回相邻实体、关系和方向；未知节点返回「无邻居或节点不存在」 |
| nowledge_mem_skill | name | 读取内置技能正文；只允许 7 个固定名字，拒绝路径穿越 |

保存时把 source_umo 与 sender_hash（sha1 前 16 位，非匿名化）写入 metadata；服务端顶层不接受这三个字段，实测会被静默忽略。

所有工具只返回字符串：成功是正文，失败以 `error:` 开头，不向对话抛异常。查询类工具 member 可用；`nowledge_mem_save` 与 `nowledge_mem_update` 仍受 `save_admin_only` 控制。

### 斜杠命令

命令参数用 AstrBot 的 `GreedyStr`，空参数返回用法提示，不会抛异常。

- `/mem_search <query>`：固定 `scope=memories`、`limit=5`；空结果 `no matches`，失败 `error:`。
- `/mem_save [fact|context] <text>`：默认 `fact`；`context ` 前缀改为 `context` 并去掉前缀；权限走 `save_admin_only`。
- `/mem_status`：等价 `nowledge_mem_status`。
- `/mem_sum`：仅管理员；用当前模型把最近若干轮（默认 10 轮，可配 `mem_sum_rounds` 3-30）蒸馏成多条原子记忆后逐条写入，条数上限 8；报告里给真实覆盖轮数与丢弃计数。

### 技能包

插件目录 `skills/` 下放 7 个 `SKILL.md`，内容与注入的 guidance 对齐：

- `read-working-memory`：会话开始时读 Working Memory；已注入 Bundle 时不要重复读。
- `search-memory`：检索路由、recorded 时间窗、有界 thread 读取与图谱探索。
- `distill-memory`：保存时机、类型、重要度、去重与安全红线。
- `status`：状态解读与排障。
- `save-thread`：自动 thread 同步与结构化交接（目标/决定/改动面/风险/下一步）。
- `check-integration`：记忆功能自检手册（连接、鉴权、检索链路、空间配置与常见故障）。
- `explore-graph`：用 `nowledge_mem_graph` 展开记忆邻居与关系，渐进加深，不承诺可视化网页。

模型可调用 `nowledge_mem_skill(name)` 按需读取正文。

### 离线 outbox

capture 的 pending 从 AstrBot KV 改为落盘：

```text
data/plugin_data/nowledge_mem_astrbot/outbox/<identity>/<umo>.json
```

- 同目录临时文件 + `os.replace` 原子写；文件权限 0600，目录 0700。
- umo 先做 URL 编码再当文件名；超长改用 sha1；以 `.` 或 `_` 开头的值额外拼 8 位哈希，避免不同 umo 撞成同一文件。
- 只保存重试真正用到的字段；`umo` 存在文件里，用于启动加载时还原键。
- 单个身份最多 50 个会话文件（每个 umo 一个），超出按 mtime 淘汰最旧。
- 启动时 `on_astrbot_loaded` 扫描加载；损坏文件只记 warning，不删除。
- 重试成功删除文件；失败保留并沿用原来的 50 次 / 24 小时上限。
- 旧版本留在 KV 里的 pending 会在首次读取时自动迁移到磁盘，然后删掉旧键。

### 游标与 append/reset

- 游标用 `last_confirmed_hash` + `confirmed_count` + `prefix_fingerprint`；优先按 `confirmed_count` 续接，重复 assistant 不会丢轮；前缀变化或窗口缩水才全量 reset。
- 当前回合末尾没有 assistant（LLM 报错轮）时不同步，避免建半截 thread。
- tool 消息映射为 assistant：`[Tool <name>] <结果全文，上限 20000 字>`。
- 独立 part 的 `<system_reminder>...</system_reminder>` 整段删除；字符串只删尾部且含 `Current datetime` 的提醒，用户正文中间或自己写的标签保留。
- 标记为临时（`_no_save` / `mark_as_temp`）的 part 不进入记忆。

### 失败重试

- 失败批次把 `mode/messages/expected/idempotency_key/window/created_at` 写入 pending；下一次 `on_llm_request` 在同一个 per-thread 锁内重读并按原 mode 重试；conflict 用 window reset 恢复；上限 50 次或 24 小时。
- append 接受 `checkpointed`，或 `deduplicated + messages_added=0 + already_existed=true`；reset 不校验 `append_mode` 白名单。

### 身份隔离

thread_id 与 KV key 都包含 `umo + space_id + agent_id + api_url + sha1(api_key)[:8]` 指纹：

```text
key_digest = sha1(api_key)[:8]
thread_id  = astrbot-<sha1(umo|space|agent|api_url|key_digest)[:16]>
identity   = sha1("|" + space + "|" + agent + "|" + api_url + "|" + key_digest)[:16]
key        = cursor:<identity>:<umo>
```

换 Space / Agent / API 地址 / API Key 会得到新 thread 与新 cursor，不会复用旧 `remote_count`；旧 thread 保留，新 thread 第一次同步从 reset 开始。

## 已知限制

- 只处理文本：图片、语音等非文本 part 不进入记忆。
- 并发锁是进程内的 `asyncio.Lock`，多进程共享同一插件 KV 时同一 thread 可能并发写入。
- 一个会话对应一个 thread，跨会话不合并。
- 每轮召回固定走 `mode=normal`，工具可显式传 `mode=deep`；不做 rerank。
- 助手回复 `<HEARTBEAT_SILENT>` 始终丢弃，与配置无关；用户消息只看是否匹配 `heartbeat_markers` 前缀，默认空列表表示不过滤用户消息。
- `nowledge_mem_update` 的 `labels` 服务端当前忽略：客户端只返回一行 `note`，不保证标签落地。
- `nowledge_mem_update` 走 `PATCH /memories/{id}`：服务端该端点不接受 `space_id`，也不返回记忆所属空间；只要拿到 `memory_id`，就能改到其他空间的记忆。客户端无法在更新前确认归属，只能依赖 API Key 在服务端的权限范围。
- `/mem_sum` 每次调用消耗一次模型请求的 token；一次最多写入 8 条；提示词预算 7000 字符，超出时丢弃最老的消息；非法条目丢弃并在报告里计数。

## 测试

```bash
uv run --with pytest --with aiohttp python -m pytest tests -q
```

可选：用真实 AstrBot 源码跑集成测试（未设置 `ASTRBOT_SOURCE` 时自动 skip）：

```bash
ASTRBOT_SOURCE=/path/to/astrbot uv run --with pytest --with aiohttp --with pydantic python -m pytest tests -q
```

插件级注册/权限测试需要 AstrBot 的完整依赖，必须用 AstrBot 自己的 venv 解释器跑，否则整模块 skip：

```bash
ASTRBOT_SOURCE=/path/to/astrbot/app uv run --python /path/to/astrbot/app/.venv/bin/python --with pytest --with aiohttp --with pydantic python -m pytest tests -q
```

只读 smoke（需要服务可达）：

```bash
NMEM_API_URL=... NMEM_API_KEY=... python scripts/smoke_readonly.py
```
