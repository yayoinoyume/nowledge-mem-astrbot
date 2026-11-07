---
name: status
description: 检查 Nowledge Mem 的连接、身份与同步状态。当记忆工具报错、搜索结果异常、刚接入新环境，或用户问「连上了吗」「配置对不对」「为什么搜不到」时使用；日常对话不要反复调用。
---

# 检查连接状态

## 何时使用
- `nowledge_mem_search`、`nowledge_mem_save` 返回 `error:`
- 搜索长期为空，怀疑是连接或配置问题，而不是真的没有记忆
- 首次接入一个新环境，或用户问「连上了吗」「配置对吗」
- 同步看起来停了：`pending` 一直不清零、`last_sync` 很久没更新

## 怎么调用
`nowledge_mem_status()`，不需要参数。

## 返回字段怎么读
| 字段 | 含义 |
| --- | --- |
| `connected` | 是否连上服务 |
| `api_url` | 服务地址，只显示协议、主机与端口，不含凭据 |
| `space` | 当前记忆空间 |
| `agent` | 当前 Agent 身份 |
| `identity` | 由空间、Agent、地址与凭据指纹算出的隔离标识 |
| `api_key` | `configured` 或 `missing`，只报状态不回显内容 |
| `server_version` | 服务端版本，拿不到时为空 |
| `cli_version` | 命令行版本；本插件不依赖命令行，通常为空或 `n/a` |
| `attention_has_pending_reviews` | 是否有待审阅项，只报有无，不是精确计数 |
| `pending` | 本会话是否有待同步批次（0/1） |
| `last_sync` | 上次同步成功时间，从未同步为 `never` |
| `plugin_version` | 插件版本 |
| `backend` | 实际走通的后端路径 |
| `error` | 失败原因，成功时不出现 |

## 判断顺序
1. `api_key: missing`：配置里没有凭据，先让用户补配置
2. `connected: false` 且有 `error`：服务不可达、鉴权失败或超时；按 `error` 原文说明，不要笼统说「网络问题」
3. `connected: true` 但 `pending` 持续为 1：写入侧受阻，提醒用户检查服务与网络
4. `connected: true` 且 `pending: 0`：连接正常；搜不到更可能是关键词问题或确实没有记忆

## 边界
- 只做诊断，不改配置、不重启服务、不循环重试
- 不把地址以外的配置、凭据、内部标识贴给用户
- 不要因为一次 `status` 失败就断定整个记忆库为空
- 给用户 2-3 条可操作建议即可，不要假装已经修好
