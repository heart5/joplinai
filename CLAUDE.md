---
jupyter:
  jupytext:
    formats: ipynb,md
    split_at_heading: true
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

Joplinai is an AI-powered knowledge retrieval and Q&A system for [Joplin](https://joplinapp.org/) notes. It vectorizes Joplin notes into ChromaDB, enables semantic search, and provides a multi-user web chat interface with admin management.

## Deployment — Two Servers

| Server | Hostname | Role |
|--------|----------|------|
| 腾讯云 (TC) | `tc` (122.51.102.233:2202) | Joplin Server (CLI, `joplin.xiloong.fans`), 向量数据库 (ChromaDB), 数据中心 (center_api), 定时向量化 (joplinai-sync), 用户管理 |
| 恒创云 (HCX) | long9.org (149.30.242.156) | Joplin Server (Docker, `joplin.strcoder.com`), Ollama LLM 推理 (端口 11434), Q&A API, Web 门户 |

两台 Joplin Server 互为备份：HCX 运行 Docker 版（主），TC 运行 CLI 版（备）。各自通过 Apache 反代对外暴露 HTTPS 端点,本机 client 直连 localhost:41184。

TC 是数据核心+向量化，HCX 是推理+门户。TC 向量化时通过公网 IP 调用 HCX 的 Ollama 生成嵌入向量（已迁移至硅基流动云端为主，HCX Ollama 为回落）。HCX 的 Web 门户通过 HTTP API Key 认证远程调用 TC 的 center_api。

## Architecture — Four Independent Services

All services are Flask apps。No Docker/containerization in production (docker-compose.yml for local dev only)。

| Service | File | Port | Host | Purpose |
|---------|------|------|------|---------|
| Web Portal | `joplin_web_app.py` | 127.0.0.1:5001 | HCX | User login, Q&A chat UI, admin panel, 综合运行面板 |
| Q&A API | `joplin_qa_api.py` | 127.0.0.1:5000 | HCX | Internal HTTP API for vector search + LLM Q&A |
| Center API | `aimod/center_api/` | 0.0.0.0:5003 | TC | Centralized data service for multi-host sharing, 含 monitor/system 端点供面板查询 |
| Vectorization Engine | `joplinai.py` | — | TC | 分块编排 / AI增强 / 写 ChromaDB / 报告。joplinai-sync.timer 每8小时触发 |
| Embedding Inference | SiliconFlow Cloud | — | Cloud | 嵌入向量生成 (`BAAI/bge-large-zh-v1.5`)，回落 HCX Ollama |
| Vision Inference | SiliconFlow Cloud | — | Cloud | 笔记图片描述 (`Qwen/Qwen3-VL-8B-Instruct`)，资源ID缓存 |

Center API 的 gunicorn 入口：`"aimod.center_api:create_app()"`（app factory 模式）。
Web App 入口：`joplin_web_app:app`（模块级 `app = create_app()`，勿放 `if __name__` 内——gunicorn 不执行该块）。

Data flow: `joplinai.py` → TC 本地 (Joplin Server / ChromaDB / center_api) + 外部 SiliconFlow Cloud (嵌入+Vision) + 外部 Cloud API (AI增强, 默认 DeepSeek)。嵌入回落 HCX Ollama。各 `aimod/*_client.py` 通过 HTTP + API Key → `aimod/center_api/` → `data/joplinai_center.db`

向量化优化：chunk 内容未变仅元数据（tags/summary）变更时，跳过嵌入生成，仅通过 `VectorDBManager.batch_update_chunks_metadata()` 一次性批量更新 ChromaDB metadata（不含 embeddings），避免不必要的远程序列调用。内容未变+增强完成时走 `_process_metadata_only_fast_path` 快速路径，完全跳过重复分块/嵌入/增强。

`joplinai_center.db` 包含 10 张表：
- 缓存与历史：`enhance_cache`、`notebook_history`、`global_run_history`
- 笔记状态：`note_process_state`（含 `enhance_config` 字段追踪摘要/标签模型，模型变更时自动触发重处理；`meta_hash` 仅含标签+笔记本元数据，不含增强配置）
- 笔记本级增强策略覆盖：`enhance_override` 云端 JSON 配置，格式 `{"笔记本标题": {"summary_model": "cloud|ollama|none", "tags_model": "cloud|ollama|none"}}`，`_resolve_enhance_config()` 合并全局配置与笔记本级覆盖
- 用户管理：`users`、`sessions`、`audit_log`、`qa_history`、`chat_sessions`

Center client 拆分为 4 个独立文件 (`aimod/`): `CacheClient`（纯远程，无本地回退），`HistoryClient`（远程优先+本地回退），`ProcessStateClient`（纯远程），`UserManagerClient`（远程优先+本地回退）。非 center_api 模块（`report_writer`、`user_manager`）通过设备 ID 判断本机是否数据中心：比对 `center_host_deviceid` → 匹配则直连 `127.0.0.1:5003`，否则走云端配置 `joplinai_center_url`；URL 为空时回退 localhost。

### Source Layout

```
src/                    # 核心模块 (jupytext paired with .ipynb)
├── config_manager.py       # 云端配置热更新单例
├── queryanswer.py          # QA 入口（拆分自原大文件）
├── user_manager.py         # 多用户管理 (admin/team_leader/team_member)
├── pathmagic.py            # 项目路径上下文
├── report_writer.py        # 统一报告模块
├── qa_system.py            # Q&A 系统核心（拆分自 queryanswer.py）
├── prompt_manager.py       # Prompt 模板管理（拆分自 queryanswer.py）
├── qa_config.py            # QA 配置常量（拆分自 queryanswer.py）
├── cli.py                  # CLI 参数解析与 main()
├── web_app/                # Web 门户包
│   ├── __init__.py         # create_app() 工厂
│   ├── routes.py           # 页面路由
│   ├── auth.py             # 用户认证
│   ├── admin.py            # 管理面板
│   ├── api.py              # 内部 API
│   └── dashboard_routes.py # 综合运行面板（聚合 TC monitor/system API）
├── __init__.py
tests/                  # 测试目录
joplinai.py             # 向量化 CLI 入口 + joplinai.ipynb
joplin_qa_api.py        # Q&A API 服务入口 + joplin_qa_api.ipynb
joplin_web_app.py       # Web 门户入口  + joplin_web_app.ipynb
pathmagic.py            # 根路径上下文  + pathmagic.ipynb
# 注意：deploy/ 目录已移除，TC 同步改为手动操作，不在 git push 后自动触发
aimod/                  # AI 核心包
├── __init__.py             # get_logger(name) 统一日志工厂
├── embedding_generator.py  # 文本分块 + 嵌入生成
├── text_splitter.py        # 文本切分器（拆分自 embedding_generator）
├── vector_db_manager.py    # ChromaDB CRUD
├── cache_manager.py        # SQLite LRU 缓存（本地回退用）
├── note_enhancer.py         # AI增强 摘要/标签增强（cloud/local/none）
├── run_tracker.py          # 运行数据采集与历史记录
├── runner_config.py        # 运行器配置
├── center_api/             # 数据中心 API 包
│   ├── __init__.py         # create_app() 工厂 + 蓝图注册
│   ├── cache_routes.py     # /cache/* 端点
│   ├── history_routes.py   # /history/* 端点
│   ├── state_routes.py     # /state/* 端点
│   ├── user_routes.py      # /users/* /auth/* 端点
│   ├── monitor_routes.py   # /monitor/* 端点（笔记监测数据）
│   └── system_routes.py    # /system/* 端点（系统资源/微信健康）
func/                   # 工具子模块 (git submodule heart5/func)
static/                 # 前端静态资源
templates/              # Jinja2 模板
data/                   # 运行时数据 (gitignored)
log/                    # 日志 (gitignored)
.github/workflows/      # CI (flake8 lint + pytest)
.env                    # 环境变量 (gitignored)
```

### Core Modules (`aimod/`)

- `embedding_generator.py` — EmbeddingGenerator 类，向量嵌入入口
- `text_splitter.py` — 文本切分器，按句子边界切分
- `vector_db_manager.py` — VectorDBManager 类，ChromaDB CRUD
- `cache_manager.py` — SQLite LRU 缓存（本地回退用）
- `note_enhancer.py` — AI增强 摘要/标签增强（cloud/local/none），含 SiliconFlow Vision 图片描述（策略模式 `_VisionClient` / `_SiliconFlowVisionClient`，按 `resource_id+model` 缓存）
- `run_tracker.py` — RunTracker 类，运行数据采集和历史记录 (remote-first)
- `cache_client.py` — CacheClient，增强缓存纯远程（无本地回退）
- `history_client.py` — HistoryClient，运行历史远程存储
- `state_client.py` — ProcessStateClient，笔记处理状态远程存储（纯远程，无本地 fallback）
- `user_client.py` — UserManagerClient，用户管理远程存储
- `report_writer.py` — 统一报告生成（在 `src/`）
- `center_api/` — 数据中心 Flask 包（6 个端点蓝图：cache, history, state, user, monitor, system）
- `dashboard_routes.py` — 综合运行面板，聚合 TC monitor/system API + HCX 本地资源状态（在 `src/web_app/`）
- `__init__.py` — `get_logger(name)` 统一日志工厂，自动继承 func/logme 的 handler/level

### Utility Submodule (`func/`)

Proper git submodule registered in `.gitmodules` pointing to `heart5/func`. Clone with `git submodule update --init` to fetch. Key modules:
- `jpfuncs.py` — Joplin API wrapper (CRUD for notes, notebooks, tags)，`_validate_uuid()` 校验所有 `parent_id` 参数
- `configpr.py` — INI config reader
- `first.py` — Project root detection (looks for `rootfile` marker)
- `logme.py` — Logging setup
- `datatools.py` — Content hashing, cloud key retrieval

### User System (`src/user_manager.py`)

Remote-first + local SQLite fallback. Three roles: `admin`, `team_leader`, `team_member`. Notebook-level access control via `allowed_notebooks` JSON field. The module-level `USER_MANAGER` singleton auto-detects remote availability and returns either `UserManagerClient` (remote) or `UserManager` (local).

User data is centralized in TC's `joplinai_center_db` and accessed remotely via HTTP API Key auth. Local `data/joplinai_users.db` serves as fallback.

## How to Run

All commands run from the project root directory.

```bash
# 1. Vectorize notes (CLI, run first to populate ChromaDB)
python joplinai.py
# With specific note IDs (comma-separated, or set cloud item `imp_note_ids`):
python joplinai.py --note_ids "id1,id2,id3"

# 2. Start Q&A API middleware (starts on configured port)
python joplin_qa_api.py

# 3. Start web portal
python joplin_web_app.py
```

Production: systemd services 需手动重启，git push 后不会自动同步至 TC（安全考量）：
```bash
# HCX（本机）— 重启 QA_API + Web_App
sudo systemctl restart joplinai-qa-api joplinai-web-app

# TC（远程）— 先 SSH 手动 git pull，再重启
ssh tc "cd /home/baiyefeng/work/joplinai && git pull && sudo systemctl restart --no-block joplinai-center-api"
```

## Key Code Patterns

- **`pathmagic.Context()`**: All modules use `with pathmagic.Context():` to ensure both the project root and `src/` are on `sys.path` before importing project-local modules. Always wrap project imports in this context manager.
- **Jupytext paired notebooks**：编辑 `.py` 源文件后，jupytext 自动生成/更新同目录同名 `.ipynb` 供 Jupyter Notebook 阅览。`.py` → `ipynb`（percent 格式）。`.ipynb` 永不入库。配置：`jupytext.toml`（默认 `formats = "ipynb,py:percent"`）+ `.git/hooks/pre-commit`（staged .py/.md 自动 `jupytext --sync`）。手动同步：`jupytext --sync <file>`。
  - **重要 `.md` 文档同样关联 `.ipynb`**：`CLAUDE.md`、`README.md`、`docs/CHANGELOG.md`、`docs/TECHNICAL_MANUAL.md`、`docs/QA_PIPELINE.md`、`docs/CHUNK_EMBED_SYSTEM.md` 通过 YAML frontmatter 声明 `formats: ipynb,md`。内容变更后必须同步 ipynb（pre-commit hook 自动执行）。
  - `# %%` 标记代码 cell，`# %% [markdown]` 标记 markdown cell。**关键规则**：markdown cell 后的下一段代码前必须有显式 `# %%`，否则 jupytext sync 会把代码当作 markdown 注释掉。
      - **自动检测**：`tools/check_jupytext_comment.py` 扫描被误注释的 Flask route/endpoint。pre-commit hook（`.git/hooks/pre-commit`）在 jupytext sync 后自动执行，CI 兜底。
- **`__all__`**: 所有 `aimod/`、`src/`、`aimod/center_api/` 的公开模块都有 `__all__` 明确导出列表。
- **`__repr__`**: 5 个关键数据类有 `__repr__`：`CacheResult`, `VectorDBManager`, `EmbeddingGenerator`, `RunTracker`, `QASystem`。
- **Type annotations**: 所有公开函数有返回类型标注 (`-> None`, `-> str`, `-> Optional[str]` 等)。
- **增量处理三级决策**：`process_notes_incremental` 提交前计算 `content_unchanged`（内容哈希匹配且非强制更新）和 `needs_re_enhance`（增强缺失或模型配置变更），传入 `process_note_chunks`。内容未变+增强完成→`_process_metadata_only_fast_path` 跳过全部分块/嵌入/增强，直接 `batch_update_chunks_metadata`。内容变更→完整路径（分块+嵌入+增强+入库）。增强缺失但内容未变→正常路径重增强但可跳过嵌入生成（TODO）。
- **块级失败重试**：`process_note_chunks` 第一轮失败块记录 `{chunk_data, reason, embedding, enhanced_metadata}`，区分 `"embedding"`（需重新生成嵌入）和 `"upsert"`（复用已有嵌入）。`process_notes_incremental` 完成后以 `max_workers=2` 执行 2 轮×3s 间隔重试。
- **`aimod.get_logger(name)`**: 统一日志工厂，自动继承 `func/logme` 的 handler 和 level。用法：`from aimod import get_logger; logger = get_logger(__name__)`。
- **Cloud config**: Configuration is fetched dynamically via `getinivaluefromcloud()` from an INI stored in a Joplin note. The `ConfigManager` singleton in `src/config_manager.py` handles hot-reloading (5-minute check interval). Key cloud items: `imp_nbs` (notebook titles), `imp_note_ids` (specific note IDs as virtual notebook), both comma-separated. `ConfigManager.get_all()` is the alias for `get_config_snapshot()`.
- **Inter-service auth**: `joplin_web_app.py` calls `joplin_qa_api.py` using an API key from the shared cloud config (`X-API-Key` header).
- **Gunicorn entry points**: `aimod.center_api:create_app()` (app factory), `joplin_web_app:app` (module-level instance), `joplin_qa_api:app` (module-level instance).

## Model Strategy

**核心原则：实时走云端，离线走本地。** HCX 服务器无 GPU（纯 CPU 8核），7B+ 模型本地推理需 6-12 分钟，对 web 场景不可用。

### 本地模型（Ollama, CPU-only）

| 模型 | 大小 | 用途 | 速度 |
|------|------|------|------|
| `dengcao/bge-large-zh-v1.5` | 651 MB | 文本嵌入（向量化） | ~2秒/次 |
| `qwen2.5:1.5b` | 986 MB | 标签提取、内容分类 | ~40-80秒/条 |
| `minicpm-v:latest` | 5.5 GB | (已废弃) 笔记图片描述，不可用 | - |

- **`qwen2.5:1.5b`** 通过 `enhance_ollama_chat_model` 配置键指定，供向量化管线离线批量任务使用
- **无本地 QA 回退**：`qa_ollama_chat_model=none`，Cloud API 不可用时直接返回服务不可用，不尝试本地模型
- 之前使用后删除的模型：`qwen:1.8b`（架构旧）、`qwen3:8b`（12分钟太慢）、`deepseek-r1:7b`（RAG误判）、`gemma3:4b-it-qat`（中文幻觉）

### 云端模型（OpenAI-compatible API，默认 DeepSeek）

| 模型 | 用途 | 速度 |
|------|------|------|
| `deepseek-v4-flash` | QA 对话主路 + AI增强（`cloud_model=deepseek-v4-flash`） | ~5秒/次 |
| `deepseek-v4-pro` | 图片精细描述（`vision_model`） | ~数秒/图 |
| `BAAI/bge-large-zh-v1.5` | 文本嵌入（硅基流动云端，与本地同模型同维度） | ~0.1秒/次 |
| `Qwen/Qwen3-VL-8B-Instruct` | 笔记图片描述（硅基流动云端视觉，默认） | ~1-5秒/图 |

可通过 `cloud_api_url` + `cloud_api_key` + `cloud_model` 切换至 Qwen/ChatGPT 等兼容提供者。
嵌入可通过 `embedding_provider=siliconflow` 切至云端（硅基流动），`_FallbackClient` 自动回落本地 Ollama。

### 配置要点

| 配置键 | 当前值 | 说明 |
|--------|--------|------|
| `qa_ollama_chat_model` | `none` | 无本地 QA 回退 |
| `cloud_model` | `deepseek-v4-flash` | 云端大模型（QA+增强） |
| `cloud_api_url` | `https://api.deepseek.com/v1/chat/completions` | 云端 API 端点 |
| `cloud_api_key` | 回退 `deepseek_token` | 云端 API Key |
| `summary_model` | `cloud` | 摘要模型: cloud/ollama/none |
| `tags_model` | `cloud` | 标签模型: cloud/ollama/none |
| `enhance_ollama_chat_model` | `qwen2.5:1.5b` | Ollama 标签/分类模型 |
| `validation_threshold` | `300` | 增强缓存验证阈值，命中>N次触发重验 |
| `vision_enabled` | `false` | 图片视觉描述（SiliconFlow云端），默认关闭 |
| `hyde_enabled` | `true` | HyDE 假设文档嵌入增强检索 |
| `rerank_enabled` | `true` | LLM 精排候选块重排序 |
| `embedding_provider` | `siliconflow` | 嵌入后端: ollama(本地) / siliconflow(云端+回落) |
| `siliconflow_api_key` | `sk-xxx` | 硅基流动 API Key（嵌入+Vision共用） |
| `vision_model` | (空=默认) | Vision 模型，默认 `Qwen/Qwen3-VL-8B-Instruct`（可选 `Qwen/Qwen3-VL-32B-Instruct`） |

## Production Incidents

### 规则：远程状态加载不得静默降级

2026-05-30 center_api 不可达导致全量重处理事故（详见 `docs/INCIDENTS.md`）的核心教训：

**center_api 的 client（`ProcessStateClient`、`CacheClient` 等）在连接失败时，禁止返回空/默认值让调用方以为"没有历史数据"。**

- `ProcessStateClient.batch_load()` 失败→抛 `CenterAPIUnreachableError`，sync 中止。仅 `--enable_force_update` 模式放行。
- `CacheClient.get()` 失败→重试 3 次后退化为 miss（可接受，AI 增强可重跑），但不可丢失已处理状态。
- `_request()` 全部方法必须有重试（3 次指数退避）。
- systemd 服务依赖：`Requires=joplinai-center-api.service`，禁止仅 `Wants`。

**原则**：对状态加载而言，宁可不跑也不能错跑。"连接失败=返回空"等价于"忘掉所有历史，从零开始"。

### 规则：状态持久化不得使用全量替换

2026-05-30 batch_save DELETE ALL + INSERT ALL 导致 940 条状态丢失事故（详见 `docs/INCIDENTS.md#inc-002`）：

**`note_process_state` 的写入必须用 upsert/merge，禁止 `DELETE + 重新 INSERT`。**
当前表有 `PRIMARY KEY (model_name, note_id)`，用 `INSERT ... ON CONFLICT DO UPDATE`。

原因：sync 可能只处理笔记本子集。DELETE ALL 会把未处理笔记的状态也删掉。同样规则适用于所有多记录持久化场景——缓存表、配置表等。

## Testing & CI

```bash
python -m pytest tests/ -v --tb=short   # 运行测试
flake8 . --max-line-length=100 --ignore=E402,W503,E203,E501 --exclude=func/  # lint
```

CI (`.github/workflows/ci.yml`): push/PR 触发，`check_jupytext_comment.py` → flake8 lint → pytest (Python 3.10)。
Pre-commit (`.pre-commit-config.yaml`): jupytext 误注释检测 + flake8。

## Known Issues & Technical Debt

- **`static/favicon.ico*`**: 3 domain-specific favicon copies at 3.5MB each. Only the default `favicon.ico` should be tracked; the `_for_*` variants are in `.gitignore`.
- **TC git pull 可能需要代理**: 直连偶尔失败（GnuTLS recv error），回退方案是 `HTTPS_PROXY=http://127.0.0.1:7890 git pull`。
- **HCX `joplinai-qa-api`/`joplinai-web-app` 代码更新后需手动重启**: gunicorn 进程不自动 reload，git pull 后必须 `sudo systemctl restart`。
- **SSH 远程 systemctl restart 会 hang**: `systemctl restart` 默认等待服务停止→启动，SSH 会话会阻塞超时。远程重启必须加 `--no-block`：`ssh tc "sudo systemctl restart --no-block <svc>"`。
- **git push 后不自动同步 TC**: 出于安全考量，推送代码后 TC 需要手动 SSH 登录执行 `git pull` + `systemctl restart --no-block`。不做自动部署。
- **TC 配置更新流程**：修改云端 INI → `ssh tc "source /usr/miniconda3/etc/profile.d/conda.sh && conda activate newlsp && joplin sync"` → `ssh tc "sudo systemctl restart --no-block joplinai-sync"`。joplinai-sync 是 timer 触发的 oneshot 服务，需手动 start/restart。
- **center_api 日志查看**：TC 上 `sudo journalctl -u joplinai-center-api -f`。logger 配置在 `aimod/center_api/__init__.py`：`propagate=False` 避免 `func/logme` root handler 双重输出。
- **ChromaDB 在 TC 以 Docker 运行**：`docker run chromadb/chroma`，端口映射 8009→8000。配置中 `chroma_port=8009`，但 `vector_db_manager.py` 硬编码默认 8000——依赖云端配置覆盖。直接 `chromadb.HttpClient(host='127.0.0.1', port=8009)` 才能连上。
- **Ollama 在 HCX 以公网 IP 暴露**：TC 向量化通过 `149.30.242.156:11434` 远程调 HCX 的 Ollama 生成嵌入。端口 11434 需对外网开放。

## Git

- Branch: `main`
- Remote: `origin` (GitHub: `heart5/joplinai`)
- `.gitignore` covers: `.claude/`, `log/`, `data/`, `*.ipynb`（全部屏蔽，永不入库——ipynb 由 jupytext 从 .py 自动生成），`__pycache__/`, backup files (`*.bak`), oversized favicon copies, `.env`
- `jupytext.toml`: py↔ipynb 配对配置，`.git/hooks/pre-commit`: 提交 .py 时自动同步 ipynb
- `.pre-commit-config.yaml`: flake8 pre-commit hook

## Configuration

Main config stored in cloud-synced Joplin note (INI format). Local override: `data/joplinai.ini`. Key settings: Joplin API token, Ollama model name, embedding model, ChromaDB path, Q&A prompts, user session settings.

**数据中心配置**：`center_host_deviceid`（数据中心主机的设备 ID，客户端通过对比本机 deviceid 自动判断是否直连 localhost）、`joplinai_center_url`（非数据中心主机配，指向 TC 公网IP；未配则回退 localhost）、`joplinai_center_api_key`（认证密钥，所有客户端通用）。

**Joplin 连接配置** (`data/joplinai.ini` `[joplin]` section)：`local_server`（本机运行 Joplin Server 时设为 `true`，直连 localhost:41184 免绕公网）、`fallback_url`（回退 Joplin Server 的 HTTPS 反代端点）、`fallback_token`（对应 API token）。

**模型配置**：`qa_ollama_chat_model`（本地 QA 对话模型，当前 `none` 无本地回退）、`enhance_ollama_chat_model`（Ollama 标签/分类小模型）、`ollama_embedding_model`（嵌入模型）、`vision_model`（视觉模型）、`cloud_model` / `summary_model` / `tags_model`（cloud/ollama/none 三态切换）、`cloud_api_url` / `cloud_api_key`（云端 API 端点/密钥，支持切换提供者）。详见上方 Model Strategy 章节。QA 检索链路详见 `docs/QA_PIPELINE.md`。

### Remote Joplin fallback

`func/jpfuncs.getapi()` 连接策略（`local_server` 优先 → fallback_url → 本地 CLI）：

```ini
[joplin]
# 本机运行 Joplin Server 时设为 true，直连 localhost:41184 免绕公网
local_server = true
# 回退/远程 Joplin Server URL
fallback_url = https://joplin.xiloong.fans
fallback_token = <api-token>
```

两台 Joplin Server 的反代端点:
- **HCX (Docker)**: `https://joplin.strcoder.com` → `127.0.0.1:41184`
- **TC (CLI)**: `https://joplin.xiloong.fans` → `127.0.0.1:41184`

本机 `local_server=true` 时直连 localhost，不通才走 fallback_url 连对端。全部失败抛出 `JoplinUnreachableError`（而非 `exit(1)`）。

## Dependencies (no requirements.txt — install manually)

Core: `flask`, `chromadb`, `ollama`, `requests`, `jinja2`, `werkzeug`
Dev: `flake8` (lint), `pytest` (test)
