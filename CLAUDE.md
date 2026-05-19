---
jupyter:
  jupytext:
    formats: ipynb,md
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
| 腾讯云 (TC) | `tc` (122.51.102.233:2202) | Joplin Server, 向量数据库 (ChromaDB), 数据中心 (center_api), 定时向量化 (joplinai-sync), 用户管理 |
| 恒创云 (HCX) | long9.org (149.30.242.156) | Ollama LLM 推理 (端口 11434), Q&A API, Web 门户 |

TC 是数据核心+向量化，HCX 是推理+门户。TC 向量化时通过公网 IP 调用 HCX 的 Ollama 生成嵌入向量。HCX 的 Web 门户通过 HTTP API Key 认证远程调用 TC 的 center_api。

## Architecture — Four Independent Services

All services are Flask apps。No Docker/containerization in production (docker-compose.yml for local dev only)。

| Service | File | Port | Host | Purpose |
|---------|------|------|------|---------|
| Web Portal | `joplin_web_app.py` | 127.0.0.1:5001 | HCX | User login, Q&A chat UI, admin panel |
| Q&A API | `joplin_qa_api.py` | 127.0.0.1:5000 | HCX | Internal HTTP API for vector search + LLM Q&A |
| Center API | `aimod/center_api/` | 0.0.0.0:5003 | TC | Centralized data service for multi-host sharing |
| Vectorization Engine | `joplinai.py` | — | TC | 分块编排 / AI增强 / 写 ChromaDB / 报告。joplinai-sync.timer 每8小时触发 |
| Embedding Inference | Ollama | 11434 | HCX | 嵌入向量生成 (`dengcao/bge-large-zh-v1.5`)，TC 经公网 IP 远程调用 |

Center API 的 gunicorn 入口：`"aimod.center_api:create_app()"`（app factory 模式）。
Web App 入口：`joplin_web_app:app`（模块级 `app = create_app()`，勿放 `if __name__` 内——gunicorn 不执行该块）。

Data flow: `joplinai.py` → TC 本地 (Joplin Server / ChromaDB / center_api) + 远程 HCX (Ollama embedding / 11434) + 外部 Cloud API (AI增强, 默认 DeepSeek)。各 `aimod/*_client.py` 通过 HTTP + API Key → `aimod/center_api/` → `data/joplinai_center.db`

向量化优化：chunk 内容未变仅元数据（tags/summary）变更时，跳过嵌入生成，仅通过 `VectorDBManager.update_chunk_metadata()` 更新 ChromaDB metadata（不含 embeddings），避免不必要的远程序列调用。

`joplinai_center.db` 包含 10 张表：
- 缓存与历史：`enhance_cache`、`probe_cache`、`notebook_history`、`global_run_history`
- 笔记状态：`note_process_state`（含 `enhance_config` 字段追踪摘要/标签模型，模型变更时自动触发重处理；`meta_hash` 仅含标签+笔记本元数据，不含增强配置）
- 笔记本级增强策略覆盖：`enhance_override` 云端 JSON 配置，格式 `{"笔记本标题": {"summary_model": "cloud|ollama|none", "tags_model": "cloud|ollama|none"}}`，`_resolve_enhance_config()` 合并全局配置与笔记本级覆盖
- 用户管理：`users`、`sessions`、`audit_log`、`qa_history`、`chat_sessions`

Center client 拆分为 5 个独立文件 (`aimod/`): `CacheClient`（纯远程，无本地回退），`ProbeCacheClient`（远程优先+本地回退），`HistoryClient`（远程优先+本地回退），`ProcessStateClient`（纯远程），`UserManagerClient`（远程优先+本地回退）。URL 发现逻辑：云端 `joplinai_center_url` 未配置则本机为生产主机走 `127.0.0.1:5003`。

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
│   └── api.py              # 内部 API
├── __init__.py
tests/                  # 测试目录
joplinai.py             # 向量化 CLI 入口 + joplinai.ipynb
joplin_qa_api.py        # Q&A API 服务入口 + joplin_qa_api.ipynb
joplin_web_app.py       # Web 门户入口  + joplin_web_app.ipynb
pathmagic.py            # 根路径上下文  + pathmagic.ipynb
deploy/                 # systemd service/timer 文件 + deploy.sh
aimod/                  # AI 核心包
├── __init__.py             # get_logger(name) 统一日志工厂
├── embedding_generator.py  # 文本分块 + 嵌入生成
├── chunk_optimizer.py      # 自适应分块策略（拆分自 embedding_generator）
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
│   └── user_routes.py      # /users/* /auth/* 端点
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
- `chunk_optimizer.py` — 自适应分块策略，探测最优块大小
- `text_splitter.py` — 文本切分器，按句子边界切分
- `vector_db_manager.py` — VectorDBManager 类，ChromaDB CRUD
- `cache_manager.py` — SQLite LRU 缓存（本地回退用）
- `note_enhancer.py` — AI增强 摘要/标签增强（cloud/local/none），含 Ollama vision 描述
- `run_tracker.py` — RunTracker 类，运行数据采集和历史记录 (remote-first)
- `cache_client.py` — CacheClient，增强缓存纯远程（无本地回退）
- `history_client.py` — HistoryClient，运行历史远程存储
- `probe_client.py` — ProbeCacheClient，探测缓存远程存储
- `state_client.py` — ProcessStateClient，笔记处理状态远程存储（纯远程，无本地 fallback）
- `user_client.py` — UserManagerClient，用户管理远程存储
- `report_writer.py` — 统一报告生成（在 `src/`）
- `center_api/` — 数据中心 Flask 包（5 个端点蓝图）
- `__init__.py` — `get_logger(name)` 统一日志工厂，自动继承 func/logme 的 handler/level

### Utility Submodule (`func/`)

Proper git submodule registered in `.gitmodules` pointing to `heart5/func`. Clone with `git submodule update --init` to fetch. Key modules:
- `jpfuncs.py` — Joplin API wrapper (CRUD for notes, notebooks, tags)
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

Production: systemd services managed via `deploy/deploy.sh`:
```bash
./deploy/deploy.sh hcx           # 恒创云：重启 QA_API + Web_App
./deploy/deploy.sh tc            # 腾讯云：git push → git pull(直连→代理→rsync兜底) → 重启 center_api
./deploy/deploy.sh hcx --dry-run # 仅预览
```

## Key Code Patterns

- **`pathmagic.Context()`**: All modules use `with pathmagic.Context():` to ensure both the project root and `src/` are on `sys.path` before importing project-local modules. Always wrap project imports in this context manager.
- **Jupytext paired notebooks**：编辑 `.py` 源文件后，jupytext 自动生成/更新同目录同名 `.ipynb` 供 Jupyter Notebook 阅览。`.py` → `ipynb`（percent 格式）。`.ipynb` 永不入库。配置：`jupytext.toml`（默认 `formats = "ipynb,py:percent"`）+ `.git/hooks/pre-commit`（staged .py/.md 自动 `jupytext --sync`）。手动同步：`jupytext --sync <file>`。
  - **重要 `.md` 文档同样关联 `.ipynb`**：`CLAUDE.md`、`README.md`、`docs/CHANGELOG.md`、`docs/TECHNICAL_MANUAL.md` 通过 YAML frontmatter 声明 `formats: ipynb,md`。内容变更后必须同步 ipynb（pre-commit hook 自动执行）。
  - `# %%` 标记代码 cell，`# %% [markdown]` 标记 markdown cell。**关键规则**：markdown cell 后的下一段代码前必须有显式 `# %%`，否则 jupytext sync 会把代码当作 markdown 注释掉。
      - **自动检测**：`tools/check_jupytext_comment.py` 扫描被误注释的 Flask route/endpoint。pre-commit hook（`.git/hooks/pre-commit`）在 jupytext sync 后自动执行，CI 兜底。
- **`__all__`**: 所有 `aimod/`、`src/`、`aimod/center_api/` 的公开模块都有 `__all__` 明确导出列表。
- **`__repr__`**: 5 个关键数据类有 `__repr__`：`CacheResult`, `VectorDBManager`, `EmbeddingGenerator`, `RunTracker`, `QASystem`。
- **Type annotations**: 所有公开函数有返回类型标注 (`-> None`, `-> str`, `-> Optional[str]` 等)。
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
| `minicpm-v:latest` | 5.5 GB | 笔记图片描述（`vision_enabled=False` 默认关闭） | ~数分钟/图 |

- **`qwen2.5:1.5b`** 通过 `enhance_ollama_chat_model` 配置键指定，供向量化管线离线批量任务使用
- **无本地 QA 回退**：`qa_ollama_chat_model=none`，Cloud API 不可用时直接返回服务不可用，不尝试本地模型
- 之前使用后删除的模型：`qwen:1.8b`（架构旧）、`qwen3:8b`（12分钟太慢）、`deepseek-r1:7b`（RAG误判）、`gemma3:4b-it-qat`（中文幻觉）

### 云端模型（OpenAI-compatible API，默认 DeepSeek）

| 模型 | 用途 | 速度 |
|------|------|------|
| `deepseek-chat` | QA 对话主路（`cloud_model=deepseek-chat`） | ~5秒/次 |
| `deepseek-v4-pro` | 图片精细描述（备用） | ~数秒/图 |

可通过 `cloud_api_url` + `cloud_api_key` + `cloud_model` 切换至 Qwen/ChatGPT 等兼容提供者。

### 配置要点

| 配置键 | 当前值 | 说明 |
|--------|--------|------|
| `qa_ollama_chat_model` | `none` | 无本地 QA 回退 |
| `cloud_model` | `deepseek-chat` | 云端大模型 |
| `cloud_api_url` | `https://api.deepseek.com/v1/chat/completions` | 云端 API 端点 |
| `cloud_api_key` | 回退 `deepseek_token` | 云端 API Key |
| `summary_model` | `cloud` | 摘要模型: cloud/ollama/none |
| `tags_model` | `cloud` | 标签模型: cloud/ollama/none |
| `enhance_ollama_chat_model` | `qwen2.5:1.5b` | Ollama 标签/分类模型 |
| `vision_enabled` | `false` | CPU 跑视觉太慢，默认关闭 |

## Testing & CI

```bash
python -m pytest tests/ -v --tb=short   # 运行测试
flake8 . --max-line-length=100 --ignore=E402,W503,E203,E501 --exclude=func/  # lint
```

CI (`.github/workflows/ci.yml`): push/PR 触发，`check_jupytext_comment.py` → flake8 lint → pytest (Python 3.10)。
Pre-commit (`.pre-commit-config.yaml`): jupytext 误注释检测 + flake8。

## Known Issues & Technical Debt

- **`static/favicon.ico*`**: 3 domain-specific favicon copies at 3.5MB each. Only the default `favicon.ico` should be tracked; the `_for_*` variants are in `.gitignore`.
- **TC git pull 可能需要代理**: 直连偶尔失败（GnuTLS recv error），回退方案是 `HTTPS_PROXY=http://127.0.0.1:7890 git pull`。`deploy/deploy.sh` 已包含三级回退策略。
- **HCX `joplin-qa-api`/`joplin-web-app` 代码更新后需手动重启**: gunicorn 进程不自动 reload，git pull 后必须 `sudo systemctl restart`。
- **SSH 远程 systemctl restart 会 hang**: `systemctl restart` 默认等待服务停止→启动，SSH 会话会阻塞超时。远程重启必须加 `--no-block`：`ssh tc "sudo systemctl restart --no-block <svc>"`。本地 `deploy.sh` 已内置此参数。
- **TC 配置更新流程**：修改云端 INI → `ssh tc "source /usr/miniconda3/etc/profile.d/conda.sh && conda activate newlsp && joplin sync"` → `ssh tc "sudo systemctl restart --no-block joplinai-sync"`。joplinai-sync 是 timer 触发的 oneshot 服务，需手动 start/restart。
- **center_api 日志查看**：TC 上 `sudo journalctl -u joplinai-center-api -f`。logger 配置在 `aimod/center_api/__init__.py`：`propagate=False` 避免 `func/logme` root handler 双重输出。
- **ChromaDB 在 TC 以 Docker 运行**：`docker run chromadb/chroma`，端口映射 8009→8000。配置中 `chroma_port=8009`，但 `vector_db_manager.py` 硬编码默认 8000——依赖云端配置覆盖。直接 `chromadb.HttpClient(host='127.0.0.1', port=8009)` 才能连上。
- **Ollama 在 HCX 以公网 IP 暴露**：TC 向量化通过 `149.30.242.156:11434` 远程调 HCX 的 Ollama 生成嵌入。端口 11434 需对外网开放。

## Git

- Branch: `main`
- Remote: `origin` (GitHub: `heart5/joplinai`)
- `.gitignore` covers: `log/`, `data/`, `*.ipynb`（全部屏蔽，永不入库——ipynb 由 jupytext 从 .py 自动生成），`__pycache__/`, backup files (`*.bak`), oversized favicon copies, `.env`
- `jupytext.toml`: py↔ipynb 配对配置，`.git/hooks/pre-commit`: 提交 .py 时自动同步 ipynb
- `.pre-commit-config.yaml`: flake8 pre-commit hook

## Configuration

Main config stored in cloud-synced Joplin note (INI format). Local override: `data/joplinai.ini`. Key settings: Joplin API token, Ollama model name, embedding model, ChromaDB path, Q&A prompts, user session settings.

**数据中心配置**：`joplinai_center_url`（非生产主机配，指向 TC 公网IP；生产主机不配则自动走 localhost）、`joplinai_center_api_key`（认证密钥）、`probe_cache_limit`（探测缓存上限，默认 10000）。

**模型配置**：`qa_ollama_chat_model`（本地 QA 对话模型，当前 `none` 无本地回退）、`enhance_ollama_chat_model`（Ollama 标签/分类小模型）、`ollama_embedding_model`（嵌入模型）、`vision_model`（视觉模型）、`cloud_model` / `summary_model` / `tags_model`（cloud/ollama/none 三态切换）、`cloud_api_url` / `cloud_api_key`（云端 API 端点/密钥，支持切换提供者）。详见上方 Model Strategy 章节。

### Remote Joplin fallback

When local Joplin server is unavailable, `jpfuncs.getapi()` can fall back to a remote Joplin server. Configure in `data/joplinai.ini`:

```ini
[joplin]
fallback_url = http://<remote-host>:41184
fallback_token = <remote-api-token>
```

This is useful in multi-server deployments for resilience — if one machine's Joplin process is down, joplinai can still function by hitting the other server's Joplin API.

## Dependencies (no requirements.txt — install manually)

Core: `flask`, `chromadb`, `ollama`, `requests`, `jinja2`, `werkzeug`
Dev: `flake8` (lint), `pytest` (test)
