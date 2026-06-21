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
| 恒创云 (HCX) | long9.org (149.30.242.156) | Joplin Server (Docker, `joplin.qingxd.com`), Ollama LLM 推理 (端口 11434), Q&A API, Web 门户 |

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
- 缓存与历史：`enhance_cache`（缓存键 `{content_hash}_{task}`，`summary`/`tags`/`vision_desc` 三类 task）、`notebook_history`、`global_run_history`
- 笔记状态：`note_process_state`（含 `enhance_config` 字段追踪摘要/标签模型，模型变更时自动触发重处理；`meta_hash` 仅含标签+笔记本元数据，不含增强配置）
- 笔记本级增强策略覆盖：`enhance_override` 云端 JSON 配置，格式 `{"笔记本标题": {"summary_model": "cloud|ollama|none", "tags_model": "cloud|ollama|none"}}`，`_resolve_enhance_config()` 合并全局配置与笔记本级覆盖
- 用户管理：`users`、`sessions`、`audit_log`、`qa_history`、`chat_sessions`

Center client 拆分为 4 个独立文件 (`aimod/`): `CacheClient`（纯远程，无本地回退），`HistoryClient`（远程优先+本地回退），`ProcessStateClient`（纯远程），`UserManagerClient`（远程优先+本地回退）。非 center_api 模块（`report_writer`、`user_manager`）通过设备 ID 判断本机是否数据中心：比对 `center_host_deviceid` → 匹配则直连 `127.0.0.1:5003`，否则走云端配置 `joplinai_center_url`；URL 为空时回退 localhost。

### Source Layout

```
src/            # 核心模块 (config_manager, user_manager, qa_system, web_app/)
src/web_app/    # Web 门户 (routes, auth, admin, api, dashboard_routes)
aimod/          # AI 引擎 (embedding, vector_db, note_enhancer, center_api/)
aimod/center_api/ # 数据中心 API (6 蓝图: cache, history, state, user, monitor, system)
func/           # 工具子模块 (git submodule heart5/func)
templates/      # Jinja2 模板
static/         # 前端静态资源
data/ log/      # 运行时数据 / 日志 (gitignored)
```
Top-level 入口: `joplinai.py` (向量化), `joplin_qa_api.py` (Q&A API), `joplin_web_app.py` (Web 门户).
各模块通过 `.py` ↔ `.ipynb` jupytext 配对。详见 `docs/TECHNICAL_MANUAL.md`.

### Core Modules (`aimod/`)

向量化管线：`embedding_generator.py`（入口）→ `text_splitter.py`（切分）→ `vector_db_manager.py`（ChromaDB CRUD）。AI增强：`note_enhancer.py`（摘要/标签/视觉描述，cloud/local/none）。预处理：`image_processor.py`（>1MB压缩，>8MB跳过）、`text_preprocessor.py`（图片→描述、表格→NL）。远程存储 client：`cache_client.py`、`history_client.py`、`state_client.py`、`user_client.py`。`center_api/` 包含 6 个蓝图（cache/history/state/user/monitor/system）。`__init__.py` → `get_logger(name)` 统一日志工厂。

### Utility Submodule (`func/`)

Git submodule `heart5/func`. Clone: `git submodule update --init`. Key: `jpfuncs.py` (Joplin CRUD, `_validate_uuid()` 校验 parent_id), `configpr.py` (INI reader), `first.py` (root detection), `logme.py` (logging), `datatools.py` (hashing).

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
  - **重要 `.md` 文档同样关联 `.ipynb`**：`CLAUDE.md`、`README.md`、`docs/CHANGELOG.md`、`docs/TECHNICAL_MANUAL.md`、`docs/QA_SYSTEM.md`、`docs/CHUNK_EMBED_SYSTEM.md` 通过 YAML frontmatter 声明 `formats: ipynb,md`。内容变更后必须同步 ipynb（pre-commit hook 自动执行）。
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
- **ConfigManager 监控键必须匹配云 INI 真实键名**：`monitored_keys` 列表中的键名直接传给 `getinivaluefromcloud()` 查云 INI。若代码重命名了 CONFIG 键但云 INI 键名未改，`_fetch_config_from_cloud` 会返回 `None`，被 `update()` 覆盖掉 `qa_config.py` 的默认值。解决方案：`config.get("key") or default` 防御 None（`.get("key", default)` 只在键缺失时生效，键存在但值为 None 时返回 None）。
- **QA 上下文智能截断**：`_build_optimized_context_from_chunks` 超限时保留头（sys_prompt）尾（问题+历史+指令），只压缩中间笔记内容，确保问题永不被截断。
- **jieba 预加载**：`qa_system.py` 模块导入时执行 `jieba.initialize()`，将词典加载提前到 gunicorn worker 启动阶段，避免首次问答现场加载耗时 5 秒+。

## Model Strategy

核心原则：实时走云端，离线走本地。HCX 无 GPU（纯 CPU 8核）。QA 主路 `deepseek-v4-flash`，增强同模型。嵌入走硅基流动云端（`BAAI/bge-large-zh-v1.5`），回落本地 Ollama。本地 Ollama 模型 `qwen2.5:1.5b` 供离线标签/分类。`qa_ollama_chat_model=none` 无本地 QA 回退。

完整模型表和配置键表：见 `docs/MODEL_STRATEGY.md`。

## Production Incidents

详见 `docs/INCIDENTS.md`。两条硬规则：

1. **远程状态加载不得静默降级**：center_api client 连接失败时抛 `CenterAPIUnreachableError`，禁止返回空/默认值。宁可不跑，不能错跑。
2. **状态持久化不得全量替换**：`note_process_state` 用 upsert（`INSERT ON CONFLICT DO UPDATE`），禁止 DELETE ALL + 重新 INSERT。

## Testing & CI

```bash
python -m pytest tests/ -v --tb=short   # 运行测试
flake8 . --max-line-length=100 --ignore=E402,W503,E203,E501 --exclude=func/  # lint
```

CI (`.github/workflows/ci.yml`): push/PR 触发，`check_jupytext_comment.py` → flake8 lint → pytest (Python 3.10)。
Pre-commit (`.pre-commit-config.yaml`): jupytext 误注释检测 + flake8。

## Known Issues & Technical Debt


- **HCX 服务更新后需手动重启**: gunicorn 不自动 reload，git pull 后必须 `sudo systemctl restart joplinai-qa-api joplinai-web-app`。
- **git push 后不自动同步 TC**: 安全考量，TC 需手动 `git pull --ff-only` + `systemctl restart --no-block`。
- **TC 配置更新流程**：修改云端 INI → `ssh tc "source /usr/miniconda3/etc/profile.d/conda.sh && conda activate newlsp && joplin sync"` → `ssh tc "sudo systemctl restart --no-block joplinai-sync"`。
- **center_api 日志查看**：TC 上 `sudo journalctl -u joplinai-center-api -f`。`aimod/center_api/__init__.py` 中 `propagate=False` 避免双重输出。
- **ChromaDB metadata 键对**：含 `has_images`（`meta_hash` 不包含）、`estimated_date`、`chunk_summary`、`tags`、`meta_hash`。
- **Ollama 在 HCX 以公网暴露**：TC 向量化通过 `149.30.242.156:11434` 调用 HCX Ollama。

## Git

- Branch: `main`
- Remote: `origin` (GitHub: `heart5/joplinai`)
- `.gitignore` covers: `.claude/`, `log/`, `data/`, `*.ipynb`（全部屏蔽，永不入库——ipynb 由 jupytext 从 .py 自动生成），`__pycache__/`, backup files (`*.bak`), oversized favicon copies, `.env`
- `jupytext.toml`: py↔ipynb 配对配置，`.git/hooks/pre-commit`: 提交 .py 时自动同步 ipynb
- `.pre-commit-config.yaml`: flake8 pre-commit hook

## Configuration

Main config in cloud-synced Joplin note (INI format). Local override: `data/joplinai.ini`. Full model config: see `docs/MODEL_STRATEGY.md`. QA 检索链路: see `docs/QA_SYSTEM.md`.

**数据中心配置**：`center_host_deviceid`（主机设备 ID，客户端比对本机 deviceid 判断是否直连 localhost）、`joplinai_center_url`（非数据中心配）、`joplinai_center_api_key`（认证密钥）。

**Joplin 连接** (`data/joplinai.ini` `[joplin]`)：`local_server=true` 时直连 localhost:41184，不通走 `fallback_url`。全部失败抛 `JoplinUnreachableError`（非 `exit(1)`）。

反代端点：HCX Docker `https://joplin.qingxd.com`，TC CLI `https://joplin.xiloong.fans`。

## Dependencies (no requirements.txt — install manually)

Core: `flask`, `chromadb`, `ollama`, `requests`, `jinja2`, `werkzeug`
Dev: `flake8` (lint), `pytest` (test)
