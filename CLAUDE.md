# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

Joplinai is an AI-powered knowledge retrieval and Q&A system for [Joplin](https://joplinapp.org/) notes. It vectorizes Joplin notes into ChromaDB, enables semantic search, and provides a multi-user web chat interface with admin management.

## Deployment — Two Servers

| Server | Hostname | Role |
|--------|----------|------|
| 腾讯云 (TC) | `tc` (122.51.102.233:2202) | Joplin Server, 向量数据库 (ChromaDB), 数据中心 (center_api), 用户管理 |
| 恒创云 (HCX) | long9.org | Ollama LLM 推理, Q&A API, Web 门户 |

TC 是数据核心，HCX 是推理+门户。HCX 的 Web 门户通过 HTTP API Key 认证远程调用 TC 的 center_api 和 Q&A API。

## Architecture — Four Independent Services

All services are Flask apps。No Docker/containerization。

| Service | File | Port | Host | Purpose |
|---------|------|------|------|---------|
| Web Portal | `joplin_web_app.py` | 127.0.0.1:5001 | HCX | User login, Q&A chat UI, admin panel |
| Q&A API | `joplin_qa_api.py` | 127.0.0.1:5000 | HCX | Internal HTTP API for vector search + LLM Q&A |
| Center API | `joplinai_center_api.py` | 0.0.0.0:5003 | TC | Centralized data service for multi-host sharing |
| Vectorization CLI | `joplinai.py` | — | HCX | Chunks Joplin notes, generates embeddings, stores in ChromaDB |

Data flow: `joplinai.py` → `deepseek_enhancer.py` / `embedding_generator.py` / `run_tracker.py` / `report_writer.py` → `DeepSeekCacheClient` / `ProbeCacheClient` / `HistoryClient` / `ProcessStateClient` (HTTP + API key auth) → `joplinai_center_api.py` → `data/joplinai_center.db`

`joplinai_center.db` 包含 10 张表：
- 缓存与历史：`deepseek_cache`、`probe_cache`、`notebook_history`、`global_run_history`
- 笔记状态：`note_process_state`
- 用户管理：`users`、`sessions`、`audit_log`、`qa_history`、`chat_sessions`

Center client (`aimod/center_client.py`) provides `DeepSeekCacheClient`, `ProbeCacheClient`, `HistoryClient`, `ProcessStateClient`, `UserManagerClient` — all remote-first with local fallback. URL 发现逻辑：云端 `joplinai_center_url` 未配置则本机为生产主机走 `127.0.0.1:5003`。

### Source Layout

```
src/              # .py source files (jupytext paired with .ipynb in same dir)
├── config_manager.py    + config_manager.ipynb
├── queryanswer.py       + queryanswer.ipynb
├── user_manager.py      + user_manager.ipynb
├── pathmagic.py         + pathmagic.ipynb
└── report_writer.py     # Unified report module
joplinai.py            # Vectorization CLI     + joplinai.ipynb
joplin_qa_api.py       # Q&A API service      + joplin_qa_api.ipynb
joplin_web_app.py      # Web portal           + joplin_web_app.ipynb
joplinai_center_api.py # Center API service   + joplinai_center_api.ipynb
pathmagic.py           # Root path context    + pathmagic.ipynb
deploy/                # systemd service files (TC: center-api; HCX: web-app, qa-api)
aimod/                 # AI core modules
├── center_client.py         # 数据中心客户端 (DeepSeek + Probe + History)
func/                 # Utility submodule (heart5/func)
static/               # Frontend assets
templates/            # Jinja2 templates
data/                 # Runtime data (gitignored)
log/                  # Logs (gitignored)
```

### Core Modules (`aimod/`)

- `embedding_generator.py` — Text chunking + embedding via Ollama models
- `vector_db_manager.py` — ChromaDB CRUD operations
- `cache_manager.py` — SQLite-based LRU cache for AI calls (本地回退用)
- `center_client.py` — 数据中心客户端 (`DeepSeekCacheClient` + `ProbeCacheClient` + `HistoryClient` + `ProcessStateClient` + `UserManagerClient`), 全部 remote-first + local fallback
- `deepseek_enhancer.py` — Optional DeepSeek API for enhanced summaries/tags
- `run_tracker.py` — Vectorization run data collector and history recorder (remote-first, local fallback)
- `report_writer.py` — Unified report generation from center_api stats endpoints (in `src/`)

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

## Key Code Patterns

- **`pathmagic.context()`**: All modules use `with pathmagic.context():` to ensure both the project root and `src/` are on `sys.path` before importing project-local modules. Always wrap project imports in this context manager.
- **Jupytext paired notebooks**：编辑 `.py`/`.md` 源文件后，jupytext 自动生成/更新同目录同名 `.ipynb` 供 Jupyter Notebook 阅览。`.py` → `ipynb`（percent 格式）、`.md` → `ipynb`（markdown 格式），两种独立配对互不干扰。`.ipynb` 永不入库。配置：`jupytext.toml`（默认 `formats = "ipynb,py:percent"`）+ `.git/hooks/pre-commit`（staged .py/.md 自动 `jupytext --sync`）。手动同步：`jupytext --sync <file>`。
- **Cloud config**: Configuration is fetched dynamically via `getinivaluefromcloud()` from an INI stored in a Joplin note. The `ConfigManager` singleton in `src/config_manager.py` handles hot-reloading (5-minute check interval). Key cloud items: `imp_nbs` (notebook titles), `imp_note_ids` (specific note IDs as virtual notebook), both comma-separated.
- **Inter-service auth**: `joplin_web_app.py` calls `joplin_qa_api.py` using an API key from the shared cloud config (`X-API-Key` header).
- **No tests directory**: No formal test framework. Test-adjacent files are scratchpad notebooks.

## Known Issues & Technical Debt

- **`static/favicon.ico*`**: 3 domain-specific favicon copies at 3.5MB each. Only the default `favicon.ico` should be tracked; the `_for_*` variants are in `.gitignore`.

## Git

- Branch: `main`
- Remote: `origin` (GitHub: `heart5/joplinai`)
- `.gitignore` covers: `log/`, `data/`, `*.ipynb`（全部屏蔽，永不入库——ipynb 由 jupytext 从 .py 自动生成），`__pycache__/`, backup files (`*.bak`), oversized favicon copies
- `jupytext.toml`: py↔ipynb 配对配置，`.git/hooks/pre-commit`: 提交 .py 时自动同步 ipynb

## Configuration

Main config stored in cloud-synced Joplin note (INI format). Local override: `data/joplinai.ini`. Key settings: Joplin API token, Ollama model name, embedding model, ChromaDB path, Q&A prompts, user session settings.

**数据中心配置**：`joplinai_center_url`（非生产主机配，指向 TC 公网IP；生产主机不配则自动走 localhost）、`joplinai_center_api_key`（认证密钥）、`probe_cache_limit`（探测缓存上限，默认 10000）。

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
