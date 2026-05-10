# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

Joplinai is an AI-powered knowledge retrieval and Q&A system for [Joplin](https://joplinapp.org/) notes. It vectorizes Joplin notes into ChromaDB, enables semantic search, and provides a multi-user web chat interface with admin management.

## Architecture — Four Independent Services

All services are Flask apps run directly with `python <file>`. No Docker/containerization.

| Service | File | Port | Purpose |
|---------|------|------|---------|
| Web Portal | `joplin_web_app.py` | 127.0.0.1:5001 | User login, Q&A chat UI, admin panel |
| Q&A API | `joplin_qa_api.py` | dynamic (from config) | Internal HTTP API for vector search + LLM Q&A |
| DeepSeek Cache API | `deepseek_cache_api.py` | 127.0.0.1:5003 | Centralized DeepSeek summary/tags cache for multi-host vectorization |
| Vectorization CLI | `joplinai.py` | — | Chunks Joplin notes, generates embeddings, stores in ChromaDB |

Data flow: `joplinai.py` → `deepseek_enhancer.py` → `RemoteCacheClient` (HTTP + API key auth) → `deepseek_cache_api.py` → SQLite

Cache client (`aimod/deepseek_cache_client.py`) uses remote-first with local SQLite fallback when the remote service is unreachable or not configured.

### Source Layout

```
src/              # .py source files (jupytext paired with .ipynb in same dir)
├── config_manager.py    + config_manager.ipynb
├── queryanswer.py       + queryanswer.ipynb
├── user_manager.py      + user_manager.ipynb
└── pathmagic.py         + pathmagic.ipynb
joplinai.py            # Vectorization CLI     + joplinai.ipynb
joplin_qa_api.py       # Q&A API service      + joplin_qa_api.ipynb
joplin_web_app.py      # Web portal           + joplin_web_app.ipynb
deepseek_cache_api.py  # Cache API service
pathmagic.py           # Root path context    + pathmagic.ipynb
deploy/                # systemd service files
aimod/                 # AI core modules
├── deepseek_cache_client.py  # Remote cache client with local fallback
func/                 # Utility submodule (heart5/func)
static/               # Frontend assets
templates/            # Jinja2 templates
data/                 # Runtime data (gitignored)
log/                  # Logs (gitignored)
```

### Core Modules (`aimod/`)

- `embedding_generator.py` — Text chunking + embedding via Ollama models
- `vector_db_manager.py` — ChromaDB CRUD operations
- `cache_manager.py` — SQLite-based LRU cache for AI calls
- `deepseek_cache_client.py` — Remote cache client with local SQLite fallback
- `deepseek_enhancer.py` — Optional DeepSeek API for enhanced summaries/tags
- `aitaskreporter.py` — Vectorization run reports and trend analytics

### Utility Submodule (`func/`)

Proper git submodule registered in `.gitmodules` pointing to `heart5/func`. Clone with `git submodule update --init` to fetch. Key modules:
- `jpfuncs.py` — Joplin API wrapper (CRUD for notes, notebooks, tags)
- `configpr.py` — INI config reader
- `first.py` — Project root detection (looks for `rootfile` marker)
- `logme.py` — Logging setup
- `datatools.py` — Content hashing, cloud key retrieval

### User System (`src/user_manager.py`)

SQLite-based (`data/joplinai_users.db`). Three roles: `admin`, `team_leader`, `team_member`. Notebook-level access control via `allowed_notebooks` JSON field.

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
- **Jupytext paired notebooks**: `.py` files are paired with `.ipynb` files in the same directory. Edits to the `.py` file are the source of truth. To sync: `jupytext --sync <file>.py`.
- **Cloud config**: Configuration is fetched dynamically via `getinivaluefromcloud()` from an INI stored in a Joplin note. The `ConfigManager` singleton in `src/config_manager.py` handles hot-reloading (5-minute check interval). Key cloud items: `imp_nbs` (notebook titles), `imp_note_ids` (specific note IDs as virtual notebook), both comma-separated.
- **Inter-service auth**: `joplin_web_app.py` calls `joplin_qa_api.py` using an API key from the shared cloud config (`X-API-Key` header).
- **No tests directory**: No formal test framework. Test-adjacent files are scratchpad notebooks.

## Known Issues & Technical Debt

- **`static/favicon.ico*`**: 3 domain-specific favicon copies at 3.5MB each. Only the default `favicon.ico` should be tracked; the `_for_*` variants are in `.gitignore`.

## Git

- Branch: `main`
- Remote: `origin` (GitHub: `heart5/joplinai`)
- `.gitignore` covers: `log/`, `data/`, `*.ipynb` (except `/*.ipynb` and `src/*.ipynb`), `__pycache__/`, debug scripts (`test_qwen.py`), backup files (`*.bak`), oversized favicon copies

## Configuration

Main config stored in cloud-synced Joplin note (INI format). Local override: `data/joplinai.ini`. Key settings: Joplin API token, Ollama model name, embedding model, ChromaDB path, Q&A prompts, user session settings.

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
