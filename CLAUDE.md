# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project Overview

Joplinai is an AI-powered knowledge retrieval and Q&A system for [Joplin](https://joplinapp.org/) notes. It vectorizes Joplin notes into ChromaDB, enables semantic search, and provides a multi-user web chat interface with admin management.

## Architecture — Three Independent Services

All services are Flask apps run directly with `python <file>`. No Docker/containerization.

| Service | File | Port | Purpose |
|---------|------|------|---------|
| Web Portal | `joplin_web_app.py` | 127.0.0.1:5001 | User login, Q&A chat UI, admin panel |
| Q&A API | `src/joplin_qa_api.py` | dynamic (from config) | Internal HTTP API for vector search + LLM Q&A |
| Vectorization CLI | `src/joplinai.py` | — | Chunks Joplin notes, generates embeddings, stores in ChromaDB |

Data flow: Browser → `joplin_web_app.py` (Flask sessions) → HTTP (API key auth) → `src/joplin_qa_api.py` → ChromaDB + Ollama

### Source Layout

```
src/              # .py source files (jupytext paired with notebooks/)
├── config_manager.py
├── joplinai.py
├── joplin_qa_api.py
├── queryanswer.py
├── joplin_web_app.py
├── user_manager.py
└── pathmagic.py
notebooks/        # .ipynb files (paired via jupytext.toml)
aimod/            # AI core modules (embedding, vector DB, cache, etc.)
func/             # Utility submodule (heart5/func)
static/           # Frontend assets
templates/        # Jinja2 templates
data/             # Runtime data (gitignored)
log/              # Logs (gitignored)
```

### Core Modules (`aimod/`)

- `embedding_generator.py` — Text chunking + embedding via Ollama models
- `vector_db_manager.py` — ChromaDB CRUD operations
- `cache_manager.py` — SQLite-based LRU cache for AI calls
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
python src/joplinai.py

# 2. Start Q&A API middleware (starts on configured port)
python src/joplin_qa_api.py

# 3. Start web portal
python joplin_web_app.py
```

## Key Code Patterns

- **`pathmagic.context()`**: All modules use `with pathmagic.context():` to ensure both the project root and `src/` are on `sys.path` before importing project-local modules. Always wrap project imports in this context manager.
- **Jupytext paired notebooks**: `.py` files in `src/` are paired with `.ipynb` files in `notebooks/` via `jupytext.toml`. Edits to the `.py` file are the source of truth. Current jupytext version: 1.19.1. To sync: `jupytext --sync jupytext.toml`.
- **Cloud config**: Configuration is fetched dynamically via `getinivaluefromcloud()` from an INI stored in a Joplin note. The `ConfigManager` singleton in `src/config_manager.py` handles hot-reloading (5-minute check interval).
- **Inter-service auth**: `joplin_web_app.py` calls `joplin_qa_api.py` using an API key from the shared cloud config (`X-API-Key` header).
- **No tests directory**: No formal test framework. Test-adjacent files are scratchpad notebooks.

## Known Issues & Technical Debt

- **`config_manager.py` `_generate_change_summary` bug**: The `old_config` parameter is not used — the function compares `new_config` against itself instead of the previous snapshot. Always returns empty change summary.
- **`_qa_system_instances` memory leak**: Global dict in `joplin_qa_api.py` accumulates instances per session_id with no eviction policy. Long-running deployments should add TTL-based cleanup.
- **`static/favicon.ico*`**: 3 domain-specific favicon copies at 3.5MB each. Only the default `favicon.ico` should be tracked; the `_for_*` variants are in `.gitignore`.

## Git

- Branch: `main`
- Remote: `origin` (GitHub: `heart5/joplinai`)
- `.gitignore` covers: `log/`, `data/`, `*.ipynb` (except `notebooks/*.ipynb`), `__pycache__/`, debug scripts (`test_qwen.py`), backup files (`*.bak`), oversized favicon copies

## Configuration

Main config stored in cloud-synced Joplin note (INI format). Local override: `data/joplinai.ini`. Key settings: Joplin API token, Ollama model name, embedding model, ChromaDB path, Q&A prompts, user session settings.

## Dependencies (no requirements.txt — install manually)

Core: `flask`, `chromadb`, `ollama`, `requests`, `jinja2`, `werkzeug`
