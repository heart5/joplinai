# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Joplinai is an AI-powered knowledge retrieval and Q&A system for [Joplin](https://joplinapp.org/) notes. It vectorizes Joplin notes into ChromaDB, enables semantic search, and provides a multi-user web chat interface with admin management.

## Architecture — Three Independent Services

All services are Flask apps run directly with `python <file>`. No Docker/containerization.

| Service | File | Port | Purpose |
|---------|------|------|---------|
| Web Portal | `web_app.py` | 127.0.0.1:5001 | User login, Q&A chat UI, admin panel |
| Q&A API | `joplin_qa_api.py` | dynamic (from config) | Internal HTTP API for vector search + LLM Q&A |
| Vectorization CLI | `joplinai.py` | — | Chunks Joplin notes, generates embeddings, stores in ChromaDB |

Data flow: Browser → `web_app.py` (Flask sessions) → HTTP (API key auth) → `joplin_qa_api.py` → ChromaDB + Ollama

### Core Modules (`aimod/`)

- `embedding_generator.py` — Text chunking + embedding via Ollama models
- `vector_db_manager.py` — ChromaDB CRUD operations
- `cache_manager.py` — SQLite-based LRU cache for AI calls
- `deepseek_enhancer.py` — Optional DeepSeek API for enhanced summaries/tags
- `aitaskreporter.py` — Vectorization run reports and trend analytics

### Utility Submodule (`func/`)

Separate git submodule (`heart5/func`). Key modules:
- `jpfuncs.py` — Joplin API wrapper (CRUD for notes, notebooks, tags)
- `configpr.py` — INI config reader
- `first.py` — Project root detection (looks for `rootfile` marker)
- `logme.py` — Logging setup
- `datatools.py` — Content hashing, cloud key retrieval

### User System (`user_manager.py`)

SQLite-based (`data/joplinai_users.db`). Three roles: `admin`, `team_leader`, `team_member`. Notebook-level access control via `allowed_notebooks` JSON field.

## How to Run

```bash
# 1. Vectorize notes (CLI, run first to populate ChromaDB)
python joplinai.py

# 2. Start Q&A API middleware (starts on configured port)
python joplin_qa_api.py

# 3. Start web portal
python web_app.py
```

## Key Code Patterns

- **`pathmagic.context()`**: All modules use `with pathmagic.context():` to ensure the project root is on `sys.path` before importing project-local modules. Always wrap project imports in this context manager.
- **Jupytext paired notebooks**: Every `.py` file is paired with a `.ipynb` via jupytext (percent format). Edits to the `.py` file are the source of truth.
- **Cloud config**: Configuration is fetched dynamically via `getinivaluefromcloud()` from an INI stored in a Joplin note. The `ConfigManager` singleton in `config_manager.py` handles hot-reloading.
- **Inter-service auth**: `web_app.py` calls `joplin_qa_api.py` using an API key from the shared cloud config (`X-API-Key` header).
- **No tests directory**: No formal test framework. Test-adjacent files are scratchpad notebooks.

## Configuration

Main config stored in cloud-synced Joplin note (INI format). Local override: `data/joplinai.ini`. Key settings: Joplin API token, Ollama model name, embedding model, ChromaDB path, Q&A prompts, user session settings.

## Dependencies (no requirements.txt)

Core: `flask`, `chromadb`, `ollama`, `requests`, `jinja2`
