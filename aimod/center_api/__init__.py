# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Joplinai 数据中心 — 共享基础设施

# %%
import argparse
import configparser
import logging
import os
import sqlite3
from functools import wraps
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request

# %%
__all__ = ["create_app", "main", "require_auth", "DB_PATH", "VALIDATION_THRESHOLD"]

log = logging.getLogger("joplinai_center_api")
log.propagate = False
_log_handler = logging.StreamHandler()
_log_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
log.addHandler(_log_handler)
log.setLevel(logging.INFO)

# %%
DB_PATH = Path(__file__).parent.parent.parent / "data" / "joplinai_center.db"
VALIDATION_THRESHOLD = 5000
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_probe_set_counter = 0
_probe_cache_limit = 10000


def _get_api_key() -> Optional[str]:
    """1. 环境变量 → 2. 云端配置 → 3. 本地 INI 回退"""
    env_key = os.getenv("JOPLINAI_CENTER_API_KEY")
    if env_key:
        return env_key
    try:
        import pathmagic
        with pathmagic.Context():
            from func.jpfuncs import getinivaluefromcloud  # noqa: E402
        key = getinivaluefromcloud("joplinai", "joplinai_center_api_key")
        if key:
            log.info("API Key 从云端配置读取成功")
            return key
    except BaseException as e:
        log.warning(f"云端配置读取失败（Joplin 可能未就绪）: {type(e).__name__}: {e}")
    local_ini = DB_PATH.parent / "joplinai.ini"
    if local_ini.exists():
        cp = configparser.ConfigParser()
        cp.read(local_ini)
        fallback = cp.get("joplinai", "joplinai_center_api_key", fallback=None)
        if fallback:
            log.info("API Key 从本地 INI 回退读取成功")
            return fallback
    return None


CENTER_API_KEY = _get_api_key()
log.info(f"数据中心 API Key {'已配置' if CENTER_API_KEY else '未配置!'}")

# %% [markdown]
# # 统一数据库初始化

# %%
def _init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS enhance_cache (
            cache_key TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            task TEXT NOT NULL,
            result TEXT NOT NULL,
            created_at DATETIME NOT NULL,
            last_accessed DATETIME NOT NULL,
            last_validated_at DATETIME,
            hit_count INTEGER DEFAULT 0,
            total_hits INTEGER DEFAULT 0,
            validation_result TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_enh_hash_task ON enhance_cache(content_hash, task)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_enh_last_accessed ON enhance_cache(last_accessed)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS probe_cache (
            text_md5      TEXT PRIMARY KEY,
            safe_len      INTEGER NOT NULL,
            snippet       TEXT    NOT NULL,
            model_name    TEXT    NOT NULL,
            chunk_size    INTEGER NOT NULL,
            created_at    TEXT    NOT NULL,
            last_accessed TEXT    NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_probe_last_accessed ON probe_cache(last_accessed)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS notebook_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            notebook_title TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            total_notes INTEGER DEFAULT 0,
            updated_count INTEGER DEFAULT 0,
            failed_count INTEGER DEFAULT 0,
            notes_added_count INTEGER DEFAULT 0,
            notes_removed_count INTEGER DEFAULT 0,
            total_chunks INTEGER DEFAULT 0,
            chunks_upserted INTEGER DEFAULT 0,
            chunks_skipped INTEGER DEFAULT 0,
            chunks_orphans_cleaned INTEGER DEFAULT 0,
            notes_added_list TEXT,
            notes_removed_list TEXT,
            failed_notes_list TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_nb_history_timestamp ON notebook_history(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_nb_history_notebook ON notebook_history(notebook_title)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS global_run_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE NOT NULL,
            timestamp DATETIME NOT NULL,
            embedding_model TEXT NOT NULL,
            notebook_count INTEGER DEFAULT 0,
            total_notes_processed INTEGER DEFAULT 0,
            total_chunks_processed INTEGER DEFAULT 0,
            total_notes_added INTEGER DEFAULT 0,
            total_notes_removed INTEGER DEFAULT 0,
            success BOOLEAN DEFAULT TRUE,
            error_message TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_global_run_timestamp ON global_run_history(timestamp)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS note_process_state (
            model_name TEXT NOT NULL,
            note_id    TEXT NOT NULL,
            state_json TEXT NOT NULL,
            updated_at DATETIME NOT NULL,
            PRIMARY KEY (model_name, note_id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_nps_model ON note_process_state(model_name)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            display_name TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin', 'team_leader', 'team_member')),
            allowed_notebooks TEXT,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            action TEXT NOT NULL,
            details TEXT,
            ip_address TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS qa_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_id TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_qa_session ON qa_history(session_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_qa_user ON qa_history(user_id)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_id TEXT NOT NULL UNIQUE,
            name TEXT DEFAULT '新对话',
            is_active INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_sessions(user_id)")
    return conn


_init_db()
log.info("统一数据库 joplinai_center.db 初始化完成")


# %% [markdown]
# # 认证装饰器

# %%
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get("X-API-Key")
        if not CENTER_API_KEY:
            return f(*args, **kwargs)
        if not api_key or api_key != CENTER_API_KEY:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


# %% [markdown]
# # Flask 应用工厂

# %%
def create_app():
    app = Flask(__name__)

    from aimod.center_api.cache_routes import cache_bp
    from aimod.center_api.history_routes import history_bp
    from aimod.center_api.state_routes import state_bp
    from aimod.center_api.user_routes import user_bp

    app.register_blueprint(cache_bp)
    app.register_blueprint(history_bp)
    app.register_blueprint(state_bp)
    app.register_blueprint(user_bp)

    @app.route("/health")
    def health():
        try:
            conn = _init_db()
            conn.execute("SELECT 1")
            conn.close()
            return jsonify({"status": "healthy", "db": "ok", "timestamp": __import__("datetime").datetime.now().isoformat()})
        except Exception as e:
            return jsonify({"status": "unhealthy", "error": str(e)}), 500

    return app


# %%
def main():
    parser = argparse.ArgumentParser(description="Joplinai 数据中心 API 服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5003)
    args = parser.parse_args()
    log.info(f"启动 Joplinai 数据中心 API 服务于 http://{args.host}:{args.port}")
    _init_db()
    log.info("统一数据库 joplinai_center.db 初始化完成")
    app = create_app()
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
