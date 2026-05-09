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
# # DeepSeek 缓存集中服务
# API Key 从云端配置读取，SQLite 操作独立（不依赖 cache_manager 避免级联导入）

# %%
import argparse
import configparser
import logging
import os
import sqlite3
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, jsonify, request

# %%
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("deepseek_cache_api")

app = Flask(__name__)

# %%
DB_PATH = Path(__file__).parent / "data" / ".deepseek_cache" / "deepseek_cache.db"
VALIDATION_THRESHOLD = 5000
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _get_api_key() -> Optional[str]:
    """1. 环境变量 → 2. 云端配置 → 3. 本地 INI 回退"""
    env_key = os.getenv("DEEPSEEK_CACHE_API_KEY")
    if env_key:
        return env_key
    try:
        import pathmagic
        with pathmagic.context():
            from func.jpfuncs import getinivaluefromcloud  # noqa: E402
        key = getinivaluefromcloud("joplinai", "deepseek_cache_api_key")
        if key:
            log.info("API Key 从云端配置读取成功")
            return key
    except BaseException as e:
        # BaseException 捕获 SystemExit（jpfuncs 模块级 getapi() 失败时会 sys.exit）
        log.warning(f"云端配置读取失败（Joplin 可能未就绪）: {type(e).__name__}: {e}")
    local_ini = Path(__file__).parent / "data" / "joplinai.ini"
    if local_ini.exists():
        cp = configparser.ConfigParser()
        cp.read(local_ini)
        fallback = cp.get("joplinai", "deepseek_cache_api_key", fallback=None)
        if fallback:
            log.info("API Key 从本地 INI 回退读取成功")
            return fallback
    return None


CACHE_API_KEY = _get_api_key()
log.info(f"缓存服务 API Key {'已配置' if CACHE_API_KEY else '未配置!'}")

# %% [markdown]
# # SQLite 操作

# %%
def _db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS processing_cache (
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
    conn.execute("CREATE INDEX IF NOT EXISTS idx_hash_task ON processing_cache(content_hash, task)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON processing_cache(last_accessed)")
    conn.commit()
    return conn


def cache_get(content_hash: str, task: str) -> dict:
    cache_key = f"{content_hash}_{task}"
    conn = _db()
    row = conn.execute(
        "SELECT result, hit_count, total_hits FROM processing_cache "
        "WHERE cache_key=? AND (julianday('now')-julianday(created_at))<90",
        (cache_key,),
    ).fetchone()
    if not row:
        conn.close()
        return {"content": None, "requires_validation": False, "cache_key": cache_key,
                "current_hit_count": 0, "total_hits": 0}

    result, hit_count, total_hits = row
    new_hit = hit_count + 1
    new_total = total_hits + 1
    should_validate = new_hit >= VALIDATION_THRESHOLD
    now = datetime.now().isoformat()

    if should_validate:
        conn.execute(
            "UPDATE processing_cache SET hit_count=0, total_hits=?, last_accessed=?, "
            "last_validated_at=?, validation_result='pending' WHERE cache_key=?",
            (new_total, now, now, cache_key),
        )
    else:
        conn.execute(
            "UPDATE processing_cache SET hit_count=?, total_hits=?, last_accessed=? WHERE cache_key=?",
            (new_hit, new_total, now, cache_key),
        )
    conn.commit()
    conn.close()
    return {"content": result, "requires_validation": should_validate, "cache_key": cache_key,
            "current_hit_count": 0 if should_validate else new_hit, "total_hits": new_total}


def cache_set(content_hash: str, task: str, result: str):
    cache_key = f"{content_hash}_{task}"
    now = datetime.now().isoformat()
    conn = _db()
    conn.execute(
        "INSERT OR REPLACE INTO processing_cache "
        "(cache_key, content_hash, task, result, created_at, last_accessed, "
        "last_validated_at, hit_count, total_hits, validation_result) "
        "VALUES (?,?,?,?,?,?,NULL,0,0,NULL)",
        (cache_key, content_hash, task, result, now, now),
    )
    count = conn.execute("SELECT COUNT(*) FROM processing_cache").fetchone()[0]
    if count > 50000:
        conn.execute("DELETE FROM processing_cache WHERE cache_key IN "
                     "(SELECT cache_key FROM processing_cache ORDER BY last_accessed ASC LIMIT 1000)")
    conn.commit()
    conn.close()


def cache_validate(cache_key: str, new_result: Optional[str], success: bool):
    now = datetime.now().isoformat()
    conn = _db()
    if not success:
        conn.execute("UPDATE processing_cache SET last_validated_at=?, validation_result='failed' WHERE cache_key=?",
                     (now, cache_key))
    elif new_result is None:
        conn.execute("UPDATE processing_cache SET last_validated_at=?, validation_result='valid' WHERE cache_key=?",
                     (now, cache_key))
    else:
        conn.execute("UPDATE processing_cache SET result=?, created_at=?, last_validated_at=?, validation_result='updated' WHERE cache_key=?",
                     (new_result, now, now, cache_key))
    conn.commit()
    conn.close()


def cache_stats(cache_key: str = None) -> dict:
    conn = _db()
    if cache_key:
        row = conn.execute("SELECT * FROM processing_cache WHERE cache_key=?", (cache_key,)).fetchone()
        conn.close()
        return dict(row) if row else {}
    row = conn.execute(
        "SELECT COUNT(*) as total, COALESCE(SUM(total_hits),0) as total_hits, "
        "COALESCE(SUM(hit_count),0) as current_hits FROM processing_cache"
    ).fetchone()
    conn.close()
    return {"total": row[0], "total_hits": row[1], "current_hits": row[2]}


# %% [markdown]
# # Flask 端点

# %%
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key")
        if not CACHE_API_KEY or key != CACHE_API_KEY:
            return jsonify({"error": "Invalid or missing API Key"}), 401
        return f(*args, **kwargs)
    return decorated


@app.route("/health")
def health():
    return jsonify({"service": "DeepSeek Cache API", "status": "running"})


@app.route("/cache/get", methods=["POST"])
@require_auth
def api_cache_get():
    data = request.get_json(force=True)
    return jsonify(cache_get(data["content_hash"], data["task"]))


@app.route("/cache/set", methods=["POST"])
@require_auth
def api_cache_set():
    data = request.get_json(force=True)
    cache_set(data["content_hash"], data["task"], data["result"])
    return jsonify({"ok": True})


@app.route("/cache/validate", methods=["POST"])
@require_auth
def api_cache_validate():
    data = request.get_json(force=True)
    cache_validate(data["cache_key"], data.get("new_result"), data["validation_successful"])
    return jsonify({"ok": True})


@app.route("/cache/stats")
@require_auth
def api_cache_stats():
    key = request.args.get("cache_key")
    return jsonify(cache_stats(cache_key=key))


# %%
def main():
    parser = argparse.ArgumentParser(description="DeepSeek 缓存 API 服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5003)
    args = parser.parse_args()
    log.info(f"启动 DeepSeek 缓存 API 服务于 http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
