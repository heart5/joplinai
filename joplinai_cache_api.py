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
# # Joplinai 统一缓存服务
# DeepSeek 摘要/标签缓存 + 自适应探测结果缓存，按 domain 隔离。
# API Key 从云端配置读取，SQLite 操作独立。

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
log = logging.getLogger("joplinai_cache_api")

app = Flask(__name__)

# %%
DB_PATH = Path(__file__).parent / "data" / ".deepseek_cache" / "deepseek_cache.db"
VALIDATION_THRESHOLD = 5000
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# 探测缓存淘汰计数器
_probe_set_counter = 0


def _get_api_key() -> Optional[str]:
    """1. 环境变量 → 2. 云端配置 → 3. 本地 INI 回退"""
    env_key = os.getenv("JOPLINAI_CACHE_API_KEY")
    if env_key:
        return env_key
    try:
        import pathmagic
        with pathmagic.context():
            from func.jpfuncs import getinivaluefromcloud  # noqa: E402
        key = getinivaluefromcloud("joplinai", "joplinai_cache_api_key")
        if key:
            log.info("API Key 从云端配置读取成功")
            return key
    except BaseException as e:
        log.warning(f"云端配置读取失败（Joplin 可能未就绪）: {type(e).__name__}: {e}")
    local_ini = Path(__file__).parent / "data" / "joplinai.ini"
    if local_ini.exists():
        cp = configparser.ConfigParser()
        cp.read(local_ini)
        fallback = cp.get("joplinai", "joplinai_cache_api_key", fallback=None)
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
    # 探测缓存表
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
    conn.commit()
    return conn


# %% [markdown]
# ## DeepSeek 缓存操作

# %%
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
# ## 探测缓存操作

# %%
def _get_probe_cache_limit() -> int:
    try:
        import pathmagic
        with pathmagic.context():
            from func.jpfuncs import getinivaluefromcloud
        val = getinivaluefromcloud("joplinai", "probe_cache_limit")
        return int(val) if val else 10000
    except Exception:
        return 10000


_probe_cache_limit = 10000  # 模块级缓存，首次 set 时更新


def probe_cache_get(text_md5: str) -> Optional[dict]:
    conn = _db()
    row = conn.execute(
        "SELECT safe_len, snippet, model_name, chunk_size, created_at, last_accessed "
        "FROM probe_cache WHERE text_md5=?",
        (text_md5,),
    ).fetchone()
    if not row:
        conn.close()
        return None
    now = datetime.now().isoformat()
    conn.execute("UPDATE probe_cache SET last_accessed=? WHERE text_md5=?", (now, text_md5))
    conn.commit()
    conn.close()
    return {
        "safe_len": row[0],
        "snippet": row[1],
        "model_name": row[2],
        "chunk_size": row[3],
        "timestamp": row[4],
    }


def probe_cache_set(text_md5: str, safe_len: int, snippet: str,
                    model_name: str, chunk_size: int):
    global _probe_set_counter, _probe_cache_limit
    now = datetime.now().isoformat()
    conn = _db()
    conn.execute(
        "INSERT OR REPLACE INTO probe_cache "
        "(text_md5, safe_len, snippet, model_name, chunk_size, created_at, last_accessed) "
        "VALUES (?,?,?,?,?, COALESCE((SELECT created_at FROM probe_cache WHERE text_md5=?), ?), ?)",
        (text_md5, safe_len, snippet, model_name, chunk_size, text_md5, now, now),
    )
    conn.commit()

    _probe_set_counter += 1
    if _probe_set_counter % 1000 == 0:
        _probe_cache_limit = _get_probe_cache_limit()
        count = conn.execute("SELECT COUNT(*) FROM probe_cache").fetchone()[0]
        if count > _probe_cache_limit:
            delete_n = max(1, count // 10)
            conn.execute(
                "DELETE FROM probe_cache WHERE text_md5 IN ("
                "SELECT text_md5 FROM probe_cache ORDER BY last_accessed ASC LIMIT ?"
                ")", (delete_n,)
            )
            conn.commit()
            log.info(f"[探测缓存] 淘汰 {delete_n} 条（总量 {count} > 上限 {_probe_cache_limit}）")
    conn.close()


def probe_cache_stats() -> dict:
    conn = _db()
    row = conn.execute("SELECT COUNT(*) FROM probe_cache").fetchone()
    conn.close()
    return {"count": row[0], "limit": _get_probe_cache_limit()}


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


# %% [markdown]
# ## 健康检查

# %%
@app.route("/health")
def health():
    return jsonify({"service": "Joplinai Cache API", "status": "running"})


# %% [markdown]
# ## DeepSeek 缓存端点

# %%
@app.route("/cache/deepseek/get", methods=["POST"])
@require_auth
def api_cache_get():
    data = request.get_json(force=True)
    return jsonify(cache_get(data["content_hash"], data["task"]))


@app.route("/cache/deepseek/set", methods=["POST"])
@require_auth
def api_cache_set():
    data = request.get_json(force=True)
    cache_set(data["content_hash"], data["task"], data["result"])
    return jsonify({"ok": True})


@app.route("/cache/deepseek/validate", methods=["POST"])
@require_auth
def api_cache_validate():
    data = request.get_json(force=True)
    cache_validate(data["cache_key"], data.get("new_result"), data["validation_successful"])
    return jsonify({"ok": True})


@app.route("/cache/deepseek/stats")
@require_auth
def api_cache_stats():
    key = request.args.get("cache_key")
    return jsonify(cache_stats(cache_key=key))


# %% [markdown]
# ## 探测缓存端点

# %%
@app.route("/cache/probe/get/<text_md5>", methods=["GET"])
@require_auth
def api_probe_cache_get(text_md5: str):
    result = probe_cache_get(text_md5)
    if result is None:
        return jsonify({"found": False}), 404
    return jsonify({"found": True, **result})


@app.route("/cache/probe/set", methods=["POST"])
@require_auth
def api_probe_cache_set():
    data = request.get_json(force=True)
    probe_cache_set(
        text_md5=data["text_md5"],
        safe_len=data["safe_len"],
        snippet=data["snippet"],
        model_name=data["model_name"],
        chunk_size=data["chunk_size"],
    )
    return jsonify({"ok": True})


@app.route("/cache/probe/stats")
@require_auth
def api_probe_cache_stats():
    return jsonify(probe_cache_stats())


# %%
def main():
    parser = argparse.ArgumentParser(description="Joplinai 统一缓存 API 服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5003)
    args = parser.parse_args()
    log.info(f"启动 Joplinai 缓存 API 服务于 http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
