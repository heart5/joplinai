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
# # Joplinai 数据中心
# DeepSeek 摘要/标签缓存 + 自适应探测结果缓存 + 历史数据库，统一 SQLite + 按域隔离端点。
# API Key 从云端配置读取。

# %%
import argparse
import configparser
import json
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
log = logging.getLogger("joplinai_center_api")

app = Flask(__name__)

# %%
DB_PATH = Path(__file__).parent / "data" / "joplinai_center.db"
VALIDATION_THRESHOLD = 5000
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

_probe_set_counter = 0


def _get_api_key() -> Optional[str]:
    """1. 环境变量 → 2. 云端配置 → 3. 本地 INI 回退"""
    env_key = os.getenv("JOPLINAI_CENTER_API_KEY")
    if env_key:
        return env_key
    try:
        import pathmagic
        with pathmagic.context():
            from func.jpfuncs import getinivaluefromcloud  # noqa: E402
        key = getinivaluefromcloud("joplinai", "joplinai_center_api_key")
        if key:
            log.info("API Key 从云端配置读取成功")
            return key
    except BaseException as e:
        log.warning(f"云端配置读取失败（Joplin 可能未就绪）: {type(e).__name__}: {e}")
    local_ini = Path(__file__).parent / "data" / "joplinai.ini"
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
    # DeepSeek 缓存表
    conn.execute("""
        CREATE TABLE IF NOT EXISTS deepseek_cache (
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
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ds_hash_task ON deepseek_cache(content_hash, task)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ds_last_accessed ON deepseek_cache(last_accessed)")
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
    # 笔记本处理历史表
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
    # 全局运行历史表
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
    return conn


# %% [markdown]
# # DeepSeek 缓存操作

# %%
def deepseek_cache_get(content_hash: str, task: str) -> dict:
    cache_key = f"{content_hash}_{task}"
    conn = _init_db()
    row = conn.execute(
        "SELECT result, hit_count, total_hits FROM deepseek_cache "
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
            "UPDATE deepseek_cache SET hit_count=0, total_hits=?, last_accessed=?, "
            "last_validated_at=?, validation_result='pending' WHERE cache_key=?",
            (new_total, now, now, cache_key),
        )
    else:
        conn.execute(
            "UPDATE deepseek_cache SET hit_count=?, total_hits=?, last_accessed=? WHERE cache_key=?",
            (new_hit, new_total, now, cache_key),
        )
    conn.commit()
    conn.close()
    return {"content": result, "requires_validation": should_validate, "cache_key": cache_key,
            "current_hit_count": 0 if should_validate else new_hit, "total_hits": new_total}


def deepseek_cache_set(content_hash: str, task: str, result: str):
    cache_key = f"{content_hash}_{task}"
    now = datetime.now().isoformat()
    conn = _init_db()
    conn.execute(
        "INSERT OR REPLACE INTO deepseek_cache "
        "(cache_key, content_hash, task, result, created_at, last_accessed, "
        "last_validated_at, hit_count, total_hits, validation_result) "
        "VALUES (?,?,?,?,?,?,NULL,0,0,NULL)",
        (cache_key, content_hash, task, result, now, now),
    )
    count = conn.execute("SELECT COUNT(*) FROM deepseek_cache").fetchone()[0]
    if count > 50000:
        conn.execute("DELETE FROM deepseek_cache WHERE cache_key IN "
                     "(SELECT cache_key FROM deepseek_cache ORDER BY last_accessed ASC LIMIT 1000)")
    conn.commit()
    conn.close()


def deepseek_cache_validate(cache_key: str, new_result: Optional[str], success: bool):
    now = datetime.now().isoformat()
    conn = _init_db()
    if not success:
        conn.execute("UPDATE deepseek_cache SET last_validated_at=?, validation_result='failed' WHERE cache_key=?",
                     (now, cache_key))
    elif new_result is None:
        conn.execute("UPDATE deepseek_cache SET last_validated_at=?, validation_result='valid' WHERE cache_key=?",
                     (now, cache_key))
    else:
        conn.execute("UPDATE deepseek_cache SET result=?, created_at=?, last_validated_at=?, validation_result='updated' WHERE cache_key=?",
                     (new_result, now, now, cache_key))
    conn.commit()
    conn.close()


def deepseek_cache_stats(cache_key: str = None) -> dict:
    conn = _init_db()
    if cache_key:
        row = conn.execute("SELECT * FROM deepseek_cache WHERE cache_key=?", (cache_key,)).fetchone()
        conn.close()
        return dict(row) if row else {}
    row = conn.execute(
        "SELECT COUNT(*) as total, COALESCE(SUM(total_hits),0) as total_hits, "
        "COALESCE(SUM(hit_count),0) as current_hits FROM deepseek_cache"
    ).fetchone()
    conn.close()
    return {"total": row[0], "total_hits": row[1], "current_hits": row[2]}


# %% [markdown]
# # 探测缓存操作

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


_probe_cache_limit = 10000


def probe_cache_get(text_md5: str) -> Optional[dict]:
    conn = _init_db()
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
    conn = _init_db()
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
    conn = _init_db()
    row = conn.execute("SELECT COUNT(*) FROM probe_cache").fetchone()
    conn.close()
    return {"count": row[0], "limit": _get_probe_cache_limit()}


# %% [markdown]
# # 历史数据库操作

# %%
def history_add_notebook_record(data: dict):
    conn = _init_db()
    chunk_stats = data.get("chunk_stats", {})
    conn.execute(
        """INSERT INTO notebook_history (
            run_id, notebook_title, timestamp,
            total_notes, updated_count, failed_count,
            notes_added_count, notes_removed_count,
            total_chunks, chunks_upserted, chunks_skipped, chunks_orphans_cleaned,
            notes_added_list, notes_removed_list, failed_notes_list
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            data["run_id"],
            data["notebook_title"],
            data["timestamp"],
            data.get("total_notes", 0),
            data.get("updated_count", 0),
            data.get("failed_count", 0),
            data.get("notes_added_count", 0),
            data.get("notes_removed_count", 0),
            chunk_stats.get("total_chunks", 0),
            chunk_stats.get("upserted", 0),
            chunk_stats.get("skipped", 0),
            chunk_stats.get("orphans_cleaned", 0),
            json.dumps(data.get("notes_added_list", []), ensure_ascii=False),
            json.dumps(data.get("notes_removed_list", []), ensure_ascii=False),
            json.dumps(data.get("failed_notes_list", []), ensure_ascii=False),
        ),
    )
    conn.commit()
    conn.close()


def history_finalize_run(data: dict):
    conn = _init_db()
    conn.execute(
        """INSERT OR REPLACE INTO global_run_history (
            run_id, timestamp, embedding_model, notebook_count,
            total_notes_processed, total_chunks_processed,
            total_notes_added, total_notes_removed,
            success, error_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            data["run_id"],
            data["timestamp"],
            data["embedding_model"],
            data["notebook_count"],
            data["total_notes_processed"],
            data["total_chunks_processed"],
            data["total_notes_added"],
            data["total_notes_removed"],
            data.get("success", True),
            data.get("error_message"),
        ),
    )
    conn.commit()
    conn.close()


def history_cumulative_stats(days: int = None) -> dict:
    conn = _init_db()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    time_condition = ""
    params = []
    if days:
        time_condition = "WHERE timestamp >= datetime('now', ?)"
        params.append(f"-{days} days")

    cursor.execute(
        f"""SELECT
            COUNT(DISTINCT run_id) as total_runs,
            COUNT(DISTINCT notebook_title) as total_notebooks_touched,
            SUM(total_notes) as total_notes_processed_all_time,
            SUM(total_chunks) as total_chunks_processed_all_time,
            SUM(notes_added_count) as total_notes_added_all_time,
            SUM(notes_removed_count) as total_notes_removed_all_time,
            SUM(chunks_upserted) as total_chunks_updated_all_time,
            SUM(chunks_orphans_cleaned) as total_orphans_cleaned_all_time
        FROM notebook_history {time_condition}""",
        params,
    )
    cumulative = dict(cursor.fetchone())

    cursor.execute("""
        SELECT strftime('%Y-%W', timestamp) as week,
            COUNT(DISTINCT run_id) as runs_count,
            SUM(total_notes) as notes_processed,
            SUM(total_chunks) as chunks_processed,
            SUM(notes_added_count) as notes_added,
            SUM(notes_removed_count) as notes_removed
        FROM notebook_history
        WHERE timestamp >= datetime('now', '-90 days')
        GROUP BY week ORDER BY week DESC LIMIT 12
    """)
    weekly_trends = [dict(row) for row in cursor.fetchall()]

    cursor.execute(
        f"""SELECT notebook_title, COUNT(*) as process_count,
            SUM(total_notes) as total_notes, SUM(total_chunks) as total_chunks,
            MAX(timestamp) as last_processed
        FROM notebook_history {time_condition}
        GROUP BY notebook_title ORDER BY process_count DESC LIMIT 10""",
        params,
    )
    top_notebooks = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return {
        "cumulative": cumulative,
        "weekly_trends": weekly_trends,
        "top_notebooks": top_notebooks,
        "analysis_period": f"最近{days}天" if days else "全部历史",
    }


def history_change_analysis(notebook_title: str = None, days: int = 30) -> dict:
    conn = _init_db()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    where_conditions = ["timestamp >= datetime('now', ?)"]
    params = [f"-{days} days"]
    if notebook_title:
        where_conditions.append("notebook_title = ?")
        params.append(notebook_title)
    where_clause = " AND ".join(where_conditions)

    cursor.execute(
        f"SELECT notes_added_list, notes_removed_list FROM notebook_history WHERE {where_clause}",
        params,
    )

    all_added = []
    all_removed = []
    for row in cursor.fetchall():
        if row["notes_added_list"]:
            all_added.extend(json.loads(row["notes_added_list"]))
        if row["notes_removed_list"]:
            all_removed.extend(json.loads(row["notes_removed_list"]))

    unique_added = list(set(all_added))
    unique_removed = list(set(all_removed))
    frequently_changed = list(set(unique_added) & set(unique_removed))

    conn.close()
    return {
        "analysis_period": f"最近{days}天",
        "notebook": notebook_title or "全局",
        "unique_notes_added": unique_added,
        "unique_notes_removed": unique_removed,
        "added_count": len(unique_added),
        "removed_count": len(unique_removed),
        "net_growth": len(unique_added) - len(unique_removed),
        "frequently_changed_notes": frequently_changed,
        "frequently_changed_count": len(frequently_changed),
    }


def history_efficiency_metrics(days: int = 30) -> dict:
    conn = _init_db()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(
        """SELECT
            AVG(total_notes) as avg_notes_per_run,
            AVG(total_chunks) as avg_chunks_per_run,
            SUM(chunks_upserted) * 100.0 / NULLIF(SUM(total_chunks), 0) as avg_update_rate_percent,
            SUM(chunks_skipped) * 100.0 / NULLIF(SUM(total_chunks), 0) as avg_skip_rate_percent,
            SUM(notes_added_count) * 100.0 / NULLIF(SUM(total_notes), 0) as avg_addition_rate_percent,
            SUM(notes_removed_count) * 100.0 / NULLIF(SUM(total_notes), 0) as avg_removal_rate_percent,
            COUNT(DISTINCT DATE(timestamp)) as active_days,
            COUNT(DISTINCT run_id) as total_runs,
            COUNT(DISTINCT run_id) * 1.0 / NULLIF(COUNT(DISTINCT DATE(timestamp)), 1) as avg_runs_per_day
        FROM notebook_history
        WHERE timestamp >= datetime('now', ?)""",
        [f"-{days} days"],
    )
    metrics = dict(cursor.fetchone())

    cursor.execute(
        """SELECT COUNT(*) as total_runs,
            SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_runs
        FROM global_run_history
        WHERE timestamp >= datetime('now', ?)""",
        [f"-{days} days"],
    )
    run_stats = cursor.fetchone()
    if run_stats and run_stats["total_runs"] > 0:
        metrics["success_rate_percent"] = (run_stats["successful_runs"] * 100.0) / run_stats["total_runs"]
    else:
        metrics["success_rate_percent"] = 0.0

    conn.close()

    for key in list(metrics.keys()):
        if metrics[key] is None:
            if "percent" in key or "rate" in key:
                metrics[key] = 0.0
            elif "avg" in key:
                metrics[key] = 0.0
            else:
                metrics[key] = 0
    for key in list(metrics.keys()):
        if "percent" in key or "rate" in key:
            metrics[key] = round(float(metrics[key]), 2)
        elif isinstance(metrics[key], (int, float)):
            metrics[key] = round(float(metrics[key]), 2)

    return metrics


# %% [markdown]
# # Flask 端点

# %%
def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key")
        if not CENTER_API_KEY or key != CENTER_API_KEY:
            return jsonify({"error": "Invalid or missing API Key"}), 401
        return f(*args, **kwargs)
    return decorated


# %% [markdown]
# ## 健康检查

# %%
@app.route("/health")
def health():
    return jsonify({"service": "Joplinai Center API", "status": "running"})


# %% [markdown]
# ## DeepSeek 缓存端点

# %%
@app.route("/cache/deepseek/get", methods=["POST"])
@require_auth
def api_ds_cache_get():
    data = request.get_json(force=True)
    return jsonify(deepseek_cache_get(data["content_hash"], data["task"]))


@app.route("/cache/deepseek/set", methods=["POST"])
@require_auth
def api_ds_cache_set():
    data = request.get_json(force=True)
    deepseek_cache_set(data["content_hash"], data["task"], data["result"])
    return jsonify({"ok": True})


@app.route("/cache/deepseek/validate", methods=["POST"])
@require_auth
def api_ds_cache_validate():
    data = request.get_json(force=True)
    deepseek_cache_validate(data["cache_key"], data.get("new_result"), data["validation_successful"])
    return jsonify({"ok": True})


@app.route("/cache/deepseek/stats")
@require_auth
def api_ds_cache_stats():
    key = request.args.get("cache_key")
    return jsonify(deepseek_cache_stats(cache_key=key))


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


# %% [markdown]
# ## 历史数据库端点

# %%
@app.route("/history/notebook_record", methods=["POST"])
@require_auth
def api_history_notebook_record():
    history_add_notebook_record(request.get_json(force=True))
    return jsonify({"ok": True})


@app.route("/history/finalize_run", methods=["POST"])
@require_auth
def api_history_finalize_run():
    history_finalize_run(request.get_json(force=True))
    return jsonify({"ok": True})


@app.route("/history/cumulative_stats", methods=["GET"])
@require_auth
def api_history_cumulative_stats():
    days = request.args.get("days", type=int)
    return jsonify(history_cumulative_stats(days=days))


@app.route("/history/change_analysis", methods=["GET"])
@require_auth
def api_history_change_analysis():
    notebook = request.args.get("notebook")
    days = request.args.get("days", default=30, type=int)
    return jsonify(history_change_analysis(notebook_title=notebook, days=days))


@app.route("/history/efficiency_metrics", methods=["GET"])
@require_auth
def api_history_efficiency_metrics():
    days = request.args.get("days", default=30, type=int)
    return jsonify(history_efficiency_metrics(days=days))


# %%
def main():
    parser = argparse.ArgumentParser(description="Joplinai 数据中心 API 服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5003)
    args = parser.parse_args()
    log.info(f"启动 Joplinai 数据中心 API 服务于 http://{args.host}:{args.port}")
    _init_db()
    log.info("统一数据库 joplinai_center.db 初始化完成")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
