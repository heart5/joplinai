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
import secrets
import sqlite3
from datetime import datetime, timedelta
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
    # 笔记处理状态表（按模型+笔记ID分片）
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
    # 用户管理表（与 joplinai_users.db schema 一致）
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


# gunicorn 导入时不执行 main()，需在模块级确保 DB 初始化
_init_db()
log.info("统一数据库 joplinai_center.db 初始化完成")

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


def deepseek_cache_report() -> dict:
    """综合缓存报告数据（5维度统计）"""
    conn = _init_db()
    conn.row_factory = sqlite3.Row
    report = {}

    # basic_stats
    row = conn.execute("SELECT COUNT(*) as total FROM deepseek_cache").fetchone()
    total_entries = row[0] if row else 0
    row = conn.execute("SELECT COALESCE(SUM(total_hits),0) as total_hits FROM deepseek_cache").fetchone()
    total_hits = row[0] if row else 0
    row = conn.execute("SELECT COALESCE(SUM(hit_count),0) as current_hits FROM deepseek_cache").fetchone()
    current_hits = row[0] if row else 0
    avg_hits = total_hits / total_entries if total_entries > 0 else 0
    tasks = [dict(r) for r in conn.execute("""
        SELECT task, COUNT(*) as count, COALESCE(SUM(total_hits),0) as hits,
               AVG(total_hits) as avg_hits
        FROM deepseek_cache GROUP BY task ORDER BY count DESC
    """).fetchall()]
    report["basic_stats"] = {
        "total_entries": total_entries, "total_hits": total_hits,
        "current_hits": current_hits, "avg_hits_per_entry": avg_hits,
        "task_distribution": tasks,
    }

    # time_analysis
    creation_by_day = [dict(r) for r in conn.execute("""
        SELECT DATE(created_at) as date, COUNT(*) as count
        FROM deepseek_cache GROUP BY DATE(created_at) ORDER BY date DESC LIMIT 30
    """).fetchall()]
    access_by_day = [dict(r) for r in conn.execute("""
        SELECT DATE(last_accessed) as date, COUNT(*) as count
        FROM deepseek_cache GROUP BY DATE(last_accessed) ORDER BY date DESC LIMIT 30
    """).fetchall()]
    age_dist = [dict(r) for r in conn.execute("""
        SELECT CASE
            WHEN julianday('now') - julianday(created_at) <= 1 THEN '1天内'
            WHEN julianday('now') - julianday(created_at) <= 7 THEN '7天内'
            WHEN julianday('now') - julianday(created_at) <= 30 THEN '30天内'
            WHEN julianday('now') - julianday(created_at) <= 90 THEN '90天内'
            ELSE '超过90天'
        END as age_group, COUNT(*) as count,
        AVG(total_hits) as avg_hits
        FROM deepseek_cache GROUP BY age_group
        ORDER BY CASE age_group
            WHEN '1天内' THEN 1 WHEN '7天内' THEN 2 WHEN '30天内' THEN 3
            WHEN '90天内' THEN 4 ELSE 5 END
    """).fetchall()]
    row = conn.execute("""
        SELECT COUNT(*) FROM deepseek_cache
        WHERE julianday('now') - julianday(last_accessed) <= 7
    """).fetchone()
    recent_active = row[0] if row else 0
    report["time_analysis"] = {
        "creation_by_day": creation_by_day, "access_by_day": access_by_day,
        "age_distribution": age_dist, "recent_active": recent_active,
    }

    # validation_analysis
    val_states = [dict(r) for r in conn.execute("""
        SELECT COALESCE(validation_result,'not_validated') as validation_state,
               COUNT(*) as count, AVG(total_hits) as avg_hits,
               AVG(julianday('now') - julianday(created_at)) as avg_age_days
        FROM deepseek_cache GROUP BY validation_state ORDER BY count DESC
    """).fetchall()]
    row = conn.execute("SELECT COUNT(*) FROM deepseek_cache WHERE hit_count >= ?", (4000,)).fetchone()
    nearing = row[0] if row else 0
    row = conn.execute("""
        SELECT MAX(last_validated_at) FROM deepseek_cache WHERE last_validated_at IS NOT NULL
    """).fetchone()
    last_val = row[0] if row else None
    report["validation_analysis"] = {
        "validation_states": val_states, "nearing_validation": nearing,
        "last_validation_time": last_val,
    }

    # performance_metrics
    top = [dict(r) for r in conn.execute("""
        SELECT cache_key, task, total_hits, hit_count, created_at, last_accessed,
               substr(result, 1, 30) as result_preview
        FROM deepseek_cache ORDER BY total_hits DESC LIMIT 10
    """).fetchall()]
    row = conn.execute("""
        SELECT COUNT(*) FROM deepseek_cache
        WHERE julianday('now') - julianday(last_accessed) > 30 AND total_hits = 0
    """).fetchone()
    stale = row[0] if row else 0
    # 不调用 _fetch_scalar，直接 COUNT
    count_row = conn.execute("SELECT COUNT(*) FROM deepseek_cache").fetchone()
    count_val = count_row[0] if count_row else 0
    access_by_hour = [dict(r) for r in conn.execute("""
        SELECT strftime('%H', last_accessed) as hour, COUNT(*) as access_count
        FROM deepseek_cache WHERE last_accessed IS NOT NULL
        GROUP BY hour ORDER BY hour
    """).fetchall()]
    report["performance_metrics"] = {
        "top_hitters": top, "stale_entries": stale,
        "estimated_size_mb": round(count_val * 2 / 1024, 2),
        "access_by_hour": access_by_hour,
    }

    # growth_trends
    daily = [dict(r) for r in conn.execute("""
        SELECT DATE(created_at) as date, COUNT(*) as new_entries
        FROM deepseek_cache WHERE DATE(created_at) >= DATE('now', '-30 days')
        GROUP BY DATE(created_at) ORDER BY date
    """).fetchall()]
    cum = [dict(r) for r in conn.execute("""
        SELECT DATE(created_at) as date,
               SUM(COUNT(*)) OVER (ORDER BY DATE(created_at)) as cumulative
        FROM deepseek_cache WHERE DATE(created_at) >= DATE('now', '-30 days')
        GROUP BY DATE(created_at) ORDER BY date
    """).fetchall()]
    avg_daily = sum(d["new_entries"] for d in daily) / len(daily) if daily else 0
    report["growth_trends"] = {
        "daily_growth": daily, "cumulative_growth": cum,
        "predicted_weekly_growth": round(avg_daily * 7, 1),
    }

    conn.close()
    return report


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


def probe_cache_report() -> dict:
    """探测缓存综合报告数据"""
    conn = _init_db()
    conn.row_factory = sqlite3.Row

    row = conn.execute("SELECT COUNT(*) as count FROM probe_cache").fetchone()
    total = row["count"] if row else 0

    # 按模型分布
    by_model = [dict(r) for r in conn.execute("""
        SELECT model_name, COUNT(*) as count,
               AVG(safe_len) as avg_safe_len,
               AVG(chunk_size) as avg_chunk_size,
               MAX(last_accessed) as last_used
        FROM probe_cache GROUP BY model_name ORDER BY count DESC
    """).fetchall()]

    # 按块大小分布
    by_chunk = [dict(r) for r in conn.execute("""
        SELECT chunk_size, COUNT(*) as count, AVG(safe_len) as avg_safe_len
        FROM probe_cache GROUP BY chunk_size ORDER BY chunk_size
    """).fetchall()]

    # 安全长度统计
    row = conn.execute("""
        SELECT MIN(safe_len) as min_len, MAX(safe_len) as max_len,
               AVG(safe_len) as avg_len
        FROM probe_cache
    """).fetchone()
    len_stats = dict(row) if row else {}

    # 最近活跃
    row = conn.execute("""
        SELECT COUNT(*) as cnt FROM probe_cache
        WHERE last_accessed >= datetime('now', '-7 days')
    """).fetchone()
    recent = row["cnt"] if row else 0

    # 新增趋势（最近30天）
    daily_new = [dict(r) for r in conn.execute("""
        SELECT DATE(created_at) as date, COUNT(*) as new_entries
        FROM probe_cache WHERE DATE(created_at) >= DATE('now', '-30 days')
        GROUP BY DATE(created_at) ORDER BY date
    """).fetchall()]

    conn.close()
    return {
        "total": total,
        "limit": _get_probe_cache_limit(),
        "by_model": by_model,
        "by_chunk_size": by_chunk,
        "safe_len_stats": len_stats,
        "recent_active": recent,
        "daily_new": daily_new,
    }


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
    try:
        conn = _init_db()
        conn.execute("SELECT 1 FROM deepseek_cache LIMIT 0")
        conn.close()
        db_ok = True
    except Exception:
        db_ok = False
    return jsonify({"service": "Joplinai Center API", "status": "running", "db_ok": db_ok})


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


@app.route("/cache/deepseek/report")
@require_auth
def api_ds_cache_report():
    return jsonify(deepseek_cache_report())


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


@app.route("/cache/probe/report")
@require_auth
def api_probe_cache_report():
    return jsonify(probe_cache_report())


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


# %% [markdown]
# # 笔记处理状态端点


# %%
@app.route("/state/batch_load", methods=["POST"])
@require_auth
def api_state_batch_load():
    data = request.get_json(force=True)
    model_name = data["model_name"]
    conn = _init_db()
    rows = conn.execute(
        "SELECT note_id, state_json FROM note_process_state WHERE model_name=?",
        (model_name,),
    ).fetchall()
    conn.close()
    states = {}
    virtual_collections = {}
    for note_id, state_json in rows:
        state = json.loads(state_json)
        if note_id == "__virtual_collections__":
            virtual_collections = state
        else:
            states[note_id] = state
    result = {"states": states}
    if virtual_collections:
        result["virtual_collections"] = virtual_collections
    return jsonify(result)


@app.route("/state/batch_save", methods=["POST"])
@require_auth
def api_state_batch_save():
    data = request.get_json(force=True)
    model_name = data["model_name"]
    states = data.get("states", {})
    virtual_collections = data.get("virtual_collections", {})
    now = datetime.now().isoformat()
    conn = _init_db()
    conn.execute("DELETE FROM note_process_state WHERE model_name=?", (model_name,))
    count = 0
    for note_id, note_state in states.items():
        conn.execute(
            "INSERT INTO note_process_state (model_name, note_id, state_json, updated_at) VALUES (?,?,?,?)",
            (model_name, note_id, json.dumps(note_state, ensure_ascii=False), now),
        )
        count += 1
    if virtual_collections:
        conn.execute(
            "INSERT INTO note_process_state (model_name, note_id, state_json, updated_at) VALUES (?,?,?,?)",
            (model_name, "__virtual_collections__", json.dumps(virtual_collections, ensure_ascii=False), now),
        )
        count += 1
    conn.commit()
    conn.close()
    return jsonify({"ok": True, "count": count})


@app.route("/state/<model_name>/<note_id>", methods=["GET"])
@require_auth
def api_state_get_note(model_name: str, note_id: str):
    conn = _init_db()
    row = conn.execute(
        "SELECT state_json FROM note_process_state WHERE model_name=? AND note_id=?",
        (model_name, note_id),
    ).fetchone()
    conn.close()
    if row:
        return jsonify({"found": True, "state": json.loads(row[0])})
    return jsonify({"found": False}), 404


@app.route("/state/delete_model", methods=["POST"])
@require_auth
def api_state_delete_model():
    data = request.get_json(force=True)
    model_name = data["model_name"]
    conn = _init_db()
    cursor = conn.execute("DELETE FROM note_process_state WHERE model_name=?", (model_name,))
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return jsonify({"ok": True, "deleted": deleted})


# %% [markdown]
# ## 认证端点


# %%
@app.route("/auth/verify", methods=["POST"])
@require_auth
def api_auth_verify():
    data = request.get_json(force=True)
    conn = _init_db()
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT id, username, display_name, role FROM users WHERE username=? AND password_hash=? AND is_active=1",
        (data["username"], data["password_hash"]),
    ).fetchone()
    if row:
        user = dict(row)
        conn.execute("UPDATE users SET last_login=? WHERE id=?", (datetime.now().isoformat(), user["id"]))
        conn.commit()
        conn.close()
        return jsonify({"found": True, "user": user})
    conn.close()
    return jsonify({"found": False}), 404


@app.route("/auth/create_session", methods=["POST"])
@require_auth
def api_auth_create_session():
    data = request.get_json(force=True)
    session_id = secrets.token_urlsafe(32)
    expires_at = datetime.now() + timedelta(hours=data.get("duration_hours", 24))
    conn = _init_db()
    conn.execute("INSERT INTO sessions (session_id, user_id, expires_at) VALUES (?,?,?)",
                 (session_id, data["user_id"], expires_at))
    conn.commit()
    conn.close()
    return jsonify({"session_id": session_id})


@app.route("/auth/validate_session", methods=["POST"])
@require_auth
def api_auth_validate_session():
    data = request.get_json(force=True)
    conn = _init_db()
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT u.id, u.username, u.display_name, u.role FROM sessions s "
        "JOIN users u ON s.user_id=u.id "
        "WHERE s.session_id=? AND s.expires_at>? AND u.is_active=1",
        (data["session_id"], datetime.now()),
    ).fetchone()
    conn.close()
    if row:
        return jsonify({"valid": True, "user": dict(row)})
    return jsonify({"valid": False}), 404


@app.route("/auth/delete_session", methods=["POST"])
@require_auth
def api_auth_delete_session():
    data = request.get_json(force=True)
    conn = _init_db()
    conn.execute("DELETE FROM sessions WHERE session_id=?", (data["session_id"],))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


# %% [markdown]
# ## 用户CRUD端点


# %%
@app.route("/users", methods=["GET"])
@require_auth
def api_users_list():
    conn = _init_db()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, username, display_name, role, is_active, created_at, last_login FROM users ORDER BY id"
    ).fetchall()
    conn.close()
    return jsonify({"users": [dict(r) for r in rows]})


@app.route("/users/<username>", methods=["GET"])
@require_auth
def api_users_get(username: str):
    conn = _init_db()
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT id, username, display_name, role, allowed_notebooks, is_active, created_at, last_login FROM users WHERE username=?",
        (username,),
    ).fetchone()
    conn.close()
    if row:
        user = dict(row)
        try:
            user["allowed_notebooks"] = json.loads(user["allowed_notebooks"] or "[]")
        except Exception:
            user["allowed_notebooks"] = []
        return jsonify({"found": True, "user": user})
    return jsonify({"found": False}), 404


@app.route("/users/create", methods=["POST"])
@require_auth
def api_users_create():
    data = request.get_json(force=True)
    notebooks_json = json.dumps(data.get("allowed_notebooks", []), ensure_ascii=False)
    conn = _init_db()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, display_name, role, allowed_notebooks) VALUES (?,?,?,?,?)",
            (data["username"], data["password_hash"], data["display_name"], data.get("role", "team_member"), notebooks_json),
        )
        conn.commit()
        conn.close()
        return jsonify({"ok": True})
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"ok": False, "error": "用户名已存在"}), 409


@app.route("/users/delete", methods=["POST"])
@require_auth
def api_users_delete():
    data = request.get_json(force=True)
    conn = _init_db()
    row = conn.execute("SELECT id FROM users WHERE username=?", (data["target_username"],)).fetchone()
    if not row:
        conn.close()
        return jsonify({"ok": False, "error": "用户不存在"}), 404
    user_id = row[0]
    conn.execute("DELETE FROM qa_history WHERE session_id IN (SELECT session_id FROM chat_sessions WHERE user_id=?)", (user_id,))
    conn.execute("DELETE FROM chat_sessions WHERE user_id=?", (user_id,))
    conn.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
    conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


# %% [markdown]
# ## 用户更新端点


# %%
@app.route("/users/update_role", methods=["POST"])
@require_auth
def api_users_update_role():
    data = request.get_json(force=True)
    conn = _init_db()
    cursor = conn.execute("UPDATE users SET role=? WHERE username=?", (data["new_role"], data["target_username"]))
    ok = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return jsonify({"ok": ok})


@app.route("/users/update_permissions", methods=["POST"])
@require_auth
def api_users_update_permissions():
    data = request.get_json(force=True)
    updates = []
    params = []
    if "role" in data and data["role"] is not None:
        updates.append("role=?")
        params.append(data["role"])
    if "allowed_notebooks" in data and data["allowed_notebooks"] is not None:
        updates.append("allowed_notebooks=?")
        params.append(json.dumps(data["allowed_notebooks"], ensure_ascii=False))
    if not updates:
        return jsonify({"ok": True})
    params.append(data["username"])
    conn = _init_db()
    conn.execute(f"UPDATE users SET {', '.join(updates)} WHERE username=?", params)
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


@app.route("/users/reset_password", methods=["POST"])
@require_auth
def api_users_reset_password():
    data = request.get_json(force=True)
    conn = _init_db()
    cursor = conn.execute("UPDATE users SET password_hash=? WHERE username=?", (data["new_password_hash"], data["target_username"]))
    ok = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return jsonify({"ok": ok})


@app.route("/users/toggle_active", methods=["POST"])
@require_auth
def api_users_toggle_active():
    data = request.get_json(force=True)
    conn = _init_db()
    cursor = conn.execute("UPDATE users SET is_active=? WHERE username=?", (1 if data["is_active"] else 0, data["target_username"]))
    ok = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return jsonify({"ok": ok})


@app.route("/users/update_display_name", methods=["POST"])
@require_auth
def api_users_update_display_name():
    data = request.get_json(force=True)
    conn = _init_db()
    cursor = conn.execute("UPDATE users SET display_name=? WHERE username=?", (data["new_display_name"], data["target_username"]))
    ok = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return jsonify({"ok": ok})


# %% [markdown]
# ## 聊天会话端点


# %%
@app.route("/chat_sessions/<int:user_id>", methods=["GET"])
@require_auth
def api_chat_sessions_list(user_id: int):
    conn = _init_db()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT cs.id, cs.session_id, cs.name, cs.is_active, cs.created_at, cs.updated_at, "
        "(SELECT COUNT(*) FROM qa_history WHERE session_id=cs.session_id) as message_count "
        "FROM chat_sessions cs WHERE cs.user_id=? ORDER BY cs.updated_at DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    return jsonify({"sessions": [dict(r) for r in rows]})


@app.route("/chat_sessions/create", methods=["POST"])
@require_auth
def api_chat_sessions_create():
    data = request.get_json(force=True)
    session_id = f"chat_{data['user_id']}_{secrets.token_urlsafe(16)}"
    conn = _init_db()
    conn.execute("INSERT INTO chat_sessions (user_id, session_id, name) VALUES (?,?,?)",
                 (data["user_id"], session_id, data.get("name", "新对话")))
    conn.commit()
    conn.close()
    return jsonify({"ok": True, "session_id": session_id})


@app.route("/chat_sessions/create_with_id", methods=["POST"])
@require_auth
def api_chat_sessions_create_with_id():
    data = request.get_json(force=True)
    conn = _init_db()
    conn.execute("INSERT OR IGNORE INTO chat_sessions (user_id, session_id, name) VALUES (?,?,?)",
                 (data["user_id"], data["session_id"], data.get("name", "默认对话")))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


@app.route("/chat_sessions/rename", methods=["POST"])
@require_auth
def api_chat_sessions_rename():
    data = request.get_json(force=True)
    conn = _init_db()
    cursor = conn.execute("UPDATE chat_sessions SET name=?, updated_at=? WHERE session_id=?",
                          (data["new_name"], datetime.now().isoformat(), data["session_id"]))
    ok = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return jsonify({"ok": ok})


@app.route("/chat_sessions/delete", methods=["POST"])
@require_auth
def api_chat_sessions_delete():
    data = request.get_json(force=True)
    conn = _init_db()
    cursor = conn.execute("DELETE FROM chat_sessions WHERE session_id=?", (data["session_id"],))
    ok = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return jsonify({"ok": ok})


@app.route("/chat_sessions/set_active", methods=["POST"])
@require_auth
def api_chat_sessions_set_active():
    data = request.get_json(force=True)
    conn = _init_db()
    conn.execute("UPDATE chat_sessions SET is_active=0 WHERE user_id=?", (data["user_id"],))
    conn.execute("UPDATE chat_sessions SET is_active=1, updated_at=? WHERE session_id=?",
                 (datetime.now().isoformat(), data["session_id"]))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


@app.route("/chat_sessions/<int:user_id>/active", methods=["GET"])
@require_auth
def api_chat_sessions_active(user_id: int):
    conn = _init_db()
    row = conn.execute("SELECT session_id FROM chat_sessions WHERE user_id=? AND is_active=1", (user_id,)).fetchone()
    if not row:
        row = conn.execute("SELECT session_id FROM chat_sessions WHERE user_id=? ORDER BY updated_at DESC LIMIT 1", (user_id,)).fetchone()
    conn.close()
    if row:
        return jsonify({"found": True, "session_id": row[0]})
    return jsonify({"found": False}), 404


# %% [markdown]
# ## 问答历史端点


# %%
@app.route("/qa/save", methods=["POST"])
@require_auth
def api_qa_save():
    data = request.get_json(force=True)
    metadata_json = json.dumps(data.get("metadata")) if data.get("metadata") else None
    conn = _init_db()
    conn.execute("INSERT INTO qa_history (user_id, session_id, question, answer, metadata) VALUES (?,?,?,?,?)",
                 (data["user_id"], data["session_id"], data["question"], data["answer"], metadata_json))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


@app.route("/qa/<int:user_id>", methods=["GET"])
@require_auth
def api_qa_history(user_id: int):
    session_id = request.args.get("session_id")
    limit = request.args.get("limit", 50, type=int)
    offset = request.args.get("offset", 0, type=int)
    conn = _init_db()
    conn.row_factory = sqlite3.Row
    if session_id:
        rows = conn.execute(
            "SELECT session_id, question, answer, metadata, created_at FROM qa_history "
            "WHERE session_id=? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (session_id, limit, offset),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT session_id, question, answer, metadata, created_at FROM qa_history "
            "WHERE user_id=? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (user_id, limit, offset),
        ).fetchall()
    conn.close()
    history = []
    for row in rows:
        item = dict(row)
        if item["metadata"]:
            try:
                item["metadata"] = json.loads(item["metadata"])
            except Exception:
                item["metadata"] = {}
        history.append(item)
    return jsonify({"history": history})


@app.route("/qa/by_session/<session_id>", methods=["GET"])
@require_auth
def api_qa_by_session(session_id: str):
    """按 session_id 查询历史（用于 restore_history_for_session）"""
    conn = _init_db()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT question, answer, created_at FROM qa_history WHERE session_id=? ORDER BY created_at ASC",
        (session_id,),
    ).fetchall()
    conn.close()
    return jsonify({"history": [{"timestamp": r["created_at"], "question": r["question"], "answer": r["answer"]} for r in rows]})


# %% [markdown]
# ## 审计日志端点


# %%
@app.route("/audit/log", methods=["POST"])
@require_auth
def api_audit_log():
    data = request.get_json(force=True)
    conn = _init_db()
    conn.execute("INSERT INTO audit_log (user_id, action, details, ip_address) VALUES (?,?,?,?)",
                 (data.get("user_id"), data["action"], data.get("details", ""), data.get("ip_address", "")))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})


@app.route("/audit/logs", methods=["GET"])
@require_auth
def api_audit_logs():
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 50, type=int)
    username = request.args.get("username")
    action = request.args.get("action")
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    conn = _init_db()
    conn.row_factory = sqlite3.Row
    conditions = []
    params = []
    if username:
        conditions.append("a.user_id IN (SELECT id FROM users WHERE username LIKE ?)")
        params.append(f"%{username}%")
    if action:
        conditions.append("a.action=?")
        params.append(action)
    if start_date:
        conditions.append("a.timestamp >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("a.timestamp <= ?")
        params.append(f"{end_date} 23:59:59")
    where_clause = " AND ".join(conditions) if conditions else "1=1"
    total = conn.execute(f"SELECT COUNT(*) FROM audit_log a WHERE {where_clause}", params).fetchone()[0]
    offset = (page - 1) * per_page
    rows = conn.execute(
        f"SELECT a.id, a.user_id, COALESCE(u.username, '(已删除用户)') as username, "
        f"COALESCE(u.display_name, '未知') as display_name, a.action, a.details, a.ip_address, a.timestamp "
        f"FROM audit_log a LEFT JOIN users u ON a.user_id=u.id "
        f"WHERE {where_clause} ORDER BY a.timestamp DESC LIMIT ? OFFSET ?",
        params + [per_page, offset],
    ).fetchall()
    conn.close()
    return jsonify({"total": total, "logs": [dict(r) for r in rows], "page": page, "per_page": per_page,
                    "total_pages": max(1, (total + per_page - 1) // per_page)})


@app.route("/audit/actions", methods=["GET"])
@require_auth
def api_audit_actions():
    conn = _init_db()
    rows = conn.execute("SELECT DISTINCT action FROM audit_log ORDER BY action").fetchall()
    conn.close()
    return jsonify({"actions": [r[0] for r in rows]})


@app.route("/audit/clear", methods=["POST"])
@require_auth
def api_audit_clear():
    data = request.get_json(force=True)
    cutoff = datetime.now() - timedelta(days=data.get("before_days", 90))
    conn = _init_db()
    cursor = conn.execute("DELETE FROM audit_log WHERE timestamp < ?", (cutoff.strftime("%Y-%m-%d %H:%M:%S"),))
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return jsonify({"ok": True, "deleted": deleted})


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
