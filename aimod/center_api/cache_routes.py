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
# # Cache Blueprint — DeepSeek 缓存 + 探测缓存端点

# %%
import json
import sqlite3
from datetime import datetime
from typing import Optional

from flask import Blueprint, jsonify, request

# %%
from aimod.center_api import (
    DB_PATH,
    VALIDATION_THRESHOLD,
    _init_db,
    _probe_set_counter,
    _probe_cache_limit,
    log,
    require_auth,
)

# %%
import pathmagic

__all__ = ["cache_bp"]

cache_bp = Blueprint("cache", __name__)


# %% [markdown]
# # DeepSeek 缓存操作

# %%
def deepseek_cache_get(content_hash: str, task: str) -> dict:
    cache_key = f"{content_hash}_{task}"
    conn = _init_db()
    row = conn.execute(
        "SELECT result, hit_count, total_hits FROM deepseek_cache "
        "WHERE cache_key=? AND (julianday('now') - julianday(created_at)) < 90",
        (cache_key,),
    ).fetchone()

    if not row:
        conn.close()
        return {"found": False, "cache_key": cache_key}

    cached_result, current_hit_count, total_hits = row
    new_hit_count = current_hit_count + 1
    new_total_hits = total_hits + 1
    now_iso = datetime.now().isoformat()
    should_validate = new_hit_count >= VALIDATION_THRESHOLD

    if should_validate:
        conn.execute(
            "UPDATE deepseek_cache SET hit_count=0, total_hits=?, last_accessed=?, "
            "last_validated_at=?, validation_result='pending' WHERE cache_key=?",
            (new_total_hits, now_iso, now_iso, cache_key),
        )
    else:
        conn.execute(
            "UPDATE deepseek_cache SET hit_count=?, total_hits=?, last_accessed=? WHERE cache_key=?",
            (new_hit_count, new_total_hits, now_iso, cache_key),
        )
    conn.commit()
    conn.close()
    return {
        "found": True,
        "cache_key": cache_key,
        "content": cached_result,
        "requires_validation": should_validate,
        "current_hit_count": 0 if should_validate else new_hit_count,
        "total_hits": new_total_hits,
    }


def deepseek_cache_set(content_hash: str, task: str, result: str):
    cache_key = f"{content_hash}_{task}"
    now_iso = datetime.now().isoformat()
    conn = _init_db()
    conn.execute(
        "INSERT OR REPLACE INTO deepseek_cache "
        "(cache_key, content_hash, task, result, created_at, last_accessed, "
        "last_validated_at, hit_count, total_hits, validation_result) "
        "VALUES (?,?,?,?,?,?,NULL,0,0,NULL)",
        (cache_key, content_hash, task, result, now_iso, now_iso),
    )
    conn.commit()
    conn.close()


def deepseek_cache_validate(cache_key: str, new_result: Optional[str], success: bool):
    conn = _init_db()
    now_iso = datetime.now().isoformat()
    if not success:
        conn.execute(
            "UPDATE deepseek_cache SET last_validated_at=?, validation_result='failed' WHERE cache_key=?",
            (now_iso, cache_key),
        )
    elif new_result is None:
        conn.execute(
            "UPDATE deepseek_cache SET last_validated_at=?, validation_result='valid' WHERE cache_key=?",
            (now_iso, cache_key),
        )
    else:
        conn.execute(
            "UPDATE deepseek_cache SET result=?, created_at=?, last_validated_at=?, validation_result='updated' WHERE cache_key=?",
            (new_result, now_iso, now_iso, cache_key),
        )
    conn.commit()
    conn.close()


def deepseek_cache_stats(cache_key: str = None) -> dict:
    conn = _init_db()
    conn.row_factory = sqlite3.Row
    if cache_key:
        row = conn.execute("SELECT * FROM deepseek_cache WHERE cache_key=?", (cache_key,)).fetchone()
        conn.close()
        return dict(row) if row else {}
    else:
        row = conn.execute(
            "SELECT COUNT(*) as total, SUM(total_hits) as total_hits, SUM(hit_count) as current_hits FROM deepseek_cache"
        ).fetchone()
        stats = dict(row) if row else {}
        rows = conn.execute(
            "SELECT validation_result, COUNT(*) as count FROM deepseek_cache "
            "WHERE validation_result IS NOT NULL GROUP BY validation_result"
        ).fetchall()
        stats["validation_breakdown"] = {r["validation_result"]: r["count"] for r in rows}
        conn.close()
        return stats


def deepseek_cache_report() -> dict:
    conn = _init_db()
    conn.row_factory = sqlite3.Row
    report = {}
    row = conn.execute("SELECT COUNT(*) as total FROM deepseek_cache").fetchone()
    report["total"] = row["total"] if row else 0
    row = conn.execute(
        "SELECT COUNT(*) as active FROM deepseek_cache "
        "WHERE last_accessed >= datetime('now', '-7 days')"
    ).fetchone()
    report["recent_active"] = row["active"] if row else 0
    report["validation_threshold"] = VALIDATION_THRESHOLD
    report["avg_hits"] = conn.execute(
        "SELECT AVG(total_hits) FROM deepseek_cache"
    ).fetchone()[0] or 0

    report["by_task"] = [dict(r) for r in conn.execute(
        "SELECT task, COUNT(*) as count FROM deepseek_cache GROUP BY task"
    ).fetchall()]
    report["validation_status"] = [dict(r) for r in conn.execute(
        "SELECT validation_result, COUNT(*) as count FROM deepseek_cache "
        "WHERE validation_result IS NOT NULL GROUP BY validation_result"
    ).fetchall()]
    report["hit_distribution"] = [dict(r) for r in conn.execute(
        "SELECT CASE WHEN total_hits<10 THEN '<10' WHEN total_hits<100 THEN '10-100' "
        "ELSE '>100' END as range, COUNT(*) as count FROM deepseek_cache GROUP BY range"
    ).fetchall()]

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
        with pathmagic.Context():
            from func.jpfuncs import getinivaluefromcloud
        val = getinivaluefromcloud("joplinai", "probe_cache_limit")
        return int(val) if val else 10000
    except Exception:
        return 10000


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
        "safe_len": row[0], "snippet": row[1], "model_name": row[2],
        "chunk_size": row[3], "timestamp": row[4],
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
    conn = _init_db()
    conn.row_factory = sqlite3.Row

    row = conn.execute("SELECT COUNT(*) as count FROM probe_cache").fetchone()
    total = row["count"] if row else 0

    by_model = [dict(r) for r in conn.execute("""
        SELECT model_name, COUNT(*) as count, AVG(safe_len) as avg_safe_len,
               AVG(chunk_size) as avg_chunk_size, MAX(last_accessed) as last_used
        FROM probe_cache GROUP BY model_name ORDER BY count DESC
    """).fetchall()]

    by_chunk = [dict(r) for r in conn.execute("""
        SELECT chunk_size, COUNT(*) as count, AVG(safe_len) as avg_safe_len
        FROM probe_cache GROUP BY chunk_size ORDER BY chunk_size
    """).fetchall()]

    row = conn.execute(
        "SELECT MIN(safe_len) as min_len, MAX(safe_len) as max_len, AVG(safe_len) as avg_len FROM probe_cache"
    ).fetchone()
    len_stats = dict(row) if row else {}

    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM probe_cache WHERE last_accessed >= datetime('now', '-7 days')"
    ).fetchone()
    recent = row["cnt"] if row else 0

    daily_new = [dict(r) for r in conn.execute("""
        SELECT DATE(created_at) as date, COUNT(*) as new_entries
        FROM probe_cache WHERE DATE(created_at) >= DATE('now', '-30 days')
        GROUP BY DATE(created_at) ORDER BY date
    """).fetchall()]

    conn.close()
    return {
        "total": total, "limit": _get_probe_cache_limit(),
        "by_model": by_model, "by_chunk_size": by_chunk,
        "safe_len_stats": len_stats, "recent_active": recent,
        "daily_new": daily_new,
    }


# %% [markdown]
# # Flask 端点

@cache_bp.route("/cache/deepseek/get", methods=["POST"])
@require_auth
def api_ds_cache_get():
    data = request.get_json(force=True)
    return jsonify(deepseek_cache_get(data["content_hash"], data["task"]))


@cache_bp.route("/cache/deepseek/set", methods=["POST"])
@require_auth
def api_ds_cache_set():
    data = request.get_json(force=True)
    deepseek_cache_set(data["content_hash"], data["task"], data["result"])
    return jsonify({"ok": True})


@cache_bp.route("/cache/deepseek/validate", methods=["POST"])
@require_auth
def api_ds_cache_validate():
    data = request.get_json(force=True)
    deepseek_cache_validate(data["cache_key"], data.get("new_result"), data["success"])
    return jsonify({"ok": True})


@cache_bp.route("/cache/deepseek/stats")
@require_auth
def api_ds_cache_stats():
    cache_key = request.args.get("cache_key")
    return jsonify(deepseek_cache_stats(cache_key))


@cache_bp.route("/cache/deepseek/report")
@require_auth
def api_ds_cache_report():
    return jsonify(deepseek_cache_report())


@cache_bp.route("/cache/probe/get/<text_md5>", methods=["GET"])
@require_auth
def api_probe_cache_get(text_md5: str):
    result = probe_cache_get(text_md5)
    if result:
        return jsonify(result)
    return jsonify({"found": False}), 404


@cache_bp.route("/cache/probe/set", methods=["POST"])
@require_auth
def api_probe_cache_set():
    data = request.get_json(force=True)
    probe_cache_set(
        data["text_md5"], data["safe_len"], data["snippet"],
        data["model_name"], data["chunk_size"],
    )
    return jsonify({"ok": True})


@cache_bp.route("/cache/probe/stats")
@require_auth
def api_probe_cache_stats():
    return jsonify(probe_cache_stats())


@cache_bp.route("/cache/probe/report")
@require_auth
def api_probe_cache_report():
    return jsonify(probe_cache_report())
