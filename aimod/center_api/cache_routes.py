# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     split_at_heading: true
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
# # Cache Blueprint — 增强缓存 + 探测缓存端点

# %%
import itertools
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
    log,
    require_auth,
)

# %%
import pathmagic

__all__ = ["cache_bp"]

cache_bp = Blueprint("cache", __name__)


# %% [markdown]
# # 增强缓存操作

# %%
def enhance_cache_get(content_hash: str, task: str, model: str = "") -> dict:
    """按 content_hash + task 列匹配（不依赖 cache_key 格式）。"""
    conn = _init_db()
    row = conn.execute(
        "SELECT result, hit_count, total_hits, cache_key FROM enhance_cache "
        "WHERE content_hash=? AND task=? ORDER BY created_at DESC LIMIT 1",
        (content_hash, task),
    ).fetchone()

    if not row:
        conn.close()
        return {"found": False, "cache_key": f"{content_hash}_{task}"}

    cached_result, current_hit_count, total_hits, effective_key = row
    new_hit_count = current_hit_count + 1
    new_total_hits = total_hits + 1
    now_iso = datetime.now().isoformat()
    should_validate = new_hit_count >= VALIDATION_THRESHOLD

    if should_validate:
        conn.execute(
            "UPDATE enhance_cache SET hit_count=0, total_hits=?, last_accessed=?, "
            "last_validated_at=?, validation_result='pending' WHERE cache_key=?",
            (new_total_hits, now_iso, now_iso, effective_key),
        )
    else:
        conn.execute(
            "UPDATE enhance_cache SET hit_count=?, total_hits=?, last_accessed=? WHERE cache_key=?",
            (new_hit_count, new_total_hits, now_iso, effective_key),
        )
    conn.commit()
    conn.close()
    return {
        "found": True,
        "cache_key": effective_key,
        "content": cached_result,
        "requires_validation": should_validate,
        "current_hit_count": 0 if should_validate else new_hit_count,
        "total_hits": new_total_hits,
    }


_enhance_set_counter = itertools.count()


def _get_cache_limit() -> int:
    try:
        import pathmagic
        with pathmagic.Context():
            from func.jpfuncs import getinivaluefromcloud
        val = getinivaluefromcloud("joplinai", "cache_limit")
        return int(val) if val else 50000
    except Exception:
        return 50000


def enhance_cache_set(content_hash: str, task: str, result: str, model: str = ""):
    cache_key = f"{content_hash}_{task}_{model}" if model else f"{content_hash}_{task}"
    now_iso = datetime.now().isoformat()
    conn = _init_db()
    conn.execute(
        "INSERT OR REPLACE INTO enhance_cache "
        "(cache_key, content_hash, task, result, created_at, last_accessed, "
        "last_validated_at, hit_count, total_hits, validation_result) "
        "VALUES (?,?,?,?,?,?,NULL,0,0,NULL)",
        (cache_key, content_hash, task, result, now_iso, now_iso),
    )
    conn.commit()

    if next(_enhance_set_counter) % 1000 == 0:
        limit = _get_cache_limit()
        count = conn.execute("SELECT COUNT(*) FROM enhance_cache").fetchone()[0]
        if count > limit:
            delete_n = max(1, count // 10)
            conn.execute(
                "DELETE FROM enhance_cache WHERE cache_key IN ("
                "SELECT cache_key FROM enhance_cache ORDER BY last_accessed ASC LIMIT ?"
                ")", (delete_n,),
            )
            conn.commit()
            log.info(f"[增强缓存] 淘汰 {delete_n} 条（总量 {count} > 上限 {limit}）")

    conn.close()


def enhance_cache_validate(cache_key: str, new_result: Optional[str], success: bool):
    conn = _init_db()
    now_iso = datetime.now().isoformat()
    if not success:
        conn.execute(
            "UPDATE enhance_cache SET last_validated_at=?, validation_result='failed' WHERE cache_key=?",
            (now_iso, cache_key),
        )
    elif new_result is None:
        conn.execute(
            "UPDATE enhance_cache SET last_validated_at=?, validation_result='valid' WHERE cache_key=?",
            (now_iso, cache_key),
        )
    else:
        conn.execute(
            "UPDATE enhance_cache SET result=?, created_at=?, last_validated_at=?, validation_result='updated' WHERE cache_key=?",
            (new_result, now_iso, now_iso, cache_key),
        )
    conn.commit()
    conn.close()


def enhance_cache_stats(cache_key: str = None) -> dict:
    conn = _init_db()
    conn.row_factory = sqlite3.Row
    if cache_key:
        row = conn.execute("SELECT * FROM enhance_cache WHERE cache_key=?", (cache_key,)).fetchone()
        conn.close()
        return dict(row) if row else {}
    else:
        row = conn.execute(
            "SELECT COUNT(*) as total, SUM(total_hits) as total_hits, SUM(hit_count) as current_hits FROM enhance_cache"
        ).fetchone()
        stats = dict(row) if row else {}
        rows = conn.execute(
            "SELECT validation_result, COUNT(*) as count FROM enhance_cache "
            "WHERE validation_result IS NOT NULL GROUP BY validation_result"
        ).fetchall()
        stats["validation_breakdown"] = {r["validation_result"]: r["count"] for r in rows}
        conn.close()
        return stats


def enhance_cache_report() -> dict:
    conn = _init_db()
    conn.row_factory = sqlite3.Row
    report = {}
    row = conn.execute("SELECT COUNT(*) as total FROM enhance_cache").fetchone()
    report["total"] = row["total"] if row else 0
    row = conn.execute(
        "SELECT COUNT(*) as active FROM enhance_cache "
        "WHERE last_accessed >= datetime('now', '-7 days')"
    ).fetchone()
    report["recent_active"] = row["active"] if row else 0
    report["validation_threshold"] = VALIDATION_THRESHOLD
    report["avg_hits"] = conn.execute(
        "SELECT AVG(total_hits) FROM enhance_cache"
    ).fetchone()[0] or 0

    report["by_task"] = [dict(r) for r in conn.execute(
        "SELECT task, COUNT(*) as count FROM enhance_cache GROUP BY task"
    ).fetchall()]

    # 按模型细分：从 cache_key 解析模型名（格式: {hash}_{task}_{model}）
    # 迁移日 (2026-05-19) 前的无模型历史记录 → 推断为 deepseek-chat；之后的保留"未知"
    model_counts = {}
    rows = conn.execute(
        "SELECT cache_key, task, created_at FROM enhance_cache"
    ).fetchall()
    for r in rows:
        ck, task, created = r["cache_key"], r["task"], r["created_at"]
        if f"_{task}_" in ck:
            model = ck.split(f"_{task}_", 1)[1]
        elif task.startswith("vision_desc:"):
            model = task.split(":", 1)[1]
        elif created and created < "2026-05-19":
            model = "deepseek-chat"
        else:
            model = "未知"
        model_counts[model] = model_counts.get(model, 0) + 1
    report["by_model"] = sorted(
        [{"model": m, "count": c} for m, c in model_counts.items()],
        key=lambda x: x["count"], reverse=True,
    )
    report["validation_status"] = [dict(r) for r in conn.execute(
        "SELECT validation_result, COUNT(*) as count FROM enhance_cache "
        "WHERE validation_result IS NOT NULL GROUP BY validation_result"
    ).fetchall()]
    report["hit_distribution"] = [dict(r) for r in conn.execute(
        "SELECT CASE WHEN total_hits<10 THEN '<10' WHEN total_hits<100 THEN '10-100' "
        "ELSE '>100' END as range, COUNT(*) as count FROM enhance_cache GROUP BY range"
    ).fetchall()]

    daily = [dict(r) for r in conn.execute("""
        SELECT DATE(created_at) as date, COUNT(*) as new_entries
        FROM enhance_cache WHERE DATE(created_at) >= DATE('now', '-30 days')
        GROUP BY DATE(created_at) ORDER BY date
    """).fetchall()]
    cum = [dict(r) for r in conn.execute("""
        SELECT DATE(created_at) as date,
               SUM(COUNT(*)) OVER (ORDER BY DATE(created_at)) as cumulative
        FROM enhance_cache WHERE DATE(created_at) >= DATE('now', '-30 days')
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
# # Flask 端点

# %%
@cache_bp.route("/cache/enhance/get", methods=["POST"])
@require_auth
def api_enh_cache_get():
    data = request.get_json(force=True)
    return jsonify(enhance_cache_get(data["content_hash"], data["task"], data.get("model", "")))


@cache_bp.route("/cache/enhance/set", methods=["POST"])
@require_auth
def api_enh_cache_set():
    data = request.get_json(force=True)
    enhance_cache_set(data["content_hash"], data["task"], data["result"], data.get("model", ""))
    log.info(f"缓存写入: task={data['task']}")
    return jsonify({"ok": True})


@cache_bp.route("/cache/enhance/validate", methods=["POST"])
@require_auth
def api_enh_cache_validate():
    data = request.get_json(force=True)
    enhance_cache_validate(data["cache_key"], data.get("new_result"), data["success"])
    return jsonify({"ok": True})


@cache_bp.route("/cache/enhance/stats")
@require_auth
def api_enh_cache_stats():
    cache_key = request.args.get("cache_key")
    return jsonify(enhance_cache_stats(cache_key))


@cache_bp.route("/cache/enhance/report")
@require_auth
def api_enh_cache_report():
    return jsonify(enhance_cache_report())

