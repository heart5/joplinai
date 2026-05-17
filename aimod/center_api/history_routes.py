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
# # History Blueprint — 笔记本处理历史端点

# %%
import json
import sqlite3

from flask import Blueprint, jsonify, request

# %%
from aimod.center_api import _init_db, log, require_auth

__all__ = ["history_bp"]

history_bp = Blueprint("history", __name__)


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
            data["run_id"], data["notebook_title"], data["timestamp"],
            data.get("total_notes", 0), data.get("updated_count", 0),
            data.get("failed_count", 0), data.get("notes_added_count", 0),
            data.get("notes_removed_count", 0),
            chunk_stats.get("total_chunks", 0), chunk_stats.get("upserted", 0),
            chunk_stats.get("skipped", 0), chunk_stats.get("orphans_cleaned", 0),
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
            data["run_id"], data["timestamp"], data["ollama_embedding_model"],
            data["notebook_count"], data["total_notes_processed"],
            data["total_chunks_processed"], data["total_notes_added"],
            data["total_notes_removed"], data.get("success", True),
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

    all_added, all_removed = [], []
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
#
# %%
@history_bp.route("/history/notebook_record", methods=["POST"])
@require_auth
def api_history_notebook_record():
    data = request.get_json(force=True)
    chunk_stats = data.get("chunk_stats", {})
    log.info(
        f"笔记本处理完成: {data.get('notebook_title', '?')}, "
        f"笔记={data.get('updated_count', 0)}更新/{data.get('failed_count', 0)}失败/{data.get('total_notes', 0)}总, "
        f"块={chunk_stats.get('upserted', 0)}增/{chunk_stats.get('skipped', 0)}跳/{chunk_stats.get('orphans_cleaned', 0)}清"
    )
    history_add_notebook_record(data)
    return jsonify({"ok": True})
#
#
@history_bp.route("/history/finalize_run", methods=["POST"])
@require_auth
def api_history_finalize_run():
    data = request.get_json(force=True)
    history_finalize_run(data)
    return jsonify({"ok": True})
#
#
@history_bp.route("/history/cumulative_stats", methods=["GET"])
@require_auth
def api_history_cumulative_stats():
    days = request.args.get("days", type=int)
    return jsonify(history_cumulative_stats(days))
#
#
@history_bp.route("/history/change_analysis", methods=["GET"])
@require_auth
def api_history_change_analysis():
    notebook_title = request.args.get("notebook_title")
    days = request.args.get("days", 30, type=int)
    return jsonify(history_change_analysis(notebook_title, days))
#
#
@history_bp.route("/history/efficiency_metrics", methods=["GET"])
@require_auth
def api_history_efficiency_metrics():
    days = request.args.get("days", 30, type=int)
    return jsonify(history_efficiency_metrics(days))
