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
# # Monitor Blueprint — 笔记监测数据 API

# %%
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

from flask import Blueprint, jsonify, request

# %%
from aimod.center_api import log, require_auth

__all__ = ["monitor_bp"]

monitor_bp = Blueprint("monitor", __name__)

MONITOR_DB = Path("/home/baiyefeng/codebase/happyjoplin/data/monitor.db")


def _get_monitor_conn():
    conn = sqlite3.connect(str(MONITOR_DB))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# %% [markdown]
# # /monitor/status — 综合状态

# %%
@monitor_bp.route("/monitor/status")
@require_auth
def monitor_status():
    conn = _get_monitor_conn()
    try:
        active = conn.execute("SELECT COUNT(*) as cnt FROM notes WHERE is_active=1").fetchone()["cnt"]
        persons = [r["person"] for r in conn.execute(
            "SELECT DISTINCT person FROM notes WHERE person!='' AND is_active=1"
        ).fetchall()]
        last_snapshot = conn.execute(
            "SELECT captured_at FROM snapshots ORDER BY captured_at DESC LIMIT 1"
        ).fetchone()
        last_collect = last_snapshot["captured_at"] if last_snapshot else None
        pending = conn.execute("SELECT COUNT(*) as cnt FROM pending_changes").fetchone()["cnt"]
        alerts = conn.execute("SELECT COUNT(*) as cnt FROM content_alerts WHERE resolved=0").fetchone()["cnt"]
        spark_total = conn.execute("SELECT COUNT(*) as cnt FROM spark_log").fetchone()["cnt"]
        return jsonify({
            "status": "ok",
            "active_notes": active,
            "persons": persons,
            "person_count": len(persons),
            "last_collect": last_collect,
            "pending_changes": pending,
            "unresolved_alerts": alerts,
            "spark_total": spark_total,
        })
    finally:
        conn.close()


# %% [markdown]
# # /monitor/heatmap — 全员热力图数据

# %%
@monitor_bp.route("/monitor/heatmap")
@require_auth
def monitor_heatmap():
    weeks = request.args.get("weeks", 12, type=int)
    end_date = date.today()
    start_date = end_date - timedelta(weeks=weeks)
    conn = _get_monitor_conn()
    try:
        persons = [r["person"] for r in conn.execute(
            "SELECT DISTINCT person FROM notes WHERE person!='' AND is_active=1 ORDER BY person"
        ).fetchall()]
        notes = conn.execute(
            "SELECT note_id, title, person FROM notes WHERE is_active=1"
        ).fetchall()
        person_notes = {}
        for n in notes:
            person_notes.setdefault(n["person"], []).append(n["note_id"])

        rows = conn.execute(
            """SELECT ds.note_id, ds.entry_date, ds.word_count, n.person
               FROM daily_stats ds JOIN notes n ON ds.note_id=n.note_id
               WHERE ds.entry_date >= ? AND ds.entry_date <= ? AND n.is_active=1
               ORDER BY ds.entry_date""",
            (start_date.isoformat(), end_date.isoformat()),
        ).fetchall()

        # Build per-person daily map: {person: {date_str: total_wc}}
        person_daily = {p: {} for p in persons}
        for r in rows:
            d = r["entry_date"]
            person_daily[r["person"]][d] = person_daily[r["person"]].get(d, 0) + r["word_count"]

        # Build date list
        dates = []
        d = start_date
        while d <= end_date:
            dates.append(d.isoformat())
            d += timedelta(days=1)

        return jsonify({
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "persons": persons,
            "dates": dates,
            "data": person_daily,
        })
    finally:
        conn.close()


# %% [markdown]
# # /monitor/spark/today — 今日火花摘语

# %%
@monitor_bp.route("/monitor/spark/today")
@require_auth
def monitor_spark_today():
    person_filter = request.args.get("person")
    today_str = date.today().isoformat()
    conn = _get_monitor_conn()
    try:
        query = "SELECT person, quote_text, source_date FROM spark_log WHERE used_date=?"
        params = [today_str]
        if person_filter:
            query += " AND person=?"
            params.append(person_filter)
        rows = conn.execute(query, params).fetchall()
        quotes = [{"person": r["person"], "text": r["quote_text"], "source_date": r["source_date"]} for r in rows]
        return jsonify({"date": today_str, "quotes": quotes, "count": len(quotes)})
    finally:
        conn.close()


# %% [markdown]
# # /monitor/spark/pool — 摘语池概览

# %%
@monitor_bp.route("/monitor/spark/pool")
@require_auth
def monitor_spark_pool():
    conn = _get_monitor_conn()
    try:
        total = conn.execute("SELECT COUNT(*) as cnt FROM spark_log").fetchone()["cnt"]
        year_rows = conn.execute(
            "SELECT source_date, COUNT(*) as cnt FROM spark_log WHERE source_date!='' GROUP BY source_date ORDER BY source_date"
        ).fetchall()
        by_year = {}
        for r in year_rows:
            y = r["source_date"][:4] if len(r["source_date"]) >= 4 else "unknown"
            by_year[y] = by_year.get(y, 0) + r["cnt"]
        person_rows = conn.execute(
            "SELECT person, COUNT(*) as cnt FROM spark_log GROUP BY person ORDER BY cnt DESC"
        ).fetchall()
        by_person = {r["person"]: r["cnt"] for r in person_rows}
        return jsonify({
            "total": total,
            "by_year": by_year,
            "by_person": by_person,
        })
    finally:
        conn.close()
