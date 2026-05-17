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
# # HistoryClient
# 历史数据库客户端 — 远程优先 + 本地 SQLite 回退

# %%
import json
import logging
import sqlite3
from typing import Dict, Optional

import requests

# %%
import pathmagic

with pathmagic.Context():
    try:
        from func.first import getdirmain
        from func.logme import log
    except ImportError as e:
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger(__name__)
        log.error(f"导入项目模块失败: {e}")

CENTER_DB_PATH = getdirmain() / "data" / "joplinai_center.db"


# %%
__all__ = ["HistoryClient"]

class HistoryClient:
    """历史数据库客户端 — 远程优先 + 本地 SQLite 回退"""

    def __init__(self, remote_url: str, api_key: str):
        self.remote_url = remote_url.rstrip("/")
        self.auth_headers = {"X-API-Key": api_key}
        self._local_db = str(CENTER_DB_PATH)

    def _request(self, method: str, path: str, **kwargs) -> Optional[requests.Response]:
        try:
            resp = requests.request(
                method,
                f"{self.remote_url}{path}",
                headers=self.auth_headers,
                timeout=5,
                **kwargs,
            )
            if resp.ok:
                return resp
            log.warning(f"远程历史 {method} {path} 返回 {resp.status_code}")
        except Exception as e:
            log.warning(f"远程历史 {method} {path} 失败: {e}")
        return None

    # ---- 写入 ----

    def add_notebook_record(self, notebook_title: str, stats: Dict,
                            run_id: str, timestamp: str):
        chunk_stats = stats.get("chunk_stats", {})
        payload = {
            "run_id": run_id,
            "notebook_title": notebook_title,
            "timestamp": timestamp,
            "total_notes": stats.get("total_notes", 0),
            "updated_count": stats.get("updated_count", 0),
            "failed_count": len(stats.get("failed_notes", [])),
            "notes_added_count": len(stats.get("notes_added", [])),
            "notes_removed_count": len(stats.get("notes_removed", [])),
            "chunk_stats": {
                "total_chunks": chunk_stats.get("total_chunks", 0),
                "upserted": chunk_stats.get("upserted", 0),
                "skipped": chunk_stats.get("skipped", 0),
                "orphans_cleaned": chunk_stats.get("orphans_cleaned", 0),
            },
            "notes_added_list": stats.get("notes_added", []),
            "notes_removed_list": stats.get("notes_removed", []),
            "failed_notes_list": stats.get("failed_notes", []),
        }
        if self._request("POST", "/history/notebook_record", json=payload):
            log.debug(f"远程历史写入: {notebook_title}")
            return
        self._local_add_notebook_record(notebook_title, stats, run_id, timestamp)

    def finalize_run(self, run_id: str, timestamp: str, ollama_embedding_model: str,
                     notebook_count: int, total_notes_processed: int,
                     total_chunks_processed: int, total_notes_added: int,
                     total_notes_removed: int, success: bool = True,
                     error_message: str = None):
        payload = {
            "run_id": run_id,
            "timestamp": timestamp,
            "ollama_embedding_model": ollama_embedding_model,
            "notebook_count": notebook_count,
            "total_notes_processed": total_notes_processed,
            "total_chunks_processed": total_chunks_processed,
            "total_notes_added": total_notes_added,
            "total_notes_removed": total_notes_removed,
            "success": success,
            "error_message": error_message,
        }
        if self._request("POST", "/history/finalize_run", json=payload):
            log.info(f"远程历史 finalize: {run_id}")
            return
        self._local_finalize_run(run_id, timestamp, ollama_embedding_model, notebook_count,
                                 total_notes_processed, total_chunks_processed,
                                 total_notes_added, total_notes_removed, success, error_message)

    # ---- 查询 ----

    def get_cumulative_stats(self, days: int = None) -> Dict:
        params = {}
        if days:
            params["days"] = days
        resp = self._request("GET", "/history/cumulative_stats", params=params)
        if resp is not None:
            return resp.json()
        return self._local_get_cumulative_stats(days)

    def get_change_analysis(self, notebook_title: str = None, days: int = None) -> Dict:
        params = {}
        if days:
            params["days"] = days
        if notebook_title:
            params["notebook"] = notebook_title
        resp = self._request("GET", "/history/change_analysis", params=params)
        if resp is not None:
            return resp.json()
        return self._local_get_change_analysis(notebook_title, days)

    def get_efficiency_metrics(self, days: int = None) -> Dict:
        params = {"days": days} if days else {}
        resp = self._request("GET", "/history/efficiency_metrics", params=params)
        if resp is not None:
            return resp.json()
        return self._local_get_efficiency_metrics(days)

    # ---- 本地回退 ----

    def _local_add_notebook_record(self, notebook_title: str, stats: Dict,
                                   run_id: str, timestamp: str):
        try:
            chunk_stats = stats.get("chunk_stats", {})
            conn = sqlite3.connect(self._local_db)
            conn.execute("""INSERT INTO notebook_history (
                run_id, notebook_title, timestamp,
                total_notes, updated_count, failed_count,
                notes_added_count, notes_removed_count,
                total_chunks, chunks_upserted, chunks_skipped, chunks_orphans_cleaned,
                notes_added_list, notes_removed_list, failed_notes_list
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                run_id, notebook_title, timestamp,
                stats.get("total_notes", 0), stats.get("updated_count", 0),
                len(stats.get("failed_notes", [])),
                len(stats.get("notes_added", [])), len(stats.get("notes_removed", [])),
                chunk_stats.get("total_chunks", 0), chunk_stats.get("upserted", 0),
                chunk_stats.get("skipped", 0), chunk_stats.get("orphans_cleaned", 0),
                json.dumps(stats.get("notes_added", []), ensure_ascii=False),
                json.dumps(stats.get("notes_removed", []), ensure_ascii=False),
                json.dumps(stats.get("failed_notes", []), ensure_ascii=False),
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            log.error(f"本地历史写入失败: {e}")

    def _local_finalize_run(self, run_id: str, timestamp: str, ollama_embedding_model: str,
                            notebook_count: int, total_notes_processed: int,
                            total_chunks_processed: int, total_notes_added: int,
                            total_notes_removed: int, success: bool, error_message: str):
        try:
            conn = sqlite3.connect(self._local_db)
            conn.execute("""INSERT OR REPLACE INTO global_run_history (
                run_id, timestamp, embedding_model, notebook_count,
                total_notes_processed, total_chunks_processed,
                total_notes_added, total_notes_removed, success, error_message
            ) VALUES (?,?,?,?,?,?,?,?,?,?)""", (
                run_id, timestamp, ollama_embedding_model, notebook_count,
                total_notes_processed, total_chunks_processed,
                total_notes_added, total_notes_removed, success, error_message,
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            log.error(f"本地历史 finalize 失败: {e}")

    def _local_get_cumulative_stats(self, days: int = None) -> Dict:
        try:
            conn = sqlite3.connect(self._local_db)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            time_condition = ""
            params = []
            if days:
                time_condition = "WHERE timestamp >= datetime('now', ?)"
                params.append(f"-{days} days")
            cursor.execute(f"""SELECT
                COUNT(DISTINCT run_id) as total_runs,
                COUNT(DISTINCT notebook_title) as total_notebooks_touched,
                SUM(total_notes) as total_notes_processed_all_time,
                SUM(total_chunks) as total_chunks_processed_all_time,
                SUM(notes_added_count) as total_notes_added_all_time,
                SUM(notes_removed_count) as total_notes_removed_all_time,
                SUM(chunks_upserted) as total_chunks_updated_all_time,
                SUM(chunks_orphans_cleaned) as total_orphans_cleaned_all_time
            FROM notebook_history {time_condition}""", params)
            cumulative = dict(cursor.fetchone())
            cursor.execute("""SELECT strftime('%Y-%W', timestamp) as week,
                COUNT(DISTINCT run_id) as runs_count, SUM(total_notes) as notes_processed,
                SUM(total_chunks) as chunks_processed, SUM(notes_added_count) as notes_added,
                SUM(notes_removed_count) as notes_removed
            FROM notebook_history WHERE timestamp >= datetime('now', '-90 days')
            GROUP BY week ORDER BY week DESC LIMIT 12""")
            weekly_trends = [dict(row) for row in cursor.fetchall()]
            cursor.execute(f"""SELECT notebook_title, COUNT(*) as process_count,
                SUM(total_notes) as total_notes, SUM(total_chunks) as total_chunks,
                MAX(timestamp) as last_processed
            FROM notebook_history {time_condition}
            GROUP BY notebook_title ORDER BY process_count DESC LIMIT 10""", params)
            top_notebooks = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return {"cumulative": cumulative, "weekly_trends": weekly_trends,
                    "top_notebooks": top_notebooks,
                    "analysis_period": f"最近{days}天" if days else "全部历史"}
        except Exception as e:
            log.error(f"本地累积统计失败: {e}")
            return {}

    def _local_get_change_analysis(self, notebook_title: str = None, days: int = None) -> Dict:
        try:
            conn = sqlite3.connect(self._local_db)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            where = []
            params = []
            if days:
                where.append("timestamp >= datetime('now', ?)")
                params.append(f"-{days} days")
            if notebook_title:
                where.append("notebook_title = ?")
                params.append(notebook_title)
            cursor.execute(
                f"SELECT notes_added_list, notes_removed_list FROM notebook_history WHERE {' AND '.join(where)}" if where else
                "SELECT notes_added_list, notes_removed_list FROM notebook_history",
                params)
            all_added, all_removed = [], []
            for row in cursor.fetchall():
                if row["notes_added_list"]:
                    all_added.extend(json.loads(row["notes_added_list"]))
                if row["notes_removed_list"]:
                    all_removed.extend(json.loads(row["notes_removed_list"]))
            unique_added = list(set(all_added))
            unique_removed = list(set(all_removed))
            conn.close()
            return {
                "analysis_period": f"最近{days}天",
                "notebook": notebook_title or "全局",
                "unique_notes_added": unique_added,
                "unique_notes_removed": unique_removed,
                "added_count": len(unique_added),
                "removed_count": len(unique_removed),
                "net_growth": len(unique_added) - len(unique_removed),
                "frequently_changed_notes": list(set(unique_added) & set(unique_removed)),
                "frequently_changed_count": len(set(unique_added) & set(unique_removed)),
            }
        except Exception as e:
            log.error(f"本地变动分析失败: {e}")
            return {}

    def _local_get_efficiency_metrics(self, days: int = None) -> Dict:
        try:
            conn = sqlite3.connect(self._local_db)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if days:
                cursor.execute("""SELECT
                    AVG(total_notes) as avg_notes_per_run, AVG(total_chunks) as avg_chunks_per_run,
                    SUM(chunks_upserted)*100.0/NULLIF(SUM(total_chunks),0) as avg_update_rate_percent,
                    SUM(chunks_skipped)*100.0/NULLIF(SUM(total_chunks),0) as avg_skip_rate_percent,
                    SUM(notes_added_count)*100.0/NULLIF(SUM(total_notes),0) as avg_addition_rate_percent,
                    SUM(notes_removed_count)*100.0/NULLIF(SUM(total_notes),0) as avg_removal_rate_percent,
                    COUNT(DISTINCT DATE(timestamp)) as active_days,
                    COUNT(DISTINCT run_id) as total_runs,
                    COUNT(DISTINCT run_id)*1.0/NULLIF(COUNT(DISTINCT DATE(timestamp)),1) as avg_runs_per_day
                FROM notebook_history WHERE timestamp >= datetime('now', ?)""", [f"-{days} days"])
            else:
                cursor.execute("""SELECT
                    AVG(total_notes) as avg_notes_per_run, AVG(total_chunks) as avg_chunks_per_run,
                    SUM(chunks_upserted)*100.0/NULLIF(SUM(total_chunks),0) as avg_update_rate_percent,
                    SUM(chunks_skipped)*100.0/NULLIF(SUM(total_chunks),0) as avg_skip_rate_percent,
                    SUM(notes_added_count)*100.0/NULLIF(SUM(total_notes),0) as avg_addition_rate_percent,
                    SUM(notes_removed_count)*100.0/NULLIF(SUM(total_notes),0) as avg_removal_rate_percent,
                    COUNT(DISTINCT DATE(timestamp)) as active_days,
                    COUNT(DISTINCT run_id) as total_runs,
                    COUNT(DISTINCT run_id)*1.0/NULLIF(COUNT(DISTINCT DATE(timestamp)),1) as avg_runs_per_day
                FROM notebook_history""")
            metrics = dict(cursor.fetchone())
            if days:
                cursor.execute("""SELECT COUNT(*) as total_runs,
                    SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) as successful_runs
                FROM global_run_history WHERE timestamp >= datetime('now', ?)""", [f"-{days} days"])
            else:
                cursor.execute("""SELECT COUNT(*) as total_runs,
                    SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) as successful_runs
                FROM global_run_history""")
            rs = cursor.fetchone()
            metrics["success_rate_percent"] = (rs["successful_runs"]*100.0/rs["total_runs"]) if rs and rs["total_runs"]>0 else 0.0
            conn.close()
            for k in list(metrics.keys()):
                if metrics[k] is None:
                    metrics[k] = 0.0 if ("percent" in k or "rate" in k or "avg" in k) else 0
                if "percent" in k or "rate" in k:
                    metrics[k] = round(float(metrics[k]), 2)
                elif isinstance(metrics[k], (int, float)):
                    metrics[k] = round(float(metrics[k]), 2)
            return metrics
        except Exception as e:
            log.error(f"本地效率指标失败: {e}")
            return {
                "avg_notes_per_run": 0.0, "avg_chunks_per_run": 0.0,
                "avg_update_rate_percent": 0.0, "avg_skip_rate_percent": 0.0,
                "avg_addition_rate_percent": 0.0, "avg_removal_rate_percent": 0.0,
                "active_days": 0, "total_runs": 0, "avg_runs_per_day": 0.0,
                "success_rate_percent": 0.0,
            }
