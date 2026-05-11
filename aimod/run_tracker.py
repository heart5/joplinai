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
# # Joplin笔记知识库AI系统运行追踪

# %%
# 运行数据采集与历史记录 — 采集每次向量化运行的逐笔记本数据，
# 写入历史数据库（remote-first + local fallback），供报告模块查询。

# %% [markdown]
# # 导入库
# %%
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# %%
import pathmagic

with pathmagic.context():
    try:
        from func.logme import log
        from func.first import getdirmain
    except ImportError as e:
        import logging

        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger(__name__)
        log.error(f"导入项目模块失败: {e}")


# %% [markdown]
# # JoplinAIRunTracker

# %%
class JoplinAIRunTracker:
    """Joplin AI任务运行追踪器
    核心功能：
    1. 采集每次运行的逐笔记本处理数据。
    2. 写入历史数据库（remote-first + local fallback）。
    3. 提供历史数据查询接口供报告模块使用。
    """

# %% [markdown]
# ## __init__(self, config: Dict)

    # %%
    def __init__(self, config: Dict, history_client=None):
        self.config = config
        self.history_client = history_client  # HistoryClient 实例（远程优先）
        self.task_records = []  # 本次运行的内存记录
        self.summary_data = {}  # 本次运行按笔记本汇总的数据
        self.global_chunk_stats = {
            "total_chunks_processed": 0,
            "chunks_upserted": 0,
            "chunks_skipped": 0,
            "orphan_chunks_cleaned": 0,
        }
        self.history_db_path = getdirmain() / "data" / "joplinai_center.db"
        if not self.history_client:
            self._init_history_db()

# %% [markdown]
# ## _init_history_db(self)
    # %%
    def _init_history_db(self):
        """初始化历史数据库（如果不存在则创建）"""
        try:
            conn = sqlite3.connect(self.history_db_path)
            cursor = conn.cursor()

            cursor.execute("""
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
            cursor.execute("""
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
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nb_history_timestamp ON notebook_history(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nb_history_notebook ON notebook_history(notebook_title)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_global_run_timestamp ON global_run_history(timestamp)")
            conn.commit()
            conn.close()
            log.info(f"历史数据库初始化完成: {self.history_db_path}")
        except Exception as e:
            log.error(f"初始化历史数据库失败: {e}", exc_info=True)

# %% [markdown]
# ## add_notebook_record(self, notebook_title: str, stats: Dict)
    # %%
    def add_notebook_record(self, notebook_title: str, stats: Dict):
        """添加单个笔记本的处理记录到内存，并立即保存到历史数据库（远程优先）"""
        record = {
            "notebook": notebook_title,
            "timestamp": datetime.now().isoformat(),
            "stats": stats,
        }
        self.task_records.append(record)
        self.summary_data[notebook_title] = stats

        chunk_stats = stats.get("chunk_stats", {})
        self.global_chunk_stats["total_chunks_processed"] += chunk_stats.get("total_chunks", 0)
        self.global_chunk_stats["chunks_upserted"] += chunk_stats.get("upserted", 0)
        self.global_chunk_stats["chunks_skipped"] += chunk_stats.get("skipped", 0)
        self.global_chunk_stats["orphan_chunks_cleaned"] += chunk_stats.get("orphans_cleaned", 0)

        run_id = self._get_current_run_id()
        timestamp = datetime.now().isoformat()
        if self.history_client:
            self.history_client.add_notebook_record(notebook_title, stats, run_id, timestamp)
            return

        try:
            conn = sqlite3.connect(self.history_db_path)
            cursor = conn.cursor()
            notes_added = stats.get("notes_added", [])
            notes_removed = stats.get("notes_removed", [])
            failed_notes = stats.get("failed_notes", [])
            cursor.execute(
                """
                INSERT INTO notebook_history (
                    run_id, notebook_title, timestamp,
                    total_notes, updated_count, failed_count,
                    notes_added_count, notes_removed_count,
                    total_chunks, chunks_upserted, chunks_skipped, chunks_orphans_cleaned,
                    notes_added_list, notes_removed_list, failed_notes_list
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id, notebook_title, timestamp,
                    stats.get("total_notes", 0), stats.get("updated_count", 0),
                    len(failed_notes), len(notes_added), len(notes_removed),
                    chunk_stats.get("total_chunks", 0), chunk_stats.get("upserted", 0),
                    chunk_stats.get("skipped", 0), chunk_stats.get("orphans_cleaned", 0),
                    json.dumps(notes_added, ensure_ascii=False),
                    json.dumps(notes_removed, ensure_ascii=False),
                    json.dumps(failed_notes, ensure_ascii=False),
                ),
            )
            conn.commit()
            conn.close()
            log.debug(f"笔记本【{notebook_title}】处理记录已保存到本地历史数据库")
        except Exception as e:
            log.error(f"保存笔记本记录到历史数据库失败: {e}", exc_info=True)

# %% [markdown]
# ## finalize_run(self, success: bool = True, error_msg: str = None)
    # %%
    def finalize_run(self, success: bool = True, error_msg: str = None):
        """完成本次运行，保存全局运行记录到历史数据库（远程优先）"""
        run_id = self._get_current_run_id()
        timestamp = datetime.now().isoformat()
        embedding_model = self.config["embedding_model"]
        notebook_count = len(self.summary_data)

        total_notes_processed = sum(s.get("total_notes", 0) for s in self.summary_data.values())
        total_notes_added = sum(len(s.get("notes_added", [])) for s in self.summary_data.values())
        total_notes_removed = sum(len(s.get("notes_removed", [])) for s in self.summary_data.values())
        total_chunks = self.global_chunk_stats["total_chunks_processed"]

        if self.history_client:
            self.history_client.finalize_run(
                run_id=run_id, timestamp=timestamp,
                embedding_model=embedding_model, notebook_count=notebook_count,
                total_notes_processed=total_notes_processed,
                total_chunks_processed=total_chunks,
                total_notes_added=total_notes_added,
                total_notes_removed=total_notes_removed,
                success=success, error_message=error_msg,
            )
            log.info(f"本次运行全局记录已保存到远程历史数据库，Run ID: {run_id}")
            return

        try:
            conn = sqlite3.connect(self.history_db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO global_run_history (
                    run_id, timestamp, embedding_model, notebook_count,
                    total_notes_processed, total_chunks_processed,
                    total_notes_added, total_notes_removed,
                    success, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id, timestamp, embedding_model, notebook_count,
                    total_notes_processed, total_chunks,
                    total_notes_added, total_notes_removed,
                    success, error_msg,
                ),
            )
            conn.commit()
            conn.close()
            log.info(f"本次运行全局记录已保存到本地历史数据库，Run ID: {run_id}")
        except Exception as e:
            log.error(f"保存全局运行记录失败: {e}", exc_info=True)

# %% [markdown]
# ## get_snapshot(self) -> Dict
    # %%
    def get_snapshot(self) -> Dict:
        """返回当前运行的快照数据，供报告模块使用"""
        return {
            "summary_data": self.summary_data,
            "global_chunk_stats": self.global_chunk_stats,
            "history_db_path": self.history_db_path,
        }

# %% [markdown]
# ## _get_current_run_id(self) -> str
    # %%
    def _get_current_run_id(self) -> str:
        """生成当前运行的唯一ID"""
        model_part = self.config["embedding_model"].split("/")[-1].replace(":", "_")[:10]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{model_part}"

# %% [markdown]
# ## 历史数据分析方法组

# %% [markdown]
# ### get_cumulative_stats(self, days: int = None) -> Dict
    # %%
    def get_cumulative_stats(self, days: int = None) -> Dict:
        """获取累积统计（远程优先）"""
        if self.history_client:
            result = self.history_client.get_cumulative_stats(days)
            if result:
                return result
        try:
            conn = sqlite3.connect(self.history_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            time_condition = ""
            params = []
            if days:
                time_condition = "WHERE timestamp >= datetime('now', ?)"
                params.append(f"-{days} days")
            cursor.execute(
                f"""
                SELECT
                    COUNT(DISTINCT run_id) as total_runs,
                    COUNT(DISTINCT notebook_title) as total_notebooks_touched,
                    SUM(total_notes) as total_notes_processed_all_time,
                    SUM(total_chunks) as total_chunks_processed_all_time,
                    SUM(notes_added_count) as total_notes_added_all_time,
                    SUM(notes_removed_count) as total_notes_removed_all_time,
                    SUM(chunks_upserted) as total_chunks_updated_all_time,
                    SUM(chunks_orphans_cleaned) as total_orphans_cleaned_all_time
                FROM notebook_history
                {time_condition}
                """,
                params,
            )
            cumulative = dict(cursor.fetchone())
            cursor.execute("""
                SELECT
                    strftime('%Y-%W', timestamp) as week,
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
                f"""
                SELECT
                    notebook_title, COUNT(*) as process_count,
                    SUM(total_notes) as total_notes, SUM(total_chunks) as total_chunks,
                    MAX(timestamp) as last_processed
                FROM notebook_history
                {time_condition}
                GROUP BY notebook_title ORDER BY process_count DESC LIMIT 10
                """,
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
        except Exception as e:
            log.error(f"获取累积统计失败: {e}", exc_info=True)
            return {}

# %% [markdown]
# ### get_change_analysis(self, notebook_title: str = None, days: int = 30) -> Dict
    # %%
    def get_change_analysis(self, notebook_title: str = None, days: int = 30) -> Dict:
        """分析指定笔记本或全局的笔记动态变化（远程优先）"""
        if self.history_client:
            result = self.history_client.get_change_analysis(notebook_title, days)
            if result:
                return result
        try:
            conn = sqlite3.connect(self.history_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            where_conditions = ["timestamp >= datetime('now', ?)"]
            params = [f"-{days} days"]
            if notebook_title:
                where_conditions.append("notebook_title = ?")
                params.append(notebook_title)
            where_clause = " AND ".join(where_conditions)
            cursor.execute(
                f"""
                SELECT notes_added_list, notes_removed_list
                FROM notebook_history WHERE {where_clause}
                """,
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
        except Exception as e:
            log.error(f"获取变化分析失败: {e}", exc_info=True)
            return {}

# %% [markdown]
# ### get_efficiency_metrics(self, days: int = 30) -> Dict
    # %%
    def get_efficiency_metrics(self, days: int = 30) -> Dict:
        """获取处理效率指标（远程优先）"""
        if self.history_client:
            result = self.history_client.get_efficiency_metrics(days)
            if result:
                return result
        try:
            conn = sqlite3.connect(self.history_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
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
                WHERE timestamp >= datetime('now', ?)
                """,
                [f"-{days} days"],
            )
            metrics = dict(cursor.fetchone())
            cursor.execute(
                """
                SELECT
                    COUNT(*) as total_runs,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_runs
                FROM global_run_history
                WHERE timestamp >= datetime('now', ?)
                """,
                [f"-{days} days"],
            )
            run_stats = cursor.fetchone()
            if run_stats and run_stats['total_runs'] > 0:
                metrics['success_rate_percent'] = (run_stats['successful_runs'] * 100.0) / run_stats['total_runs']
            else:
                metrics['success_rate_percent'] = 0.0
            conn.close()
            for key in list(metrics.keys()):
                if metrics[key] is None:
                    if 'percent' in key or 'rate' in key:
                        metrics[key] = 0.0
                    elif 'avg' in key:
                        metrics[key] = 0.0
                    else:
                        metrics[key] = 0
            for key in list(metrics.keys()):
                if 'percent' in key or 'rate' in key:
                    metrics[key] = round(float(metrics[key]), 2)
                elif isinstance(metrics[key], (int, float)):
                    metrics[key] = round(float(metrics[key]), 2)
            return metrics
        except Exception as e:
            log.error(f"获取效率指标失败: {e}", exc_info=True)
            return {
                'avg_notes_per_run': 0.0, 'avg_chunks_per_run': 0.0,
                'avg_update_rate_percent': 0.0, 'avg_skip_rate_percent': 0.0,
                'avg_addition_rate_percent': 0.0, 'avg_removal_rate_percent': 0.0,
                'active_days': 0, 'total_runs': 0, 'avg_runs_per_day': 0.0,
                'success_rate_percent': 0.0,
            }
