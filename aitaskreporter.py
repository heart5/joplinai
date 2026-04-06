# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Joplin笔记知识库AI系统运行分析报告系统

# %%
# aitaskreport.py

# %%
# 增强版：Joplin笔记知识库AI系统运行分析报告系统（含历史数据仓库与趋势分析）
# 功能：监控文本块粒度、追踪笔记进出动态、永久保存历史记录、分析动态变化趋势。

# %% [markdown]
# # 导入库
# %%
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# %%
try:
    from func.configpr import (
        findvaluebykeyinsection,
        getcfpoptionvalue,
        setcfpoptionvalue,
    )
    from func.first import dirmainpath, getdirmain
    from func.getid import getdeviceid, getdevicename, gethostuser
    from func.jpfuncs import (
        createnote,
        getinivaluefromcloud,
        getnote,
        jpapi,
        searchnotebook,
        searchnotes,
        updatenote_body,
    )
    from func.logme import log
    from func.sysfunc import execcmd, not_IPython
    from func.wrapfuncs import timethis
except ImportError as e:
    import logging

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    log.error(f"导入项目模块失败: {e}")


# %% [markdown]
# # JoplinAITaskReporter (历史数据仓库与趋势分析版)
# %%
class JoplinAITaskReporter:
    """Joplin AI任务报告管理器 - 历史数据仓库与趋势分析版
    核心功能：
    1. 永久保存每次运行的详细处理记录到SQLite数据库。
    2. 累积统计所有历史数据，计算总量、增长趋势。
    3. 生成包含历史对比和动态变化分析的增强报告。
    """

# %% [markdown]
# ## \_\_init__(self, config: Dict)

    # %%
    def __init__(self, config: Dict):
        self.config = config
        self.task_records = []  # 本次运行的内存记录
        self.summary_data = {}  # 本次运行按笔记本汇总的数据
        # 全局块统计（本次运行）
        self.global_chunk_stats = {
            "total_chunks_processed": 0,
            "chunks_upserted": 0,
            "chunks_skipped": 0,
            "orphan_chunks_cleaned": 0,
        }
        # 历史数据库路径
        self.history_db_path = getdirmain() / "data" / "joplinai_history.db"
        # 初始化历史数据库
        self._init_history_db()

# %% [markdown]
# ## _init_history_db(self)
    # %%
    def _init_history_db(self):
        """初始化历史数据库（如果不存在则创建）"""
        try:
            conn = sqlite3.connect(self.history_db_path)
            cursor = conn.cursor()

            # 表1: 笔记本处理历史 (notebook_history)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS notebook_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,               -- 本次运行的唯一标识（如时间戳+模型）
                    notebook_title TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,        -- 处理完成时间
                    
                    -- 笔记维度统计
                    total_notes INTEGER DEFAULT 0,
                    updated_count INTEGER DEFAULT 0,    -- 内容变更的笔记数
                    failed_count INTEGER DEFAULT 0,
                    notes_added_count INTEGER DEFAULT 0,
                    notes_removed_count INTEGER DEFAULT 0,
                    
                    -- 文本块维度统计
                    total_chunks INTEGER DEFAULT 0,
                    chunks_upserted INTEGER DEFAULT 0,
                    chunks_skipped INTEGER DEFAULT 0,
                    chunks_orphans_cleaned INTEGER DEFAULT 0,
                    
                    -- 原始数据（JSON格式，用于详细清单）
                    notes_added_list TEXT,              -- JSON数组
                    notes_removed_list TEXT,            -- JSON数组
                    failed_notes_list TEXT,             -- JSON数组
                    
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 表2: 全局运行历史 (global_run_history)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS global_run_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT UNIQUE NOT NULL,
                    timestamp DATETIME NOT NULL,
                    embedding_model TEXT NOT NULL,
                    notebook_count INTEGER DEFAULT 0,   -- 本次处理的笔记本数
                    
                    -- 全局汇总（本次运行）
                    total_notes_processed INTEGER DEFAULT 0,
                    total_chunks_processed INTEGER DEFAULT 0,
                    total_notes_added INTEGER DEFAULT 0,
                    total_notes_removed INTEGER DEFAULT 0,
                    
                    -- 运行状态
                    success BOOLEAN DEFAULT TRUE,
                    error_message TEXT,
                    
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 创建索引以加速时间范围查询
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_nb_history_timestamp ON notebook_history(timestamp)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_nb_history_notebook ON notebook_history(notebook_title)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_global_run_timestamp ON global_run_history(timestamp)"
            )

            conn.commit()
            conn.close()
            log.info(f"历史数据库初始化完成: {self.history_db_path}")

        except Exception as e:
            log.error(f"初始化历史数据库失败: {e}", exc_info=True)

# %% [markdown]
# ## add_notebook_record(self, notebook_title: str, stats: Dict)
    # %%
    def add_notebook_record(self, notebook_title: str, stats: Dict):
        """添加单个笔记本的处理记录到内存，并立即保存到历史数据库"""
        # 1. 保存到内存（原有逻辑）
        record = {
            "notebook": notebook_title,
            "timestamp": datetime.now().isoformat(),
            "stats": stats,
        }
        self.task_records.append(record)
        self.summary_data[notebook_title] = stats

        # 更新全局块统计（内存）
        chunk_stats = stats.get("chunk_stats", {})
        self.global_chunk_stats["total_chunks_processed"] += chunk_stats.get(
            "total_chunks", 0
        )
        self.global_chunk_stats["chunks_upserted"] += chunk_stats.get("upserted", 0)
        self.global_chunk_stats["chunks_skipped"] += chunk_stats.get("skipped", 0)
        self.global_chunk_stats["orphan_chunks_cleaned"] += chunk_stats.get(
            "orphans_cleaned", 0
        )

        # 2. 立即持久化到历史数据库
        try:
            conn = sqlite3.connect(self.history_db_path)
            cursor = conn.cursor()

            # 生成本次运行的唯一ID（例如：时间戳+模型名哈希前8位）
            run_id = self._get_current_run_id()

            # 准备数据
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
                    run_id,
                    notebook_title,
                    datetime.now().isoformat(),
                    stats.get("total_notes", 0),
                    stats.get("updated_count", 0),
                    len(failed_notes),
                    len(notes_added),
                    len(notes_removed),
                    chunk_stats.get("total_chunks", 0),
                    chunk_stats.get("upserted", 0),
                    chunk_stats.get("skipped", 0),
                    chunk_stats.get("orphans_cleaned", 0),
                    json.dumps(notes_added, ensure_ascii=False),
                    json.dumps(notes_removed, ensure_ascii=False),
                    json.dumps(failed_notes, ensure_ascii=False),
                ),
            )

            conn.commit()
            conn.close()
            log.debug(f"笔记本【{notebook_title}】处理记录已保存到历史数据库")

        except Exception as e:
            log.error(f"保存笔记本记录到历史数据库失败: {e}", exc_info=True)

# %% [markdown]
# ## finalize_run(self, success: bool = True, error_msg: str = None)
    # %%
    def finalize_run(self, success: bool = True, error_msg: str = None):
        """完成本次运行，保存全局运行记录到历史数据库"""
        try:
            conn = sqlite3.connect(self.history_db_path)
            cursor = conn.cursor()

            run_id = self._get_current_run_id()

            # 计算本次运行的全局汇总
            total_notes_processed = sum(
                s.get("total_notes", 0) for s in self.summary_data.values()
            )
            total_notes_added = sum(
                len(s.get("notes_added", [])) for s in self.summary_data.values()
            )
            total_notes_removed = sum(
                len(s.get("notes_removed", [])) for s in self.summary_data.values()
            )

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
                    run_id,
                    datetime.now().isoformat(),
                    self.config["embedding_model"],
                    len(self.summary_data),
                    total_notes_processed,
                    self.global_chunk_stats["total_chunks_processed"],
                    total_notes_added,
                    total_notes_removed,
                    success,
                    error_msg,
                ),
            )

            conn.commit()
            conn.close()
            log.info(f"本次运行全局记录已保存到历史数据库，Run ID: {run_id}")

        except Exception as e:
            log.error(f"保存全局运行记录失败: {e}", exc_info=True)

# %% [markdown]
# ## _get_current_run_id(self) -> str
    # %%
    def _get_current_run_id(self) -> str:
        """生成当前运行的唯一ID"""
        # 格式: YYYYMMDD_HHMMSS_模型名简写
        model_part = (
            self.config["embedding_model"].split("/")[-1].replace(":", "_")[:10]
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{model_part}"

# %% [markdown]
# ## 历史数据分析方法组
# %% [markdown]
# ### get_cumulative_stats(self, days: int = None) -> Dict
    # %%
    def get_cumulative_stats(self, days: int = None) -> Dict:
        """获取累积统计（可指定最近N天）"""
        try:
            conn = sqlite3.connect(self.history_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # 构建时间条件
            time_condition = ""
            params = []
            if days:
                time_condition = "WHERE timestamp >= datetime('now', ?)"
                params.append(f"-{days} days")

            # 1. 累积总量
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

            # 2. 近期趋势（最近30天，按周聚合）
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
                GROUP BY week
                ORDER BY week DESC
                LIMIT 12
            """)

            weekly_trends = [dict(row) for row in cursor.fetchall()]

            # 3. 最活跃的笔记本（按处理次数）
            cursor.execute(
                f"""
                SELECT 
                    notebook_title,
                    COUNT(*) as process_count,
                    SUM(total_notes) as total_notes,
                    SUM(total_chunks) as total_chunks,
                    MAX(timestamp) as last_processed
                FROM notebook_history
                {time_condition}
                GROUP BY notebook_title
                ORDER BY process_count DESC
                LIMIT 10
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
        """分析指定笔记本或全局的笔记动态变化"""
        try:
            conn = sqlite3.connect(self.history_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # 构建查询条件
            where_conditions = ["timestamp >= datetime('now', ?)"]
            params = [f"-{days} days"]

            if notebook_title:
                where_conditions.append("notebook_title = ?")
                params.append(notebook_title)

            where_clause = " AND ".join(where_conditions)

            # 获取新增/移除笔记的详细列表（去重）
            cursor.execute(
                f"""
                SELECT 
                    notes_added_list,
                    notes_removed_list
                FROM notebook_history
                WHERE {where_clause}
            """,
                params,
            )

            all_added = []
            all_removed = []
            for row in cursor.fetchall():
                added = (
                    json.loads(row["notes_added_list"])
                    if row["notes_added_list"]
                    else []
                )
                removed = (
                    json.loads(row["notes_removed_list"])
                    if row["notes_removed_list"]
                    else []
                )
                all_added.extend(added)
                all_removed.extend(removed)

            # 去重并计数
            unique_added = list(set(all_added))
            unique_removed = list(set(all_removed))

            # 计算净增长
            net_growth = len(unique_added) - len(unique_removed)

            # 查找频繁变化的笔记（既被添加过又被移除过）
            frequently_changed = list(set(unique_added) & set(unique_removed))

            conn.close()

            return {
                "analysis_period": f"最近{days}天",
                "notebook": notebook_title or "全局",
                "unique_notes_added": unique_added,
                "unique_notes_removed": unique_removed,
                "added_count": len(unique_added),
                "removed_count": len(unique_removed),
                "net_growth": net_growth,
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
        """获取处理效率指标"""
        try:
            conn = sqlite3.connect(self.history_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
    
            cursor.execute(
                """
                SELECT 
                    -- 平均每次运行处理的数据量
                    AVG(total_notes) as avg_notes_per_run,
                    AVG(total_chunks) as avg_chunks_per_run,
                    
                    -- 块处理效率（更新率 vs 跳过率）
                    SUM(chunks_upserted) * 100.0 / NULLIF(SUM(total_chunks), 0) as avg_update_rate_percent,
                    SUM(chunks_skipped) * 100.0 / NULLIF(SUM(total_chunks), 0) as avg_skip_rate_percent,
                    
                    -- 笔记变化率
                    SUM(notes_added_count) * 100.0 / NULLIF(SUM(total_notes), 0) as avg_addition_rate_percent,
                    SUM(notes_removed_count) * 100.0 / NULLIF(SUM(total_notes), 0) as avg_removal_rate_percent,
                    
                    -- 运行频率
                    COUNT(DISTINCT DATE(timestamp)) as active_days,
                    COUNT(DISTINCT run_id) as total_runs,
                    COUNT(DISTINCT run_id) * 1.0 / NULLIF(COUNT(DISTINCT DATE(timestamp)), 1) as avg_runs_per_day
                    
                FROM notebook_history
                WHERE timestamp >= datetime('now', ?)
                """,
                [f"-{days} days"],
            )
    
            metrics = dict(cursor.fetchone())
            
            # 计算处理成功率（基于全局运行记录）
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
            
            # === 关键修复：确保所有值不为None ===
            # 处理可能的None值
            for key in list(metrics.keys()):
                if metrics[key] is None:
                    if 'percent' in key or 'rate' in key:
                        metrics[key] = 0.0
                    elif 'avg' in key:
                        metrics[key] = 0.0
                    else:
                        metrics[key] = 0
            
            # 格式化百分比
            for key in list(metrics.keys()):
                if 'percent' in key or 'rate' in key:
                    metrics[key] = round(float(metrics[key]), 2)
                elif isinstance(metrics[key], (int, float)):
                    metrics[key] = round(float(metrics[key]), 2)
            
            return metrics
            
        except Exception as e:
            log.error(f"获取效率指标失败: {e}", exc_info=True)
            # 返回安全的默认值
            return {
                'avg_notes_per_run': 0.0,
                'avg_chunks_per_run': 0.0,
                'avg_update_rate_percent': 0.0,
                'avg_skip_rate_percent': 0.0,
                'avg_addition_rate_percent': 0.0,
                'avg_removal_rate_percent': 0.0,
                'active_days': 0,
                'total_runs': 0,
                'avg_runs_per_day': 0.0,
                'success_rate_percent': 0.0
            }

# %% [markdown]
# ## generate_markdown_report(self) -> str
    # %%
    def generate_markdown_report(self) -> str:
        """生成Markdown格式的增强统计报告（包含历史趋势分析）"""
        md_lines = ["# 📊 Joplin笔记向量化处理统计报告（历史分析版）\n"]
        md_lines.append(
            f"*报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*  |  "
            f"*嵌入模型：{self.config['embedding_model']}*  |  "
            f"*历史数据库：{self.history_db_path.name}*\n"
        )

        # === 第一部分：本次运行快照 ===
        md_lines.append("## 1. 本次运行快照\n")

        total_notes = sum(s.get("total_notes", 0) for s in self.summary_data.values())
        total_updated = sum(
            s.get("updated_count", 0) for s in self.summary_data.values()
        )
        total_failed = sum(
            len(s.get("failed_notes", [])) for s in self.summary_data.values()
        )
        total_notes_added = sum(
            len(s.get("notes_added", [])) for s in self.summary_data.values()
        )
        total_notes_removed = sum(
            len(s.get("notes_removed", [])) for s in self.summary_data.values()
        )

        md_lines.append(f"- **处理笔记本数**：`{len(self.summary_data)}` 个")
        md_lines.append(f"- **涉及笔记总数**：`{total_notes}` 条")
        md_lines.append(
            f"- **笔记动态变化**：新增 `{total_notes_added}` 条，移除 `{total_notes_removed}` 条"
        )
        md_lines.append(f"- **本次内容更新**：`{total_updated}` 条笔记的内容发生了变更")
        md_lines.append(f"- **处理异常笔记**：`{total_failed}` 条\n")

        # 文本块粒度统计
        md_lines.append(
            f"- **处理总块数**：`{self.global_chunk_stats['total_chunks_processed']}`"
        )
        md_lines.append(
            f"- **向量库更新**：新增或更新 `{self.global_chunk_stats['chunks_upserted']}` 个块"
        )
        md_lines.append(
            f"- **内容未变更**：跳过 `{self.global_chunk_stats['chunks_skipped']}` 个块"
        )
        md_lines.append(
            f"- **清理孤儿块**：移除 `{self.global_chunk_stats['orphan_chunks_cleaned']}` 个块\n"
        )

        # === 第二部分：历史累积统计 ===
        md_lines.append("## 2. 历史累积统计（全部数据）\n")

        cumulative = self.get_cumulative_stats(days=None)
        if cumulative:
            cum = cumulative.get("cumulative", {})
            md_lines.append(f"- **总运行次数**：`{cum.get('total_runs', 0)}` 次")
            md_lines.append(
                f"- **涉及笔记本数**：`{cum.get('total_notebooks_touched', 0)}` 个"
            )
            md_lines.append(
                f"- **累计处理笔记**：`{cum.get('total_notes_processed_all_time', 0)}` 条"
            )
            md_lines.append(
                f"- **累计处理文本块**：`{cum.get('total_chunks_processed_all_time', 0)}` 个"
            )
            md_lines.append(
                f"- **累计新增笔记**：`{cum.get('total_notes_added_all_time', 0)}` 条"
            )
            md_lines.append(
                f"- **累计移除笔记**：`{cum.get('total_notes_removed_all_time', 0)}` 条"
            )
            md_lines.append(
                f"- **累计更新块数**：`{cum.get('total_chunks_updated_all_time', 0)}` 个"
            )
            md_lines.append(
                f"- **累计清理孤儿块**：`{cum.get('total_orphans_cleaned_all_time', 0)}` 个\n"
            )
        else:
            md_lines.append("*历史数据统计暂不可用*\n")

        # === 第三部分：近期趋势分析（最近30天） ===
        md_lines.append("## 3. 近期趋势分析（最近30天）\n")

        trend_30d = self.get_cumulative_stats(days=30)
        if trend_30d and trend_30d.get("cumulative"):
            cum_30d = trend_30d["cumulative"]
            md_lines.append(f"- **运行次数**：`{cum_30d.get('total_runs', 0)}` 次")
            md_lines.append(
                f"- **处理笔记**：`{cum_30d.get('total_notes_processed_all_time', 0)}` 条"
            )
            md_lines.append(
                f"- **处理文本块**：`{cum_30d.get('total_chunks_processed_all_time', 0)}` 个"
            )
            md_lines.append(
                f"- **新增笔记**：`{cum_30d.get('total_notes_added_all_time', 0)}` 条"
            )
            md_lines.append(
                f"- **移除笔记**：`{cum_30d.get('total_notes_removed_all_time', 0)}` 条\n"
            )

            # 周趋势表格
            if trend_30d.get("weekly_trends"):
                md_lines.append("**周度处理趋势：**")
                md_lines.append(
                    "| 周次 | 运行次数 | 处理笔记 | 处理块数 | 新增笔记 | 移除笔记 |"
                )
                md_lines.append("|:---|:---|:---|:---|:---|:---|")
                for week_data in trend_30d["weekly_trends"][:6]:  # 显示最近6周
                    md_lines.append(
                        f"| {week_data['week']} | {week_data['runs_count']} | "
                        f"{week_data['notes_processed']} | {week_data['chunks_processed']} | "
                        f"{week_data['notes_added']} | {week_data['notes_removed']} |"
                    )
                md_lines.append("")
        else:
            md_lines.append("*30天趋势数据暂不可用*\n")

        # === 第四部分：效率指标分析 ===
        md_lines.append("## 4. 系统效率指标（最近30天）\n")
        
        efficiency = self.get_efficiency_metrics(days=30)
        if efficiency:
            # 安全获取并格式化数值，确保不是None
            avg_notes = efficiency.get('avg_notes_per_run', 0)
            avg_chunks = efficiency.get('avg_chunks_per_run', 0)
            update_rate = efficiency.get('avg_update_rate_percent', 0)
            skip_rate = efficiency.get('avg_skip_rate_percent', 0)
            add_rate = efficiency.get('avg_addition_rate_percent', 0)
            remove_rate = efficiency.get('avg_removal_rate_percent', 0)
            success_rate = efficiency.get('success_rate_percent', 0)
            avg_runs_per_day = efficiency.get('avg_runs_per_day', 0)
            
            # 确保所有值都是数字类型
            avg_notes = float(avg_notes or 0)
            avg_chunks = float(avg_chunks or 0)
            avg_runs_per_day = float(avg_runs_per_day or 0)
            
            md_lines.append(f"- **平均每次运行**：处理 `{avg_notes:.1f}` 条笔记，`{avg_chunks:.1f}` 个文本块")
            md_lines.append(f"- **块处理效率**：更新率 `{update_rate}%`，跳过率 `{skip_rate}%`")
            md_lines.append(f"- **笔记变动率**：新增率 `{add_rate}%`，移除率 `{remove_rate}%`")
            md_lines.append(f"- **运行稳定性**：成功率 `{success_rate}%`，日均运行 `{avg_runs_per_day:.1f}` 次\n")
        else:
            md_lines.append("*效率指标暂不可用*\n")

        # === 第五部分：动态变化深度分析 ===
        md_lines.append("## 5. 笔记动态变化深度分析（最近30天）\n")

        change_analysis = self.get_change_analysis(days=30)
        if change_analysis:
            md_lines.append(
                f"- **唯一新增笔记**：`{change_analysis['added_count']}` 条"
            )
            md_lines.append(
                f"- **唯一移除笔记**：`{change_analysis['removed_count']}` 条"
            )
            md_lines.append(f"- **净增长笔记**：`{change_analysis['net_growth']}` 条")
            md_lines.append(
                f"- **频繁变动笔记**：`{change_analysis['frequently_changed_count']}` 条（既被添加过又被移除过）\n"
            )

            # 显示频繁变动的笔记标题（前5条）
            if change_analysis["frequently_changed_notes"]:
                md_lines.append("**频繁变动笔记示例：**")
                for note in change_analysis["frequently_changed_notes"][:5]:
                    md_lines.append(f"- `{note}`")
                if len(change_analysis["frequently_changed_notes"]) > 5:
                    md_lines.append(
                        f"- ... 等 {len(change_analysis['frequently_changed_notes'])} 条"
                    )
                md_lines.append("")
        else:
            md_lines.append("*变化分析数据暂不可用*\n")

        # === 第六部分：最活跃笔记本排名 ===
        md_lines.append("## 6. 最活跃笔记本排名（历史累计）\n")

        if trend_30d and trend_30d.get("top_notebooks"):
            md_lines.append(
                "| 笔记本 | 处理次数 | 累计笔记数 | 累计块数 | 最后处理时间 |"
            )
            md_lines.append("|:---|:---|:---|:---|:---|")
            for nb in trend_30d["top_notebooks"][:10]:  # 显示前10
                last_time = (
                    datetime.fromisoformat(nb["last_processed"]).strftime("%m-%d %H:%M")
                    if nb["last_processed"]
                    else "N/A"
                )
                md_lines.append(
                    f"| {nb['notebook_title']} | {nb['process_count']} | "
                    f"{nb['total_notes']} | {nb['total_chunks']} | {last_time} |"
                )
            md_lines.append("")
        else:
            md_lines.append("*活跃笔记本数据暂不可用*\n")

        # === 第七部分：本次运行详细表格（原有功能保留） ===
        if self.summary_data:
            md_lines.append("## 7. 本次运行笔记本详情\n")
            md_lines.append(
                "| 笔记本 | 笔记统计 (总/新/移/更/败) | 文本块统计 (总/更/跳/清) | 备注 |"
            )
            md_lines.append("|:---|:---|:---|:---|")

            for notebook, stats in self.summary_data.items():
                total_notes = stats.get("total_notes", 0)
                updated = stats.get("updated_count", 0)
                failed_list = stats.get("failed_notes", [])
                failed_count = len(failed_list)
                notes_added = stats.get("notes_added", [])
                notes_removed = stats.get("notes_removed", [])

                notes_cell = f"{total_notes} / {len(notes_added)} / {len(notes_removed)} / {updated} / {failed_count}"
                chunk_stats = stats.get("chunk_stats", {})
                chunks_cell = f"{chunk_stats.get('total_chunks', 0)} / {chunk_stats.get('upserted', 0)} / {chunk_stats.get('skipped', 0)} / {chunk_stats.get('orphans_cleaned', 0)}"

                remark_parts = []
                if failed_list:
                    remark_parts.append(f"失败:{len(failed_list)}条")
                if notes_added:
                    remark_parts.append(f"新增:{len(notes_added)}条")
                if notes_removed:
                    remark_parts.append(f"移除:{len(notes_removed)}条")
                remark = "；".join(remark_parts) if remark_parts else "正常"

                md_lines.append(
                    f"| {notebook} | {notes_cell} | {chunks_cell} | {remark} |"
                )
            md_lines.append("")

        # === 第八部分：数据维护提示 ===
        md_lines.append("## 8. 数据维护\n")
        md_lines.append(f"- **历史数据库位置**：`{self.history_db_path}`")
        md_lines.append(f"- **当前数据库大小**：`{self._get_db_size_mb():.2f} MB`")
        md_lines.append("- **数据保留策略**：永久保留，支持时间范围查询")
        md_lines.append("- **清理建议**：如需清理，可直接备份或删除数据库文件\n")

        return "\n".join(md_lines)

# %% [markdown]
# ## _get_db_size_mb(self) -> float
    # %%
    def _get_db_size_mb(self) -> float:
        """获取历史数据库文件大小（MB）"""
        if self.history_db_path.exists():
            return self.history_db_path.stat().st_size / (1024 * 1024)
        return 0.0

# %% [markdown]
# ## update_joplin_note(self, report_content: str) -> bool
    # %%
    def update_joplin_note(self, report_content: str) -> bool:
        """将报告更新到Joplin笔记（原有逻辑，保持兼容）"""
        try:
            if not (
                note_id := getcfpoptionvalue("joplinai", "aitaskreport", "note_id")
            ):
                notebook_title = "ewmobile"
                if not (notebook_id := searchnotebook(notebook_title)):
                    notebook_id = jpapi.add_notebook(title=notebook_title)
                note_title = "JoplinAI向量化处理报告（历史分析版）"
                if existing_notes := searchnotes(note_title, parent_id=notebook_id):
                    note = existing_notes[0]
                    note_id = note.id
                    setcfpoptionvalue("joplinai", "aitaskreport", "note_id", note_id)
                else:
                    note_id = createnote(
                        note_title, report_content, parent_id=notebook_id
                    )
                    setcfpoptionvalue("joplinai", "aitaskreport", "note_id", note_id)
                    log.info(f"创建Joplin统计笔记: {note_title} (ID: {note_id})")
            else:
                updatenote_body(note_id, report_content)
                log.info(f"更新Joplin统计笔记: {note_id}")
            return True
        except Exception as e:
            log.error(f"更新Joplin统计笔记失败: {e}")
            return False
