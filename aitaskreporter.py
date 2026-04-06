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
# # 导入库
# %%
# aitaskreport.py
# 增强版：Joplin笔记知识库AI系统运行分析报告系统
# 新增功能：监控文本块粒度、追踪笔记进出动态、反映整体变化情况

# %%
from datetime import datetime
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
# # JoplinAITaskReporter (增强版)
# %%
class JoplinAITaskReporter:
    """Joplin AI任务报告管理器 - 增强版
    处理向量化任务的统计与笔记同步，支持文本块粒度监控和笔记动态变化追踪。
    """

    def __init__(self, config: Dict):
        self.config = config
        self.task_records = []  # 存储每次运行的任务记录
        self.summary_data = {}  # 按笔记本汇总的数据
        # 新增：用于跨笔记本的全局统计
        self.global_chunk_stats = {
            "total_chunks_processed": 0,
            "chunks_upserted": 0,
            "chunks_skipped": 0,
            "orphan_chunks_cleaned": 0,
        }

    def add_notebook_record(self, notebook_title: str, stats: Dict):
        """添加单个笔记本的处理记录
        期望 stats 字典包含来自 process_notes_incremental 的增强信息：
            - 原有字段: total_notes, updated_count, failed_notes, new_time_notes
            - 新增字段（建议在 joplinai.py 中补充）:
                * `chunk_stats`: {
                    "total_chunks": X,       # 本次处理涉及的总块数
                    "upserted": Y,           # 新增或更新的块数
                    "skipped": Z,            # 哈希未变跳过的块数
                    "orphans_cleaned": W,    # 清理的孤儿块数
                  }
                * `notes_added`: [title1, title2],     # 本次新增的笔记标题列表
                * `notes_removed`: [title3, title4],   # 本次移除的笔记标题列表
        """
        record = {
            "notebook": notebook_title,
            "timestamp": datetime.now().isoformat(),
            "stats": stats,
        }
        self.task_records.append(record)
        self.summary_data[notebook_title] = stats

        # 更新全局块统计
        chunk_stats = stats.get("chunk_stats", {})
        self.global_chunk_stats["total_chunks_processed"] += chunk_stats.get(
            "total_chunks", 0
        )
        self.global_chunk_stats["chunks_upserted"] += chunk_stats.get("upserted", 0)
        self.global_chunk_stats["chunks_skipped"] += chunk_stats.get("skipped", 0)
        self.global_chunk_stats["orphan_chunks_cleaned"] += chunk_stats.get(
            "orphans_cleaned", 0
        )

    def generate_markdown_report(self) -> str:
        """生成Markdown格式的增强统计报告"""
        md_lines = ["# 📊 Joplin笔记向量化处理统计报告（增强版）\n"]
        md_lines.append(
            f"*报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*  |  *嵌入模型：{self.config['embedding_model']}*\n"
        )

        # 1. 整体概览（核心指标）
        md_lines.append("## 1. 整体处理概览\n")
        total_notes = sum(s.get("total_notes", 0) for s in self.summary_data.values())
        total_updated = sum(
            s.get("updated_count", 0) for s in self.summary_data.values()
        )
        total_failed = sum(
            len(s.get("failed_notes", [])) for s in self.summary_data.values()
        )

        # 计算新增和移除的笔记总数
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

        # 2. 文本块粒度统计
        md_lines.append("## 2. 文本块处理详情\n")
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

        # 3. 按笔记本详细统计（增强表格）
        if self.summary_data:
            md_lines.append("## 3. 笔记本处理详情\n")
            # 列：笔记本 | 笔记总数 | 新增笔记 | 移除笔记 | 更新笔记 | 失败笔记 | 文本块(总/更/跳/清)
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

                # 笔记统计单元格
                notes_cell = f"{total_notes} / {len(notes_added)} / {len(notes_removed)} / {updated} / {failed_count}"

                # 文本块统计单元格
                chunk_stats = stats.get("chunk_stats", {})
                chunk_total = chunk_stats.get("total_chunks", 0)
                chunk_upserted = chunk_stats.get("upserted", 0)
                chunk_skipped = chunk_stats.get("skipped", 0)
                chunk_orphans = chunk_stats.get("orphans_cleaned", 0)
                chunks_cell = f"{chunk_total} / {chunk_upserted} / {chunk_skipped} / {chunk_orphans}"

                # 备注（例如列出失败或新增的笔记，截断显示）
                remark_parts = []
                if failed_list:
                    remark_parts.append(
                        f"失败: {', '.join(failed_list[:2])}"
                        + (f" 等{len(failed_list)}条" if len(failed_list) > 2 else "")
                    )
                if notes_added:
                    remark_parts.append(
                        f"新增: {', '.join(notes_added[:2])}"
                        + (f" 等{len(notes_added)}条" if len(notes_added) > 2 else "")
                    )
                if notes_removed:
                    remark_parts.append(
                        f"移除: {', '.join(notes_removed[:2])}"
                        + (
                            f" 等{len(notes_removed)}条"
                            if len(notes_removed) > 2
                            else ""
                        )
                    )
                remark = "; ".join(remark_parts) if remark_parts else "正常"

                md_lines.append(
                    f"| {notebook} | {notes_cell} | {chunks_cell} | {remark} |"
                )

        # 4. 变化清单（可选详细列表）
        # 如果变化较多，可以单独列出
        all_notes_added = []
        all_notes_removed = []
        for stats in self.summary_data.values():
            all_notes_added.extend(stats.get("notes_added", []))
            all_notes_removed.extend(stats.get("notes_removed", []))

        if all_notes_added or all_notes_removed:
            md_lines.append("\n## 4. 笔记变动清单\n")
            if all_notes_added:
                md_lines.append(f"**新增的笔记 ({len(all_notes_added)}条):**")
                for title in all_notes_added:
                    md_lines.append(f"- `{title}`")
                md_lines.append("")
            if all_notes_removed:
                md_lines.append(f"**移除的笔记 ({len(all_notes_removed)}条):**")
                for title in all_notes_removed:
                    md_lines.append(f"- ~~{title}~~")
                md_lines.append("")

        # 5. 最近处理历史（优化显示）
        md_lines.append("\n## 5. 最近处理历史\n")
        md_lines.append("| 时间 | 笔记本 | 笔记结果 | 块结果 |")
        md_lines.append("|:---|:---|:---|:---|")

        for record in self.task_records[-10:]:  # 显示最近10条
            notebook = record["notebook"]
            stats = record["stats"]
            # 笔记摘要
            notes_summary = (
                f"{stats.get('total_notes', 0)}总/{stats.get('updated_count', 0)}更"
            )
            # 块摘要
            chunk_stats = stats.get("chunk_stats", {})
            chunks_summary = f"{chunk_stats.get('total_chunks', 0)}总/{chunk_stats.get('upserted', 0)}更"

            md_lines.append(
                f"| {record['timestamp'][:19]} | {notebook} | {notes_summary} | {chunks_summary} |"
            )

        return "\n".join(md_lines)

    def update_joplin_note(self, report_content: str) -> bool:
        """将报告更新到Joplin笔记"""
        try:
            notebook_title = "AI知识库"
            notebook_id = searchnotebook(notebook_title)
            if not notebook_id:
                notebook_id = jpapi.add_notebook(title=notebook_title)

            note_title = "JoplinAI向量化处理报告（增强版）"
            existing_notes = searchnotes(note_title, parent_id=notebook_id)

            if existing_notes:
                note = existing_notes
                updatenote_body(note.id, report_content)
                log.info(f"更新Joplin统计笔记: {note_title}")
            else:
                note_id = createnote(note_title, report_content, parent_id=notebook_id)
                log.info(f"创建Joplin统计笔记: {note_title} (ID: {note_id})")

            return True
        except Exception as e:
            log.error(f"更新Joplin统计笔记失败: {e}")
            return False
