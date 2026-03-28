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
    )
    from func.logme import log
    from func.sysfunc import execcmd, not_IPython
    from func.wrapfuncs import timethis
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    log.error(f"导入项目模块失败: {e}")


# %% [markdown]
# # JoplinAITaskReporter

# %%
class JoplinAITaskReporter:
    """Joplin AI任务报告管理器 - 处理向量化任务的统计与笔记同步"""

    def __init__(self, config: Dict):
        self.config = config
        self.task_records = []  # 存储每次运行的任务记录
        self.summary_data = {}  # 按笔记本汇总的数据

    def add_notebook_record(self, notebook_title: str, stats: Dict):
        """添加单个笔记本的处理记录"""
        record = {
            "notebook": notebook_title,
            "timestamp": datetime.now().isoformat(),
            "stats": stats,  # 包含 total_notes, updated_count, failed_notes 等
        }
        self.task_records.append(record)
        self.summary_data[notebook_title] = stats

    def generate_markdown_report(self) -> str:
        """生成Markdown格式的统计报告（参考hostconfig的表格设计）"""
        md_lines = ["# Joplin笔记向量化处理统计报告\n"]
        md_lines.append(f"*生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

        # 1. 总体概览
        md_lines.append("## 1. 处理概览\n")
        total_notes = sum(s.get("total_notes", 0) for s in self.summary_data.values())
        total_updated = sum(
            s.get("updated_count", 0) for s in self.summary_data.values()
        )
        total_failed = sum(
            len(s.get("failed_notes", [])) for s in self.summary_data.values()
        )

        md_lines.append(f"- **处理笔记本数**：{len(self.summary_data)} 个")
        md_lines.append(f"- **涉及笔记总数**：{total_notes} 条")
        md_lines.append(f"- **本次更新笔记**：{total_updated} 条")
        md_lines.append(f"- **处理失败笔记**：{total_failed} 条\n")

        # 2. 按笔记本详细统计（表格）
        if self.summary_data:
            md_lines.append("## 2. 笔记本处理详情\n")
            md_lines.append(
                "| 笔记本 | 笔记总数 | 更新数量 | 失败数量 | 失败笔记列表 |"
            )
            md_lines.append("|:---|:---:|:---:|:---:|:---|")

            for notebook, stats in self.summary_data.items():
                total = stats.get("total_notes", 0)
                updated = stats.get("updated_count", 0)
                failed_list = stats.get("failed_notes", [])
                failed_count = len(failed_list)
                # 失败笔记列表，截断显示
                failed_preview = ", ".join(failed_list[:3]) if failed_list else "无"
                if len(failed_list) > 3:
                    failed_preview += f" 等{len(failed_list)}条"

                md_lines.append(
                    f"| {notebook} | {total} | {updated} | {failed_count} | {failed_preview} |"
                )

        # 3. 最近处理历史
        md_lines.append("\n## 3. 最近处理历史\n")
        md_lines.append("| 时间 | 笔记本 | 结果摘要 |")
        md_lines.append("|:---|:---|:---|")

        for record in self.task_records[-10:]:  # 显示最近10条
            notebook = record["notebook"]
            stats = record["stats"]
            summary = f"总数{stats.get('total_notes', 0)}，更新{stats.get('updated_count', 0)}，失败{len(stats.get('failed_notes', []))}"
            md_lines.append(f"| {record['timestamp'][:19]} | {notebook} | {summary} |")

        return "\n".join(md_lines)

    def update_joplin_note(self, report_content: str) -> bool:
        """将报告更新到Joplin笔记（参考hostconfig.py的update_joplin_note方法）"""
        try:
            # 查找或创建笔记本（例如，可以放在“系统管理”或“AI处理”笔记本下）
            notebook_title = "AI知识库"  # 可根据配置调整
            notebook_id = searchnotebook(notebook_title)
            if not notebook_id:
                notebook_id = jpapi.add_notebook(title=notebook_title)

            # 查找或创建笔记
            note_title = "JoplinAI向量化处理报告"
            existing_notes = searchnotes(note_title, parent_id=notebook_id)

            if existing_notes:
                # 更新现有笔记
                note = existing_notes[0]
                updatenote_body(note.id, report_content)
                log.info(f"更新Joplin统计笔记: {note_title}")
            else:
                # 创建新笔记
                note_id = createnote(note_title, report_content, parent_id=notebook_id)
                log.info(f"创建Joplin统计笔记: {note_title} (ID: {note_id})")

            return True
        except Exception as e:
            log.error(f"更新Joplin统计笔记失败: {e}")
            return False
