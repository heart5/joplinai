# -*- coding: utf-8 -*-
# %%
"""统一报告模块 — 从 center_api stats 端点获取数据 → Markdown 格式化 → 写入 Joplin"""
# %% [markdown]
# # ReportWriter — 统一报告生成与写入

# %%
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import pathmagic

with pathmagic.Context():
    from func.logme import log


# %% [markdown]
# # ReportWriter

# %%
__all__ = ["ReportWriter", "main"]

class ReportWriter:
    """统一报告生成器 — 从 stats 端点取数据，格式化后写入 Joplin"""

    def __init__(self, config: Dict, history_client=None, cache_client=None):
        self.config = config
        self.history = history_client
        self.cache = cache_client      # CacheClient

# %% [markdown]
#     # ## generate_vectorization_report

    # %%
    def generate_vectorization_report(self, snapshot: Dict) -> str:
        """生成向量化处理报告

        snapshot = {
            "summary_data": {notebook_title: {...}},
            "global_chunk_stats": {...},
            "history_db_path": Path,
        }
        """
        summary_data = snapshot.get("summary_data", {})
        chunk_stats = snapshot.get("global_chunk_stats", {})
        history_db_path = snapshot.get("history_db_path", Path("."))

        md_lines = ["# 📊 Joplin笔记向量化处理统计报告（历史分析版）\n"]
        md_lines.append(
            f"*报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*  |  "
            f"*嵌入模型：{self.config.get('ollama_embedding_model', 'N/A')}*  |  "
            f"*历史数据库：{history_db_path.name}*\n"
        )

        # === Part 1: 本次运行快照 ===
        md_lines.append("## 1. 本次运行快照\n")
        total_notes = sum(s.get("total_notes", 0) for s in summary_data.values())
        total_updated = sum(s.get("updated_count", 0) for s in summary_data.values())
        total_failed = sum(len(s.get("failed_notes", [])) for s in summary_data.values())
        total_notes_added = sum(len(s.get("notes_added", [])) for s in summary_data.values())
        total_notes_removed = sum(len(s.get("notes_removed", [])) for s in summary_data.values())
        md_lines.append(f"- **处理笔记本数**：`{len(summary_data)}` 个")
        md_lines.append(f"- **涉及笔记总数**：`{total_notes}` 条")
        md_lines.append(f"- **笔记动态变化**：新增 `{total_notes_added}` 条，移除 `{total_notes_removed}` 条")
        md_lines.append(f"- **本次内容更新**：`{total_updated}` 条笔记的内容发生了变更")
        md_lines.append(f"- **处理异常笔记**：`{total_failed}` 条\n")
        md_lines.append(f"- **处理总块数**：`{chunk_stats.get('total_chunks_processed', 0)}`")
        md_lines.append(f"- **向量库更新**：新增或更新 `{chunk_stats.get('chunks_upserted', 0)}` 个块")
        md_lines.append(f"- **内容未变更**：跳过 `{chunk_stats.get('chunks_skipped', 0)}` 个块")
        md_lines.append(f"- **清理孤儿块**：移除 `{chunk_stats.get('orphan_chunks_cleaned', 0)}` 个块\n")

        # === Part 2: 历史累积统计 ===
        md_lines.append("## 2. 历史累积统计（全部数据）\n")
        cumulative = self._safe_call(self.history, "get_cumulative_stats", days=None)
        if cumulative:
            cum = cumulative.get("cumulative", {})
            md_lines.append(f"- **总运行次数**：`{cum.get('total_runs', 0)}` 次")
            md_lines.append(f"- **涉及笔记本数**：`{cum.get('total_notebooks_touched', 0)}` 个")
            md_lines.append(f"- **累计处理笔记**：`{cum.get('total_notes_processed_all_time', 0)}` 条")
            md_lines.append(f"- **累计处理文本块**：`{cum.get('total_chunks_processed_all_time', 0)}` 个")
            md_lines.append(f"- **累计新增笔记**：`{cum.get('total_notes_added_all_time', 0)}` 条")
            md_lines.append(f"- **累计移除笔记**：`{cum.get('total_notes_removed_all_time', 0)}` 条")
            md_lines.append(f"- **累计更新块数**：`{cum.get('total_chunks_updated_all_time', 0)}` 个")
            md_lines.append(f"- **累计清理孤儿块**：`{cum.get('total_orphans_cleaned_all_time', 0)}` 个\n")
        else:
            md_lines.append("*历史数据统计暂不可用*\n")

        # === Part 3: 近期趋势分析 ===
        md_lines.append("## 3. 近期趋势分析（最近30天）\n")
        trend_30d = self._safe_call(self.history, "get_cumulative_stats", days=30)
        if trend_30d and trend_30d.get("cumulative"):
            cum_30d = trend_30d["cumulative"]
            md_lines.append(f"- **运行次数**：`{cum_30d.get('total_runs', 0)}` 次")
            md_lines.append(f"- **处理笔记**：`{cum_30d.get('total_notes_processed_all_time', 0)}` 条")
            md_lines.append(f"- **处理文本块**：`{cum_30d.get('total_chunks_processed_all_time', 0)}` 个")
            md_lines.append(f"- **新增笔记**：`{cum_30d.get('total_notes_added_all_time', 0)}` 条")
            md_lines.append(f"- **移除笔记**：`{cum_30d.get('total_notes_removed_all_time', 0)}` 条\n")
            if trend_30d.get("weekly_trends"):
                md_lines.append("**周度处理趋势：**")
                md_lines.append("| 周次 | 运行次数 | 处理笔记 | 处理块数 | 新增笔记 | 移除笔记 |")
                md_lines.append("|:---|:---|:---|:---|:---|:---|")
                for w in trend_30d["weekly_trends"][:6]:
                    md_lines.append(
                        f"| {w['week']} | {w['runs_count']} | {w['notes_processed']} | "
                        f"{w['chunks_processed']} | {w['notes_added']} | {w['notes_removed']} |"
                    )
                md_lines.append("")
        else:
            md_lines.append("*30天趋势数据暂不可用*\n")

        # === Part 4: 效率指标 ===
        md_lines.append("## 4. 系统效率指标（最近30天）\n")
        efficiency = self._safe_call(self.history, "get_efficiency_metrics", days=30)
        if efficiency:
            avg_notes = float(efficiency.get("avg_notes_per_run", 0) or 0)
            avg_chunks = float(efficiency.get("avg_chunks_per_run", 0) or 0)
            update_rate = efficiency.get("avg_update_rate_percent", 0)
            skip_rate = efficiency.get("avg_skip_rate_percent", 0)
            add_rate = efficiency.get("avg_addition_rate_percent", 0)
            remove_rate = efficiency.get("avg_removal_rate_percent", 0)
            success_rate = efficiency.get("success_rate_percent", 0)
            avg_runs = float(efficiency.get("avg_runs_per_day", 0) or 0)
            md_lines.append(f"- **平均每次运行**：处理 `{avg_notes:.1f}` 条笔记，`{avg_chunks:.1f}` 个文本块")
            md_lines.append(f"- **块处理效率**：更新率 `{update_rate}%`，跳过率 `{skip_rate}%`")
            md_lines.append(f"- **笔记变动率**：新增率 `{add_rate}%`，移除率 `{remove_rate}%`")
            md_lines.append(f"- **运行稳定性**：成功率 `{success_rate}%`，日均运行 `{avg_runs:.1f}` 次\n")
        else:
            md_lines.append("*效率指标暂不可用*\n")

        # === Part 5: 动态变化分析 ===
        md_lines.append("## 5. 笔记动态变化深度分析（最近30天）\n")
        change = self._safe_call(self.history, "get_change_analysis", days=30)
        if change:
            md_lines.append(f"- **唯一新增笔记**：`{change['added_count']}` 条")
            md_lines.append(f"- **唯一移除笔记**：`{change['removed_count']}` 条")
            md_lines.append(f"- **净增长笔记**：`{change['net_growth']}` 条")
            md_lines.append(f"- **频繁变动笔记**：`{change['frequently_changed_count']}` 条（既被添加过又被移除过）\n")
            if change.get("frequently_changed_notes"):
                md_lines.append("**频繁变动笔记示例：**")
                for note in change["frequently_changed_notes"][:5]:
                    md_lines.append(f"- `{note}`")
                if len(change["frequently_changed_notes"]) > 5:
                    md_lines.append(f"- ... 等 {len(change['frequently_changed_notes'])} 条")
                md_lines.append("")
        else:
            md_lines.append("*变化分析数据暂不可用*\n")

        # === Part 6: 最活跃笔记本排名 ===
        md_lines.append("## 6. 最活跃笔记本排名（历史累计）\n")
        if trend_30d and trend_30d.get("top_notebooks"):
            md_lines.append("| 笔记本 | 处理次数 | 累计笔记数 | 累计块数 | 最后处理时间 |")
            md_lines.append("|:---|:---|:---|:---|:---|")
            for nb in trend_30d["top_notebooks"][:10]:
                last_time = (
                    datetime.fromisoformat(nb["last_processed"]).strftime("%m-%d %H:%M")
                    if nb.get("last_processed") else "N/A"
                )
                md_lines.append(
                    f"| {nb['notebook_title']} | {nb['process_count']} | "
                    f"{nb['total_notes']} | {nb['total_chunks']} | {last_time} |"
                )
            md_lines.append("")
        else:
            md_lines.append("*活跃笔记本数据暂不可用*\n")

        # === Part 7: 本次运行笔记本详情 ===
        if summary_data:
            md_lines.append("## 7. 本次运行笔记本详情\n")
            md_lines.append("| 笔记本 | 笔记统计 (总/新/移/更/败) | 文本块统计 (总/更/跳/清) | 备注 |")
            md_lines.append("|:---|:---|:---|:---|")
            for notebook, stats in summary_data.items():
                notes_total = stats.get("total_notes", 0)
                updated = stats.get("updated_count", 0)
                failed_list = stats.get("failed_notes", [])
                failed_count = len(failed_list)
                notes_added = stats.get("notes_added", [])
                notes_removed = stats.get("notes_removed", [])
                notes_cell = f"{notes_total} / {len(notes_added)} / {len(notes_removed)} / {updated} / {failed_count}"
                cs = stats.get("chunk_stats", {})
                chunks_cell = f"{cs.get('total_chunks', 0)} / {cs.get('upserted', 0)} / {cs.get('skipped', 0)} / {cs.get('orphans_cleaned', 0)}"
                remark_parts = []
                if failed_list:
                    remark_parts.append(f"失败:{len(failed_list)}条")
                if notes_added:
                    remark_parts.append(f"新增:{len(notes_added)}条")
                if notes_removed:
                    remark_parts.append(f"移除:{len(notes_removed)}条")
                remark = "；".join(remark_parts) if remark_parts else "正常"
                md_lines.append(f"| {notebook} | {notes_cell} | {chunks_cell} | {remark} |")
            md_lines.append("")

        # === Part 8: 数据维护 ===
        md_lines.append("## 8. 数据维护\n")
        md_lines.append(f"- **历史数据库位置**：`{history_db_path}`")
        db_size = history_db_path.stat().st_size / (1024 * 1024) if history_db_path.exists() else 0
        md_lines.append(f"- **当前数据库大小**：`{db_size:.2f} MB`")
        md_lines.append("- **数据保留策略**：永久保留，支持时间范围查询")
        md_lines.append("- **清理建议**：如需清理，可直接备份或删除数据库文件\n")

        return "\n".join(md_lines)

# %% [markdown]
#     # ## generate_cache_report

    # %%
    def generate_cache_report(self) -> str:
        """生成 AI增强缓存分析报告"""
        if not self.cache:
            return "*缓存报告：cache_client 未配置*\n"

        report = self.cache.get_report()
        if not report:
            return "*缓存报告数据暂不可用*\n"

        md_lines = ["# 📊 AI增强缓存分析报告"]
        md_lines.append(f"**生成时间**: {datetime.now().isoformat()}")
        md_lines.append("")

        total = report.get("total", 0)
        recent_active = report.get("recent_active", 0)
        avg_hits = report.get("avg_hits", 0)
        validation_threshold = report.get("validation_threshold", 0)
        by_task = report.get("by_task", [])
        validation_status = report.get("validation_status", [])
        hit_distribution = report.get("hit_distribution", [])
        growth = report.get("growth_trends", {})

        # 执行摘要
        validated_count = sum(
            s.get("count", 0) for s in validation_status
        )
        md_lines.append("## 🎯 执行摘要")
        md_lines.append(f"- **总缓存条目**: {total}")
        md_lines.append(f"- **平均命中率**: {avg_hits:.2f} hits/entry")
        md_lines.append(f"- **近期活跃（7天）**: {recent_active} entries")
        md_lines.append(f"- **验证覆盖率**: {validated_count} entries validated")
        md_lines.append(f"- **验证阈值**: {validation_threshold} hits")
        md_lines.append(f"- **预测周增长**: {growth.get('predicted_weekly_growth', 0)} entries/week")
        md_lines.append("")

        # 任务类型分布
        md_lines.append("## 📈 任务类型分布")
        if by_task:
            md_lines.append("| 任务类型 | 条目数 |")
            md_lines.append("|----------|--------|")
            for task in by_task:
                md_lines.append(f"| {task['task']} | {task['count']} |")
        else:
            md_lines.append("*暂无数据*")
        md_lines.append("")

        # 按模型细分
        by_model = report.get("by_model", [])
        if by_model:
            md_lines.append("## 🤖 按模型细分")
            md_lines.append("| 模型 | 条目数 |")
            md_lines.append("|------|--------|")
            for m in by_model:
                md_lines.append(f"| {m['model']} | {m['count']} |")
            md_lines.append("")

        # 验证状态分析
        md_lines.append("## ✅ 验证状态分析")
        if validation_status:
            md_lines.append("| 验证结果 | 条目数 |")
            md_lines.append("|----------|--------|")
            for state in validation_status:
                md_lines.append(f"| {state['validation_result']} | {state['count']} |")
        else:
            md_lines.append("*暂无验证数据*")
        md_lines.append("")

        # 命中分布
        md_lines.append("## 🎯 命中次数分布")
        if hit_distribution:
            md_lines.append("| 命中范围 | 条目数 |")
            md_lines.append("|----------|--------|")
            for h in hit_distribution:
                md_lines.append(f"| {h['range']} | {h['count']} |")
        else:
            md_lines.append("*暂无数据*")
        md_lines.append("")

        # 增长趋势
        md_lines.append("## 📊 增长趋势（最近30天）")
        md_lines.append(f"- **预测周增长**: {growth.get('predicted_weekly_growth', 0)} 条目")
        md_lines.append("")
        daily_growth = growth.get("daily_growth", [])
        if daily_growth:
            md_lines.append("### 每日新增缓存")
            md_lines.append("| 日期 | 新增条目 | 累计总数 |")
            md_lines.append("|------|----------|----------|")
            cum_by_date = {c["date"]: c["cumulative"] for c in growth.get("cumulative_growth", [])}
            for d in reversed(daily_growth[-10:]):
                cum_val = cum_by_date.get(d["date"], "N/A")
                md_lines.append(f"| {d['date']} | {d['new_entries']} | {cum_val} |")
        else:
            md_lines.append("*暂无增长数据*")
        md_lines.append("")

        # 洞察建议
        md_lines.append("## 💡 洞察与建议")
        insights = []
        if recent_active / max(total, 1) < 0.3 and total > 100:
            insights.append("**活跃度低**: 近期活跃缓存比例较低，考虑优化缓存策略或检查数据新鲜度")
        if avg_hits > 100:
            insights.append("**高命中率**: 缓存命中率很高，系统运行良好")
        if not insights:
            insights.append("缓存系统运行良好，继续保持当前策略")
        for insight in insights:
            md_lines.append(f"- {insight}")

        return "\n".join(md_lines)

# %% [markdown]
#     # ## write_to_joplin

    # %%
    def write_to_joplin(self, content: str, note_title: str,
                        notebook: str = "ewmobile", config_key: str = "vectorization_report") -> bool:
        """将报告写入 Joplin 笔记

        config_key 用于区分不同报告类型，避免 note_id 缓存冲突。
        默认 "vectorization_report"，缓存报告用 "enhance_cache_report" / "probe_cache_report"。
        """
        try:
            with pathmagic.Context():
                from func.jpfuncs import (
                    createnote, searchnotes, updatenote_body,
                    searchnotebook, getcfpoptionvalue, setcfpoptionvalue,
                )
            # 检查是否已缓存 note_id
            note_id = getcfpoptionvalue("joplinai", f"report_{config_key}", "note_id")
            if note_id:
                updatenote_body(note_id, content)
                log.info(f"已更新 Joplin 报告笔记: {note_title}")
                return True

            # 搜索或创建
            if not (notebook_id := searchnotebook(notebook)):
                with pathmagic.Context():
                    from func.jpfuncs import jpapi
                notebook_id = jpapi.add_notebook(title=notebook)
            existing = searchnotes(note_title, parent_id=notebook_id)
            if existing:
                note_id = existing[0].id
                updatenote_body(note_id, content)
            else:
                note_id = createnote(title=note_title, body=content, parent_id=notebook_id)
            setcfpoptionvalue("joplinai", f"report_{config_key}", "note_id", note_id)
            log.info(f"报告已写入 Joplin 笔记: {note_title}")
            return True
        except Exception as e:
            log.error(f"写入 Joplin 报告失败: {e}")
            return False

# %% [markdown]
#     # ## _safe_call

    # %%
    @staticmethod
    def _safe_call(client, method: str, **kwargs) -> Optional[Dict]:
        """安全调用客户端方法，失败返回 None"""
        if client is None:
            return None
        try:
            fn = getattr(client, method, None)
            if fn is None:
                return None
            return fn(**kwargs)
        except Exception as e:
            log.warning(f"调用 {method} 失败: {e}")
            return None


# %% [markdown]
# # CLI 入口 — 生成缓存报告并写入 Joplin

# %%
def _init_clients():
    """初始化 data center 客户端（从云端配置读取 URL/Key）"""
    import pathmagic
    with pathmagic.Context():
        from func.jpfuncs import getinivaluefromcloud
        from aimod.cache_client import CacheClient

    remote_url = getinivaluefromcloud("joplinai", "joplinai_center_url")
    if not remote_url:
        remote_url = "http://127.0.0.1:5003"
    api_key = getinivaluefromcloud("joplinai", "joplinai_center_api_key")
    if not api_key:
        raise RuntimeError("未配置 joplinai_center_api_key")

    cache_client = CacheClient(remote_url, api_key)
    return cache_client


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Joplinai 缓存报告生成器")
    parser.add_argument(
        "--type", type=str, default="enhance_cache",
        choices=["enhance_cache"],
        help="报告类型 (default: enhance_cache)",
    )
    parser.add_argument(
        "--output", type=str, default="joplin",
        choices=["joplin", "stdout"],
        help="输出目标 (default: joplin)",
    )
    args = parser.parse_args()

    import pathmagic
    with pathmagic.Context():
        from func.jpfuncs import getinivaluefromcloud
        from func.logme import log

    cache_client = _init_clients()
    config = {}
    writer = ReportWriter(config, cache_client=cache_client)

    log.info("生成 AI增强缓存报告...")
    content = writer.generate_cache_report()
    title = "AI增强缓存分析报告"
    config_key = "enhance_cache_report"
    if writer.cache:
        try:
            stats = writer.cache.get_report()
            if stats:
                total = stats.get("total", 0)
                recent = stats.get("recent_active", 0)
                growth = stats.get("growth_trends", {}) or {}
                log.info(
                    f"AI增强缓存统计: 总条目={total}, "
                    f"近期活跃(7天)={recent}, "
                    f"预测周增长={growth.get('predicted_weekly_growth', 0)}"
                )
        except Exception as e:
            log.warning(f"获取 AI增强缓存统计失败: {e}")

    if args.output == "stdout":
        print(content)
    else:
        ok = writer.write_to_joplin(content, title, config_key=config_key)
    if ok:
        log.info(f"《{title}》已写入 Joplin")
    else:
        log.error(f"《{title}》写入 Joplin 失败")

    log.info("缓存报告生成完成")


if __name__ == "__main__":
    main()
