# -*- coding: utf-8 -*-
"""统一报告模块 — 从 center_api stats 端点获取数据 → Markdown 格式化 → 写入 Joplin"""
# %% [markdown]
# # ReportWriter — 统一报告生成与写入

# %%
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import pathmagic

with pathmagic.context():
    from func.logme import log


# %% [markdown]
# # ReportWriter

# %%
class ReportWriter:
    """统一报告生成器 — 从 stats 端点取数据，格式化后写入 Joplin"""

    def __init__(self, config: Dict, history_client=None, cache_client=None, probe_client=None):
        self.config = config
        self.history = history_client
        self.cache = cache_client      # DeepSeekCacheClient
        self.probe = probe_client      # ProbeCacheClient

    # %% [markdown]
    # ## generate_vectorization_report

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
            f"*嵌入模型：{self.config.get('embedding_model', 'N/A')}*  |  "
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
    # ## generate_cache_report

    # %%
    def generate_cache_report(self) -> str:
        """生成 DeepSeek 缓存分析报告"""
        if not self.cache:
            return "*缓存报告：cache_client 未配置*\n"

        report = self.cache.get_report()
        if not report:
            return "*缓存报告数据暂不可用*\n"

        md_lines = ["# 📊 DeepSeek 缓存分析报告"]
        md_lines.append(f"**生成时间**: {datetime.now().isoformat()}")
        md_lines.append("")

        # 执行摘要
        basic = report.get("basic_stats", {})
        time_ana = report.get("time_analysis", {})
        valid_ana = report.get("validation_analysis", {})
        perf = report.get("performance_metrics", {})
        growth = report.get("growth_trends", {})

        md_lines.append("## 🎯 执行摘要")
        md_lines.append(f"- **总缓存条目**: {basic.get('total_entries', 0)}")
        md_lines.append(f"- **总命中次数**: {basic.get('total_hits', 0)}")
        md_lines.append(f"- **平均命中率**: {basic.get('avg_hits_per_entry', 0):.2f} hits/entry")
        md_lines.append(f"- **近期活跃**: {time_ana.get('recent_active', 0)} entries (last 7 days)")
        validated_count = sum(
            v.get("count", 0) for v in valid_ana.get("validation_states", [])
            if v.get("validation_state") != "not_validated"
        )
        md_lines.append(f"- **验证覆盖率**: {validated_count} entries validated")
        md_lines.append(f"- **估算大小**: {perf.get('estimated_size_mb', 0)} MB")
        md_lines.append(f"- **预测周增长**: {growth.get('predicted_weekly_growth', 0)} entries/week")
        md_lines.append("")

        # 基础统计
        md_lines.append("## 📈 基础统计")
        md_lines.append(f"- **总记录数**: {basic.get('total_entries', 0)}")
        md_lines.append(f"- **总命中次数**: {basic.get('total_hits', 0)}")
        md_lines.append(f"- **当前周期命中**: {basic.get('current_hits', 0)}")
        md_lines.append(f"- **平均每条目命中**: {basic.get('avg_hits_per_entry', 0):.2f}")
        md_lines.append("")
        md_lines.append("### 任务类型分布")
        md_lines.append("| 任务类型 | 条目数 | 总命中 | 平均命中 |")
        md_lines.append("|----------|--------|--------|----------|")
        for task in basic.get("task_distribution", []):
            md_lines.append(f"| {task['task']} | {task['count']} | {task.get('hits', 0) or 0} | {task.get('avg_hits', 0) or 0:.1f} |")
        md_lines.append("")

        # 时间分析
        md_lines.append("## ⏰ 时间分析")
        md_lines.append(f"- **最近活跃缓存（7天内）**: {time_ana.get('recent_active', 0)}")
        md_lines.append("")
        md_lines.append("### 缓存年龄分布")
        md_lines.append("| 年龄分组 | 条目数 | 平均命中 |")
        md_lines.append("|----------|--------|----------|")
        for age in time_ana.get("age_distribution", []):
            md_lines.append(f"| {age['age_group']} | {age['count']} | {age.get('avg_hits', 0) or 0:.1f} |")
        md_lines.append("")

        # 验证分析
        md_lines.append("## ✅ 验证状态分析")
        md_lines.append("### 验证结果分布")
        md_lines.append("| 验证状态 | 条目数 | 平均命中 | 平均年龄(天) |")
        md_lines.append("|----------|--------|----------|--------------|")
        for state in valid_ana.get("validation_states", []):
            md_lines.append(
                f"| {state['validation_state']} | {state['count']} | "
                f"{state.get('avg_hits', 0) or 0:.1f} | {state.get('avg_age_days', 0) or 0:.1f} |"
            )
        md_lines.append("")
        md_lines.append(f"- **接近验证阈值**: {valid_ana.get('nearing_validation', 0)} 条")
        md_lines.append(f"- **最后验证时间**: {valid_ana.get('last_validation_time') or '从未验证'}")
        md_lines.append("")

        # 性能指标
        md_lines.append("## 🚀 性能指标")
        md_lines.append(f"- **估算缓存大小**: {perf.get('estimated_size_mb', 0)} MB")
        md_lines.append(f"- **陈旧条目（30天未访问且零命中）**: {perf.get('stale_entries', 0)}")
        md_lines.append("")
        md_lines.append("### 高命中缓存（Top 10）")
        md_lines.append("| 内容预览 | 任务 | 总命中 | 周期命中 | 创建时间 | 最近命中 |")
        md_lines.append("|----------|------|--------|----------|----------|----------|")
        for hit in perf.get("top_hitters", [])[:10]:
            created = (hit.get("created_at") or "")[:10]
            accessed = (hit.get("last_accessed") or "")[:10]
            md_lines.append(
                f"| `{hit.get('result_preview', '')}` | {hit.get('task', '')} | "
                f"{hit.get('total_hits', 0)} | {hit.get('hit_count', 0)} | {created} | {accessed} |"
            )
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

        # 洞察建议
        md_lines.append("")
        md_lines.append("## 💡 洞察与建议")
        insights = []
        if perf.get("stale_entries", 0) > 100:
            insights.append("**清理建议**: 发现较多陈旧条目，考虑运行 `cleanup_old_entries()` 或调整清理策略")
        if valid_ana.get("nearing_validation", 0) > 50:
            insights.append("**验证提醒**: 大量缓存接近验证阈值，建议安排批量验证")
        total = basic.get("total_entries", 1)
        if time_ana.get("recent_active", 0) / max(total, 1) < 0.3:
            insights.append("**活跃度低**: 近期活跃缓存比例较低，考虑优化缓存策略或检查数据新鲜度")
        if not insights:
            insights.append("缓存系统运行良好，继续保持当前策略")
        for insight in insights:
            md_lines.append(f"- {insight}")

        # 数据快照
        md_lines.append("")
        md_lines.append("## 🔍 数据快照")
        md_lines.append("```json")
        md_lines.append(
            json.dumps(
                {
                    "timestamp": datetime.now().isoformat(),
                    "total_entries": basic.get("total_entries", 0),
                    "total_hits": basic.get("total_hits", 0),
                    "validation_states": [
                        s.get("validation_state") for s in valid_ana.get("validation_states", [])
                    ],
                },
                indent=2,
            )
        )
        md_lines.append("```")

        return "\n".join(md_lines)

    # %% [markdown]
    # ## generate_probe_report

    # %%
    def generate_probe_report(self) -> str:
        """生成文本块长度探测缓存报告"""
        if not self.probe:
            return "*探测缓存报告：probe_client 未配置*\n"

        report = self.probe.get_report()
        if not report:
            return "*探测缓存报告数据暂不可用*\n"

        md_lines = ["# 📊 文本块长度探测缓存报告"]
        md_lines.append(f"**生成时间**: {datetime.now().isoformat()}")
        md_lines.append("")

        md_lines.append("## 📈 基础统计")
        md_lines.append(f"- **总条目数**: {report.get('total', 0)}")
        md_lines.append(f"- **上限**: {report.get('limit', 0)}")
        md_lines.append(f"- **近期活跃（7天）**: {report.get('recent_active', 0)}")

        len_stats = report.get("safe_len_stats", {})
        if len_stats:
            md_lines.append(f"- **安全长度范围**: {len_stats.get('min_len', '?')} ~ {len_stats.get('max_len', '?')} 字符")
            md_lines.append(f"- **平均安全长度**: {len_stats.get('avg_len', 0):.0f} 字符")
        md_lines.append("")

        # 按模型分布
        by_model = report.get("by_model", [])
        if by_model:
            md_lines.append("## 🤖 按模型分布")
            md_lines.append("| 模型 | 条目数 | 平均安全长度 | 平均块大小 | 最后使用 |")
            md_lines.append("|------|--------|-------------|-----------|---------|")
            for m in by_model:
                last = (m.get("last_used") or "")[:10]
                md_lines.append(
                    f"| {m['model_name']} | {m['count']} | "
                    f"{m.get('avg_safe_len', 0):.0f} | {m.get('avg_chunk_size', 0):.0f} | {last} |"
                )
            md_lines.append("")

        # 按块大小分布
        by_chunk = report.get("by_chunk_size", [])
        if by_chunk:
            md_lines.append("## 📐 按块大小分布")
            md_lines.append("| 块大小 | 条目数 | 平均安全长度 |")
            md_lines.append("|--------|--------|-------------|")
            for c in by_chunk:
                md_lines.append(f"| {c['chunk_size']} | {c['count']} | {c.get('avg_safe_len', 0):.0f} |")
            md_lines.append("")

        # 增长趋势
        daily = report.get("daily_new", [])
        if daily:
            md_lines.append("## 📊 每日新增（最近30天）")
            md_lines.append("| 日期 | 新增条目 |")
            md_lines.append("|------|----------|")
            for d in reversed(daily[-10:]):
                md_lines.append(f"| {d['date']} | {d['new_entries']} |")
            md_lines.append("")

        return "\n".join(md_lines)

    # %% [markdown]
    # ## write_to_joplin

    # %%
    def write_to_joplin(self, content: str, note_title: str,
                        notebook: str = "ewmobile", config_key: str = "vectorization_report") -> bool:
        """将报告写入 Joplin 笔记

        config_key 用于区分不同报告类型，避免 note_id 缓存冲突。
        默认 "vectorization_report"，缓存报告用 "deepseek_cache_report" / "probe_cache_report"。
        """
        try:
            with pathmagic.context():
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
                with pathmagic.context():
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
    # ## _safe_call

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
    with pathmagic.context():
        from func.jpfuncs import getinivaluefromcloud
        from aimod.center_client import DeepSeekCacheClient, ProbeCacheClient

    remote_url = getinivaluefromcloud("joplinai", "joplinai_center_url")
    if not remote_url:
        remote_url = "http://127.0.0.1:5003"
    api_key = getinivaluefromcloud("joplinai", "joplinai_center_api_key")
    if not api_key:
        raise RuntimeError("未配置 joplinai_center_api_key")

    cache_client = DeepSeekCacheClient(remote_url, api_key)
    probe_client = ProbeCacheClient(remote_url, api_key)
    return cache_client, probe_client


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Joplinai 缓存报告生成器")
    parser.add_argument(
        "--type", type=str, default="all",
        choices=["deepseek_cache", "probe_cache", "all"],
        help="报告类型 (default: all)",
    )
    parser.add_argument(
        "--output", type=str, default="joplin",
        choices=["joplin", "stdout"],
        help="输出目标 (default: joplin)",
    )
    args = parser.parse_args()

    import pathmagic
    with pathmagic.context():
        from func.jpfuncs import getinivaluefromcloud
        from func.logme import log

    cache_client, probe_client = _init_clients()
    config = {}
    writer = ReportWriter(config, cache_client=cache_client, probe_client=probe_client)

    report_types = (
        ["deepseek_cache", "probe_cache"] if args.type == "all" else [args.type]
    )

    for rtype in report_types:
        if rtype == "deepseek_cache":
            log.info("生成 DeepSeek 缓存报告...")
            content = writer.generate_cache_report()
            title = "DeepSeek缓存分析报告"
            config_key = "deepseek_cache_report"
        else:
            log.info("生成探测缓存报告...")
            content = writer.generate_probe_report()
            title = "文本块长度探测缓存报告"
            config_key = "probe_cache_report"

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
