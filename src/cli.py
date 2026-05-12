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
# # joplinai.py CLI 入口 — parse_args + main

# %%
import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from typing import Optional, Tuple

# %%
import pathmagic

with pathmagic.Context():
    try:
        from aimod.history_client import HistoryClient
        from aimod.run_tracker import RunTracker
        from func.first import getdirmain
        from func.jpfuncs import getinivaluefromcloud
        from func.logme import log
        from func.wrapfuncs import timethis
        from joplinai import CONFIG, add_file_lock, process_notes_incremental
    except ImportError as e:
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger(__name__)
        log.error(f"导入项目模块失败: {e}")


# %% [markdown]
# # parse_args()
#
# __all__ = ["parse_args", "main"]
#
#
# def parse_args():
#     parser = argparse.ArgumentParser(description="Joplin笔记向量化处理工具")
#     parser.add_argument(
#         "--model", type=str, default=CONFIG["embedding_model"],
#         help=f"Ollama嵌入模型名称（默认：{CONFIG['embedding_model']}）",
#     )
#     parser.add_argument(
#         "--notebook_titles", type=str, default=CONFIG["notebook_titles"],
#         help=f"笔记本名称（用逗号分割）（默认：{CONFIG['notebook_titles']}）",
#     )
#     parser.add_argument(
#         "--workers", type=int, default=CONFIG["max_workers"],
#         help=f"并发数（默认：{CONFIG['max_workers']}）",
#     )
#     parser.add_argument(
#         "--enable_deepseek_summary", action="store_true",
#         default=CONFIG["enable_deepseek_summary"],
#         help=f"开启deepseek摘要支持（默认：{CONFIG['enable_deepseek_summary']}）",
#     )
#     parser.add_argument(
#         "--enable_deepseek_tags", action="store_true",
#         default=CONFIG["enable_deepseek_tags"],
#         help=f"开启deepseek标签支持（默认：{CONFIG['enable_deepseek_tags']}）",
#     )
#     parser.add_argument(
#         "--enable_force_update", action="store_true",
#         default=CONFIG["force_update"],
#         help=f"强制更新开关（默认：{CONFIG['force_update']}）",
#     )
#     parser.add_argument(
#         "--batch-size", type=int, default=0,
#         help="分批处理时每轮处理的笔记本数，需配合 --enable_force_update 使用（默认：0=不分批）",
#     )
#     parser.add_argument(
#         "--note_ids", type=str, default="",
#         help="指定笔记ID列表（用\",\"或\"，\"分割），作为虚拟笔记集处理（默认：空）",
#     )
#     return parser.parse_args()


# %% [markdown]
# # main()
#
#
# @timethis
# def main() -> None:
#     """主函数：执行增量处理"""
#     args = parse_args()
#     dynamic_config = CONFIG.copy()
#
#     dynamic_config["embedding_model"] = args.model
#     model_name_str = (
#         dynamic_config.get("embedding_model")
#         .replace(":", "_").replace("/", "_").replace("-", "_")
#     )
#     dynamic_config["state_path"] = (
#         getdirmain() / "data" / f"joplin_process_state_{model_name_str}.json"
#     )
#
#     dynamic_config["max_workers"] = args.workers
#     dynamic_config["enable_deepseek_summary"] = args.enable_deepseek_summary
#     dynamic_config["enable_deepseek_tags"] = args.enable_deepseek_tags
#     if args.enable_force_update:
#         dynamic_config["force_update"] = True
#     else:
#         dynamic_config["force_update"] = getinivaluefromcloud("joplinai", "force_update")
#
#     dynamic_config["batch_size"] = args.batch_size
#
#     notebook_titles_str = dynamic_config["notebook_titles"]
#     if args.notebook_titles != dynamic_config["notebook_titles"]:
#         notebook_titles_str = args.notebook_titles
#     elif cloud_imp_nbs := getinivaluefromcloud("joplinai", "imp_nbs"):
#         notebook_titles_str = cloud_imp_nbs
#     dynamic_config["notebook_titles"] = notebook_titles_str
#
#     note_ids_str = args.note_ids or getinivaluefromcloud("joplinai", "imp_note_ids") or ""
#
#     log.info(
#         f"动态配置：模型={dynamic_config['embedding_model']}, "
#         f"处理状态文件={dynamic_config['state_path']}, "
#         f"ollama server={dynamic_config['ollama_host']}, "
#         f"ollama port={dynamic_config['ollama_port']}, "
#         f"chroma server={dynamic_config['chroma_server_host']}, "
#         f"chroma port={dynamic_config['chroma_server_port']}, "
#         f"笔记本={dynamic_config['notebook_titles']}, "
#         f"指定笔记ID={'有' if note_ids_str else '无'}, "
#         f"并发数={dynamic_config['max_workers']}，"
#         f"使能deepseek摘要功能为{dynamic_config['enable_deepseek_summary']}，"
#         f"使能deepseek标签功能为{dynamic_config['enable_deepseek_tags']}，"
#         f"强制更新为{dynamic_config['force_update']}"
#     )
#
#     # ==== 1. 文件锁 ====
#     model_name = dynamic_config["embedding_model"]
#     model_name_str2 = model_name.replace(":", "_").replace("/", "_").replace("-", "_")
#     lock_name = f"joplinai_{model_name_str2}.lock"
#     lock_file, lock_acquired = add_file_lock(model_name, lock_name, timeout=10800)
#     if not lock_acquired:
#         sys.exit(1)
#
#     # 初始化历史数据库客户端（远程优先）
#     history_client = None
#     try:
#         remote_url = getinivaluefromcloud("joplinai", "joplinai_center_url")
#         if not remote_url:
#             remote_url = "http://127.0.0.1:5003"
#         api_key = getinivaluefromcloud("joplinai", "joplinai_center_api_key")
#         if remote_url and api_key:
#             history_client = HistoryClient(remote_url, api_key)
#     except Exception:
#         pass
#
#     # 初始化笔记状态客户端（远程优先）
#     state_client = None
#     try:
#         remote_url = getinivaluefromcloud("joplinai", "joplinai_center_url")
#         if not remote_url:
#             remote_url = "http://127.0.0.1:5003"
#         api_key = getinivaluefromcloud("joplinai", "joplinai_center_api_key")
#         if remote_url and api_key:
#             from aimod.state_client import ProcessStateClient
#             state_client = ProcessStateClient(remote_url, api_key)
#     except Exception:
#         pass
#     dynamic_config["state_client"] = state_client
#
#     task_reporter = RunTracker(dynamic_config, history_client=history_client)
#
#     log.info("===== 启动Joplin笔记向量化处理 =====")
#     notebook_titles = [
#         title.strip()
#         for title in re.split(r"[,，]", notebook_titles_str.strip())
#         if title.strip()
#     ]
#     total_notebook_count = len(notebook_titles)
#     try:
#         batch_size = dynamic_config.get("batch_size", 0)
#         is_batch = batch_size > 0 and dynamic_config.get("force_update", False)
#         batch_progress = None
#         resume_index = 0
#         if is_batch:
#             batch_progress = dynamic_config["state_path"].with_suffix(".batch_progress")
#             if batch_progress.exists():
#                 try:
#                     with open(batch_progress) as f:
#                         progress = json.load(f)
#                     resume_index = progress.get("next_index", 0)
#                     log.info(
#                         f"分批处理检查点: 从第 {resume_index + 1} 个笔记本继续"
#                         f"（共 {total_notebook_count} 个）"
#                     )
#                 except Exception as e:
#                     log.warning(f"读取分批处理检查点失败: {e}，从头开始")
#             batch = notebook_titles[resume_index:resume_index + batch_size]
#             if not batch:
#                 log.info("分批处理: 所有笔记本已处理完毕")
#                 batch_progress.unlink()
#                 return
#             notebook_titles = batch
#             log.info(f"本轮分批处理 {len(batch)} 个笔记本: {batch}")
#         elif batch_size > 0:
#             log.warning("--batch-size 需配合 --enable_force_update 使用，本次忽略分批设置")
#
#         checkpoint_file = dynamic_config["state_path"].with_suffix(".checkpoint")
#         if checkpoint_file.exists():
#             with open(checkpoint_file, "r") as f:
#                 checkpoint = json.load(f)
#             notebook_titles = notebook_titles[checkpoint["last_processed_index"]:]
#
#         for i, notebook_title in enumerate(notebook_titles, 1):
#             log.info(f"开始处理笔记本（{i}/{len(notebook_titles)}）: 【{notebook_title}】…………")
#             stats = process_notes_incremental(
#                 notebook_title=notebook_title, config=dynamic_config
#             )
#             task_reporter.add_notebook_record(notebook_title, stats)
#
#             if i % 2 == 0:
#                 checkpoint_data = {
#                     "notebook": notebook_title,
#                     "last_processed_index": i,
#                     "timestamp": datetime.now().isoformat(),
#                 }
#                 with open(checkpoint_file, "w") as f:
#                     json.dump(checkpoint_data, f)
#
#         log.info("===== 所有笔记本处理完成 =====")
#
#         if note_ids_str.strip():
#             note_ids = [
#                 nid.strip()
#                 for nid in re.split(r"[,，]", note_ids_str.strip())
#                 if nid.strip()
#             ]
#             if note_ids:
#                 log.info(f"开始处理虚拟笔记集【[指定笔记]】，共 {len(note_ids)} 条指定笔记ID")
#                 v_stats = process_notes_incremental(
#                     "[指定笔记]", dynamic_config, note_ids=note_ids
#                 )
#                 task_reporter.add_notebook_record("[指定笔记]", v_stats)
#                 log.info("虚拟笔记集处理完成。")
#
#         if is_batch:
#             new_index = resume_index + len(notebook_titles)
#             if new_index >= total_notebook_count:
#                 if batch_progress.exists():
#                     batch_progress.unlink()
#                 log.info("分批处理完成: 所有笔记本已处理完毕")
#             else:
#                 with open(batch_progress, "w") as f:
#                     json.dump({
#                         "next_index": new_index,
#                         "timestamp": datetime.now().isoformat(),
#                     }, f)
#                 log.info(f"分批处理检查点已保存: 下次运行将从第 {new_index + 1} 个笔记本继续")
#
#         task_reporter.finalize_run(success=True)
#         from src.report_writer import ReportWriter
#         writer = ReportWriter(dynamic_config, history_client, cache_client=None)
#         snapshot = task_reporter.get_snapshot()
#         report_content = writer.generate_vectorization_report(snapshot)
#         success = writer.write_to_joplin(report_content, "JoplinAI向量化处理报告（历史分析版）")
#         if success:
#             log.info("处理统计报告已成功更新至Joplin笔记。")
#         else:
#             log.warning("处理统计报告更新至Joplin笔记失败。")
#     except Exception as e:
#         log.critical(f"主流程执行失败: {e}", exc_info=True)
#         return
#
#     if checkpoint_file.exists():
#         checkpoint_file.unlink()
