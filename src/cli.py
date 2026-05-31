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
# # joplinai.py CLI 入口 — parse_args + main

# %%
import argparse
import logging
import re
import sys
from datetime import datetime

# %%
import pathmagic

with pathmagic.Context():
    try:
        from aimod.history_client import HistoryClient
        from aimod.run_tracker import RunTracker
        from func.getid import getdeviceid
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

# %%
__all__ = ["parse_args", "main"]


def parse_args():
    parser = argparse.ArgumentParser(description="Joplin笔记向量化处理工具")
    parser.add_argument(
        "--model", type=str, default=CONFIG["ollama_embedding_model"],
        help=f"Ollama嵌入模型名称（默认：{CONFIG['ollama_embedding_model']}）",
    )
    parser.add_argument(
        "--notebook_titles", type=str, default=CONFIG["notebook_titles"],
        help=f"笔记本名称（用逗号分割）（默认：{CONFIG['notebook_titles']}）",
    )
    parser.add_argument(
        "--workers", type=int, default=CONFIG["max_workers"],
        help=f"并发数（默认：{CONFIG['max_workers']}）",
    )
    parser.add_argument(
        "--summary_model", type=str,
        default=CONFIG.get("summary_model", "cloud"),
        choices=["cloud", "ollama", "none"],
        help=f"摘要模型: cloud/ollama/none（默认：{CONFIG.get('summary_model', 'cloud')}）",
    )
    parser.add_argument(
        "--tags_model", type=str,
        default=CONFIG.get("tags_model", "cloud"),
        choices=["cloud", "ollama", "none"],
        help=f"标签模型: cloud/ollama/none（默认：{CONFIG.get('tags_model', 'cloud')}）",
    )
    parser.add_argument(
        "--enable_force_update", action="store_true",
        default=CONFIG["force_update"],
        help=f"强制更新开关（默认：{CONFIG['force_update']}）",
    )
    parser.add_argument(
        "--batch-size", type=int, default=0,
        help="分批处理时每轮处理的笔记本数，需配合 --enable_force_update 使用（默认：0=不分批）",
    )
    parser.add_argument(
        "--note_ids", type=str, default="",
        help="指定笔记ID列表（用\",\"或\"，\"分割），作为虚拟笔记集处理（默认：空）",
    )
    return parser.parse_args()


# %% [markdown]
# # _resolve_center_url()


# %%
def _resolve_center_url() -> str:
    """解析 center_api URL，本机为数据中心时走 localhost。

    云端配置项 center_host_deviceid 与 func.getid.getdeviceid() 比对，
    匹配则返回 http://127.0.0.1:5003，否则走云端 joplinai_center_url。
    """
    center_deviceid = getinivaluefromcloud("joplinai", "center_host_deviceid")
    if center_deviceid:
        local_id = getdeviceid()
        if local_id and str(local_id) == str(center_deviceid):
            log.info(f"本机 deviceid 匹配 center_host_deviceid，center_url 走 localhost")
            return "http://127.0.0.1:5003"
    remote_url = getinivaluefromcloud("joplinai", "joplinai_center_url")
    if not remote_url:
        remote_url = "http://127.0.0.1:5003"
    return remote_url


# %% [markdown]
# # main()


# %%
@timethis
def main() -> None:
    """主函数：执行增量处理"""
    args = parse_args()
    dynamic_config = CONFIG.copy()

    dynamic_config["ollama_embedding_model"] = args.model

    dynamic_config["max_workers"] = args.workers
    dynamic_config["summary_model"] = args.summary_model
    dynamic_config["tags_model"] = args.tags_model
    if args.enable_force_update:
        dynamic_config["force_update"] = True
    else:
        dynamic_config["force_update"] = getinivaluefromcloud("joplinai", "force_update")

    dynamic_config["batch_size"] = args.batch_size

    notebook_titles_str = dynamic_config["notebook_titles"]
    if args.notebook_titles != dynamic_config["notebook_titles"]:
        notebook_titles_str = args.notebook_titles
    elif cloud_imp_nbs := getinivaluefromcloud("joplinai", "imp_nbs"):
        notebook_titles_str = cloud_imp_nbs
    dynamic_config["notebook_titles"] = notebook_titles_str

    note_ids_str = args.note_ids or getinivaluefromcloud("joplinai", "imp_note_ids") or ""

    # 区分用户意图：显式传参 vs 默认全量模式
    user_explicit_notebooks = args.notebook_titles != CONFIG["notebook_titles"]
    user_explicit_note_ids = bool(args.note_ids)

    log.info(
        f"动态配置：模型={dynamic_config['ollama_embedding_model']}, "
        f"ollama={dynamic_config['ollama_host']}, "
        f"chroma server={dynamic_config['chroma_server_host']}, "
        f"chroma port={dynamic_config['chroma_server_port']}, "
        f"笔记本={dynamic_config['notebook_titles']}, "
        f"指定笔记ID={'有' if note_ids_str else '无'}, "
        f"并发数={dynamic_config['max_workers']}，"
        f"摘要模型={dynamic_config['summary_model']}，"
        f"标签模型={dynamic_config['tags_model']}，"
        f"强制更新为{dynamic_config['force_update']}"
    )

    # ==== 1. 文件锁 ====
    from func.datatools import normalize_collection_name
    model_name = dynamic_config.get("siliconflow_embedding_model") or dynamic_config["ollama_embedding_model"]
    model_name_str2 = normalize_collection_name(model_name)
    lock_name = f"joplinai_{model_name_str2}.lock"
    lock_file, lock_acquired = add_file_lock(model_name, lock_name, timeout=10800)
    if not lock_acquired:
        sys.exit(1)

    # 初始化历史数据库客户端（远程优先）
    remote_url = _resolve_center_url()
    api_key = getinivaluefromcloud("joplinai", "joplinai_center_api_key")
    history_client = None
    try:
        if remote_url and api_key:
            history_client = HistoryClient(remote_url, api_key)
    except Exception:
        pass

    # 初始化笔记状态客户端（远程优先）
    state_client = None
    try:
        if remote_url and api_key:
            from aimod.state_client import ProcessStateClient
            state_client = ProcessStateClient(remote_url, api_key)
    except Exception:
        pass
    dynamic_config["state_client"] = state_client

    task_reporter = RunTracker(dynamic_config, history_client=history_client)

    log.info("===== 启动Joplin笔记向量化处理 =====")
    notebook_titles = [
        title.strip()
        for title in re.split(r"[,，]", notebook_titles_str.strip())
        if title.strip()
    ]
    total_notebook_count = len(notebook_titles)
    try:
        batch_size = dynamic_config.get("batch_size", 0)
        is_batch = batch_size > 0 and dynamic_config.get("force_update", False)
        resume_index = 0
        if is_batch:
            if state_client:
                bp = state_client.load_run_state(model_name_str2, "batch_progress")
                if bp:
                    resume_index = bp.get("next_index", 0)
                    log.info(
                        f"分批处理检查点: 从第 {resume_index + 1} 个笔记本继续"
                        f"（共 {total_notebook_count} 个）"
                    )
            else:
                log.error("state_client 未配置，无法读取分批处理进度")
            batch = notebook_titles[resume_index:resume_index + batch_size]
            if not batch:
                log.info("分批处理: 所有笔记本已处理完毕")
                if state_client:
                    state_client.delete_run_state(model_name_str2, "batch_progress")
                return
            notebook_titles = batch
            log.info(f"本轮分批处理 {len(batch)} 个笔记本: {batch}")
        elif batch_size > 0:
            log.warning("--batch-size 需配合 --enable_force_update 使用，本次忽略分批设置")

        if state_client:
            cp = state_client.load_run_state(model_name_str2, "checkpoint")
            if cp:
                notebook_titles = notebook_titles[cp["last_processed_index"]:]

        if not user_explicit_note_ids:
            for i, notebook_title in enumerate(notebook_titles, 1):
                log.info(f"开始处理笔记本（{i}/{len(notebook_titles)}）: 【{notebook_title}】…………")
                stats = process_notes_incremental(
                    notebook_title=notebook_title, config=dynamic_config
                )
                log.info(f"[笔记本进度: {i}/{len(notebook_titles)} ({i * 100 // len(notebook_titles)}%)] 【{notebook_title}】处理完成")
                task_reporter.add_notebook_record(notebook_title, stats)

                if i % 2 == 0 and state_client:
                    checkpoint_data = {
                        "notebook": notebook_title,
                        "last_processed_index": i,
                        "timestamp": datetime.now().isoformat(),
                    }
                    state_client.save_run_state(model_name_str2, "checkpoint", checkpoint_data)

            log.info("===== 所有笔记本处理完成 =====")
        else:
            log.info("===== 显式指定 --note_ids，跳过物理笔记本处理 =====")

        skip_virtual = user_explicit_notebooks and not user_explicit_note_ids
        if note_ids_str.strip() and not skip_virtual:
            note_ids = [
                nid.strip()
                for nid in re.split(r"[,，]", note_ids_str.strip())
                if nid.strip()
            ]
            if note_ids:
                log.info(f"开始处理虚拟笔记集【[指定笔记]】，共 {len(note_ids)} 条指定笔记ID")
                v_stats = process_notes_incremental(
                    "[指定笔记]", dynamic_config, note_ids=note_ids
                )
                task_reporter.add_notebook_record("[指定笔记]", v_stats)
                log.info("虚拟笔记集处理完成。")
        elif skip_virtual:
            log.info("显式指定 --notebook_titles，跳过虚拟笔记集【[指定笔记]】。")

        if is_batch:
            new_index = resume_index + len(notebook_titles)
            if new_index >= total_notebook_count:
                if state_client:
                    state_client.delete_run_state(model_name_str2, "batch_progress")
                log.info("分批处理完成: 所有笔记本已处理完毕")
            elif state_client:
                state_client.save_run_state(model_name_str2, "batch_progress", {
                    "next_index": new_index,
                    "timestamp": datetime.now().isoformat(),
                })
                log.info(f"分批处理检查点已保存: 下次运行将从第 {new_index + 1} 个笔记本继续")

        task_reporter.finalize_run(success=True)
        from src.report_writer import ReportWriter
        writer = ReportWriter(dynamic_config, history_client, cache_client=None)
        snapshot = task_reporter.get_snapshot()
        report_content = writer.generate_vectorization_report(snapshot)
        success = writer.write_to_joplin(report_content, "JoplinAI向量化处理报告（历史分析版）")
        if success:
            log.info("处理统计报告已成功更新至Joplin笔记。")
        else:
            log.warning("处理统计报告更新至Joplin笔记失败。")
    except Exception as e:
        log.critical(f"主流程执行失败: {e}", exc_info=True)
        return

    if state_client:
        state_client.delete_run_state(model_name_str2, "checkpoint")
