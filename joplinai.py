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
# # joplinai AI增强Joplin笔记管理（向量化+增强更新+并发加速）

# %% [markdown]
# """
# joplinai.py | 优化版：AI增强Joplin笔记管理（向量化+增量更新+并发加速）
# 核心功能：将Joplin笔记向量化存储到ChromaDB，支持语义检索；仅增量处理更新笔记。
# """

# %% [markdown]
# ## 导入核心库

# %% [markdown]
# ### 导入系统库

# %%
import argparse
import hashlib
import json
import logging
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
import ollama
from chromadb.config import Settings

# %% [markdown]
# ### 项目模块导入（根据实际路径调整）

# %%
try:
    from embedding_generator import EmbeddingGenerator
    from func.configpr import (
        findvaluebykeyinsection,
        getcfpoptionvalue,
        setcfpoptionvalue,
    )
    from func.first import dirmainpath, getdirmain
    from func.getid import getdeviceid, getdevicename, gethostuser
    from func.jpfuncs import (
        createnote,
        get_notes_in_notebook_by_title,
        get_tag_titles,
        getinivaluefromcloud,
        getnote,
        jpapi,
        searchnotebook,
        searchnotes,
        updatenote_body,
        updatenote_title,
    )
    from func.logme import log
    from func.sysfunc import execcmd, not_IPython
    from func.wrapfuncs import timethis
    from vector_db_manager import VectorDBManager
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    log.error(f"导入项目模块失败: {e}")

# %% [markdown]
# ## 核心配置（根据需求调整）

# %%
CONFIG = {
    "notebook_titles": "顺风顺水, 日新月异, 运营管理",  # 改为笔记本名称列表字符串
    "embedding_model": "dengcao/bge-large-zh-v1.5",  # 嵌入模型（Ollama本地模型，优先选dengcao/bge-large-zh-v1.5
    # "embedding_model": "qwen:1.8b",  # 嵌入模型（Ollama本地模型，优先选nomic-embed-text）
    "chunk_size": 2000,  # 文本分块大小（字符数，根据模型上下文调整，nomic支持8192）
    "max_context": 4000,  # 模型最大上下文（字符数，nomic-embed-text为8192）
    "concurrency_type": "thread",  # 固定使用多线程，移除 process 选项
    "max_workers": min(
        8, (os.cpu_count() or 1) * 2
    ),  # 动态设置最大工作者数：CPU核心数 * 2，上限为16
    "db_path": getdirmain() / "data" / "joplin_vector_db",  # ChromaDB存储路径
    "enable_deepseek_embed": False,  # 是否用DeepSeek嵌入替代本地嵌入（增强向量质量）
    "enable_deepseek_summary": False,  # 是否用DeepSeek生成摘要（增强笔记元数据）
    "enable_deepseek_tags": False,  # 是否用DeepSeek提取标签（增强笔记标签）
    "deepseek_api_key": getinivaluefromcloud("joplinai", "deepseek_token"),
    "deepseek_chat_model": "deepseek-chat",  # 修正模型名称
    "deepseek_embed_model": "deepseek-embedding",
    "force_update": False,  # 新增：强制更新开关，默认关闭
}

model_name_str = (
    CONFIG.get("embedding_model").replace(":", "_").replace("/", "_").replace("-", "_")
)
CONFIG["state_path"] = (
    getdirmain() / "data" / f"joplin_process_state_{model_name_str}.json"
)  # 处理状态文件路径


# %% [markdown]
# ## 功能函数集
# %% [markdown]
# ### 工具小集合
# %% [markdown]
# #### clean_text(text: str) -> str
# %%
def clean_text(text: str) -> str:
    """清理笔记文本：移除图片、格式符号、多余换行"""
    if not text:
        return ""
    # 移除图片链接：![alt](url)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # 移除Markdown格式符号：#、`、*、>、~、-等
    text = re.sub(r"[#*`>~-]", "", text)
    # 合并多余换行（3个以上换行→2个）
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# %% [markdown]
# #### compute_content_hash(title: str, body: str) -> str
# %%
def compute_content_hash(title: str, body: str) -> str:
    """计算笔记内容哈希（用于增量更新判断）"""
    content = f"{title}{body}"
    return hashlib.md5(content.encode("utf-8")).hexdigest()


# %% [markdown]
# #### load_process_state(state_path: Path) -> Dict[str, Dict]
# %%
def load_process_state(state_path: Path) -> Dict[str, Dict]:
    """加载处理状态（笔记ID→{更新时间, 哈希, 处理时间}）"""
    if state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.error(f"加载状态文件失败: {e}，将重置状态")
    return {}


# %% [markdown]
# #### save_process_state(state: Dict, state_path: Path)
# %%
def save_process_state(state: Dict, state_path: Path):
    """保存处理状态（增强序列化）"""

    def serialize(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        if isinstance(obj, (list, tuple)):
            return [serialize(item) for item in obj]
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        return str(obj)

    try:
        serialized_state = serialize(state)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(serialized_state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error(f"保存状态文件失败: {e}")


# %% [markdown]
# ### 笔记处理核心逻辑（增量更新+入库）
# %% [markdown]
# #### process_single_note(note, vector_db: VectorDBManager, model_name: str, config: Dict) -> bool
# %%
def process_single_note(
    note,
    vector_db: VectorDBManager,
    embedding_generator: EmbeddingGenerator,
    config: Dict,
) -> bool:
    """处理单条笔记（清理→嵌入→入库），返回是否成功"""
    try:
        # 获取笔记完整信息（含更新时间）
        note_detail = getnote(note.id)
        if not note_detail:
            log.warning(f"笔记《{note.title}》（{note.id}） 获取失败，跳过")
            return False

        # 计算内容哈希（标题+正文）
        current_hash = compute_content_hash(note.title, note.body)
        current_update_time = (
            note_detail.updated_time
        )  # Joplin笔记的更新时间（Unix时间戳）

        # 清理文本（标题+正文）
        cleaned_body = clean_text(note.body)
        text = f"{note.title}\n{cleaned_body}"

        # 生成嵌入
        embedding = EmbeddingGenerator(config["embedding_model"]).get_merged_embedding(
            text, config["enable_deepseek_embed"]
        )
        if not embedding:  # 嵌入生成失败
            log.error(f"笔记《{note.title}》（{note.id}） 嵌入生成失败，跳过")
            return False

        # 获取标签
        local_tags = get_tag_titles(note.id)  # 项目函数：获取笔记标签列表
        # -------------------------- DeepSeek增强加工（可选） --------------------------
        enhanced_metadata = {}
        # enhanced_metadata["note_id"] = note.id
        try:
            if config["enable_deepseek_summary"] and config["deepseek_api_key"]:
                from deepseek_enhancer import deepseek_process_note

                summary = deepseek_process_note(
                    text,
                    task="summary",
                    model=config.get("deepseek_chat_model", "deepseek-chat"),
                )
                enhanced_metadata["summary"] = summary or ""  # 存入摘要
        except Exception as e:
            log.error(f"笔记《{note.title}》（{note.id}） 嵌入生成失败，跳过。{e}")
            return false

        try:
            if config["enable_deepseek_tags"] and config["deepseek_api_key"]:
                tags_str = deepseek_process_note(
                    text,
                    task="tags",
                    model=config.get("deepseek_chat_model", "deepseek-chat"),
                )
                deepseek_tags = (
                    [t.strip() for t in tags_str.split(",")] if tags_str else []
                )
                # 合并本地标签与DeepSeek标签（去重）
                enhanced_tags = list(set(local_tags + deepseek_tags))
            else:
                enhanced_tags = local_tags

            enhanced_metadata["tags"] = ",".join(enhanced_tags)
        except Exception as e:
            log.error(f"笔记《{note.title}》（{note.id}） 嵌入生成失败，跳过。{e}")
            return false

        log.debug(f"笔记《{note.title}》（{note.id}）增强元数据: {enhanced_metadata}")

        log.info(
            f"笔记《{note.title}》（{note.id}）准备入库，嵌入维度：{len(embedding)}"
        )
        # ------ 入库（本地向量库+增强元数据），存在则更新，不存在则新增 ------
        vector_db.upsert_note(
            note_id=note.id,
            text=text,
            embedding=embedding,
            tags=enhanced_tags,
            metadata=enhanced_metadata,
        )
        log.info(f"笔记《{note.title}》（{note.id}）向量化处理完成！")
        return True

    except Exception as e:
        log.error(
            f"向量化处理笔记《{note.title}》（{note.id}）失败: {e}", exc_info=True
        )
        return False


# %% [markdown]
# #### process_notes_incremental(notebook_title: str, config: Dict)
# %%
def process_notes_incremental(notebook_title: str, config: Dict):
    """增量处理笔记本笔记（修复时间处理问题）"""
    # 动态获取模型最大上下文
    model_name = config["embedding_model"]
    max_context = EmbeddingGenerator(config["embedding_model"]).chunk_size
    chunk_size = min(config["chunk_size"], max_context // 2)
    log.info(
        f"使用模型“{model_name}”，动态分块配置：chunk_size={chunk_size}，max_context={max_context}"
    )

    # 初始化向量数据库（在整个处理过程中只初始化一次）
    if not hasattr(process_notes_incremental, "vector_db"):
        process_notes_incremental.vector_db = VectorDBManager(
            config["db_path"], model_name, True
        )
        log.info(
            f"向量数据库初始化完成，集合: {process_notes_incremental.vector_db.collection_name}"
        )

    vector_db = process_notes_incremental.vector_db

    # 加载处理状态
    process_state = load_process_state(config["state_path"])
    # 获取强制更新配置
    force_update = config.get("force_update", False)

    # 获取笔记本所有笔记
    notes = get_notes_in_notebook_by_title(notebook_title=notebook_title)
    if not notes:
        log.info(f"笔记本 【{notebook_title}】 无笔记，跳过处理")
        return {}

    log.info(f"开始增量处理笔记本 【{notebook_title}】，共 {len(notes)} 条笔记")
    updated_count = 0
    new_time_notes = []
    failed_notes = []

    with ThreadPoolExecutor(max_workers=config["max_workers"]) as executor:
        # 提交所有笔记处理任务
        future_to_note = {}
        for i, note in enumerate(notes):
            # 检查是否需要处理（增量更新判断）
            note_id = note.id
            note_detail = getnote(note_id)
            if not note_detail:
                continue

            # 时间格式转换，统一时间格式为时间戳
            current_update_time = note_detail.updated_time
            if isinstance(current_update_time, datetime):
                current_update_time = current_update_time.timestamp()
            elif isinstance(current_update_time, str):
                current_update_time = datetime.fromisoformat(
                    current_update_time
                ).timestamp()

            current_hash = compute_content_hash(note.title, note.body)
            last_state = process_state.get(note_id, {})

            # 只有需要更新的笔记才提交处理，为了方便调试，增加了云端配置的强制更新选项
            if force_update or not (
                last_state.get("update_time") == current_update_time
                and last_state.get("hash") == current_hash
            ):
                # 提交任务到线程池/进程池
                future = executor.submit(
                    process_single_note,
                    note,
                    vector_db,
                    model_name,
                    config,
                )
                future_to_note[future] = (note_id, current_update_time, current_hash)
                log.info(
                    f"开始处理笔记本【{notebook_title}】下的第（{i + 1}/{len(notes)}）条笔记: 《{note.title}》…………"
                )
                new_time_notes.append(note.title)

        # 收集处理结果
        for future in as_completed(future_to_note):
            note_id, update_time, content_hash = future_to_note[future]
            try:
                success = future.result()
                if success:
                    # 更新处理状态
                    process_state[note_id] = {
                        "update_time": float(update_time),
                        "hash": content_hash,
                        "processed_time": datetime.now().timestamp(),
                    }
                    updated_count += 1
                else:
                    failed_notes.append(note.title)
                    log.error(f"向量化处理笔记 《{note.title}》 时可能异常")
            except Exception as e:
                log.error(f"并发处理笔记 《{note.title}》 异常: {e}")
                failed_notes.append(note.title)

    # 保存状态
    save_process_state(process_state, config["state_path"])
    log.info(
        f"增量处理笔记本【{notebook_title}】中的笔记完成情况小结：新日期需要更新 {len(new_time_notes)} 条，成功 {updated_count} 条，失败 {len(failed_notes)} 条（总计 {len(notes)} 条）"
    )
    if failed_notes:
        log.warning(
            f"笔记本【{notebook_title}】中增量处理（向量化）失败的笔记: {set(failed_notes)}"
        )

    # 在函数末尾，整理并返回统计信息
    stats = {
        "notebook_title": notebook_title,
        "total_notes": len(notes),
        "updated_count": updated_count,
        "failed_notes": failed_notes,  # 这是一个列表
        "new_time_notes": new_time_notes,  # 需要更新的笔记标题列表
        "process_time": datetime.now().isoformat(),
    }

    log.info(f"笔记本【{notebook_title}】处理完成。统计：{stats}")
    return stats


# %% [markdown]
# ### 主流程入口

# %% [markdown]
# #### parse_args()


# %%
def parse_args():
    parser = argparse.ArgumentParser(description="Joplin笔记向量化处理工具")
    parser.add_argument(
        "--model",
        type=str,
        default=CONFIG["embedding_model"],
        help=f"Ollama嵌入模型名称（默认：{CONFIG['embedding_model']}）",
    )
    parser.add_argument(
        "--notebook_titles",
        type=str,
        default=CONFIG["notebook_titles"],
        help=f"笔记本名称（用“,”分割）（默认：{CONFIG['notebook_titles']}）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=CONFIG["max_workers"],
        help=f"并发数（默认：{CONFIG['max_workers']}）",
    )
    parser.add_argument(
        "--enable_deepseek_embed",
        action="store_true",
        default=CONFIG["enable_deepseek_embed"],
        help=f"开启deepseek嵌入支持（默认：{CONFIG['enable_deepseek_embed']}）",
    )
    parser.add_argument(
        "--enable_deepseek_summary",
        action="store_true",
        default=CONFIG["enable_deepseek_summary"],
        help=f"开启deepseek摘要支持（默认：{CONFIG['enable_deepseek_summary']}）",
    )
    parser.add_argument(
        "--enable_deepseek_tags",
        action="store_true",
        default=CONFIG["enable_deepseek_tags"],
        help=f"开启deepseek标签支持（默认：{CONFIG['enable_deepseek_tags']}）",
    )
    parser.add_argument(
        "--enable_force_update",
        action="store_true",
        default=CONFIG["force_update"],
        help=f"开启deepseek标签支持（默认：{CONFIG['force_update']}）",
    )
    return parser.parse_args()


# %% [markdown]
# #### main()


# %%
@timethis
def main():
    """主函数：执行增量处理"""
    args = parse_args()
    # 动态覆盖配置
    dynamic_config = CONFIG.copy()

    # 生成按照模型区分的状态文件名称
    dynamic_config["embedding_model"] = args.model
    model_name_str = (
        dynamic_config.get("embedding_model")
        .replace(":", "_")
        .replace("/", "_")
        .replace("-", "_")
    )
    dynamic_config["state_path"] = (
        getdirmain() / "data" / f"joplin_process_state_{model_name_str}.json"
    )

    dynamic_config["max_workers"] = args.workers
    dynamic_config["enable_deepseek_embed"] = args.enable_deepseek_embed
    dynamic_config["enable_deepseek_summary"] = args.enable_deepseek_summary
    dynamic_config["enable_deepseek_tags"] = args.enable_deepseek_tags
    if args.enable_force_update:
        dynamic_config["force_update"] = True
    else:
        dynamic_config["force_update"] = getinivaluefromcloud(
            "joplinai", "force_update"
        )

    # 处理笔记本列表字符串
    if args.notebook_titles != dynamic_config["notebook_titles"]:
        notebook_titles_str = args.notebook_titles
    elif notebook_titles_str := getinivaluefromcloud("joplinai", "imp_nbs"):
        pass
    else:
        notebook_titles_str = dynamic_config["notebook_titles"]
    dynamic_config["notebook_titles"] = notebook_titles_str
    log.info(
        f"动态配置：模型={dynamic_config['embedding_model']}, \
        处理状态文件={dynamic_config['state_path']}, \
        笔记本={dynamic_config['notebook_titles']}, \
        并发数={dynamic_config['max_workers']}， \
        使能deepseek嵌入模型为{dynamic_config['enable_deepseek_embed']}， \
        使能deepseek摘要功能为{dynamic_config['enable_deepseek_summary']}， \
        使能deepseek标签功能为{dynamic_config['enable_deepseek_tags']}， \
        强制更新为{dynamic_config['force_update']} \
        "
    )
    # 初始化任务报告器
    from aitaskreporter import JoplinAITaskReporter

    task_reporter = JoplinAITaskReporter(dynamic_config)

    log.info("===== 启动Joplin笔记向量化处理 =====")
    notebook_titles = [
        title.strip()
        for title in re.split(r"[,，]", notebook_titles_str.strip())
        if title.strip()
    ]
    try:
        # 添加检查点
        checkpoint_file = dynamic_config["state_path"].with_suffix(".checkpoint")
        if checkpoint_file.exists():
            with open(checkpoint_file, "r") as f:
                checkpoint = json.load(f)
            # 从断点处恢复处理
            notebook_titles = notebook_titles[checkpoint["last_processed_index"] :]

        for i, notebook_title in enumerate(notebook_titles):
            log.info(
                f"开始处理笔记本（{i + 1}/{len(notebook_titles)}）: 【{notebook_title}】…………"
            )
            # 调用原有的处理函数，但需要其返回统计信息
            # 建议修改 process_notes_incremental 使其返回统计字典
            stats = process_notes_incremental(
                notebook_title=notebook_title, config=dynamic_config
            )

            # 将统计结果添加到报告器
            task_reporter.add_notebook_record(notebook_title, stats)

            # 每处理2条笔记保存一次检查点
            if i % 2 == 0:
                checkpoint_data = {
                    "notebook": notebook_title,
                    "last_processed_index": i,
                    "timestamp": datetime.now().isoformat(),
                }
                with open(checkpoint_file, "w") as f:
                    json.dump(checkpoint_data, f)

        log.info("===== 所有笔记本处理完成 =====")

        # 生成并保存报告到Joplin
        report_content = task_reporter.generate_markdown_report()
        success = task_reporter.update_joplin_note(report_content)
        if success:
            log.info("处理统计报告已成功更新至Joplin笔记。")
        else:
            log.warning("处理统计报告更新至Joplin笔记失败。")
    except Exception as e:
        log.critical(f"主流程执行失败: {e}", exc_info=True)
        return

    # 处理完成后删除检查点
    if checkpoint_file.exists():
        checkpoint_file.unlink()


# %% [markdown]
# ## 主函数

# %%
if __name__ == "__main__":
    main()
