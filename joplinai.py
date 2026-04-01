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
import atexit
import hashlib
import json
import logging
import os
import re
import sys
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
    "notebook_titles": "顺风顺水, 日新白异, 运营管理",  # 改为笔记本名称列表字符串
    "embedding_model": "dengcao/bge-large-zh-v1.5",  # 嵌入模型（Ollama本地模型，优先选dengcao/bge-large-zh-v1.5
    # "embedding_model": "qwen:1.8b",  # 嵌入模型（Ollama本地模型，优先选nomic-embed-text）
    # "chunk_size": 512,  # 文本分块大小（字符数，根据模型上下文调整）
    # "max_context": 512,  # 模型最大上下文（字符数）
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
# %%
def clean_text_other(text: str) -> str:
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
# #### enhance_by_deepseek_for_summary_tags(note, chunk: str, config: Dict)

# %%
def enhance_by_deepseek_for_summary_tags(note, chunk: str, config: Dict):
    """DeepSeek 官方模型增强生成小结和标签（和笔记既有标签进行融合）"""

    from deepseek_enhancer import deepseek_process_note

    enhanced_metadata = {}
    if config["enable_deepseek_summary"]:
        summary = deepseek_process_note(
            chunk,
            task="summary",
            model=config.get("deepseek_chat_model", "deepseek-chat"),
            use_cache=True,  # 启用缓存
        )
        enhanced_metadata["chunk_summary"] = summary or ""  # 存入摘要

    if config["enable_deepseek_tags"]:
        tags_str = deepseek_process_note(
            chunk,
            task="tags",
            model=config.get("deepseek_chat_model", "deepseek-chat"),
            use_cache=True,  # 启用缓存
        )
        deepseek_tags = [t.strip() for t in tags_str.split(",")] if tags_str else []
        # 合并本地标签与DeepSeek标签（去重）
        local_tags = get_tag_titles(note.id)  # 项目函数：获取笔记标签列表
        enhanced_tags = list(set(local_tags + deepseek_tags))
        enhanced_metadata["tags"] = ",".join(enhanced_tags)

    return enhanced_metadata


# %% [markdown]
# #### process_note_chunks(note, vector_db: VectorDBManager, embedding_generator: EmbeddingGenerator, config: Dict,) -> bool
# %%
def process_note_chunks(
    note,
    vector_db: VectorDBManager,
    embedding_generator: EmbeddingGenerator,
    config: Dict,
) -> bool:
    """处理单条笔记（块级增量更新），返回是否成功"""
    try:
        note_detail = getnote(note.id)
        if not note_detail:
            log.warning(f"笔记《{note.title}》（{note.id}） 获取失败，跳过")
            return False

        log.info(f"开始块级增量处理笔记: 《{note.title}》 (ID: {note.id})")

        # 1. 获取此笔记在向量库中所有现有块的 块ID->哈希 映射
        existing_chunks_map = vector_db.get_existing_chunk_hashes(note.id)
        log.info(
            f"笔记《{note.title}》在向量库中存在 {len(existing_chunks_map)} 个旧块。"
        )

        # 2. 准备笔记文本并分块
        if len(note.body) < 20:
            text = note.title
        else:
            text = f"{note.title}\n{note.body}"
        local_tags = get_tag_titles(note.id)
        tags_str = ",".join(local_tags) if local_tags else ""

        # 分块（此时每个块的 metadata 已包含 content_hash）
        chunk_dicts = embedding_generator.split_into_semantic_chunks(
            text=text, note_title=note.title, note_tags=tags_str
        )
        if not chunk_dicts:
            log.warning(f"笔记《{note.title}》分块后无内容，跳过。")
            return False

        # 3. 遍历新分出的每个块，决定是否需要处理
        chunks_to_upsert = []  # 存放需要更新/插入的块信息
        new_chunk_hashes = {}  # 记录本次处理所有新块的 预期块ID -> 内容哈希
        skipped_chunks = 0

        for chunk_info in chunk_dicts:
            chunk_content = chunk_info["content"]
            base_metadata = chunk_info["metadata"]
            chunk_hash = base_metadata.get("content_hash", "")  # 从元数据中取出哈希

            # 构建此块预期的最终块ID (与原有逻辑保持一致)
            expected_chunk_id = f"{note.id}_chunk_{base_metadata['chunk_index']}"
            new_chunk_hashes[expected_chunk_id] = chunk_hash  # 记录

            # 检查是否需要处理此块
            need_process = True
            if expected_chunk_id in existing_chunks_map:
                # 如果块ID已存在，且哈希值相同，则跳过
                if existing_chunks_map[expected_chunk_id] == chunk_hash:
                    log.debug(
                        f"笔记《{note.title}》中的块 {base_metadata['chunk_index']} 内容未变，跳过嵌入生成。"
                    )
                    need_process = False
                    skipped_chunks += 1
                else:
                    log.info(
                        f"笔记《{note.title}》中的块 {base_metadata['chunk_index']} 内容哈希已变化，需要重新嵌入。"
                    )
            # 如果块ID不存在，则是全新块，需要处理

            if need_process:
                # 此块需要处理，加入待处理列表
                chunks_to_upsert.append(
                    {
                        "chunk_id": expected_chunk_id,
                        "content": chunk_content,
                        "base_metadata": base_metadata,
                    }
                )

        log.info(
            f"笔记《{note.title}》共 {len(chunk_dicts)} 个块，其中 {skipped_chunks} 个跳过，{len(chunks_to_upsert)} 个需要处理。"
        )

        # 4. 处理所有需要更新的块
        successful_upserts = 0
        for chunk_data in chunks_to_upsert:
            chunk_id = chunk_data["chunk_id"]
            chunk_content = chunk_data["content"]
            base_metadata = chunk_data["base_metadata"]

            # 生成嵌入
            embedding = embedding_generator.get_merged_embedding(
                chunk_content, config["enable_deepseek_embed"]
            )
            if not embedding:
                log.warning(
                    f"笔记《{note.title}》块 {base_metadata['chunk_index']} 嵌入生成失败，跳过此块。"
                )
                continue

            # DeepSeek 增强生成摘要和标签
            enhanced_metadata = {}
            try:
                enhanced_metadata = enhance_by_deepseek_for_summary_tags(
                    note, chunk_content, config
                )
            except Exception as e:
                log.error(
                    f"对笔记《{note.title}》的块进行DeepSeek增强时失败: {e}",
                    exc_info=True,
                )

            # 构建完整元数据
            metadata = {
                **base_metadata,  # 包含 content_hash, estimated_date 等
                **enhanced_metadata,  # 包含 chunk_summary, tags 等
                "source_note_id": note.id,
            }

            # 存入向量数据库
            try:
                vector_db.upsert_chunk(
                    chunk_id=chunk_id,
                    text=chunk_content,
                    embedding=embedding,
                    tags=[tag.strip() for tag in metadata.get("tags", "").split(",")],
                    metadata=metadata,
                )
                successful_upserts += 1
                log.info(
                    f"笔记《{note.title}》的块 {base_metadata['chunk_index']} 向量化入库更新成功。"
                )
            except Exception as e:
                log.error(f"笔记《{note.title}》存储块失败: {e}", exc_info=True)

        # 5. 智能清理“孤儿块”
        # 找出那些存在于 existing_chunks_map，但不在本次新块列表 new_chunk_hashes 中的块ID
        orphan_chunk_ids = [
            chunk_id
            for chunk_id in existing_chunks_map
            if chunk_id not in new_chunk_hashes
        ]
        if orphan_chunk_ids:
            deleted_count = vector_db.delete_chunks_by_id_list(orphan_chunk_ids)
            log.info(
                f"清理了笔记《{note.title}》相关的 {deleted_count} 个孤儿块（ID不在新分块中）。"
            )
        else:
            log.info(f"未发现笔记《{note.title}》相关需要清理的孤儿块。")

        # 6. 最终判断
        total_processed = successful_upserts + skipped_chunks
        if total_processed == len(chunk_dicts):
            log.info(
                f"笔记《{note.title}》块级增量处理完成。成功更新 {successful_upserts} 个块，跳过 {skipped_chunks} 个块。"
            )
            return True
        else:
            log.error(
                f"笔记《{note.title}》处理不完整。预期{len(chunk_dicts)}块，实际处理{total_processed}块。"
            )
            return False

    except Exception as e:
        log.error(f"块级增量处理笔记《{note.title}》失败: {e}", exc_info=True)
        return False


# %% [markdown]
# #### process_notes_incremental(notebook_title: str, config: Dict)
# %%
def process_notes_incremental(notebook_title: str, config: Dict):
    """增量处理笔记本笔记（修复时间处理问题）"""
    model_name = config["embedding_model"]
    # 动态获取模型最大上下文
    # chunk_size = EmbeddingGenerator(config["embedding_model"]).chunk_size
    log.info(f"使用模型“{model_name}”")

    # 初始化向量数据库（在整个处理过程中只初始化一次）
    if not hasattr(process_notes_incremental, "vector_db"):
        process_notes_incremental.vector_db = VectorDBManager(
            config["db_path"], model_name, True
        )
        log.info(
            f"向量数据库初始化完成，集合: {process_notes_incremental.vector_db.collection_name}"
        )
    vector_db = process_notes_incremental.vector_db

    # 初始化向量数据库（在整个处理过程中只初始化一次）
    if not hasattr(process_notes_incremental, "embedding_gen"):
        process_notes_incremental.embedding_gen = EmbeddingGenerator(
            config["embedding_model"]
        )
        log.info(f"嵌入生成器初始化完成")

    embedding_gen = process_notes_incremental.embedding_gen

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
                    process_note_chunks,
                    note,
                    vector_db,
                    embedding_gen,
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
        "failed_notes": list(
            set(failed_notes)
        ),  # 这是一个集合，为了去重，因为可能是该笔记的多个块出错
        "new_time_notes": new_time_notes,  # 需要更新的笔记标题列表
        "process_time": datetime.now().isoformat(),
    }

    log.info(f"笔记本【{notebook_title}】处理完成。统计：{stats}")
    return stats


# %% [markdown]
# ### 主流程入口

# %% [markdown]
# #### add_file_lock(model_name: str, lock_name: str = "joplinai.lock", timeout: int = 3600)

# %%
def add_file_lock(
    model_name: str, lock_name: str = "joplinai.lock", timeout: int = 3600
):
    """
    在脚本入口创建文件锁，防止多实例并发运行。

    Args:
        lock_name: 锁文件名，建议与模型或配置关联以避免不同配置间的冲突。
        model_name: 嵌入模型名称
        timeout: 锁超时时间（秒），用于处理进程崩溃后锁未释放的情况。

    Returns:
        lock_file_path (Path): 锁文件路径，用于后续清理。
        acquired (bool): 是否成功获取锁。
    """
    # 确定锁文件存放目录，优先使用临时目录，其次使用脚本数据目录
    temp_dir = Path(os.getenv("TEMP", "/tmp"))
    lock_dir = temp_dir if temp_dir.exists() else getdirmain() / "data"
    lock_dir.mkdir(parents=True, exist_ok=True)

    lock_file_path = lock_dir / lock_name

    try:
        # 尝试创建锁文件（以独占模式打开，如果文件已存在则失败）
        lock_fd = os.open(lock_file_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        # 写入当前进程信息，便于调试
        with os.fdopen(lock_fd, "w") as f:
            f.write(f"pid: {os.getpid()}\n")
            f.write(f"time: {datetime.now().isoformat()}\n")
            f.write(f"model: {model_name}\n")

        log.info(f"文件锁创建成功: {lock_file_path}")

        # 注册退出时的清理函数
        def cleanup_lock():
            try:
                if lock_file_path.exists():
                    lock_file_path.unlink()
                    log.info(f"文件锁已清理: {lock_file_path}")
            except Exception as e:
                log.warning(f"清理锁文件时出错: {e}")

        atexit.register(cleanup_lock)
        return lock_file_path, True

    except FileExistsError:
        # 锁文件已存在，检查是否已超时（进程可能已崩溃）
        try:
            print(lock_file_path.stat().st_mtime)
            print(time.time() - timeout)
            if lock_file_path.stat().st_mtime < (time.time() - timeout):
                log.warning(
                    f"检测到过期的锁文件（超过{timeout}秒），将强制清理并继续。"
                )
                lock_file_path.unlink()
                # 递归调用自身以重试获取锁
                return add_file_lock(model_name, lock_name, timeout)
            else:
                # 锁有效，退出程序
                log.error(
                    f"另一个 joplinai.py 实例正在运行或上次运行未正常结束。\n"
                    f"锁文件: {lock_file_path}\n"
                    f"如需强制运行，请手动删除锁文件。"
                )
                return lock_file_path, False
        except Exception as e:
            log.error(f"检查锁文件状态时出错: {e}")
            return lock_file_path, False
    except Exception as e:
        log.error(f"创建文件锁时发生未知错误: {e}")
        return None, False

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

    # ==== 1. 文件锁：防止并发运行 ====
    # 生成与模型相关的唯一锁名，避免不同模型配置间的冲突
    model_name = dynamic_config["embedding_model"]
    model_name_str = model_name.replace(":", "_").replace("/", "_").replace("-", "_")
    lock_name = f"joplinai_{model_name_str}.lock"

    lock_file, lock_acquired = add_file_lock(
        model_name, lock_name, timeout=10800
    )  # 3小时超时
    if not lock_acquired:
        sys.exit(1)  # 获取锁失败，安全退出
    # ==== 文件锁结束 ====

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
