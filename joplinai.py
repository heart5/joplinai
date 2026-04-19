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
# # 导入库

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

# %%
try:
    from cache_manager import SQLiteCacheManager
    from embedding_generator import EmbeddingGenerator
    from func.configpr import (
        findvaluebykeyinsection,
        getcfpoptionvalue,
        setcfpoptionvalue,
    )
    from func.datatools import compute_content_hash
    from func.first import dirmainpath, getdirmain
    from func.getid import getdeviceid, getdevicename, gethostuser
    from func.jpfuncs import (
        createnote,
        get_notebook_ids_for_note,
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
# # 核心配置（根据需求调整）

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
    # "enable_deepseek_embed": False,  # 是否用DeepSeek嵌入替代本地嵌入（增强向量质量）
    "enable_deepseek_summary": False,  # 是否用DeepSeek生成摘要（增强笔记元数据）
    "enable_deepseek_tags": False,  # 是否用DeepSeek提取标签（增强笔记标签）
    "deepseek_api_key": getinivaluefromcloud("joplinai", "deepseek_token"),
    "deepseek_chat_model": "deepseek-chat",  # 修正模型名称
    "deepseek_embed_model": "deepseek-embedding",
    "force_update": False,  # 新增：强制更新开关，默认关闭
    "chunk_overlap": 50,
    # 【新增】自适应分块配置
    "enable_adaptive_chunking": getinivaluefromcloud(
        "joplinai", "enable_adaptive_chunking"
    ),
    "adaptive_cache_size": 100,
}

model_name_str = (
    CONFIG.get("embedding_model").replace(":", "_").replace("/", "_").replace("-", "_")
)
CONFIG["state_path"] = (
    getdirmain() / "data" / f"joplin_process_state_{model_name_str}.json"
)  # 处理状态文件路径

# %% [markdown]
# # 全局变量

# %%
# 确保缓存目录存在
cache_dir = getdirmain() / "data" / ".deepseek_cache"
os.makedirs(cache_dir, exist_ok=True)
cache_db_path = cache_dir / "deepseek_cache.db"

# 创建全局缓存管理器实例
global_cache_manager = SQLiteCacheManager(db_path=str(cache_db_path))


# %% [markdown]
# # 功能函数集
# %% [markdown]
# ## 工具小集合
# %% [markdown]
# ### load_process_state(state_path: Path) -> Dict[str, Dict]
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
# ### save_process_state(state: Dict, state_path: Path)
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
        log.error(f"保存状态文件{state_path}失败: {e}")


# %% [markdown]
# ### filter_notes(notes: List[Any]) -> List[Any]

# %%
def filter_notes(notes):
    """
    根据云端配置过滤需要排除的笔记
    """
    try:
        if filter_notes_titles := getinivaluefromcloud(
            "joplinai", "filter_notes_titles"
        ):
            titles_for_remove = [
                title.strip()
                for title in re.split(r"[,，]", filter_notes_titles)
                if title.strip()
            ]
            notes = [note for note in notes if note.title not in titles_for_remove]
        return notes
    except Exception as e:
        log.error(f"笔记过滤函数执行失败: {e}")
        return notes


# %% [markdown]
# ## 笔记处理核心逻辑（增量更新+入库）
# %% [markdown]
# ### process_note_chunks(note, vector_db: VectorDBManager, embedding_generator: EmbeddingGenerator, config: Dict,) -> bool
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
            return {"success": False, "chunk_stats": {}}

        log.info(f"开始块级增量处理笔记: 《{note.title}》 (ID: {note.id})")

        # 1. 获取此笔记在向量库中所有现有块的 块ID->哈希 映射
        existing_chunks_map = vector_db.get_existing_chunk_hashes_for_note(note.id)
        # print(existing_chunks_map)
        log.info(
            f"笔记《{note.title}》在向量库中存在 {len(existing_chunks_map)} 个旧块。"
        )

        # 2. 准备笔记文本并分块
        if len(note.body) < 10:
            text = note.title
        else:
            text = f"{note.title}\n{note.body}"
        local_tags = get_tag_titles(note.id)
        tags_str = ",".join(local_tags) if local_tags else ""

        notebook_dicts = get_notebook_ids_for_note(note.id)
        notebook_dict = notebook_dicts[-1] if notebook_dicts else {}
        notebook_title = next(iter(notebook_dict.values()), "")  # 获取笔记本标题

        # 开始分块
        chunk_dicts = embedding_generator.split_into_semantic_chunks(
            text=text,
            note_title=note.title,
            note_tags=tags_str,
            source_notebook_title=notebook_title,  # 新增参数
        )
        if not chunk_dicts:
            log.warning(f"笔记《{note.title}》拆分不出有效内容块，跳过。")
            return {"success": False, "chunk_stats": {}}

        # 3. 遍历新分出的每个块，决定是否需要处理
        chunks_to_upsert = []  # 存放需要更新/插入的块信息
        new_chunk_hashes = {}  # 记录本次处理所有新块的 预期块ID -> 内容哈希
        skipped_chunks = 0

        for chunk_info in chunk_dicts:
            chunk_content = chunk_info["content"]
            base_metadata = chunk_info["metadata"]
            chunk_hash = base_metadata.get("content_hash", "")  # 从元数据中取出哈希
            metadata_chunk_idx_from_one = base_metadata["chunk_index"]
            tags = base_metadata.get("source_note_tags", "")
            tags_str = (
                ",".join(sorted(tags.split(","))) if tags else ""
            )  # 排序保证一致性
            notebook_title = base_metadata.get("source_notebook_title", "")
            meta_hash = compute_content_hash(f"{tags_str}{notebook_title}")

            # 构建此块预期的最终块ID (与原有逻辑保持一致)
            expected_chunk_id = f"{note.id}_chunk_{base_metadata['chunk_index']}"
            new_chunk_hashes[expected_chunk_id] = chunk_hash  # 记录

            # 检查是否需要处理此块
            need_process = True
            if expected_chunk_id in existing_chunks_map:
                # 如果块ID已存在，且哈希值相同，则跳过
                old_c_hash = existing_chunks_map[expected_chunk_id].get(
                    "content_hash", ""
                )
                old_m_hash = existing_chunks_map[expected_chunk_id].get("meta_hash", "")
                if (old_c_hash and old_c_hash == chunk_hash) and (
                    old_m_hash and old_m_hash == meta_hash
                ):
                    log.debug(
                        f"笔记《{note.title}》中的块 {metadata_chunk_idx_from_one} 内容"
                        f"（长度：{len(chunk_content)}）未变，元数据也未变，跳过嵌入生成。"
                    )
                    need_process = False
                    skipped_chunks += 1
                else:
                    log.debug(
                        f"笔记《{note.title}》中的块 {metadata_chunk_idx_from_one} "
                        f"内容（长度：{len(chunk_content)}）"
                        f"变化={not old_c_hash or old_c_hash != chunk_hash}"
                        f"，元数据变化={not old_m_hash or old_m_hash != meta_hash}，"
                        f"执行嵌入重新入库。"
                    )
                    if not old_c_hash or old_c_hash != chunk_hash:
                        base_metadata["content_hash"] = chunk_hash
                    if not old_m_hash or old_m_hash != meta_hash:
                        base_metadata["meta_hash"] = meta_hash
            else:
                # 如果块ID不存在，则是全新块，需要处理
                log.debug(
                    f"笔记《{note.title}》中的块 {metadata_chunk_idx_from_one} "
                    f"内容（长度：{len(chunk_content)}）是新增块，执行嵌入入库。"
                )

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
            metadata_chunk_idx_from_one = base_metadata["chunk_index"]

            # 生成嵌入
            embedding = embedding_generator.get_merged_embedding(chunk_data)
            if not embedding:
                log.warning(
                    f"笔记《{note.title}》块 {metadata_chunk_idx_from_one} （长度：{len(chunk_content)}）嵌入生成失败，跳过此块。"
                )
                continue

            enhanced_metadata = {}
            if len(chunk_content) > embedding_generator.chunk_size * 0.8:
                enhanced_metadata["potential_long_chunk"] = True

            notebook_dicts = get_notebook_ids_for_note(note.id)
            notebook_dict = notebook_dicts[-1]
            try:
                notebook_id, notebook_title = next(iter(notebook_dict.items()))
            except StopIteration:
                notebook_id, notebook_title = "", ""
            # 构建完整元数据
            metadata = {
                # 包含 content_hash, estimated_date, chunk_summary, tags, meta_hash 等
                **base_metadata,
                **enhanced_metadata,
                "source_note_id": note.id,
                "source_notebook_id": notebook_id,
                "source_notebook_title": notebook_title,
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
                    f"笔记《{note.title}》的块 【{metadata_chunk_idx_from_one}/{len(chunk_dicts)}】 （长度：{len(chunk_content)}）向量化入库更新成功，文本块元数据为：{metadata}"
                )
            except Exception as e:
                log.error(
                    f"笔记《{note.title}》存储块 {metadata_chunk_idx_from_one} （长度：{len(chunk_content)}）失败: {e}",
                    exc_info=True,
                )

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
                f"清理了笔记《{note.title}》相关的 {deleted_count} 个孤儿块（ID不在新分块中），块列表：{orphan_chunk_ids}。"
            )
        else:
            log.info(f"未发现笔记《{note.title}》相关需要清理的孤儿块。")

        # 6. 最终判断
        total_processed = successful_upserts + skipped_chunks
        if total_processed == len(chunk_dicts):
            log.info(
                f"笔记《{note.title}》块级增量处理完成。成功更新 {successful_upserts} 个块，跳过 {skipped_chunks} 个块。"
            )
            # 返回详细的统计字典，而不仅仅是 True
            return {
                "success": True,
                "chunk_stats": {
                    "total_chunks": len(chunk_dicts),
                    "upserted": successful_upserts,
                    "skipped": skipped_chunks,
                    "orphans_cleaned": len(orphan_chunk_ids),  # 本次清理的孤儿块数
                },
            }
        else:
            log.error(
                f"笔记《{note.title}》处理不完整。预期{len(chunk_dicts)}块，实际处理{total_processed}块。"
            )
            return {
                "success": False,
                "chunk_stats": {
                    "total_chunks": len(chunk_dicts),
                    "upserted": successful_upserts,
                    "skipped": skipped_chunks,
                    "orphans_cleaned": len(orphan_chunk_ids),
                },
            }

    except Exception as e:
        log.error(f"块级增量处理笔记《{note.title}》失败: {e}", exc_info=True)
        return {"success": False, "chunk_stats": {}}


# %% [markdown]
# ### process_notes_incremental(notebook_title: str, config: Dict)
# %%
def process_notes_incremental(notebook_title: str, config: Dict):
    """增量处理笔记本笔记（修复时间处理问题）"""
    model_name = config["embedding_model"]
    # 动态获取模型最大上下文
    # chunk_size = EmbeddingGenerator(config["embedding_model"]).chunk_size
    log.info(f"使用模型“{model_name}”")


# %% [markdown]
# #### 初始化向量库

    # %%
    # 初始化向量数据库（在整个处理过程中只初始化一次）
    if not hasattr(process_notes_incremental, "vector_db"):
        process_notes_incremental.vector_db = VectorDBManager(
            config["db_path"], model_name, True
        )
        log.info(
            f"向量数据库初始化完成，集合: {process_notes_incremental.vector_db.collection_name}"
        )
    vector_db = process_notes_incremental.vector_db
    # 根据文本块内容更新其预估日期，所有文本块遍历一遍
    # vector_db.refresh_estimated_date()
    # 输出向量库信息
    log.info(f"向量库《{vector_db}》信息：{vector_db.get_collection_info()}")

    # 初始化向量数据库（在整个处理过程中只初始化一次）
    if not hasattr(process_notes_incremental, "embedding_gen"):
        process_notes_incremental.embedding_gen = EmbeddingGenerator(
            config,
            config["embedding_model"],
            chunk_size=config.get("chunk_size", 512),
            chunk_overlap=config.get("chunk_overlap", 50),
            # 【新增】传递自适应分块配置
            enable_adaptive_chunking=config.get("enable_adaptive_chunking", False),
            adaptive_cache_size=config.get("adaptive_cache_size", 100),
            cache_manager=global_cache_manager,  # 传入统一的缓存管理器
        )
        log.info(f"嵌入生成器初始化完成")

    embedding_gen = process_notes_incremental.embedding_gen

    # 加载处理状态
    process_state = load_process_state(config["state_path"])
    # 获取强制更新配置
    force_update = config.get("force_update", False)

    # 获取笔记本所有笔记
    notes_all = get_notes_in_notebook_by_title(notebook_title=notebook_title)
    # 过滤需要排除的笔记
    notes = filter_notes(notes_all)
    if len(notes_all) != len(notes):
        all_set = set([note.id for note in notes_all])
        filter_set = set([note.id for note in notes])
        exclude_set = all_set - filter_set
        exclude_titles = [f"《{getnote(note_id).title}》" for note_id in exclude_set]
        log.info(f"笔记本 【{notebook_title}】中因为云端排除设定被排除处理的笔记：\t{'，'.join(exclude_titles)} ")

    if not notes:
        log.info(f"笔记本 【{notebook_title}】 无笔记，跳过处理")
        return {}

# %% [markdown]
# #### 开始增量处理，多线程

    # %%
    log.info(f"开始增量处理笔记本 【{notebook_title}】，共 {len(notes)} 条笔记")
    total_chunks_for_notebook = 0
    total_upserted_for_notebook = 0
    total_skipped_for_notebook = 0
    total_orphans_cleaned_for_notebook = 0
    updated_count = 0
    new_time_notes = []
    failed_notes = []

    with ThreadPoolExecutor(max_workers=config["max_workers"]) as executor:
        # 提交所有笔记处理任务
        future_to_note = {}
        for i, note in enumerate(notes, 1):
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

            # === 【核心修改】分别计算内容哈希和元数据哈希 ===
            # 1. 获取当前笔记的标签和笔记本信息（用于计算 meta_hash）
            local_tags = get_tag_titles(note_id)  # 返回标签列表
            tags_str = ",".join(sorted(local_tags)) if local_tags else ""  # 排序保证一致性
            notebook_dicts = get_notebook_ids_for_note(note.id)
            notebook_dict = notebook_dicts[-1] if notebook_dicts else {}
            current_notebook_title = next(iter(notebook_dict.values()), "")
            # 2. 计算内容哈希 (基于标题和正文)
            current_content_hash = compute_content_hash(f"{note.title}{note.body}")
            # 3. 计算元数据哈希 (基于标签和笔记本标题)
            current_meta_hash = compute_content_hash(f"{tags_str}{current_notebook_title}")
            # === 修改结束 ===
            # 获取上一次处理的状态（兼容旧格式）
            last_state = process_state.get(note_id, {})
            last_update_time = last_state.get("update_time")
            last_content_hash = last_state.get("content_hash")
            last_meta_hash = last_state.get("meta_hash")

            # 如果状态文件是旧的（没有meta_hash字段），则视为需要更新元数据
            needs_meta_update = ("meta_hash" not in last_state)

            # 判断是否需要处理：强制更新 或 更新时间变化 或 内容哈希变化 或 元数据哈希变化
            if force_update or needs_meta_update or not (
                last_update_time == current_update_time
                and last_content_hash == current_content_hash
                and last_meta_hash == current_meta_hash
            ):
                log.info(
                    f"笔记《{note.title}》需要更新。"
                    f"原因: force={force_update}, meta_missing={needs_meta_update}, "
                    f"时间变化={last_update_time != current_update_time}, "
                    f"内容变化={last_content_hash != current_content_hash}, "
                    f"元数据变化={last_meta_hash != current_meta_hash}"
                )
                # 提交任务到线程池/进程池
                future = executor.submit(
                    process_note_chunks,
                    note,
                    vector_db,
                    embedding_gen,
                    config,
                )
                future_to_note[future] = (
                    note_id,
                    note.title,
                    current_update_time,
                    current_content_hash,
                    current_meta_hash,
                )
                log.info(
                    f"开始处理笔记本【{notebook_title}】下的第（{i}/{len(notes)}）条笔记: 《{note.title}》…………"
                )
                new_time_notes.append(note.title)

        # 收集处理结果
        for future in as_completed(future_to_note):
            note_id, note_title, update_time, content_hash, meta_hash = future_to_note[future]
            try:
                result_dict = future.result() # 现在接收的是字典
                success = result_dict.get("success", False)
                note_chunk_stats = result_dict.get("chunk_stats", {})

                # 累加块统计
                total_chunks_for_notebook += note_chunk_stats.get("total_chunks", 0)
                total_upserted_for_notebook += note_chunk_stats.get("upserted", 0)
                total_skipped_for_notebook += note_chunk_stats.get("skipped", 0)
                total_orphans_cleaned_for_notebook += note_chunk_stats.get("orphans_cleaned", 0)

                if success:
                    # 更新处理状态
                    process_state[note_id] = {
                        "note_title": note_title,
                        "update_time": float(update_time),
                        "content_hash": content_hash,
                        "meta_hash": meta_hash,
                        "processed_time": datetime.now().timestamp(),
                    }
                    updated_count += 1
                else:
                    failed_notes.append(note_title)
                    log.error(f"向量化处理笔记 《{note_title}》 时可能异常")
            except Exception as e:
                log.error(f"并发处理笔记 《{note_title}》 异常: {e}")
                failed_notes.append(note_title)
    # 保存状态
    save_process_state(process_state, config["state_path"])
    log.info(
        f"增量处理笔记本【{notebook_title}】中的笔记完成情况小结：新日期需要更新 {len(new_time_notes)} 条，成功 {updated_count} 条，失败 {len(failed_notes)} 条（总计 {len(notes)} 条）"
    )
    if failed_notes:
        log.warning(
            f"笔记本【{notebook_title}】中增量处理（向量化）失败的笔记: {set(failed_notes)}"
        )

# %% [markdown]
# #### 清理已移除当前笔记本的向量数据

    # %%
    # 在 process_notes_incremental 函数内部，所有笔记处理完成后，返回 stats 之前添加
    # ========== 新增：清理已移出当前笔记本的笔记的向量数据 ==========
    log.info(f"开始执行笔记本【{notebook_title}】的‘孤儿笔记’向量数据清理...")
    try:
        # 1. 获取当前笔记本对象及其ID
        notebook_id = searchnotebook(notebook_title) # 直接返回笔记本的id
        current_notebook_id = notebook_id

        if not current_notebook_id:
            log.warning(f"无法获取笔记本【{notebook_title}】的ID，跳过清理步骤。")
        else:
            # 2. 获取当前笔记本中所有笔记的ID集合
            current_note_ids_in_notebook = list(set(note.id for note in notes))

            # 3. 从向量库中查询所有 source_notebook_id 等于当前笔记本ID的块，并聚合出笔记ID
            # 我们需要一个辅助函数来执行这个查询。由于ChromaDB的get不支持直接按元数据值分组，
            # 我们先获取所有相关块，再在内存中聚合。
            all_results = vector_db.collection.get(
                where={"source_notebook_id": current_notebook_id},
                include=["metadatas"]
            )
            # print(f"从向量数据库中取出当前笔记本【{notebook_title}】相关的文本块结果入下：\n{all_results}")

            notes_in_vector_db = {}
            for metadata in all_results.get("metadatas", []):
                notes_in_vector_db[metadata.get("source_note_id")] = metadata.get('source_note_title')
            log.info(
                f"从向量数据库既有数据查找到的笔记本【{notebook_title}】"
                f"中的笔记有这些：{list(notes_in_vector_db.values())[-5:]}……"
            )

            # 4. 找出在向量库中存在，但已不在当前笔记本中的笔记ID
            orphan_note_ids = [note_id for id in notes_in_vector_db if note_id not in current_note_ids_in_notebook]

            # 5. 清理这些“孤儿笔记”对应的所有块
            cleaned_total_chunks = 0
            for note_id in orphan_note_ids:
                deleted_chunks_count = vector_db.delete_chunks_by_note_id(note_id)
                cleaned_total_chunks += deleted_chunks_count
                if deleted_chunks_count > 0:
                    note_title = notes_in_vector_db.get(note_id, "")
                    log.info(f"清理已移除笔记《{note_title}》（note_id={note_id}）的孤儿文本块向量数据, 删除 {deleted_chunks_count} 个块。")
                    # 可选：从处理状态中移除该笔记的记录
                    process_state.pop(note_id, None)

            if cleaned_total_chunks > 0:
                log.info(f"笔记本【{notebook_title}】清理完成。共移除 {len(orphan_note_ids)} 条笔记的向量数据，涉及 {cleaned_total_chunks} 个块。")
            else:
                log.info(f"笔记本【{notebook_title}】未发现需要清理的孤儿笔记向量数据。")

            notes_removed_titles = [notes_in_vector_db.get(nid) for nid in orphan_note_ids]

            # 计算新增笔记 (当前笔记本中的笔记ID 减去 向量库中存在的笔记ID)
            current_note_ids = set(note.id for note in notes)
            notes_added_ids = current_note_ids - set(notes_in_vector_db)
            notes_added_titles = [getnote(nid).title for nid in notes_added_ids if getnote(nid)]

    except Exception as e:
        log.error(f"执行笔记本【{notebook_title}】的清理步骤时出错: {e}", exc_info=True)
    # ========== 清理步骤结束 ==========

# %% [markdown]
# #### 统计汇总

    # %%
    # ========== 在“清理已移除当前笔记本的向量数据”步骤之后，构建最终stats之前 ==========
    # 此时我们已经有了 orphan_note_ids (被清理的笔记ID集合)
    # 和 notes_in_vector_db (向量库中属于本笔记本的笔记字典)
    # 以及 current_note_ids (当前笔记本中的笔记ID集合)

    # 1. 计算并获取“移除的笔记”标题列表
    notes_removed_titles = []
    for note_id in orphan_note_ids:
        note_title = notes_in_vector_db.get(note_id, "")
        if note_title:
            notes_removed_titles.append(note_title)
        else:
            # 笔记可能已被彻底删除，记录ID
            notes_removed_titles.append(f"[已删除的笔记: {note_id}]")

    # 2. 计算并获取“新增的笔记”标题列表
    # 新增的笔记 = 当前在笔记本中，但不在向量库历史记录中的笔记
    notes_added_ids = current_note_ids - set(notes_in_vector_db)
    notes_added_titles = []
    for note_id in notes_added_ids:
        note_detail = getnote(note_id)
        if note_detail:
            notes_added_titles.append(note_detail.title)

    # 3. 构建增强的统计字典
    stats = {
        "notebook_title": notebook_title,
        "total_notes": len(notes),
        "updated_count": updated_count,
        "failed_notes": list(set(failed_notes)),
        "new_time_notes": new_time_notes,
        # === 以下为新增字段，用于深度分析 ===
        "notes_added": notes_added_titles,      # 列表
        "notes_removed": notes_removed_titles,  # 列表
        "chunk_stats": {                        # 字典
            "total_chunks": total_chunks_for_notebook,
            "upserted": total_upserted_for_notebook,
            "skipped": total_skipped_for_notebook,
            "orphans_cleaned": total_orphans_cleaned_for_notebook,
        },
        "process_time": datetime.now().isoformat(),
    }

    log.info(f"笔记本【{notebook_title}】处理完成。")
    # 返回这个完整的 stats 字典
    return stats


# %% [markdown]
# ## 主流程入口

# %% [markdown]
# ### add_file_lock(model_name: str, lock_name: str = "joplinai.lock", timeout: int = 3600)

# %%
def add_file_lock(
    model_name: str, lock_name: str = "joplinai.lock", timeout: int = 3600
):
    """在脚本入口创建文件锁，防止多实例并发运行。

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
# ### parse_args()


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
# ### main()


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
        使能deepseek摘要功能为{dynamic_config['enable_deepseek_summary']}， \
        使能deepseek标签功能为{dynamic_config['enable_deepseek_tags']}， \
        强制更新为{dynamic_config['force_update']} \
        "
    )


# %% [markdown]
# #### 程序运行锁

    # %%
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

# %% [markdown]
# #### 给向量库项目的metadata添加两个字段

# %%
# ！！！一次性运行，给向量库项目的metadata添加notebook_id和notebook_title字段。
# vector_db = VectorDBManager(
#     dynamic_config["db_path"], dynamic_config["embedding_model"], True
# )
# vector_db.migrate_add_notebook_id(get_notebook_ids_for_note)
# ！！！运行结束！！！请注释掉上面两行代码

# %% [markdown]
# #### 初始化任务报告器并启动处理传入的笔记本列表

    # %%
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

        for i, notebook_title in enumerate(notebook_titles, 1):
            log.info(
                f"开始处理笔记本（{i}/{len(notebook_titles)}）: 【{notebook_title}】…………"
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
        # 关键步骤：通知报告器本次运行结束，保存全局记录
        task_reporter.finalize_run(success=True)

        # 生成并保存报告到joplin
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
# # 主函数

# %%
if __name__ == "__main__":
    main()
