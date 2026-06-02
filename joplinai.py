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
# # joplinai AI增强Joplin笔记管理（向量化+增强更新+并发加速）

# %% [markdown]
# """
# joplinai.py | 优化版：AI增强Joplin笔记管理（向量化+增量更新+并发加速）
# 核心功能：将Joplin笔记向量化存储到ChromaDB，支持语义检索；仅增量处理更新笔记。
# """

# %% [markdown]
# # 导入库

# %%
import atexit
import hashlib
import logging
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import chromadb
except ImportError:
    import types
    from aimod.chromadb_http import ChromaDBHttpClient
    chromadb = types.SimpleNamespace(HttpClient=ChromaDBHttpClient)
try:
    import ollama
except ImportError:
    ollama = None

# %%
import pathmagic

with pathmagic.Context():
    try:
        from aimod.embedding_generator import EmbeddingGenerator
        from aimod.state_client import CenterAPIUnreachableError
        from aimod.vector_db_manager import VectorDBManager
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
    except ImportError as e:
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger(__name__)
        log.error(f"导入项目模块失败: {e}")

# %% [markdown]
# # 核心配置（根据需求调整）

# %%
CONFIG = {
    "environment": "production",
    "ollama_host": getinivaluefromcloud("joplinai", f"ollama_host_{getdeviceid()}"),
    "chroma_server_host": getinivaluefromcloud(
        "joplinai", f"chroma_server_{getdeviceid()}"
    ),
    "chroma_server_port": getinivaluefromcloud("joplinai", "chroma_port"),
    "notebook_titles": "顺风顺水, 日新白异, 运营管理",  # 改为笔记本名称列表字符串
    "ollama_embedding_model": getinivaluefromcloud("joplinai", "ollama_embedding_model") or "dengcao/bge-large-zh-v1.5",
    # "ollama_embedding_model": "qwen:1.8b",  # 嵌入模型（Ollama本地模型，优先选nomic-embed-text）
    # "chunk_size": 512,  # 文本分块大小（字符数，根据模型上下文调整）
    # "max_context": 512,  # 模型最大上下文（字符数）
    "concurrency_type": "thread",  # 固定使用多线程，移除 process 选项
    "max_workers": 4,  # 默认：嵌入走 Ollama 本地（semaphore=2）

    # 嵌入走硅基流动云端时，并发可调高（无 Ollama 本地瓶颈）
    # 注意：CONFIG["max_workers"] 在硅基流动模式下动态调整为 8，见模块尾部
    "db_path": str(getdirmain() / "data" / "joplin_vector_db"),  # ChromaDB存储路径

    # provider-agnostic 增强模型配置: "cloud" / "ollama" / "none"
    "summary_model": getinivaluefromcloud("joplinai", "summary_model") or "cloud",
    "tags_model": getinivaluefromcloud("joplinai", "tags_model") or "cloud",
    "cloud_model": getinivaluefromcloud("joplinai", "cloud_model") or "deepseek-v4-flash",
    "enhance_ollama_chat_model": getinivaluefromcloud("joplinai", "enhance_ollama_chat_model") or "qwen2.5:1.5b",
    "cloud_api_key": getinivaluefromcloud("joplinai", "cloud_api_key") or getinivaluefromcloud("joplinai", "deepseek_token"),
    # 嵌入后端选择: ollama(默认) / siliconflow
    "embedding_provider": getinivaluefromcloud("joplinai", "embedding_provider") or "ollama",
    "siliconflow_api_key": getinivaluefromcloud("joplinai", "siliconflow_api_key") or "",
    "siliconflow_embedding_model": getinivaluefromcloud("joplinai", "siliconflow_embedding_model") or "",
    "siliconflow_embedding_chunk_size": siliconflow_embedding_chunk_size
    if (siliconflow_embedding_chunk_size := getinivaluefromcloud("joplinai", "siliconflow_embedding_chunk_size"))
    else 1500,
    "siliconflow_embedding_dimension": siliconflow_embedding_dimension
    if (siliconflow_embedding_dimension := getinivaluefromcloud("joplinai", "siliconflow_embedding_dimension"))
    else 1024,

    "force_update": False,  # 新增：强制更新开关，默认关闭
    "chunk_overlap": 50,
    # 【新增】自适应分块配置
    "enable_adaptive_chunking": getinivaluefromcloud(
        "joplinai", "enable_adaptive_chunking"
    ),
    # 【新增】图片视觉处理开关，默认开启
    "vision_enabled": getinivaluefromcloud("joplinai", "vision_enabled"),
    "vision_model": getinivaluefromcloud("joplinai", "vision_model") or "",
}

# 硅基流动云端嵌入无 Ollama 本地瓶颈（semaphore=2），并发可调高
if CONFIG.get("siliconflow_embedding_model"):
    CONFIG["max_workers"] = 8

# %% [markdown]
# # 功能函数集
# %% [markdown]
# ## 工具小集合
# %% [markdown]
# ### filter_notes(notes: List[Any]) -> List[Any]

# %%
def filter_notes(notes) -> List:
    """根据云端配置过滤需要排除的笔记"""
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

# %%
def _resolve_enhance_config(config: dict, notebook_title: str) -> dict:
    """解析笔记本级增强策略覆盖。

    enhance_override 为 JSON 键，格式如：
    {"笔记本A": {"summary_model": "none", "tags_model": "ollama"}}
    未覆盖的笔记本回退全局 summary_model/tags_model。
    """
    import json
    overrides_raw = config.get("enhance_override", "")
    overrides = json.loads(overrides_raw) if overrides_raw else {}
    nb = overrides.get(notebook_title, {})
    resolved = {**config}
    for key in ("summary_model", "tags_model"):
        if key in nb:
            resolved[key] = nb[key]
    return resolved

# %% [markdown]
# ### _process_metadata_only_fast_path(note, vector_db, existing_chunks_map) → dict
# %%
def _process_metadata_only_fast_path(
    note,
    vector_db: VectorDBManager,
    existing_chunks_map: Dict[str, Dict[str, str]],
) -> dict:
    """快速路径：内容未变仅元数据变更时，跳过重复分块/嵌入/增强，
    直接获取既有块全量元数据，重新计算 meta_hash 后批量更新。"""
    total = len(existing_chunks_map)
    log.info(
        f"笔记《{note.title}》内容未变→快速路径：跳过{total}个块的重复分块/嵌入/增强。"
    )

    # 获取最新标签和笔记本信息
    local_tags = get_tag_titles(note.id)
    new_tags_str = ",".join(sorted(local_tags)) if local_tags else ""
    notebook_dicts = get_notebook_ids_for_note(note.id)
    nb_info = notebook_dicts[-1] if notebook_dicts else {}
    try:
        nb_id, nb_title = next(iter(nb_info.items()))
    except StopIteration:
        nb_id, nb_title = "", ""

    # 获取既有块的全量元数据
    existing_metas = vector_db.get_chunks_full_metadata(note.id)
    if not existing_metas:
        log.warning(f"笔记《{note.title}》快速路径：无法获取既有块元数据，回退全量处理。")
        return {"success": False, "chunk_stats": {},
                "_fallback": True}  # 特殊标记，调用方需回退

    # 为每个块重建元数据（保留不变的字段，更新标签/笔记本/meta_hash）
    batch_ids = []
    batch_tags = []
    batch_metadatas = []
    for chunk_id, old_meta in existing_metas.items():
        old_summary = old_meta.get("summary", "") or old_meta.get("chunk_summary", "")
        new_meta_hash = compute_content_hash(
            f"{new_tags_str}{nb_title}{old_summary}"
        )
        new_metadata = {
            "chunk_index": old_meta.get("chunk_index", 1),
            "content_hash": old_meta.get("content_hash", ""),
            "meta_hash": new_meta_hash,
            "source_note_title": old_meta.get("source_note_title", note.title),
            "source_note_id": note.id,
            "source_notebook_title": nb_title,
            "source_notebook_id": nb_id,
            "source_note_tags": new_tags_str,
            "chunk_summary": old_summary,
            "estimated_date": old_meta.get("estimated_date", ""),
            "word_count": old_meta.get("word_count", 0),
            "note_author": old_meta.get("note_author", "白晔峰"),
            "note_type": old_meta.get("note_type", "个人笔记"),
            "enhanced": old_meta.get("enhanced", True),
        }
        batch_ids.append(chunk_id)
        batch_tags.append([t.strip() for t in new_tags_str.split(",") if t.strip()])
        batch_metadatas.append(new_metadata)

    # 批量更新
    try:
        vector_db.batch_update_chunks_metadata(batch_ids, batch_tags, batch_metadatas)
        log.info(
            f"笔记《{note.title}》快速路径完成：批量更新 {total} 个块元数据，"
            f"新标签=[{new_tags_str}]，笔记本=[{nb_title}]"
        )
        return {
            "success": True,
            "chunk_stats": {
                "total": total, "full_upserted": 0,
                "metadata_updated": total, "skipped": 0,
                "failed_full": 0, "failed_metadata": 0,
                "retried_success": 0, "retried_failed": 0,
            },
        }
    except Exception as e:
        log.error(f"笔记《{note.title}》快速路径批量更新 {total} 块失败: {e}", exc_info=True)
        return {
            "success": False,
            "chunk_stats": {
                "total": total, "full_upserted": 0,
                "metadata_updated": 0, "skipped": 0,
                "failed_full": 0, "failed_metadata": total,
                "retried_success": 0, "retried_failed": 0,
            },
        }

# %% [markdown]
# ### process_note_chunks(note, vector_db: VectorDBManager, embedding_generator: EmbeddingGenerator, config: Dict,) -> bool
# %%
def process_note_chunks(
    note,
    vector_db: VectorDBManager,
    embedding_generator: EmbeddingGenerator,
    config: Dict,
    content_unchanged: bool = False,
    needs_re_enhance: bool = False,
) -> dict:
    """处理单条笔记（块级增量更新），返回是否成功。

    Args:
        content_unchanged: 内容哈希未变 → 可跳过重复分块+嵌入+增强
        needs_re_enhance: 增强缺失或配置变更 → 即使内容未变也需重新增强
    """
    try:
        note_detail = getnote(note.id)
        if not note_detail:
            log.warning(f"笔记《{note.title}》（{note.id}） 获取失败，跳过")
            return {"success": False, "chunk_stats": {}}

        log.info(f"开始块级增量处理笔记: 《{note.title}》 (ID: {note.id})")

        # 1. 获取此笔记在向量库中所有现有块的 块ID->哈希 映射
        existing_chunks_map = vector_db.get_existing_chunk_hashes_for_note(note.id)
        log.info(
            f"笔记《{note.title}》在向量库中存在 {len(existing_chunks_map)} 个旧块。"
        )

        # === 快速路径：内容未变且增强已完成 → 跳过重分块，仅更新元数据 ===
        if content_unchanged and not needs_re_enhance and existing_chunks_map:
            fast_result = _process_metadata_only_fast_path(
                note, vector_db, existing_chunks_map
            )
            if not fast_result.get("_fallback"):
                return fast_result
            log.warning(
                f"笔记《{note.title}》快速路径回退→走全量处理"
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
        # 图片处理：提取图片资源 → 获取base64 → Ollama Vision生成描述
        from aimod.text_preprocessor import TextPreprocessor

        vision_enabled = config.get("vision_enabled")
        if vision_enabled is None:
            vision_enabled = True  # 默认开启
        elif isinstance(vision_enabled, str):
            vision_enabled = vision_enabled.lower() not in ("false", "0", "no", "off")
        image_descs = {}
        resource_ids = TextPreprocessor.extract_resource_ids(note.body)

        if vision_enabled and resource_ids:
            try:
                log.info(
                    f"笔记《{note.title}》检测到 {len(resource_ids)} 个图片资源"
                )
                from aimod.image_processor import ImageProcessor
                image_proc = ImageProcessor(jpapi)
                images = image_proc.fetch_images_for_note(note.id, resource_ids)
                if images:
                    log.info(
                        f"笔记《{note.title}》成功获取 {len(images)}/{len(resource_ids)} 张图片"
                    )
                    from aimod.note_enhancer import describe_images
                    image_descs = describe_images(
                        images,
                        context=note.body,
                        model=config.get("vision_model", ""),
                    )
                    if image_descs:
                        total_chars = sum(len(d) for d in image_descs.values())
                        log.info(
                            f"笔记《{note.title}》图片视觉描述生成成功"
                            f"（{len(image_descs)}张，{total_chars}字符）"
                        )
                else:
                    log.debug(f"笔记《{note.title}》图片资源获取失败，回退纯文本模式")
            except Exception as e:
                log.warning(f"笔记《{note.title}》图片处理异常，回退纯文本模式: {e}")

        # 图片描述内联替换原图片链接位置
        if image_descs:
            text = TextPreprocessor.replace_images_with_descriptions(text, image_descs)
        elif resource_ids:
            text = TextPreprocessor.remove_image_syntax(text, keep_alt=True)

        chunk_dicts = embedding_generator.split_into_semantic_chunks(
            text=text,
            note_title=note.title,
            note_tags=tags_str,
            source_notebook_title=notebook_title,
        )
        if not chunk_dicts:
            log.warning(f"笔记《{note.title}》拆分不出有效内容块，跳过。")
            return {"success": False, "chunk_stats": {}}

        # 检查增强是否完成（启用了但所有块皆未成功增强）
        enhance_enabled = config.get("summary_model") != "none" or config.get("tags_model") != "none"
        enhance_missing = False
        if enhance_enabled:
            enhance_flags = [
                c.get("metadata", {}).get("enhanced", False)
                for c in chunk_dicts
            ]
            enhance_missing = not any(enhance_flags)
            if enhance_missing:
                log.warning(
                    f"笔记《{note.title}》增强未完成"
                    f"（共 {len(chunk_dicts)} 个块全部失败），将在下次运行时重试。"
                )

        # 3. 遍历新分出的每个块，决定是否需要处理
        chunks_to_upsert = []  # 内容变更或新块 → 全量处理
        chunks_metadata_only = []  # 内容未变仅元数据变更 → 跳过嵌入
        new_chunk_hashes = {}  # 记录本次处理所有新块的 预期块ID -> 内容哈希
        skipped_chunks = 0

        for chunk_info in chunk_dicts:
            chunk_content = chunk_info["content"]
            base_metadata = chunk_info["metadata"]
            chunk_hash = base_metadata.get("content_hash", "")  # 从元数据中取出哈希
            metadata_chunk_idx_from_one = base_metadata["chunk_index"]
            tags = base_metadata.get("tags") or base_metadata.get("source_note_tags", "")
            tags_str = (
                ",".join(sorted(tags.split(","))) if tags else ""
            )  # 排序保证一致性
            notebook_title = base_metadata.get("source_notebook_title", "")
            chunk_summary = base_metadata.get("chunk_summary", "")
            meta_hash = compute_content_hash(f"{tags_str}{notebook_title}{chunk_summary}")

            # 构建此块预期的最终块ID (与原有逻辑保持一致)
            expected_chunk_id = f"{note.id}_chunk_{base_metadata['chunk_index']}"
            new_chunk_hashes[expected_chunk_id] = chunk_hash  # 记录

            # 检查是否需要处理此块
            need_process = True
            metadata_only = False
            if expected_chunk_id in existing_chunks_map:
                old_c_hash = existing_chunks_map[expected_chunk_id].get(
                    "content_hash", ""
                )
                old_m_hash = existing_chunks_map[expected_chunk_id].get("meta_hash", "")
                if old_c_hash and old_c_hash == chunk_hash:
                    if old_m_hash and old_m_hash == meta_hash:
                        need_process = False
                        skipped_chunks += 1
                    else:
                        metadata_only = True
                        log.debug(
                            f"笔记《{note.title}》中的块 {metadata_chunk_idx_from_one} "
                            f"内容未变，仅元数据变更→跳过嵌入，仅更新元数据。"
                        )
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
                log.debug(
                    f"笔记《{note.title}》中的块 {metadata_chunk_idx_from_one} "
                    f"内容（长度：{len(chunk_content)}）是新增块，执行嵌入入库。"
                )

            if need_process:
                if metadata_only:
                    chunks_metadata_only.append(
                        {
                            "chunk_id": expected_chunk_id,
                            "content": chunk_content,
                            "base_metadata": base_metadata,
                        }
                    )
                else:
                    chunks_to_upsert.append(
                        {
                            "chunk_id": expected_chunk_id,
                            "content": chunk_content,
                            "base_metadata": base_metadata,
                            "embedding": chunk_info.get("embedding"),
                        }
                    )

        total_processed = len(chunks_to_upsert) + len(chunks_metadata_only)
        log.info(
            f"笔记《{note.title}》共 {len(chunk_dicts)} 个块，其中 {skipped_chunks} 个跳过，"
            f"{total_processed} 个需要处理（{len(chunks_metadata_only)} 仅元数据更新，"
            f"{len(chunks_to_upsert)} 全量更新）。"
        )

        # 4a. 处理仅元数据更新的块（无需重新生成嵌入）
        successful_metadata_updates = 0
        if chunks_metadata_only:
            notebook_dicts = get_notebook_ids_for_note(note.id)
            nb_info = notebook_dicts[-1] if notebook_dicts else {}
            try:
                nb_id, nb_title = next(iter(nb_info.items()))
            except StopIteration:
                nb_id, nb_title = "", ""

            batch_ids = []
            batch_tags = []
            batch_metadatas = []
            for chunk_data in chunks_metadata_only:
                metadata = {
                    **chunk_data["base_metadata"],
                    "source_note_id": note.id,
                    "source_notebook_id": nb_id,
                    "source_notebook_title": nb_title,
                }
                batch_ids.append(chunk_data["chunk_id"])
                batch_tags.append(
                    [t.strip() for t in metadata.get("tags", "").split(",")]
                )
                batch_metadatas.append(metadata)

            try:
                vector_db.batch_update_chunks_metadata(
                    batch_ids, batch_tags, batch_metadatas
                )
                successful_metadata_updates = len(batch_ids)
                for i, chunk_data in enumerate(chunks_metadata_only):
                    log.info(
                        f"笔记《{note.title}》的块 "
                        f"【{chunk_data['base_metadata']['chunk_index']}/{len(chunk_dicts)}】"
                        f"（仅元数据更新）成功，元数据：{batch_metadatas[i]}"
                    )
            except Exception as e:
                log.error(
                    f"笔记《{note.title}》批量元数据更新 {len(batch_ids)} 个块失败: {e}",
                    exc_info=True,
                )

        def _upsert_with_embedding(chunk_data, embedding, extra_metadata=None):
            """将块写入 ChromaDB，返回 True 表示成功。"""
            chunk_id = chunk_data["chunk_id"]
            chunk_content = chunk_data["content"]
            base_metadata = chunk_data["base_metadata"]

            notebook_dicts = get_notebook_ids_for_note(note.id)
            notebook_dict = notebook_dicts[-1]
            try:
                nb_id, nb_title = next(iter(notebook_dict.items()))
            except StopIteration:
                nb_id, nb_title = "", ""
            metadata = {
                **base_metadata,
                **(extra_metadata or {}),
                "source_note_id": note.id,
                "source_notebook_id": nb_id,
                "source_notebook_title": nb_title,
            }
            vector_db.upsert_chunk(
                chunk_id=chunk_id,
                text=chunk_content,
                embedding=embedding,
                tags=[tag.strip() for tag in metadata.get("tags", "").split(",")],
                metadata=metadata,
            )
            return True, metadata

        # 4b. 处理需要全量更新的块（重新生成嵌入并入库）
        successful_upserts = 0
        failed_chunks = []  # P0-1: 追踪失败块 [{chunk_data, reason, embedding, enhanced_metadata}]
        for chunk_data in chunks_to_upsert:
            chunk_content = chunk_data["content"]
            base_metadata = chunk_data["base_metadata"]
            metadata_chunk_idx_from_one = base_metadata["chunk_index"]

            # 生成嵌入：优先使用分块阶段预生成的嵌入
            embedding = chunk_data.get("embedding")
            if embedding is None:
                embedding = embedding_generator.get_merged_embedding(chunk_data)
            else:
                log.info(
                    f"笔记《{note.title}》块 {metadata_chunk_idx_from_one} "
                    f"复用分块阶段预生成嵌入（{len(embedding)}维）"
                )
            if not embedding:
                log.warning(
                    f"笔记《{note.title}》块 {metadata_chunk_idx_from_one}"
                    f"（长度：{len(chunk_content)}）嵌入生成失败，跳过此块。"
                )
                failed_chunks.append(
                    {"chunk_data": chunk_data, "reason": "embedding",
                     "embedding": None, "enhanced_metadata": {}}
                )
                continue

            enhanced_metadata = {}
            if len(chunk_content) > embedding_generator.chunk_size * 0.8:
                enhanced_metadata["potential_long_chunk"] = True

            try:
                _, metadata = _upsert_with_embedding(chunk_data, embedding, enhanced_metadata)
                successful_upserts += 1
                log.info(
                    f"笔记《{note.title}》的块 【{metadata_chunk_idx_from_one}/{len(chunk_dicts)}】"
                    f"（长度：{len(chunk_content)}）向量化入库更新成功，文本块元数据为：{metadata}"
                )
            except Exception as e:
                log.error(
                    f"笔记《{note.title}》存储块 {metadata_chunk_idx_from_one}"
                    f"（长度：{len(chunk_content)}）失败: {e}",
                    exc_info=True,
                )
                failed_chunks.append(
                    {"chunk_data": chunk_data, "reason": "upsert",
                     "embedding": embedding, "enhanced_metadata": enhanced_metadata}
                )

        # 5. 智能清理"孤儿块"
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

        # 6. P0-1: 失败块自动重试（最多2轮，间隔3s）
        if failed_chunks:
            log.warning(
                f"笔记《{note.title}》有 {len(failed_chunks)} 个块首次处理失败，开始重试..."
            )
            for retry_round in range(1, 3):
                if not failed_chunks:
                    break
                time.sleep(3)
                retry_success = 0
                still_failed = []
                for fail_record in failed_chunks:
                    chunk_data = fail_record["chunk_data"]
                    chunk_idx = chunk_data["base_metadata"]["chunk_index"]
                    # 仅嵌入生成失败的块需重新生成嵌入；upsert失败的复用原嵌入
                    if fail_record["reason"] == "upsert":
                        embedding = fail_record["embedding"]
                        enhanced_metadata = fail_record.get("enhanced_metadata", {})
                    else:
                        embedding = embedding_generator.get_merged_embedding(chunk_data)
                        enhanced_metadata = {}
                        if len(chunk_data["content"]) > embedding_generator.chunk_size * 0.8:
                            enhanced_metadata["potential_long_chunk"] = True
                    if not embedding:
                        still_failed.append(fail_record)
                        continue
                    try:
                        _upsert_with_embedding(chunk_data, embedding, enhanced_metadata)
                        successful_upserts += 1
                        retry_success += 1
                        log.info(
                            f"笔记《{note.title}》块 {chunk_idx} 第{retry_round}轮重试成功"
                        )
                    except Exception as e:
                        log.error(
                            f"笔记《{note.title}》块 {chunk_idx} 第{retry_round}轮重试仍失败: {e}"
                        )
                        still_failed.append(
                            {"chunk_data": chunk_data, "reason": "upsert",
                             "embedding": embedding, "enhanced_metadata": enhanced_metadata}
                        )
                log.info(
                    f"笔记《{note.title}》第{retry_round}轮重试: "
                    f"{retry_success} 成功, {len(still_failed)} 仍失败"
                )
                failed_chunks = still_failed

        # 7. 最终判断
        failed_after_retry = len(failed_chunks)
        total_processed = successful_upserts + successful_metadata_updates + skipped_chunks
        if total_processed == len(chunk_dicts):
            log.info(
                f"笔记《{note.title}》块级增量处理完成。成功更新 {successful_upserts} 个块，"
                f"元数据更新 {successful_metadata_updates} 个块，跳过 {skipped_chunks} 个块。"
            )
            return {
                "success": True,
                "enhance_missing": enhance_missing,
                "chunk_stats": {
                    "total_chunks": len(chunk_dicts),
                    "upserted": successful_upserts,
                    "metadata_updated": successful_metadata_updates,
                    "skipped": skipped_chunks,
                    "orphans_cleaned": len(orphan_chunk_ids),
                    "failed_after_retry": failed_after_retry,
                },
            }
        else:
            log.error(
                f"笔记《{note.title}》处理不完整。预期{len(chunk_dicts)}块，实际处理{total_processed}块"
                f"（{failed_after_retry}块重试后仍失败）。"
            )
            return {
                "success": False,
                "chunk_stats": {
                    "total_chunks": len(chunk_dicts),
                    "upserted": successful_upserts,
                    "metadata_updated": successful_metadata_updates,
                    "skipped": skipped_chunks,
                    "orphans_cleaned": len(orphan_chunk_ids),
                    "failed_after_retry": failed_after_retry,
                },
            }

    except Exception as e:
        log.error(f"块级增量处理笔记《{note.title}》失败: {e}", exc_info=True)
        return {"success": False, "chunk_stats": {}}


# %% [markdown]
# ### process_notes_incremental(notebook_title: str, config: Dict)
# %%
def process_notes_incremental(notebook_title: str, config: Dict, note_ids: List[str] = None):
    """增量处理笔记本笔记。

    Args:
        notebook_title: 笔记本名称（虚拟笔记集时作为标识名）
        config: 配置字典
        note_ids: 可选，直接指定笔记ID列表，作为虚拟笔记集处理。为None时按物理笔记本处理。
    """


# %% [markdown]
# #### 初始化向量库

    # %%
    # 根据 provider 选主模型：siliconflow_embedding_model 配了即用 SF 模型，否则 Ollama
    primary_model = config.get("siliconflow_embedding_model") or config["ollama_embedding_model"]

    # 初始化向量数据库（在整个处理过程中只初始化一次）
    if not hasattr(process_notes_incremental, "vector_db"):
        process_notes_incremental.vector_db = VectorDBManager(
            config["db_path"], primary_model, True
        )
        log.info(
            f"向量数据库初始化完成，集合: {process_notes_incremental.vector_db.collection_name}"
        )
    vector_db = process_notes_incremental.vector_db
    # 根据文本块内容更新其预估日期，所有文本块遍历一遍
    # vector_db.refresh_estimated_date()
    # 输出向量库信息
    log.info(f"向量库《{vector_db}》信息：{vector_db.get_collection_info()}")

    # 初始化嵌入生成器（在整个处理过程中只初始化一次）
    if not hasattr(process_notes_incremental, "embedding_gen"):
        process_notes_incremental.embedding_gen = EmbeddingGenerator(
            config,
            primary_model,
            chunk_size=config.get("chunk_size", 512),
            chunk_overlap=config.get("chunk_overlap", 50),
            # 【新增】传递自适应分块配置
            enable_adaptive_chunking=config.get("enable_adaptive_chunking", False),
        )
        log.info(f"嵌入生成器初始化完成")

    embedding_gen = process_notes_incremental.embedding_gen

    # 本地模型可用性检查（摘要/标签配了本地模型时，整个运行周期仅检查一次）
    if not hasattr(process_notes_incremental, "ollama_checked"):
        process_notes_incremental.ollama_checked = True
        ollama_chat_model = config.get("enhance_ollama_chat_model")
        if config.get("summary_model") == "ollama" or config.get("tags_model") == "ollama":
            if ollama is None:
                log.warning("ollama 包未安装，跳过本地模型可用性检查")
            else:
                import time
                ollama_host = config.get("ollama_host", "http://149.30.242.156:11434")
                ollama_client = ollama.Client(host=ollama_host)
                for attempt in range(1, 4):
                    try:
                        result = ollama_client.list()
                        models_list = result.models if hasattr(result, 'models') else result.get("models", [])
                        installed = [m.model if hasattr(m, 'model') else m["name"] for m in models_list]
                        if ollama_chat_model in installed:
                            log.info(f"Ollama 标签/摘要模型 {ollama_chat_model} 可用")
                        else:
                            log.warning(
                                f"Ollama 模型 {ollama_chat_model} 未安装，标签/摘要将不可用。"
                                f"安装: ollama pull {ollama_chat_model}"
                            )
                        break
                    except Exception as e:
                        if attempt < 3:
                            log.warning(f"Ollama 连接失败 (第{attempt}次): {e}，2秒后重试...")
                            time.sleep(2)
                        else:
                            log.warning(f"Ollama 连接失败 (3次重试均失败): {e}，标签/摘要将不可用")

    # 获取强制更新配置（需在加载状态前读取，用于决定 center_api 不可达时的策略）
    force_update = config.get("force_update", False)

    # 加载处理状态（纯远程，无本地 fallback）
    state_client = config.get("state_client")
    if state_client:
        from func.datatools import normalize_collection_name
        model_name_str = normalize_collection_name(primary_model)
        try:
            process_state = state_client.batch_load(model_name_str)
        except CenterAPIUnreachableError:
            if force_update:
                log.warning(
                    "center_api 不可达，但因 --enable_force_update，以空状态继续运行。"
                    "本次将全量重处理所有笔记。"
                )
                process_state = {}
            else:
                raise
    else:
        log.error("state_client 未配置，无法加载处理状态，本次运行视为全新处理")
        process_state = {}
    # 重置增强调用统计
    from aimod.note_enhancer import reset_call_stats
    reset_call_stats()

    # 解析笔记本级增强策略覆盖（enhance_override JSON 键）
    effective_config = _resolve_enhance_config(config, notebook_title)
    if effective_config.get("summary_model") != config.get("summary_model") or \
       effective_config.get("tags_model") != config.get("tags_model"):
        log.info(
            f"笔记本【{notebook_title}】增强策略覆盖："
            f"摘要={effective_config.get('summary_model')}，标签={effective_config.get('tags_model')}"
        )

    # 获取笔记列表：物理笔记本或虚拟笔记集
    if note_ids is not None:
        notes_all = [getnote(nid) for nid in note_ids if getnote(nid)]
    else:
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
    total_metadata_updated_for_notebook = 0
    total_skipped_for_notebook = 0
    total_orphans_cleaned_for_notebook = 0
    total_failed_after_retry = 0
    updated_count = 0
    skipped_note_count = 0
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
            # 4. 当前增强配置标识 (模型变更时自动重处理)
            current_enhance_config = (
                f"summary={effective_config.get('summary_model', '')}"
                f"|tags={effective_config.get('tags_model', '')}"
            )
            # === 修改结束 ===
            # 获取上一次处理的状态（兼容旧格式）
            last_state = process_state.get(note_id, {})
            last_update_time = last_state.get("update_time")
            last_content_hash = last_state.get("content_hash")
            last_meta_hash = last_state.get("meta_hash")
            last_enhance_missing = last_state.get("enhance_missing", True)
            last_enhance_config = last_state.get("enhance_config", "")

            # 如果状态文件是旧的（没有meta_hash字段），则视为需要更新元数据
            needs_meta_update = ("meta_hash" not in last_state)

            # 增强未完成时也需要强制重试
            enhance_enabled = effective_config.get("summary_model") != "none" or effective_config.get("tags_model") != "none"
            needs_enhance_retry = (
                enhance_enabled and last_enhance_missing
            ) or (
                enhance_enabled and last_enhance_config != current_enhance_config
            )

            # 判断是否需要处理：强制更新 或 元数据缺失 或 增强缺失/模型变更 或 内容变更
            if force_update or needs_meta_update or needs_enhance_retry or not (
                last_update_time == current_update_time
                and last_content_hash == current_content_hash
                and last_meta_hash == current_meta_hash
            ):
                log.info(
                    f"笔记《{note.title}》需要更新。"
                    f"原因: force={force_update}, meta_missing={needs_meta_update}, "
                    f"enhance_missing={last_enhance_missing}, "
                    f"enhance_config_change={last_enhance_config != current_enhance_config}, "
                    f"时间变化={last_update_time != current_update_time}, "
                    f"内容变化={last_content_hash != current_content_hash}, "
                    f"元数据变化={last_meta_hash != current_meta_hash}"
                )
                # 计算快速路径标记
                _content_unchanged = (
                    not force_update
                    and last_content_hash == current_content_hash
                    and last_content_hash  # 有历史记录才能确认未变
                )
                _needs_re_enhance = (
                    enhance_enabled and (
                        last_enhance_missing
                        or last_enhance_config != current_enhance_config
                    )
                )
                # 提交任务到线程池/进程池
                future = executor.submit(
                    process_note_chunks,
                    note,
                    vector_db,
                    embedding_gen,
                    effective_config,
                    _content_unchanged,
                    _needs_re_enhance,
                )
                future_to_note[future] = (
                    note_id,
                    note.title,
                    current_update_time,
                    current_content_hash,
                    current_meta_hash,
                    current_enhance_config,
                )
                log.info(
                    f"开始处理笔记本【{notebook_title}】下的第（{i}/{len(notes)}）条笔记: 《{note.title}》…………"
                )
                new_time_notes.append(note.title)

        # 收集处理结果
        total_notes = len(future_to_note)
        completed_count = 0
        for future in as_completed(future_to_note):
            completed_count += 1
            note_id, note_title, update_time, content_hash, meta_hash, enhance_config = future_to_note[future]
            log.info(
                f"[进度: {completed_count}/{total_notes} ({completed_count * 100 // total_notes}%)] "
                f"《{note_title}》完成"
            )
            try:
                result_dict = future.result() # 现在接收的是字典
                success = result_dict.get("success", False)
                note_chunk_stats = result_dict.get("chunk_stats", {})

                # 累加块统计
                total_chunks_for_notebook += note_chunk_stats.get("total_chunks", 0)
                total_upserted_for_notebook += note_chunk_stats.get("upserted", 0)
                total_metadata_updated_for_notebook += note_chunk_stats.get("metadata_updated", 0)
                total_skipped_for_notebook += note_chunk_stats.get("skipped", 0)
                total_orphans_cleaned_for_notebook += note_chunk_stats.get("orphans_cleaned", 0)
                total_failed_after_retry += note_chunk_stats.get("failed_after_retry", 0)

                if success:
                    # 更新处理状态
                    process_state[note_id] = {
                        "note_title": note_title,
                        "update_time": float(update_time),
                        "content_hash": content_hash,
                        "meta_hash": meta_hash,
                        "processed_time": datetime.now().timestamp(),
                        "enhance_missing": result_dict.get("enhance_missing", False),
                        "enhance_config": enhance_config,
                    }
                    # 仅当有实际块变更（新增/更新/元数据更新/清理孤儿）时才计入 updated_count
                    note_upserted = note_chunk_stats.get("upserted", 0)
                    note_metadata = note_chunk_stats.get("metadata_updated", 0)
                    note_orphans = note_chunk_stats.get("orphans_cleaned", 0)
                    if note_upserted > 0 or note_metadata > 0 or note_orphans > 0:
                        updated_count += 1
                    else:
                        skipped_note_count += 1
                else:
                    failed_notes.append(note_title)
                    log.error(f"向量化处理笔记 《{note_title}》 时可能异常")
            except Exception as e:
                log.error(f"并发处理笔记 《{note_title}》 异常: {e}")
                failed_notes.append(note_title)
    # 保存状态（纯远程，无本地 fallback）
    state_client = config.get("state_client")
    if state_client:
        from func.datatools import normalize_collection_name
        model_name_str = normalize_collection_name(primary_model)
        state_client.batch_save(model_name_str, process_state)
    else:
        log.error("state_client 未配置，处理状态未能持久化")
    log.info(
        f"增量处理笔记本【{notebook_title}】中的笔记完成情况小结："
        f"新日期需要更新 {len(new_time_notes)} 条，"
        f"实际更新 {updated_count} 条，"
        f"无变化跳过 {skipped_note_count} 条，"
        f"失败 {len(failed_notes)} 条"
        f"（总计 {len(notes)} 条）"
    )
    if failed_notes:
        log.warning(
            f"笔记本【{notebook_title}】中增量处理（向量化）失败的笔记: {set(failed_notes)}"
        )

# %% [markdown]
# #### 清理已移除当前笔记集的向量数据

    # %%
    # ========== 清理已移出当前笔记集的笔记的向量数据 ==========
    # 初始化安全默认值
    notes_removed_titles = []
    notes_added_titles = []
    orphan_note_ids = []

    is_virtual = note_ids is not None
    log.info(f"开始执行笔记集【{notebook_title}】的'孤儿笔记'向量数据清理...")

    try:
        if not is_virtual:
            # --- 物理笔记本：按 source_notebook_id 从向量库反查 ---
            notebook_id = searchnotebook(notebook_title)
            current_notebook_id = notebook_id

            if not current_notebook_id:
                log.warning(f"无法获取笔记本【{notebook_title}】的ID，跳过清理步骤。")
                notes_in_vector_db = {}
            else:
                current_note_ids_in_notebook = list(set(note.id for note in notes))

                all_results = vector_db.collection.get(
                    where={"source_notebook_id": current_notebook_id},
                    include=["metadatas"]
                )

                notes_in_vector_db = {}
                for metadata in all_results.get("metadatas", []):
                    notes_in_vector_db[metadata.get("source_note_id")] = metadata.get('source_note_title')
                log.info(
                    f"从向量数据库既有数据查找到的笔记本【{notebook_title}】"
                    f"中的笔记有这些：{list(notes_in_vector_db.values())[-5:]}……"
                )

                orphan_note_ids = [nid for nid in notes_in_vector_db if nid not in current_note_ids_in_notebook]
        else:
            # --- 虚拟笔记集：对比上次运行的笔记ID列表 ---
            vcollections = process_state.get("_virtual_collections", {})
            last_vc = vcollections.get(notebook_title, {})
            last_note_ids = last_vc.get("note_ids", [])
            last_note_id_set = set(last_note_ids)
            current_note_id_set = set(note_ids)

            orphan_note_ids = list(last_note_id_set - current_note_id_set)
            notes_in_vector_db = {}
            for nid in last_note_ids:
                note_obj = getnote(nid)
                notes_in_vector_db[nid] = note_obj.title if note_obj else ""

            log.info(
                f"虚拟笔记集【{notebook_title}】：上次 {len(last_note_ids)} 条，"
                f"本次 {len(note_ids)} 条，移除 {len(orphan_note_ids)} 条"
            )

            # 保存本次快照到状态文件
            process_state.setdefault("_virtual_collections", {})[notebook_title] = {
                "note_ids": note_ids,
                "last_run": datetime.now().isoformat(),
            }

        # --- 以下清理和统计逻辑两种模式共用 ---
        cleaned_total_chunks = 0
        for note_id in orphan_note_ids:
            deleted_chunks_count = vector_db.delete_chunks_by_note_id(note_id)
            cleaned_total_chunks += deleted_chunks_count
            if deleted_chunks_count > 0:
                note_title = notes_in_vector_db.get(note_id, "")
                log.info(f"清理已移除笔记《{note_title}》（note_id={note_id}）的孤儿文本块向量数据, 删除 {deleted_chunks_count} 个块。")
                process_state.pop(note_id, None)

        if cleaned_total_chunks > 0:
            log.info(f"笔记集【{notebook_title}】清理完成。共移除 {len(orphan_note_ids)} 条笔记的向量数据，涉及 {cleaned_total_chunks} 个块。")
        else:
            log.info(f"笔记集【{notebook_title}】未发现需要清理的孤儿笔记向量数据。")

        # 计算移除的笔记标题
        notes_removed_titles = [
            notes_in_vector_db.get(nid, f"[已删除的笔记: {nid}]") for nid in orphan_note_ids
        ]

        # 计算新增的笔记
        if not is_virtual:
            current_note_ids = set(note.id for note in notes)
            notes_added_ids = current_note_ids - set(notes_in_vector_db)
        else:
            last_note_id_set_v = set(last_note_ids) if last_note_ids else set()
            notes_added_ids = current_note_id_set - last_note_id_set_v
        notes_added_titles = [getnote(nid).title for nid in notes_added_ids if getnote(nid)]

    except Exception as e:
        log.error(f"执行笔记集【{notebook_title}】的清理步骤时出错: {e}", exc_info=True)
    # ========== 清理步骤结束 ==========

# %% [markdown]
# #### 统计汇总

    # %%
    stats = {
        "notebook_title": notebook_title,
        "total_notes": len(notes),
        "updated_count": updated_count,
        "skipped_note_count": skipped_note_count,
        "failed_notes": list(set(failed_notes)),
        "new_time_notes": new_time_notes,
        "notes_added": notes_added_titles,
        "notes_removed": notes_removed_titles,
        "chunk_stats": {
            "total_chunks": total_chunks_for_notebook,
            "upserted": total_upserted_for_notebook,
            "skipped": total_skipped_for_notebook,
            "orphans_cleaned": total_orphans_cleaned_for_notebook,
            "failed_after_retry": total_failed_after_retry,
        },
        "process_time": datetime.now().isoformat(),
    }

    from aimod.note_enhancer import get_call_stats
    stats["enhance_stats"] = get_call_stats()

    log.info(f"笔记集【{notebook_title}】处理完成。")
    # 返回这个完整的 stats 字典
    return stats


# %% [markdown]
# ## 主流程入口

# %% [markdown]
# ### add_file_lock(model_name: str, lock_name: str = "joplinai.lock", timeout: int = 3600)

# %%
def add_file_lock(
    model_name: str, lock_name: str = "joplinai.lock", timeout: int = 3600
) -> Tuple[Optional[str], bool]:
    """在脚本入口创建文件锁，防止多实例并发运行。

    Args:
        lock_name: 锁文件名，建议与模型或配置关联以避免不同配置间的冲突。
        model_name: 嵌入模型名称
        timeout: 锁超时时间（秒），用于处理进程崩溃后锁未释放的情况。

    Returns:
        lock_file_path (Path): 锁文件路径，用于后续清理。
        acquired (bool): 是否成功获取锁。
    """
    # 确定锁文件存放目录，优先 /tmp，不可写时（如 Android Termux）回落项目 data/
    temp_dir = Path(os.getenv("TEMP", "/tmp"))
    lock_dir = temp_dir if (temp_dir.exists() and os.access(temp_dir, os.W_OK)) else getdirmain() / "data"
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

if __name__ == "__main__":
    from src.cli import main
    main()
