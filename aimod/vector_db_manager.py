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
# # 向量数据库管理器
# vector_db_manager.py

# %% [markdown]
# ## 引入重要库

# %%
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
import ollama

# %%
import pathmagic

with pathmagic.context():
    try:
        from aimod.embedding_generator import EmbeddingGenerator  # 用于调用提取函数
        from func.datatools import compute_content_hash
        from func.first import getdirmain
        from func.jpfuncs import (
            get_notebook_ids_for_note,
            get_tag_titles,
            getnote,
        )
        from func.logme import log
    except ImportError as e:
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger(__name__)
        log.error(f"导入项目模块失败: {e}")


# %% [markdown]
# ## VectorDBManager类

# %%
class VectorDBManager:
    """统一的ChromaDB向量数据库管理器"""


# %% [markdown]
# ### __init__(self, db_path: Path, embedding_model: str, for_creation: bool = False)

    # %%
    def __init__(self, db_path: Path, embedding_model: str, for_creation: bool = False):
        """初始化向量数据库管理器"""
        try:
            import pathmagic
            with pathmagic.context():
                from joplinai import CONFIG as CONFIG_JA
            config = CONFIG_JA
            # print(config)
            self.client = chromadb.HttpClient(
                host=config.get("chroma_server_host", "10.9.0.1"),
                port=config.get("chroma_server_port", 8000)
            )
            # self.client = chromadb.PersistentClient(
            #     path=str(db_path),
            # )
            self._model_dimension_cache = {}  # 添加这行：初始化缓存字典
            self.embedding_model = embedding_model
            self.collection_name = f"joplin_{embedding_model.replace(':', '_').replace('/', '_').replace('-', '_')}"

            # 先尝试获取集合，如果不存在则创建
            try:
                self.collection = self.client.get_collection(self.collection_name)
                log.info(f"成功加载现有集合: {self.collection_name}")

                # === 【关键修复】尝试验证集合可访问性，捕获索引损坏 ===
                try:
                    # 执行一个轻量级操作来验证集合是否健康
                    count = self.collection.count()
                    log.debug(f"集合《{self.collection_name}》状态健康，数量为：{count}。")

                    # 如果健康，继续原有的维度验证逻辑
                    if count > 0:
                        sample = self.collection.get(include=['embeddings', 'metadatas'], limit=1)
                        # log.debug(sample)
                        if sample and "embeddings" in sample and sample["embeddings"].any():
                            existing_dim = len(sample["embeddings"][0])
                            current_dim = self._get_model_dimension(self.embedding_model)
                            log.info(
                                f"测试向量数据集合的维度为：{existing_dim}，"
                                f"测试模型获取的向量维度为：{current_dim}。"
                            )
                            if existing_dim != current_dim:
                                log.warning(f"维度不匹配: 现有{existing_dim}D, 需要{current_dim}D")
                                if for_creation:
                                    log.info("创建模式下重建集合（因维度不匹配）")
                                    self.client.delete_collection(self.collection_name)
                                    self.collection = self.client.create_collection(
                                        name=self.collection_name,
                                        metadata={
                                            "hnsw:space": "cosine",
                                            "dimension": current_dim,
                                        },
                                    )
                except Exception as inner_e:
                    # 捕获 count() 或 get() 时的错误，这很可能意味着索引损坏
                    log.error(f"集合《{self.collection_name}》可能已损坏或无法访问: {inner_e}")
                    if for_creation:
                        log.warning(f"创建模式下，将尝试删除并重建损坏的集合《{self.collection_name}》。")
                        try:
                            self.client.delete_collection(self.collection_name)
                            log.info(f"已删除损坏的集合《{self.collection_name}》。")
                        except Exception as delete_e:
                            log.error(f"删除集合失败: {delete_e}")
                            raise RuntimeError(f"无法删除损坏的集合，请手动清理: {delete_e}")
                        # 重建集合
                        self.collection = self.client.create_collection(
                            name=self.collection_name,
                            metadata={
                                "hnsw:space": "cosine",
                                "dimension": self._get_model_dimension(self.embedding_model),
                            },
                        )
                        log.info(f"已重建集合《{self.collection_name}》。")
                    else:
                        # 非创建模式，无法自动修复，抛出更清晰的错误
                        raise RuntimeError(
                            f"集合《{self.collection_name}》已存在但可能已损坏，错误详情: {inner_e}。"
                            f"请尝试运行带有 `--force_update` 参数的 `joplinai.py` 来重建向量库。"
                        ) from inner_e

            except Exception as outer_e:
                # 此处的异常是 self.client.get_collection 失败，意味着集合真的不存在
                log.info(f"集合《{self.collection_name}》不存在，创建新集合。")
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={
                        "hnsw:space": "cosine",
                        "dimension": self._get_model_dimension(self.embedding_model),
                    },
                )
        except Exception as e:
            log.error(f"初始化向量数据库时出错：{e}")

# %% [markdown]
# ### _ensure_collection(self)

    # %%
    def _ensure_collection(self):
        """确保集合存在且维度匹配（用于joplinai.py）"""
        try:
            self.collection = self.client.get_collection(self.collection_name)
            # 检查维度是否匹配
            sample = self.collection.get(limit=1)
            if sample and "embeddings" in sample:
                existing_dim = len(sample["embeddings"][0])
                current_dim = self._get_model_dimension(self.embedding_model)
                if existing_dim != current_dim:
                    raise ValueError(
                        f"维度不匹配: 现有{existing_dim}D, 需要{current_dim}D"
                    )
        except Exception as e:
            log.warning(f"集合不存在或维度不匹配，将创建新集合: {e}")
            # 维度不匹配或集合不存在时重建
            if hasattr(self, "collection") and self.collection:
                self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "dimension": self._get_model_dimension(self.embedding_model),
                },
            )


# %% [markdown]
# ### _load_collection(self)

    # %%
    def _load_collection(self):
        """加载集合（用于joplin_qa.py）"""
        try:
            self.collection = self.client.get_collection(self.collection_name)
            log.info(f"成功加载集合: {self.collection_name}")
            
            # 获取集合统计信息
            try:
                count = self.collection.count()
                log.info(f"集合中包含 {count} 个文档")
                
                # 获取一些样本数据
                sample = self.collection.get(limit=min(3, count))
                if sample and sample["ids"]:
                    log.info(f"样本笔记ID: {sample['ids'][:3]}")
            except Exception as e:
                log.warning(f"无法获取集合统计信息: {e}")
                
        except Exception as e:
            log.error(f"加载集合失败: {e}")
            self.collection = None
            
            # 尝试创建集合（如果不存在）
            try:
                log.info(f"尝试创建集合: {self.collection_name}")
                self.collection = self.client.create_collection(self.collection_name)
                log.info(f"成功创建新集合: {self.collection_name}")
            except Exception as create_e:
                log.error(f"创建集合也失败: {create_e}")


# %% [markdown]
# ### _get_model_dimension(self, model_name: str) -> int

    # %%
    def _get_model_dimension(self, model_name: str) -> int:
        """获取模型维度"""
        # 已知模型维度映射
        known_dimensions = {
            "dengcao/bge-large-zh-v1.5": 1024,  # 根据日志显示实际是1024维
            "nomic-embed-text": 768,
            "qwen:1.8b": 2048,
            # 可以添加更多已知模型
        }

        if model_name in known_dimensions:
            dim = known_dimensions[model_name]
            self._model_dimension_cache[model_name] = dim
            # log.info(f"使用已知模型维度: {model_name} -> {dim}D")
            return dim

        if model_name in self._model_dimension_cache:
            return self._model_dimension_cache[model_name]

        # 尝试从Ollama获取模型信息
        try:
            # 通过生成一个简单嵌入来获取维度
            test_response = ollama.embeddings(model=model_name, prompt="test")
            dim = len(test_response["embedding"])
            self._model_dimension_cache[model_name] = dim
            log.info(f"通过测试嵌入获取模型维度: {model_name} -> {dim}D")
            return dim
        except Exception as e:
            log.error(f"获取模型维度失败: {e}")
            # 默认返回常见维度
            return 1024

# %% [markdown]
# ### search_similar_chunks(self, query_embedding: list, top_k: int = 10)

    # %%
    def search_similar_chunks(self, query_embedding: List[float], limit: int = 10, user_identity: Optional[Dict] = None):
        """查询相似块，支持基于用户身份的权限过滤"""
        if not self.collection:
            log.error("集合未加载")
            return []
    
        # === 【新增】调试日志：记录传入的身份信息 ===
        log.debug(f"[权限过滤] 查询请求 received. user_identity: {user_identity}")
        
        # 构建基础查询
        n_results = min(limit * 2, 50)

        # === 权限过滤逻辑 ===
        where_filter = None
        
        if user_identity:
            user_role = user_identity.get('role')
            user_display_name = user_identity.get('display_name')
            
            # 获取笔记本白名单（JSON数组格式）
            allowed_notebooks = user_identity.get('allowed_notebooks', [])
            
            log.debug(
                f"[权限过滤] 用户: {user_display_name}, 角色: {user_role}, "
                f"授权笔记本数: {len(allowed_notebooks)}"
            )
            
            if user_role == 'admin':
                # 管理员：无过滤，访问全部
                where_filter = None
                log.debug("[权限过滤] 管理员角色，无过滤。")
                
            elif user_role in ['team_leader', 'team_member']:
                # 团队领导 & 团队成员：统一使用笔记本白名单机制
                # 个人作者标识（根据您的命名规范）
                personal_author = f"{user_display_name}"
                
                if not allowed_notebooks:
                    # 如果没有授权任何笔记本，只能访问个人笔记
                    where_filter = {
                        "note_author": {"$eq": personal_author}
                    }
                    log.debug(f"[权限过滤] 无授权笔记本，仅限个人笔记。")
                    
                else:
                    # 构建复合过滤器：个人笔记 OR (团队笔记 AND 在授权笔记本中)
                    where_filter = {
                        "$or": [
                            # 条件1：个人创建的笔记
                            {"note_author": {"$eq": personal_author}},
                            
                            # 条件2：团队笔记且在授权笔记本范围内
                            {
                                "$and": [
                                    {"note_author": {"$eq": "团队_共同维护"}},
                                    {"source_notebook_title": {"$in": allowed_notebooks}}
                                ]
                            }
                        ]
                    }
                    log.debug(
                        f"[权限过滤] 应用笔记本白名单过滤，"
                        f"授权笔记本: {allowed_notebooks[:3]}{'...' if len(allowed_notebooks) > 3 else ''}"
                    )
            else:
                # 未知角色：严格限制
                where_filter = {"note_author": {"$eq": "__NO_ACCESS__"}}
                log.warning(f"[权限过滤] 未知角色 '{user_role}'，禁止访问。")
    
        # === 【新增】调试日志：显示最终发送给ChromaDB的过滤器 ===
        log.debug(f"[权限过滤] 即将发送给 ChromaDB 的 where 参数: {where_filter}")

        try:
           # ChromaDB 的正确查询方法
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_filter,  # ChromaDB 将应用此过滤器
                include=["documents", "metadatas", "distances"] # 确保包含这些字段
            )

            # 格式化返回结果
            similar_chunks = []

            # 检查结果结构
            if results and "ids" in results and results["ids"]:
                num_returned = len(results["ids"][0]) if results["ids"] and results["ids"][0] else 0
                log.debug(f"[权限过滤] ChromaDB 返回了 {num_returned} 个结果。")
                # ChromaDB返回的ids、metadatas、documents、distances都是列表的列表
                # 因为query_embeddings是单元素列表，所以取第一个元素
                ids_list = results["ids"][0] if results["ids"] else []
                metadatas_list = results["metadatas"][0] if results.get("metadatas") else []
                documents_list = results["documents"][0] if results.get("documents") else []
                distances_list = results["distances"][0] if results.get("distances") else []

                for i in range(len(ids_list)):
                    chunk_id = ids_list[i]
                    metadata = metadatas_list[i] if i < len(metadatas_list) else {}
                    document = documents_list[i] if i < len(documents_list) else ""
                    # **关键修正**：distances_list[i] 是单个距离值
                    distance = distances_list[i] if i < len(distances_list) else 0.0

                    # 将距离转换为相似度（余弦距离越小，相似度越高）
                    # 注意：余弦距离范围是[0, 2]，但通常归一化到[0, 1]
                    similarity = 1.0 - distance if distance <= 1.0 else 0.0

                    similar_chunks.append({
                        "chunk_id": chunk_id,
                        "source_note_id": metadata.get("source_note_id", ""),
                        "content": document,
                        "similarity": similarity,
                        "metadata": metadata
                    })

                log.info(f"成功检索到 {len(similar_chunks)} 个相关块")
                return similar_chunks
            else:
                log.warning("未检索到相关块")
                return []

        except Exception as e:
            log.error(f"向量搜索失败: {e}")
            import traceback
            log.error(f"详细错误堆栈:\n{traceback.format_exc()}")
            return []

# %% [markdown]
# ### upsert_chunk(self, note_id: str, text: str, embedding: List[float]

    # %%
    def upsert_chunk(self, chunk_id: str, text: str, embedding: List[float],
                     tags: List[str], metadata: Dict):
        """插入/更新笔记向量数据"""
        if not self.collection:
            log.error("集合未加载")
            return

        # 构建要存储的元数据，合并所有必要信息
        db_metadata = {
            "chunk_id": chunk_id, # 当前块的ID
            "tags": ",".join(tags),
            "summary": metadata.get("chunk_summary", ""),
            "source_note_title": metadata.get("source_note_title", ""),
            "source_note_id": metadata.get("source_note_id", ""),
            "chunk_index": metadata.get("chunk_index", 1),
            "content_hash": metadata.get("content_hash", ""),
            # === 新增字段 ===
            "meta_hash": metadata.get("meta_hash", ""),
            "source_notebook_title": metadata.get("source_notebook_title", ""),
            "source_notebook_id": metadata.get("source_notebook_id", ""),
            "note_author": metadata.get("note_author", "白晔峰"),
            "note_type": metadata.get("note_type", "个人笔记"),
        }

        # 确保使用 upsert 方法
        self.collection.upsert(
            ids=[chunk_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[db_metadata],
        )
        log.info(
            f"成功存储笔记块: {chunk_id}, 作者：{db_metadata.get('note_author')}，"
            f"来源位于【{db_metadata.get('source_notebook_title')}】笔记本中的笔记:"
            f"《{db_metadata.get('source_note_title')}》")

# %% [markdown]
# ### delete_note(self, note_id: str)

    # %%
    def delete_note(self, note_id: str):
        """删除笔记向量数据"""
        # ========== joplinai.py 的功能 ==========
        if not self.collection:
            log.error(f"集合《{self.collection_name}》未加载")
            return

        self.collection.delete(ids=[note_id])


# %% [markdown]
# ### delete_chunks_by_note_id(self, note_id: str) -> int

    # %%
    def delete_chunks_by_note_id(self, note_id: str) -> int:
        """删除属于某个笔记的所有块"""
        if not self.collection:
            return 0
        try:
            # 查询所有包含此 note_id 的块
            results = self.collection.get(where={"source_note_id": note_id})
            if results and results['ids']:
                self.collection.delete(ids=results['ids'])
                return len(results['ids'])
        except Exception as e:
            log.error(f"从集合《{self.collection_name}》按笔记ID删除块失败: {e}")
        return 0

# %% [markdown]
# ### migrate_content_hash(self,)

    # %%
    def migrate_content_hash(
        self,
    ):
        """为所有文档添加content_hash字段"""
        # 获取所有文档
        results = self.collection.get(include=["metadatas", "documents"])

        for i, (chunk_id, metadata, document) in enumerate(
            zip(results["ids"], results["metadatas"], results["documents"])
        ):
            if "content_hash" not in metadata or not metadata["content_hash"]:
                # 计算content_hash
                new_hash = hashlib.md5(document.encode("utf-8")).hexdigest()

                # 更新元数据
                updated_metadata = {**metadata, "content_hash": new_hash}
                self.collection.update(ids=[chunk_id], metadatas=[updated_metadata])

                if i % 100 == 0:
                    log.info(f"已为 {i} 个文档重新生成 content_hash 字段")

# %% [markdown]
# ### refresh_estimated_date(self,)

    # %%
    def refresh_estimated_date(
        self,
    ):
        """为所有文档添加content_hash字段"""
        # 获取所有文档
        results = self.collection.get(include=["metadatas", "documents"])

        for i, (chunk_id, metadata, document) in enumerate(
            zip(results["ids"], results["metadatas"], results["documents"]),
            1
        ):
            if "estimated_date" not in metadata or not metadata["estimated_date"]:
                unified_date_pattern_for_chunk = re.compile(
                    r"(\d{4}[-年/]\d{1,2}[-月/]\d{1,2}[日号])\s*", re.MULTILINE
                )
                date_in_chunk = unified_date_pattern_for_chunk.search(document.strip())
                if date_in_chunk:
                    estimated_date = date_in_chunk.group(1).replace("号", "日")
                    # 更新元数据
                    updated_metadata = {**metadata, "estimated_date": estimated_date}
                    self.collection.update(ids=[chunk_id], metadatas=[updated_metadata])

                    if i % 100 == 0:
                        log.info(f"已为 {i} 个文本块重新生成 estimated_date 字段或者更新其值")

# %% [markdown]
# ### get_existing_chunk_hashes_for_note(self, note_id: str) -> Dict[str, Dict[str, str]]

    # %%
    def get_existing_chunk_hashes_for_note(self, note_id: str) -> Dict[str, Dict[str, str]]:
        """
        提取指定笔记的块hash_map并返回
        查询并修复缺失的content_hash和meta_hash，惰性修复，仅针对content_hash和meta_hash字段
        """
        if not self.collection:
            return {}

        results = self.collection.get(where={"source_note_id": note_id})
        if not results or not results['ids']:
            return {}

        hash_map = {}
        # needs_update = []

        for chunk_id, metadata, document in zip(
            results['ids'], results['metadatas'], results.get('documents', [])
        ):
            content_hash = metadata.get("content_hash", "")
            meta_hash = metadata.get("meta_hash", "")

            # # 如果content_hash缺失或为空，计算并标记需要更新
            # if (not content_hash or not meta_hash) and document:
            #     current_content_hash = compute_content_hash(document)
            #     local_tags = metadata.get("tags", "")
            #     tags_str = ",".join(sorted(local_tags.split(','))) if local_tags else ""  # 排序保证一致性
            #     current_notebook_title = metadata.get("source_notebook_title", "")
            #     current_meta_hash = compute_content_hash(f"{tags_str}{current_notebook_title}")
            #     needs_update.append((chunk_id, metadata, current_content_hash, current_meta_hash))
            #     # content_hash = current_content_hash
            #     # meta_hash = current_meta_hash

            hash_map[chunk_id] = {"content_hash": content_hash, "meta_hash": meta_hash}

        # # 批量更新需要修复的文档
        # if needs_update:
        #     for chunk_id, metadata, content_hash, meta_hash in needs_update:
        #         updated_metadata = {**metadata, "content_hash": content_hash, "meta_hash": meta_hash}
        #         self.collection.update(
        #             ids=[chunk_id],
        #             metadatas=[updated_metadata]
        #         )
        #     log.info(f"修复了 {len(needs_update)} 个文档的content_hash或meta_hash字段")

        return hash_map

# %% [markdown]
# ### get_existing_chunk_hashes_for_note_other(self, note_id: str) -> Dict[str, Dict[str, str]]

    # %%
    def get_existing_chunk_hashes_for_note_other(self, note_id: str) -> Dict[str, Dict[str, str]]:
        """
        提取指定笔记的块hash_map并返回
        查询并修复缺失的content_hash和meta_hash，惰性修复，仅针对content_hash和meta_hash字段
        """
        if not self.collection:
            return {}

        results = self.collection.get(where={"source_note_id": note_id})
        if not results or not results['ids']:
            return {}

        hash_map = {}
        needs_update = []

        for chunk_id, metadata, document in zip(
            results['ids'], results['metadatas'], results.get('documents', [])
        ):
            content_hash = metadata.get("content_hash", "")
            meta_hash = metadata.get("meta_hash", "")

            # 如果content_hash缺失或为空，计算并标记需要更新
            if (not content_hash or not meta_hash) and document:
                current_content_hash = compute_content_hash(document)
                local_tags = metadata.get("tags", "")
                tags_str = ",".join(sorted(local_tags.split(','))) if local_tags else ""  # 排序保证一致性
                current_notebook_title = metadata.get("source_notebook_title", "")
                current_meta_hash = compute_content_hash(f"{tags_str}{current_notebook_title}")
                needs_update.append((chunk_id, metadata, current_content_hash, current_meta_hash))
                # content_hash = current_content_hash
                # meta_hash = current_meta_hash

            hash_map[chunk_id] = {"content_hash": content_hash, "meta_hash": meta_hash}

        # 批量更新需要修复的文档
        if needs_update:
            for chunk_id, metadata, content_hash, meta_hash in needs_update:
                updated_metadata = {**metadata, "content_hash": content_hash, "meta_hash": meta_hash}
                self.collection.update(
                    ids=[chunk_id],
                    metadatas=[updated_metadata]
                )
            log.info(f"修复了 {len(needs_update)} 个文档的content_hash或meta_hash字段")

        return hash_map

# %% [markdown]
# ### delete_chunks_by_id_list(self, chunk_id_list: List[str]) -> int

    # %%
    def delete_chunks_by_id_list(self, chunk_id_list: List[str]) -> int:
        """根据块ID列表批量删除块"""
        if not self.collection or not chunk_id_list:
            return 0
        try:
            self.collection.delete(ids=chunk_id_list)
            return len(chunk_id_list)
        except Exception as e:
            log.error(f"从集合《{self.collection_name}》中批量删除块失败: {e}")
            return 0

# %% [markdown]
# ### search_similar_notes(self, query: str, n_results: int = 5) -> List[Dict]

    # %%
    def search_similar_notes(self, query: str, n_results: int = 5) -> List[Dict]:
        """搜索与查询相似的笔记"""
        # ========== joplin_qa.py 的功能 ==========
        if not self.collection:
            log.error(f"集合《{self.collection_name}》中未加载")
            return []

        try:
            # 生成查询的嵌入向量
            query_embedding = self._generate_query_embedding(query)
            if not query_embedding:
                log.error(f"从集合《{self.collection_name}》中查询嵌入生成失败")
                return []

            log.info(f"集合《{self.collection_name}》查询嵌入维度: {len(query_embedding)}")

            # 在向量数据库中搜索
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            # 格式化结果
            similar_notes = []
            
            if not results or "ids" not in results or not results["ids"]:
                log.warning("ChromaDB返回空结果")
                return similar_notes
            
            # 处理结果结构
            ids_list = results["ids"]
            if not ids_list or len(ids_list) == 0:
                log.warning("结果ID列表为空")
                return similar_notes
            
            ids = ids_list[0] if isinstance(ids_list[0], list) else ids_list
            
            if not ids or len(ids) == 0:
                log.warning("没有找到相关笔记")
                return similar_notes
            
            log.info(f"找到 {len(ids)} 个潜在相关笔记")
            
            # 获取其他字段
            documents = results.get("documents", [])
            distances = results.get("distances", [])
            metadatas = results.get("metadatas", [])
            
            doc_list = documents[0] if documents and isinstance(documents[0], list) else []
            dist_list = distances[0] if distances and isinstance(distances[0], list) else []
            meta_list = metadatas[0] if metadatas and isinstance(metadatas[0], list) else []
            
            for i in range(min(len(ids), len(doc_list))):
                try:
                    note_id = ids[i]
                    content = doc_list[i] if i < len(doc_list) else ""
                    distance = dist_list[i] if i < len(dist_list) else 0
                    metadata = meta_list[i] if i < len(meta_list) else {}
                    
                    # 确保distance是数字而不是列表
                    if isinstance(distance, list):
                        distance = distance[0] if len(distance) > 0 else 0
                    
                    similarity = 1.0 / (1.0 + distance) if distance > 0 else 1.0
                    
                    similar_notes.append({
                        "note_id": note_id,
                        "content": content,
                        "similarity": similarity,
                        "metadata": metadata,
                    })
                    
                except Exception as e:
                    log.error(f"处理第{i + 1}个笔记时出错: {e}")
                    continue
            
            # 按相似度排序
            similar_notes.sort(key=lambda x: x["similarity"], reverse=True)
            log.info(f"成功检索到 {len(similar_notes)} 条相关笔记")
            
            return similar_notes
            
        except Exception as e:
            log.error(f"搜索失败: {e}")
            import traceback
            log.error(f"详细错误堆栈:\n{traceback.format_exc()}")
            return []

# %% [markdown]
# ### _generate_query_embedding(self, query: str) -> List[float]

    # %%
    def _generate_query_embedding(self, query: str) -> List[float]:
        """生成查询文本的嵌入向量"""
        # ========== joplin_qa.py 的功能 ==========
        try:
            log.info(f"尝试使用模型 {self.embedding_model} 生成查询嵌入")
            
            # 检查Ollama服务是否可用
            try:
                models = ollama.list()
                log.info(f"可用模型: {[m['model'] for m in models['models']]}")
            except Exception as e:
                log.error(f"无法获取Ollama模型列表: {e}")
                return []
            
            # 尝试生成嵌入，最多重试3次
            for attempt in range(3):
                try:
                    response = ollama.embeddings(model=self.embedding_model, prompt=query)
                    if "embedding" in response and response["embedding"]:
                        log.info(f"成功生成查询嵌入，维度: {len(response['embedding'])}")
                        return response["embedding"]
                    else:
                        log.warning(f"第{attempt + 1}次尝试：返回的嵌入为空")
                except Exception as e:
                    log.warning(f"第{attempt + 1}次尝试生成嵌入失败: {str(e)[:100]}")
                    time.sleep(1)

            log.error("生成查询嵌入最终失败")
            return []

        except Exception as e:
            log.error(f"生成查询嵌入失败: {e}")
            return []

# %% [markdown]
# ### get_note_by_id(self, note_id: str) -> Optional[Dict]

    # %%
    def get_note_by_id(self, note_id: str) -> Optional[Dict]:
        """根据ID获取笔记"""
        # ========== joplin_qa.py 的功能 ==========
        if not self.collection:
            return None

        try:
            results = self.collection.get(
                ids=[note_id], include=["documents", "metadatas"]
            )

            if results and results["ids"]:
                return {
                    "note_id": note_id,
                    "content": results["documents"][0] if results["documents"] else "",
                    "metadata": results["metadatas"][0] if results["metadatas"] else {},
                }
            return None
        except Exception as e:
            log.error(f"获取笔记失败: {e}")
            return None

# %% [markdown]
# ### get_collection_info(self) -> Dict

    # %%
    def get_collection_info(self) -> Dict:
        """获取集合信息"""
        # ========== 通用功能 ==========
        if not self.collection:
            return {"error": "集合未加载"}

        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "note_count": count,
                "embedding_model": self.embedding_model,
                "dimension": self._get_model_dimension(self.embedding_model),
            }
        except Exception as e:
            log.error(f"获取集合信息失败: {e}")
            return {"error": str(e)}

# %% [markdown]
# ### migrate_add_notebook_id(self, get_notebook_ids_for_note)

    # %%
    def migrate_add_notebook_id(self, get_notebook_ids_for_note):
        """
        安全迁移：为所有现有块添加 source_notebook_id 字段。
    
        参数:
            get_notebook_ids_for_note: 一个可调用函数，接收 note_id，返回其所属的笔记本列表（包含id和title）。
                                           例如: lambda notebooks_list: get_notebook_id(note_id)
        """
        if not self.collection:
            log.error("集合未加载，无法迁移。")
            return
    
        log.info("开始迁移：为所有向量数据块添加 source_notebook_id 字段...")
    
        # 获取集合中的所有数据
        try:
            results = self.collection.get(include=["metadatas", "documents"])
        except Exception as e:
            log.error(f"获取集合数据失败: {e}")
            return
    
        ids = results.get("ids", [])
        metadatas = results.get("metadatas", [])
    
        if not ids:
            log.info("集合为空，无需迁移。")
            return
    
        updated_count = 0
        error_count = 0
    
        for i, (chunk_id, metadata) in enumerate(zip(ids, metadatas)):
            try:
                # 检查是否已存在 source_notebook_id和source_notebook_title 字段，避免重复迁移
                if "source_notebook_id" in metadata and metadata["source_notebook_id"]:
                    continue
                if (
                    "source_notebook_title" in metadata
                    and metadata["source_notebook_title"]
                ):
                    continue
    
                note_id = metadata.get("source_note_id")
                note_title = metadata.get("source_note_title")
                if not note_id:
                    log.warning(
                        f"笔记 《{note_title}》 的块 {chunk_id} 缺少 source_note_id，跳过。"
                    )
                    continue
    
                # 调用外部函数获取笔记本ID
                notebook_dicts = get_notebook_ids_for_note(note_id)
                notebook_dict = notebook_dicts[-1]
                if not notebook_dict:
                    log.warning(
                        f"无法获取笔记 《{note_title}》 的笔记本ID，可能已被删除。跳过。"
                    )
                    # 可选：可以标记为待删除，或留空。这里选择留空，后续清理可能处理。
                try:
                    notebook_id, notebook_title = next(iter(notebook_dict.items()))
                except StopIteration:
                    notebook_id, notebook_title = "", ""
    
                # 更新元数据
                updated_metadata = {
                    **metadata,
                    "source_notebook_id": notebook_id,
                    "source_notebook_title": notebook_title,
                }
                self.collection.update(ids=[chunk_id], metadatas=[updated_metadata])
    
                updated_count += 1
                if updated_count % 100 == 0:
                    log.info(f"迁移进度：已更新 {updated_count} 个块。")
    
            except Exception as e:
                error_count += 1
                log.error(
                    f"迁移笔记 《{note_title}》 的块 {chunk_id} 时出错: {e}", exc_info=False
                )  # 避免堆栈过长
    
        log.info(f"迁移完成。成功更新 {updated_count} 个块，遇到 {error_count} 个错误。")

# %% [markdown]
# ### extract_unique_notebook_titles(self) -> List[str]

    # %%
    def extract_unique_notebook_titles(self) -> List[str]:
        """
        从向量库提取所有唯一的笔记本标题
        
        Returns:
            排序后的唯一笔记本标题列表
        """
        if not self.collection:
            log.error("集合未加载")
            return
    
        # 获取集合中的所有元数据
        try:
            results = self.collection.get(include=["metadatas"])
            metadatas = results.get("metadatas", [])
            
            # 提取并去重
            notebook_titles = set()
            for metadata in metadatas:
                title = metadata.get("source_notebook_title")
                if title and title.strip():
                    notebook_titles.add(title.strip())
            
            # 排序并返回
            sorted_titles = sorted(list(notebook_titles))
            log.info(f"提取到 {len(sorted_titles)} 个唯一笔记本标题")
            
            return sorted_titles
        except Exception as e:
            log.error(f"提取文本块所在笔记本（所有）标题失败: {e}")
            return []

# %% [markdown]
# ### get_notebook_statistics(self) -> Dict

    # %%
    def get_notebook_statistics(self) -> Dict:
        """
        获取笔记本统计信息
        
        Returns:
            包含统计信息的字典
        """
        titles = self.extract_unique_notebook_titles()
        
        # 按首字分组统计（便于前端展示）
        grouped = {}
        for title in titles:
            first_char = title[0] if title else "其他"
            if first_char not in grouped:
                grouped[first_char] = []
            grouped[first_char].append(title)
        
        return {
            "total_count": len(titles),
            "titles": titles,
            "grouped": grouped
        }


# %% [markdown]
# ## migrate_all_chunks_with_author(vector_db: VectorDBManager, embedding_gen: EmbeddingGenerator)

# %%
def migrate_all_chunks_with_author(
    vector_db: VectorDBManager, embedding_gen: EmbeddingGenerator
):
    """遍历向量库，为每个块重新计算并更新作者元数据。"""
    if not vector_db.collection:
        log.error("向量库未加载")
        return

    # 1. 获取所有块的数据
    results = vector_db.collection.get(include=["metadatas", "documents"])
    ids = results["ids"]
    metadatas = results["metadatas"]
    documents = results["documents"]

    log.info(f"开始处理 {len(ids)} 个文本块...")

    updated_count = 0
    for i, (chunk_id, old_meta) in enumerate(zip(ids, metadatas)):
        source_note_id = old_meta.get("source_note_id")
        if not source_note_id:
            log.warning(f"块 {chunk_id} 无 source_note_id，跳过")
            continue

        # 2. 从 Joplin 获取笔记最新信息
        try:
            note = getnote(source_note_id)
            if not note:
                log.warning(f"无法获取笔记 {source_note_id}，跳过其块 {chunk_id}")
                continue
            note_title = note.title
            local_tags = get_tag_titles(source_note_id)
            note_tags_str = ",".join(local_tags) if local_tags else ""
        except Exception as e:
            log.error(f"获取笔记 {source_note_id} 信息失败: {e}")
            continue

        # 3. 重新计算作者（使用 embedding_generator 中的方法）
        source_notebook_title = old_meta.get("source_notebook_title")
        note_author_type_meta = embedding_gen._extract_author_from_note(
            note_title, note_tags_str, source_notebook_title
        )

        # 4. 更新元数据
        new_metadata = {**old_meta, **note_author_type_meta}
        vector_db.collection.update(ids=[chunk_id], metadatas=[new_metadata])
        updated_count += 1

        if i % 100 == 0:
            log.info(
                f"已处理 {i} 个块，最近示例：{note_title} -> {note_author_type_meta}"
            )

    log.info(f"迁移完成！共更新了 {updated_count} 个文本块的作者信息。")


# %% [markdown]
# # 主函数main()

# %%
if __name__ == "__main__":
    # 初始化，配置需与主程序一致
    config_dynamic = {
        "embedding_model": "dengcao/bge-large-zh-v1.5",  # 根据实际情况修改
        "db_path": getdirmain() / "data" / "joplin_vector_db",  # ChromaDB存储路径
    }
    config = {**config_dynamic}
    print(config.get("db_path", ""), config.get("embedding_model", ""))
    vector_db = VectorDBManager(
        config.get("db_path", ""),
        config.get("embedding_model", ""),
    )
    embedding_gen = EmbeddingGenerator(config, config.get("embedding_model"))

    migrate_all_chunks_with_author(vector_db, embedding_gen)
