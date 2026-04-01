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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
import ollama

# %%
try:
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
            self.client = chromadb.PersistentClient(
                path=str(db_path),
            )
            self.embedding_model = embedding_model
            self._model_dimension_cache = {}  # 添加这行：初始化缓存字典
            self.collection_name = f"joplin_{embedding_model.replace(':', '_').replace('/', '_').replace('-', '_')}"

            # 先尝试获取集合，如果不存在则创建
            try:
                self.collection = self.client.get_collection(self.collection_name)
                log.info(f"成功加载现有集合: {self.collection_name}")

                # 验证维度匹配
                if self.collection.count() > 0:
                    sample = self.collection.get(limit=1)
                    if sample and "embeddings" in sample and sample["embeddings"]:
                        existing_dim = len(sample["embeddings"][0])
                        current_dim = self._get_model_dimension(self.embedding_model)
                        if existing_dim != current_dim:
                            log.warning(f"维度不匹配: 现有{existing_dim}D, 需要{current_dim}D")
                            if for_creation:
                                log.info("创建模式下重建集合")
                                self.client.delete_collection(self.collection_name)
                                self.collection = self.client.create_collection(
                                    name=self.collection_name,
                                    metadata={
                                        "hnsw:space": "cosine",
                                        "dimension": current_dim,
                                    },
                                )
            except Exception as e:
                log.info(f"集合不存在，创建新集合: {self.collection_name}")
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={
                        "hnsw:space": "cosine",
                        "dimension": self._get_model_dimension(self.embedding_model),
                    },
                )
                
        except Exception as e:
            log.error(f"初始化向量数据库失败: {e}")
            raise

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
            log.info(f"使用已知模型维度: {model_name} -> {dim}D")
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
    def search_similar_chunks(self, query_embedding: list, top_k: int = 10):
        """搜索最相似的文本块（使用 ChromaDB 正确 API）"""
        if not self.collection:
            log.error("集合未加载")
            return []
    
        try:
            # ChromaDB 的正确查询方法
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"] # 确保包含这些字段
            )
            
            # 格式化返回结果
            similar_chunks = []
            
            # 检查结果结构
            if results and "ids" in results and results["ids"]:
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
                        "note_id": metadata.get("note_id", ""),
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
            "chunk_id": chunk_id,           # 当前块的ID
            "tags": ",".join(tags),
            "summary": metadata.get("chunk_summary", ""),
            # 关键：保留从 joplinai.py 传入的完整元数据
            "source_note_title": metadata.get("source_note_title", ""),
            "source_note_id": metadata.get("source_note_id", ""),
            # "note_title": metadata.get("note_title", ""), # 也可能叫 note_title
            # 可以根据需要添加其他字段，如 chunk_index
            "chunk_index": metadata.get("chunk_index", 0),
        }

        # 确保使用 upsert 方法
        self.collection.upsert(
            ids=[chunk_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[db_metadata],
        )
        log.info(f"成功存储笔记块: {chunk_id}, 来源笔记: 《{db_metadata.get('source_note_title')}》")

# %% [markdown]
# ### delete_note(self, note_id: str)

    # %%
    def delete_note(self, note_id: str):
        """删除笔记向量数据"""
        # ========== joplinai.py 的功能 ==========
        if not self.collection:
            log.error("集合未加载")
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
            log.error(f"按笔记ID删除块失败: {e}")
        return 0

# %% [markdown]
# ### search_similar_notes(self, query: str, n_results: int = 5) -> List[Dict]

    # %%
    def search_similar_notes(self, query: str, n_results: int = 5) -> List[Dict]:
        """搜索与查询相似的笔记"""
        # ========== joplin_qa.py 的功能 ==========
        if not self.collection:
            log.error("向量数据库集合未加载")
            return []
        
        try:
            # 生成查询的嵌入向量
            query_embedding = self._generate_query_embedding(query)
            if not query_embedding:
                log.error("查询嵌入生成失败")
                return []
            
            log.info(f"查询嵌入维度: {len(query_embedding)}")
            
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
