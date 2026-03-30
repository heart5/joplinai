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
# # 嵌入生成器
# embedding_generator.py

# %% [markdown]
# ## 导入库

# %%
import hashlib
import logging
import time
from functools import lru_cache
from typing import List, Optional

import ollama

try:
    from func.logme import log
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    log.error(f"导入项目模块失败: {e}")


# %% [markdown]
# ## EmbeddingGenerator类

# %%
class EmbeddingGenerator:
    """嵌入生成器，支持长文本分块处理"""


# %% [markdown]
# ### __init__(self, model_name: str, chunk_size: int = 2048)

    # %%
    def __init__(self, model_name: str, chunk_size: int = 2048):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.embedding_cache = {}  # 简单缓存

# %% [markdown]
# ### get_model_max_context(self) -> int
    # %%
    def get_model_max_context(self) -> int:
        """精确获取模型上下文限制（token→字符转换）"""
        # 特殊处理已知模型
        if self.model_name == "nomic-embed-text":
            self.chunk_size = 1850  # 768 token × 3字符/token × 0.8余量 ≈ 1850
            return
        elif self.model_name == "dengcao/bge-large-zh-v1.5":
            self.chunk_size = 450 # 512 token，中文是一对一，应为是一对三或者四，实际中文为主
            return
            return
        elif self.model_name == "qwen:1.8b":
            self.chunk_size = 4000  # 2048 token × 3 × 0.8 = 4916，取4000
            return
    
        # 通用模型处理
        try:
            model_info = ollama.show(model=self.model_name)
            
            # 方法1：尝试从 parameters 字符串解析
            params_str = model_info.get("parameters", "")
            num_ctx = 2048  # 默认值
            
            if isinstance(params_str, str) and params_str:
                # 解析字符串，例如 "num_ctx 512"
                import re
                match = re.search(r'num_ctx\s+(\d+)', params_str)
                if match:
                    num_ctx = int(match.group(1))
                else:
                    # 方法2：尝试从 modelfile 解析
                    modelfile = model_info.get("modelfile", "")
                    match = re.search(r'PARAMETER num_ctx\s+(\d+)', modelfile)
                    if match:
                        num_ctx = int(match.group(1))
                    else:
                        # 方法3：尝试从 details.modelinfo 获取
                        try:
                            num_ctx = model_info.get("details", {}).get("modelinfo", {}).get("bert.context_length", 2048)
                        except:
                            pass
            else:
                # 如果 parameters 是字典（其他模型）
                num_ctx = model_info.get("parameters", {}).get("num_ctx", 1024)
                
            self.chunk_size = int(num_ctx * 3 * 0.8)
            log.info(f"模型 {self.model_name} 上下文长度: {num_ctx} tokens, 分块大小: {self.chunk_size} 字符")
            
        except Exception as e:
            log.warning(f"获取模型上下文失败({self.model_name})，使用默认值1024字符: {e}")
            self.chunk_size = 1024

# %% [markdown]
# ### get_ollama_embedding(self, text: str) -> List[float]
    # %%
    def get_ollama_embedding(self, text: str) -> List[float]:
        """调用 Ollama 生成文本嵌入。注意：传入的 text 应是已分块的适当长度文本。"""
        for attempt in range(3):
            try:
                response = ollama.embeddings(model=self.model_name, prompt=text)
                return response["embedding"]
            except Exception as e:
                log.warning(f"嵌入失败({attempt + 1}/3): {str(e)[:100]}")
                time.sleep(2**attempt)
    
        log.error(f"嵌入生成最终失败: {self.model_name}, 文本长度{len(text)}")
        return []

# %% [markdown]
# ### get_cached_embedding(self, text_hash: str) -> Optional[List[float]]

    # %%
    @lru_cache(maxsize=100)
    def get_cached_embedding(self, text_hash: str) -> Optional[List[float]]:
        """缓存嵌入结果"""
        return self.embedding_cache.get(text_hash)

# %% [markdown]
# ### get_merged_embedding(self, text: str, enable_deepseek_embed: bool = False) -> List[float]
    # %%
    def get_merged_embedding(
        self, text: str, enable_deepseek_embed: bool = False
    ) -> List[float]:
        # 计算文本哈希
        text_hash = hashlib.md5(text.encode()).hexdigest()

        # 检查缓存
        cached = self.get_cached_embedding(text_hash)
        if cached:
            log.info(f"使用缓存嵌入: {text_hash[:8]}")
            return cached

        """合并嵌入生成（本地嵌入为主，DeepSeek嵌入为增强选项）"""
        if enable_deepseek_embed:
            # 优先用DeepSeek增强嵌入
            log.info("使用DeepSeek增强嵌入")
            from deepseek_enhancer import get_deepseek_embedding

            embedding = get_deepseek_embedding(
                text, model=config.get("deepseek_embed_model", "deepseek-embedding")
            )
            # 存储到缓存
            if embedding:
                self.embedding_cache[text_hash] = embedding
                log.info(f"生成缓存嵌入: {text_hash[:8]}，{len(embedding)}")
            return embedding
        else:
            # 默认用本地Ollama嵌入（保留您已优化的分块逻辑）
            log.info("使用本地Ollama嵌入")
            self.get_model_max_context()
            chunks = self._split_text_into_chunks(text)

            if not chunks:
                return []

            log.info(
                f"文本分块: {len(chunks)}块（每块{self.chunk_size}字符），实际每块字符数量{[len(chunk) for chunk in chunks]}"
            )

            # 顺序处理（避免并发问题）
            embeddings = []
            for i in range(len(chunks)):
                emb = self.get_ollama_embedding(chunks[i])
                if emb:
                    embeddings.append(emb)
                else:  # 如果文本块中有嵌入量化操作失败，返回空列表，方便后面捕捉，避免写入残值导致误判
                    log.error(
                        f"处理文本块列表{[len(chunk) for chunk in chunks]}的第{i}块时出错，未有效获取嵌入向量"
                    )
                    return []

            if not embeddings:
                return []

            # 简单平均合并
            embedding = [sum(dim) / len(embeddings) for dim in zip(*embeddings)]
            # 存储到缓存
            if embedding:
                self.embedding_cache[text_hash] = embedding
                log.info(f"生成缓存嵌入: {text_hash[:8]}，{len(embedding)}")
            return embedding

# %% [markdown]
# ### _split_text_into_chunks(self, text: str) -> List[str]

    # %%
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """基于字符位置分块，保留完整单词"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # 计算本块的结束位置
            end = start + self.chunk_size
            
            # 如果已经到文本末尾
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # 寻找合适的断点（空格或标点）
            # 向前找最近的空格
            while end > start and text[end] not in ' \t\n.,;!?':
                end -= 1
            
            # 如果没找到空格，强制在chunk_size处截断
            if end == start:
                end = start + self.chunk_size
            
            # 添加块
            chunk = text[start:end].strip()
            if chunk:  # 避免添加空块
                chunks.append(chunk)
            
            # 更新起始位置
            start = end
        
        return chunks
