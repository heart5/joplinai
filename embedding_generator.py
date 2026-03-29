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
import logging
import time
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

# %% [markdown]
# ### get_model_max_context(self,) -> int
    # %%
    def get_model_max_context(
        self,
    ) -> int:
        """精确获取模型上下文限制（token→字符转换）"""
        # 特殊处理已知模型
        if self.model_name == "nomic-embed-text":
            self.chunk_size = 1850  # 768 token × 3字符/token × 0.8余量 ≈ 1850
        elif self.model_name == "qwen:1.8b":
            self.chunk_size = 4900  # 2048 token × 3 × 0.8 = 4916，取4900
            return
    
        # 通用模型处理
        try:
            model_info = ollama.show(model=self.model_name)
            num_ctx = model_info.get("parameters", {}).get("num_ctx", 2048)
            self.chunk_size = int(num_ctx * 3 * 0.8)
        except Exception as e:
            log.warning(f"获取模型上下文失败({self.model_name})，使用默认值2048字符: {e}")
            self.chunk_size = 2048

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
# ### get_merged_embedding(self, text: str, enable_deepseek_embed: bool = False) -> List[float]
    # %%
    def get_merged_embedding(
        self, text: str, enable_deepseek_embed: bool = False
    ) -> List[float]:
        """合并嵌入生成（本地嵌入为主，DeepSeek嵌入为增强选项）"""
        if enable_deepseek_embed:
            # 优先用DeepSeek增强嵌入
            log.info("使用DeepSeek增强嵌入")
            from deepseek_enhancer import get_deepseek_embedding
    
            return get_deepseek_embedding(
                text, model=config.get("deepseek_embed_model", "deepseek-embedding")
            )
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
            return [sum(dim) / len(embeddings) for dim in zip(*embeddings)]

# %% [markdown]
# ### _split_text_into_chunks(self, text: str) -> List[str]

    # %%
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """将文本分割成块"""
        # 实现分块逻辑
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            word_length = len(word)
            if current_length + word_length + 1 > self.chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length + 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
