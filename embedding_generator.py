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
import re
import time
from functools import lru_cache
from typing import Dict, List, Optional

import ollama
import requests

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
        self.embedding_dim = self._get_model_dimension()
        self.embedding_cache = {}  # 简单缓存

# %% [markdown]
# ### _get_model_dimension(self)

    # %%
    def _get_model_dimension(self):
        # 尝试从Ollama获取模型信息
        try:
            # 通过生成一个简单嵌入来获取维度
            test_response = ollama.embeddings(model=self.model_name, prompt="test")
            dim = len(test_response["embedding"])
            log.info(f"通过测试嵌入获取模型维度: {self.model_name} -> {dim}D")
            return dim
        except Exception as e:
            log.error(f"获取模型维度失败: {e}")
            # 默认返回常见维度
            return 1024

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
            self.chunk_size = 512 # 512 token，中文是一对一，应为是一对三或者四，实际中文为主
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
# ### _preprocess_text_for_embedding(self, text: str) -> str

    # %%
    def _preprocess_text_for_embedding(self, text: str) -> str:
        """专门为嵌入模型设计的文本预处理"""
        import re
        
        # 1. 移除或替换可能引起问题的特殊字符和模式
        # 例如：过多的换行符合并
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 例如：将多个连续相同字符缩减（针对“哈哈哈...”）
        text = re.sub(r'(.)\1{4,}', r'\1\1\1', text)  # 5个以上相同字符保留3个
        
        # 2. 移除或替换无语义的极端口语词（可选，根据需求调整）
        # 这里示例移除一些可能无实际语义的强感叹词，避免影响核心语义
        strong_interjections = [r'我操\s*', r'他妈\s*的', r'哈哈哈\s*']
        for pattern in strong_interjections:
            text = re.sub(pattern, '', text)
        
        # 3. 确保文本两端无多余空格
        text = text.strip()
        
        return text

# %% [markdown]
# ### _estimate_token_count(self, text: str) -> int

    # %%
    def _estimate_token_count(self, text: str) -> int:
        """简单估算文本的token数量（对于中文，一个粗略的方法是：字符数 * 系数）"""
        # 这是一个非常粗略的估算！对于中文，token数通常接近或少于字符数。
        # 系数可以设为0.8到1.2之间，具体取决于模型和文本。
        # 更准确的做法是使用模型的tokenizer，但这里为简单起见使用估算。
        estimated_tokens = int(len(text) * 1.2)
        return estimated_tokens

# %% [markdown]
# ### _smart_truncate(self, text: str, max_token_estimate: int) -> str

    # %%
    def _smart_truncate(self, text: str, max_token_estimate: int) -> str:
        """智能截断文本，尽量在句子或意群边界处截断"""
        # 根据字符数估算进行截断（因为token估算不精确）
        max_char_limit = int(max_token_estimate / 1.2)  # 反向估算字符数
        
        if len(text) <= max_char_limit:
            return text
        
        # 尝试在最后一个句号、问号、感叹号或换行处截断
        truncate_point = text.rfind('。', 0, max_char_limit)
        if truncate_point == -1:
            truncate_point = text.rfind('？', 0, max_char_limit)
        if truncate_point == -1:
            truncate_point = text.rfind('！', 0, max_char_limit)
        if truncate_point == -1:
            truncate_point = text.rfind('\n', 0, max_char_limit)
        if truncate_point == -1 or truncate_point < max_char_limit * 0.5:  # 如果找不到合适的点或点太靠前
            truncate_point = max_char_limit  # 硬截断
        
        return text[:truncate_point]

# %% [markdown]
# ### _aggressive_text_reduction(self, text: str) -> str

    # %%
    def _aggressive_text_reduction(self, text: str) -> str:
        """当模型明确报告超长时，采取更激进的文本缩减策略"""
        # 1. 直接截取前300个字符（保证极短）
        safe_length = 300
        if len(text) > safe_length:
            text = text[:safe_length]
            log.debug(f"激进缩减至{safe_length}字符")
        
        # 2. 进一步移除所有重复字符模式
        import re
        text = re.sub(r'(.)\1{2,}', r'\1', text)  # 3个以上相同字符保留1个
        
        return text

# %% [markdown]
# ### split_into_semantic_chunks(self, text: str, note_title: str = "", note_tags: str = "") -> List[Dict]

    # %%
    def split_into_semantic_chunks(self, text: str, note_title: str = "", note_tags: str = "") -> List[Dict]:
        """
        将文本分割成语义块，并返回块字典列表。
        每个字典包含 'content' 和初步的 'metadata'。
        """
        if not text:
            return []
        
        chunks = []
        
        # 1. 首先尝试按明显的章节或日期分割（针对日志）
        # 例如：按 "## "、"---"、日期行 "2025年11月27日" 分割
        major_sections = re.split(r'\n(?:#{1,3}\s+.*?|\-{3,}|\d{4}年\d{1,2}月\d{1,2}日.*?)\n', text)
        major_sections = [s.strip() for s in major_sections if s.strip()]
        
        for section in major_sections:
            if len(section) <= self.chunk_size:
                # 如果整个章节已经很小，直接作为一个块
                chunks.append(section)
            else:
                # 2. 对于长章节，按段落分割
                paragraphs = section.split('\n\n')
                current_chunk = ""
                for para in paragraphs:
                    if len(current_chunk) + len(para) <= self.chunk_size:
                        current_chunk += (para + "\n\n")
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = para + "\n\n"
                if current_chunk:
                    chunks.append(current_chunk.strip())
        
        # 3. 如果上述分割结果块数太少或太大，回退到句子级分块
        if len(chunks) == 0 or max(len(c) for c in chunks) > self.chunk_size * 1.5:
            log.debug("启用句子级分块回退")
            chunks = self._split_by_sentences(text)
        
        # 4. 为每个块构建初步元数据
        chunk_dicts = []
        for idx, chunk_content in enumerate(chunks):
            # 提取块内可能的关键词或日期（简单示例）
            import datetime
            date_match = re.search(r'(\d{4}年\d{1,2}月\d{1,2}日)', chunk_content)
            chunk_metadata = {
                "chunk_index": idx,
                "content": chunk_content,
                "parent_note_title": note_title,
                "parent_note_tags": note_tags,
                "estimated_date": date_match.group(1) if date_match else "",
                "word_count": len(chunk_content),
                # 后续可在 joplinai.py 中补充 DeepSeek 生成的摘要或关键词
            }
            chunk_dicts.append({
                "content": chunk_content,
                "metadata": chunk_metadata
            })
        
        log.info(f"将文本分割成 {len(chunk_dicts)} 个语义块。")
        return chunk_dicts

# %% [markdown]
# ### _split_by_sentences(self, text: str) -> List[str]

    # %%
    def _split_by_sentences(self, text: str) -> List[str]:
        """按句子分块的回退方法，保持原有逻辑但可调整。"""
        # 这里可以放入您原有的 _split_text_into_chunks 逻辑，或以下简化版：
        sentences = re.split(r'[。！？；\n]', text)
        chunks = []
        current_chunk = ""
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(current_chunk) + len(sent) <= self.chunk_size:
                current_chunk += sent + "。"
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sent + "。"
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

# %% [markdown]
# ### _split_text_into_chunks(self, text: str) -> List[str]

    # %%
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """优化分块：优先按自然段落、日期分隔符划分，其次按句子，最后才按字符。"""
        if len(text) <= self.chunk_size:
            return [text]
    
        chunks = []
        paragraphs = text.split('\n\n')  # 首先尝试按空行（段落）分割
    
        current_chunk = ""
        for para in paragraphs:
            # 如果当前段落本身就很长，尝试按句子分割
            if len(para) > self.chunk_size * 0.8:
                sentences = re.split(r'[。！？；\n]', para)
                for sent in sentences:
                    if len(current_chunk) + len(sent) <= self.chunk_size:
                        current_chunk += sent
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sent
            # 如果加上这个段落仍小于块大小，则合并
            elif len(current_chunk) + len(para) <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                # 保存当前块，开始新块
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
    
        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk.strip())
    
        # 如果按段落分割后块还是太大，则回退到原有的基于字符的智能分块
        if any(len(ch) > self.chunk_size * 1.2 for ch in chunks):
            log.info("部分块过长，启用字符级分块回退")
            return self._split_text_into_chunks_fallback(text)
    
        return chunks

# %% [markdown]
# ### _split_text_into_chunks_fallback(self, text: str) -> List[str]

    # %%
    def _split_text_into_chunks_fallback(self, text: str) -> List[str]:
        """原有的基于字符的智能分块逻辑（作为回退）
        基于字符位置分块，保留完整单词"""
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

# %% [markdown]
# ### get_ollama_embedding_safe(self, text: str, max_retries: int = 3) -> list

    # %%
    def get_ollama_embedding_safe(self, text: str, max_retries: int = 3) -> list:
        """调用本地Ollama服务生成嵌入，增加对长文本/异常文本的容错处理"""
        if not text or len(text.strip()) < 1:
            log.warning("输入文本为空或过短，返回零向量")
            # return [0.0] * self.embedding_dim
            return []
    
        # 1. 关键预处理：对文本进行清洗和规范化
        processed_text = self._preprocess_text_for_embedding(text)
    
        # 2. (可选但推荐) 估算token长度并主动截断
        # 假设模型最大上下文为512 tokens，我们设定一个安全阈值（如450 tokens）
        estimated_tokens = self._estimate_token_count(processed_text)
        safe_token_limit = int(self.chunk_size * 0.8)  # 根据模型调整，可设为配置项
        if estimated_tokens > safe_token_limit:
            log.warning(
                f"文本预估token数({estimated_tokens})超过安全阈值({safe_token_limit})，将进行智能截断"
            )
            processed_text = self._smart_truncate(processed_text, safe_token_limit)
            log.debug(f"截断后文本预览: {processed_text[:100]}...")
    
        # 3. 带重试的请求逻辑，并特别处理“长度超限”错误
        for attempt in range(max_retries):
            try:
                # 这里是您调用Ollama API的代码（示例）
                # response = ollama.embeddings(model=self.model_name, prompt=text)
                response = requests.post(
                    'http://localhost:11434/api/embeddings',
                    json={
                        "model": self.model_name,  # 例如 "dengcao/bge-large-zh-v1.5"
                        "prompt": processed_text  # 注意：某些模型可能用 "input" 而非 "prompt"
                    },
                    timeout=60
                )
    
                # 检查响应
                if response.status_code == 500:
                    error_msg = response.text.lower()
                    if "context length" in error_msg or "input length" in error_msg:
                        log.warning(
                            f"嵌入失败(尝试{attempt + 1}/{max_retries}): 模型报告输入超长。将尝试缩减文本。"
                        )
                        # 尝试更激进的文本缩减
                        processed_text = self._aggressive_text_reduction(processed_text)
                        continue  # 使用缩减后的文本重试
                    else:
                        # 其他500错误
                        log.warning(
                            f"嵌入失败(尝试{attempt + 1}/{max_retries}): {response.text}"
                        )
                        time.sleep(1)
                        continue
    
                response.raise_for_status()
                result = response.json()
                embedding = result.get("embedding")
                if embedding and len(embedding) == self.embedding_dim:
                    return embedding
                else:
                    log.warning(
                        f"返回的嵌入向量格式异常或维度不符，尝试 {attempt + 1}/{max_retries}"
                    )
                    time.sleep(1)
    
            except Exception as e:
                log.error(f"请求Ollama API失败(尝试{attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    break
                time.sleep(2**attempt)  # 指数退避
    
        # 所有重试均失败后的降级策略
        log.error(f"为文本生成嵌入最终失败，文本预览: '{text[:150]}...'。将返回安全向量。")
        # 返回一个零向量或随机向量，确保流程不中断
        # return [0.0] * self.embedding_dim
        return []

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
    @lru_cache(maxsize=200)
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
            for i, chunk in enumerate(chunks):
                emb = self.get_ollama_embedding(chunk)
                if emb:
                    embeddings.append(emb)
                else:  # 如果文本块中有嵌入量化操作失败，返回空列表，方便后面捕捉，避免写入残值导致误判
                    log.error(
                        f"处理文本块列表{[len(chunk) for chunk in chunks]}的第{i}块时出错，未有效获取嵌入向量"
                    )
                    log.debug(f"问题文本块如下：\n{chunk}\n再次用更保守又是更安全的方式处理…………")
                    emb = self.get_ollama_embedding_safe(chunk)
                    if emb:
                        embeddings.append(emb)
                    else:
                        log.error(
                            f"多次尝试处理文本块列表{[len(chunk) for chunk in chunks]}的第{i}块时出错，未有效获取嵌入向量"
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
