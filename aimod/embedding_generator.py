# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # 嵌入生成器
# embedding_generator.py

# %% [markdown]
# # 导入库

# %%
import hashlib
import logging
import re
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import ollama
import pathmagic
import requests


# %%
with pathmagic.context():
    try:
        from aimod.deepseek_enhancer import deepseek_process_note, get_cache_manager
        from func.datatools import compute_content_hash
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
    except ImportError as e:
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger(__name__)
        log.error(f"导入项目模块失败: {e}")


# %% [markdown]
# # AdaptiveChunkOptimizer类


# %%
class AdaptiveChunkOptimizer:
    """
    自适应分块优化器。
    原理：通过实际调用嵌入接口，探测给定文本在特定模型下的最大安全处理长度。
    利用本地模型零成本的优势，动态调整分块大小，避免过碎分块。
    """

    def __init__(self, embedding_generator, enabled=False):
        self.embedding_generator = embedding_generator
        self.enabled = enabled

    @staticmethod
    def _is_length_error(exception: Exception) -> bool:
        """判断异常是否为模型token上下文超限"""
        msg = str(exception).lower()
        return any(
            kw in msg
            for kw in ["context length", "input length", "too long", "exceed", "500"]
        )

    def _probe_at(self, text: str, length: int) -> Tuple[bool, Optional[List[float]]]:
        """探测指定长度是否安全。返回 (成功?, 嵌入向量或None)。"""
        try:
            emb = self.embedding_generator.get_ollama_embedding(text[:length])
            return True, emb
        except Exception as e:
            if self._is_length_error(e):
                return False, None
            raise  # 非长度错误向上抛

    def probe_max_safe_length(
        self, text: str, model_name: str, start_len: int = None
    ) -> int:
        """
        双向探测模型能安全处理的最大字符数。
        - start_len 通过 → 指数增长向上探索
        - start_len 失败 → 二分搜索向下探索
        支持 start_len 温启动。
        """
        if not self.enabled:
            return self.embedding_generator.chunk_size

        chunk_size = self.embedding_generator.chunk_size
        if start_len is None:
            # 温启动未提供：取 chunk_size 的 88%，接近上限减少探测次数
            start_len = int(chunk_size * 0.88)
        max_len = len(text)

        log.info(
            f"开始自适应分块探测: 模型={model_name}, 文本长度={max_len}字符, "
            f"起始={start_len}字符"
        )

        current_safe_len = start_len

        # Phase 1: 测试起始点
        try:
            ok, _ = self._probe_at(text, min(start_len, max_len))
        except Exception as e:
            log.warning(f"  起始探测非长度错误，回退默认chunk_size={chunk_size}: {e}")
            current_safe_len = chunk_size
            ok = False  # 触发下方回退

        if ok:
            # 起始点通过 → 指数增长向上探索
            current_safe_len = min(start_len, max_len)
            test_len = int(current_safe_len * 1.1)
            while test_len <= max_len:
                try:
                    ok, _ = self._probe_at(text, test_len)
                except Exception as e:
                    log.warning(f"  探测非长度错误，保留 {current_safe_len}字符: {e}")
                    break
                if ok:
                    current_safe_len = test_len
                    log.debug(f"  探测通过[字符→API]: {test_len}字符未超token上下文")
                    test_len = int(test_len * 1.1)
                else:
                    log.debug(
                        f"  探测失败[token超限]: {test_len}字符→超出模型token上下文"
                    )
                    break
        else:
            # 起始点失败 → 二分搜索向下探索
            lo = int(chunk_size * 0.5)  # 绝对下限
            hi = min(start_len, max_len)
            log.debug(f"  起始探测超限，二分搜索: [{lo}, {hi}]")
            while lo < hi:
                mid = (lo + hi) // 2
                try:
                    ok_mid, _ = self._probe_at(text, mid)
                except Exception:
                    break  # 非长度错误，中止搜索
                if ok_mid:
                    current_safe_len = mid
                    lo = mid + 1
                else:
                    hi = mid
            log.debug(f"  二分搜索完成: 安全上限={current_safe_len}字符")

        # 3. 确保结果在合理范围内
        final_safe_len = max(
            int(chunk_size * 0.5),
            min(current_safe_len, chunk_size * 2),
        )
        log.info(
            f"自适应分块探测完成: 建议块大小={final_safe_len}字符 (chunk_size={chunk_size}字符)"
        )

        return final_safe_len


# %% [markdown]
# # PunctuationAwareSplitter类


# %%
class PunctuationAwareSplitter:
    """
    增强型标点与结构感知分块器 (完善版)
    策略：文档结构切分 (最高优先级) -> 语义标点切分 -> 句子重叠回退 -> 硬切保底
    """

    def __init__(
        self,
        max_chunk_size: int = 512,
        min_chunk_size: int = 100,
        target_overlap_sentences: int = 1,
        fallback_overlap_chars: int = 100,
    ):
        """
        初始化分块器

        参数:
            max_chunk_size: 单个块的最大字符数（目标值）。
            min_chunk_size: 单个块的最小字符数，避免产生无意义碎片。
            target_overlap_sentences: 目标重叠句子数。
            fallback_overlap_chars: 保底重叠字符数。
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.target_overlap = target_overlap_sentences
        self.fallback_overlap = fallback_overlap_chars

        # **关键完善1：扩展权重字典，纳入文档结构标记**[6](@ref)
        # 为Markdown标题、列表项等赋予最高权重，确保它们被优先识别为切分边界。
        self.punct_weights = {
            # 文档结构标记 (最高优先级)
            "\n### ": 2.0,
            "\n## ": 1.8,
            "\n# ": 1.8,  # Markdown标题
            "\n- ": 1.5,
            "\n* ": 1.5,
            "\n+ ": 1.5,  # 无序列表项
            r"\n\d+\. ": 1.5,  # 有序列表项 (如 `1. `)
            "\n\n": 1.2,  # 段落分隔
            # 标准标点符号
            "。": 1.0,
            "！": 1.0,
            "？": 1.0,
            "；": 0.7,
            "：": 0.6,
            "，": 0.3,
            "、": 0.1,
            ".": 1.0,
            "!": 1.0,
            "?": 1.0,
            ";": 0.7,
            ",": 0.3,
        }
        # 编译正则时，需对结构标记中的特殊字符进行转义，并确保有序列表模式正确。
        escaped_keys = []
        for k in self.punct_weights.keys():
            if k.startswith(r"\n\d+\. "):  # 有序列表的正则模式单独处理
                escaped_keys.append(r"\n\d+\.\s+")
            else:
                escaped_keys.append(re.escape(k))
        self.punct_pattern = re.compile(f"({'|'.join(escaped_keys)})")

        # 句子结束符正则（用于重叠回退策略）
        self.sentence_end_pattern = re.compile(r"([。！？.!?]+|\n{2,})")

    def split(self, text: str) -> List[str]:
        """
        主分割函数：实现 **结构优先** 的四层降级分块策略。
        """
        # **关键完善2：预处理 - 按最高权重的结构标记进行首次粗分割**
        # 此步骤旨在先将文档按章节、大列表项等宏观结构拆开。
        primary_chunks = self._split_by_primary_structure(text)

        final_chunks = []
        for primary_chunk in primary_chunks:
            if len(primary_chunk) <= self.max_chunk_size:
                # 如果宏观结构块本身已满足大小要求，直接保留。
                final_chunks.append(primary_chunk.strip())
            else:
                # **关键完善3：递归应用原有智能策略**
                # 对仍然过长的宏观块，递归调用内部方法进行细粒度分割。
                # 这保证了在宏观结构内，依然遵循“语义->句子->硬切”的保底逻辑。
                sub_chunks = self._split_recursively(primary_chunk)
                final_chunks.extend(sub_chunks)

        log.info(
            f"文本【{repr(text[:30])}……】总长{len(text)}字符，"
            f"被增强型语义切割工具切割为{len(final_chunks)}块，"
            f"各块字符数: {[len(chunk) for chunk in final_chunks]}"
        )
        return final_chunks

    def _split_by_primary_structure(self, text: str) -> List[str]:
        """
        使用最高权重的文档结构标记进行第一级分割。
        此方法旨在捕获如 `### 标题`、`1. 主要事项` 这样的天然大边界。
        """
        # 查找所有高权重结构标记的位置
        positions = [0]  # 起始位置
        for match in self.punct_pattern.finditer(text):
            punct = match.group()
            if self.punct_weights.get(punct, 0) >= 1.2:  # 只对高权重标记进行切分
                positions.append(match.start())

        if len(positions) == 1:
            return [text]  # 未找到高权重结构，整个文本作为一个初级块

        # 根据找到的位置进行分割
        chunks = []
        for i in range(len(positions)):
            start = positions[i]
            end = positions[i + 1] if i + 1 < len(positions) else len(text)
            chunk = text[start:end]
            if chunk.strip():  # 避免空块
                chunks.append(chunk)
        return chunks

    def _split_recursively(self, text: str) -> List[str]:
        """
        对单个宏观结构块递归应用原有的智能滑动窗口分块逻辑。
        这是您原有 `split` 方法逻辑的移植，用于处理块内部。
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            if text_length - start <= self.min_chunk_size:
                final_chunk = text[start:].strip()
                if final_chunk:
                    chunks.append(final_chunk)
                break

            window_end = min(start + self.max_chunk_size, text_length)
            window_text = text[start:window_end]
            best_split_pos = self._find_best_split_position(window_text)

            if best_split_pos > 0:
                # 找到语义切分点
                chunk = text[start : start + best_split_pos].strip()
                if chunk:
                    chunks.append(chunk)
                start += best_split_pos
                continue

            # 未找到语义点，进行保底处理
            current_chunk = text[start:window_end].strip()
            if current_chunk:
                chunks.append(current_chunk)

            next_start = window_end
            if self.target_overlap > 0:
                # 尝试句子重叠回退
                lookback_start = max(start, window_end - self.max_chunk_size)
                lookback_text = text[lookback_start:window_end]
                sentence_ends = []
                for match in self.sentence_end_pattern.finditer(lookback_text):
                    absolute_pos = lookback_start + match.end()
                    sentence_ends.append(absolute_pos)
                sentence_ends.sort(reverse=True)

                if len(sentence_ends) >= self.target_overlap:
                    next_start = sentence_ends[self.target_overlap - 1]
                else:
                    # 字符重叠保底
                    next_start = max(
                        window_end - self.fallback_overlap, start + self.min_chunk_size
                    )

            if next_start <= start:
                next_start = window_end
            start = next_start

        return chunks

    def _find_best_split_position(self, window_text: str) -> int:
        """
        在窗口内寻找最佳切分位置（完善版）。
        现在会考虑所有已定义的标点和结构标记的权重。
        """
        punct_positions = []
        for match in self.punct_pattern.finditer(window_text):
            punct = match.group()
            pos = match.end()
            weight = self.punct_weights.get(punct, 0)
            punct_positions.append((pos, weight))

        if not punct_positions:
            return 0

        punct_positions.sort(key=lambda x: x[1], reverse=True)
        for pos, weight in punct_positions:
            if pos >= self.min_chunk_size:
                return pos
        return 0


# %% [markdown]
# # ContextAwareSplitter(PunctuationAwareSplitter)

# %%
class ContextAwareSplitter(PunctuationAwareSplitter):
    """
    上下文感知分块器：集成防过碎逻辑与即时上下文注入。
    在分割时直接为每个块添加【笔记标题 - 日期】头部。
    """

    def _inject_context(
        self, chunk_text: str, note_title: str, source_date: str
    ) -> str:
        """为单个文本块注入格式化的上下文头部。"""
        # 构建头部
        header_title_date = f"【{note_title}】\n日期：{source_date}\n\n"
        header_title = f"【{note_title}】\n\n"

        # 检查是否被注入过，是则直接返回传入文本，避免重复注入
        if chunk_text.startswith(header_title_date) or chunk_text.startswith(
            header_title
        ):
            return chunk_text

        date_pattern = re.compile(r"^#{0,3}\s*\d{4}年\d{1,2}月\d{1,2}[日号]\s*")
        match = date_pattern.match(chunk_text)
        if match:
            # 从文本中移除开头的日期模式
            chunk_text = date_pattern.sub("", chunk_text, count=1).strip()
        if source_date:
            header = header_title_date
        else:
            header = header_title

        final_text = header + chunk_text

        return final_text.strip()

# %% [markdown]
# # EmbeddingGenerator类


# %%
class EmbeddingGenerator:
    """嵌入生成器，支持长文本分块处理"""

# %% [markdown]
# ## 类常量

    # %%
    # ========== 类常量：预编译的正则表达式 ==========
    # 用于一级分割：匹配 "### 2026年4月14日" 或 "2026年4月14日" 等日期标题行
    UNIFIED_DATE_PATTERN = re.compile(
        r"^(?:###\s*)?(\d{4}[-年/]\d{1,2}[-月/]\d{1,2}[日号])\s*$", re.MULTILINE
    )
    # 用于从注入上下文的块中提取日期：匹配 "2026年4月14日"
    DATE_IN_CONTEXT_PATTERN = re.compile(
        r"(\d{4}[-年/]\d{1,2}[-月/]\d{1,2}[日号])", re.MULTILINE
    )
    # 用于从注入上下文的块中提取日期：匹配 "日期：2026年4月14日"
    # DATE_IN_CONTEXT_PATTERN = re.compile(
    #     r"日期[：:]\s*(\d{4}[-年/]\d{1,2}[-月/]\d{1,2}[日号])", re.MULTILINE
    # )
    # 通用章节分割模式：优先匹配分隔符 (***, ---)，其次匹配 # 标题
    GENERAL_SECTION_PATTERN = re.compile(
        r"\n(?:\*{3,}|\-{3,}|#{1,3}\s+.*?)\n", re.MULTILINE
    )

# %% [markdown]
# ## \_\_init__(self, config: dict, model_name: str, chunk_size: int = 1024)

    # %%
    def __init__(
        self,
        config: dict,
        model_name: str,
        chunk_size: int = 1024,
        enable_adaptive_chunking=False,
        chunk_overlap=50,
    ):
        self.config = config
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._model_dimension_cache = {}
        self.embedding_dim = self._get_model_dimension()
        self.embedding_cache = {}
        self._chunk_embedding_cache = {}
        self._set_chunk_size()
        self.adaptive_optimizer = AdaptiveChunkOptimizer(
            embedding_generator=self,
            enabled=enable_adaptive_chunking,
        )
        self.enable_adaptive_chunking = enable_adaptive_chunking

    @property
    def _safe_net_chars(self) -> int:
        """单块净文本安全字符上限（不含上下文注入头部）。

        与 _iterative_chunking 内部 margin 完全一致：
        margin = int(chunk_size * 0.85 * 0.9) = int(chunk_size * 0.765)
        注入头部后总长仍 <= 模型token上下文，不会触发API超限。
        """
        return int(self.chunk_size * 0.765)

    def _try_embed_chunk(self, chunk_text: str) -> None:
        """分块阶段预生成嵌入并缓存，后续 get_merged_embedding 可直接取用。
        失败静默跳过——嵌入阶段仍可重试。
        """
        if not self.enable_adaptive_chunking:
            return
        key = hashlib.md5(chunk_text.encode("utf-8")).hexdigest()
        if key in self._chunk_embedding_cache:
            return
        try:
            processed = self._preprocess_text_for_embedding(chunk_text)
            self._chunk_embedding_cache[key] = self.get_ollama_embedding(processed)
        except Exception:
            pass

# %% [markdown]
# ## _get_model_dimension(self)

    # %%
    def _get_model_dimension(self):
        """获取模型维度"""
        # 已知模型维度映射
        known_dimensions = {
            "dengcao/bge-large-zh-v1.5": 1024,  # 根据日志显示实际是1024维
            "nomic-embed-text": 768,
            "qwen:1.8b": 2048,
            # 可以添加更多已知模型
        }

        if self.model_name in known_dimensions:
            dim = known_dimensions[self.model_name]
            self._model_dimension_cache[self.model_name] = dim
            # log.info(f"使用已知模型维度: {model_name} -> {dim}D")
            return dim

        if self.model_name in self._model_dimension_cache:
            return self._model_dimension_cache[self.model_name]

        # 尝试从Ollama获取模型信息
        try:
            # 通过生成一个简单嵌入来获取维度
            test_response = ollama.embeddings(model=self.model_name, prompt="test")
            dim = len(test_response["embedding"])
            self._model_dimension_cache[self.model_name] = dim
            log.info(f"通过测试嵌入获取模型维度: {self.model_name} -> {dim}D")
            return dim
        except Exception as e:
            log.error(f"获取模型{self.model_name}维度失败: {e}")
            # 默认返回常见维度
            return 1024

# %% [markdown]
# ## _set_chunk_size(self) -> None
    # %%
    def _set_chunk_size(self) -> None:
        """精确获取模型上下文限制（token→字符转换）"""
        # 特殊处理已知模型
        if self.model_name == "nomic-embed-text":
            self.chunk_size = 1850  # 字符: 768token × 3字符/token × 0.8 ≈ 1850字符
            return
        elif self.model_name == "dengcao/bge-large-zh-v1.5":
            self.chunk_size = (
                512  # 字符: bge中文模型 512token上下文, 中文≈1token/字符
            )
            return
        elif self.model_name == "qwen:1.8b":
            self.chunk_size = 4000  # 字符: 2048token上下文 × 3字符/token × 0.8 ≈ 4916字符, 取4000字符
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

                match = re.search(r"num_ctx\s+(\d+)", params_str)
                if match:
                    num_ctx = int(match.group(1))
                else:
                    # 方法2：尝试从 modelfile 解析
                    modelfile = model_info.get("modelfile", "")
                    match = re.search(r"PARAMETER num_ctx\s+(\d+)", modelfile)
                    if match:
                        num_ctx = int(match.group(1))
                    else:
                        # 方法3：尝试从 details.modelinfo 获取
                        try:
                            num_ctx = (
                                model_info.get("details", {})
                                .get("modelinfo", {})
                                .get("bert.context_length", 2048)
                            )
                        except:
                            pass
            else:
                # 如果 parameters 是字典（其他模型）
                num_ctx = model_info.get("parameters", {}).get("num_ctx", 1024)

            self.chunk_size = int(num_ctx * 3 * 0.8)
            log.info(
                f"模型 {self.model_name} 上下文={num_ctx}token, chunk_size(字符)={self.chunk_size}字符"
            )

        except Exception as e:
            log.warning(
                f"获取模型上下文失败({self.model_name}), 回退默认chunk_size=1024字符: {e}"
            )
            self.chunk_size = 1024

# %% [markdown]
# ## clean_text(text: str) -> str
    # %%
    def clean_text(self, text: str) -> str:
        """
        清理笔记文本：移除图片、格式符号、多余换行
        一般用于分块之后净化文本
        """
        if not text:
            return ""

        # 1. 移除图片链接：![alt](url)
        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)

        # 2. 特别处理：移除图片链接后的多余空行
        # 将3个以上连续换行减少为2个
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 3. 移除Markdown格式符号（保留必要的标点）
        # 注意：不要移除中文标点符号
        text = re.sub(r"[#*`>~\-]", "", text)

        # 4. 移除开头和结尾的空白行
        text = text.strip()

        # 5. 如果清理后文本过短，记录警告
        if len(text) < 10:  # 少于10个字符
            log.warning(f"清理后文本过短，不到10个字符。清理前为: {text[:50]}...")

        return text

# %% [markdown]
# ## enhance_by_deepseek_for_summary_tags(chunk_content: str, note_tags: str, config: Dict)

    # %%
    def enhance_by_deepseek_for_summary_tags(self, chunk_content: str, note_tags: str, config: Dict):
        """DeepSeek 官方模型增强生成小结和标签（和笔记既有标签进行融合）

        Returns:
            enhanced_metadata: 增强后的元数据字典，包含 deepseek_enhanced 标记
        """

        # log.info(get_cache_manager().get_stats())

        enhanced_metadata = {"deepseek_enhanced": False}
        if config["enable_deepseek_summary"]:
            summary = deepseek_process_note(
                chunk_content,
                task="summary",
                model=config.get("deepseek_chat_model", "deepseek-chat"),
                use_cache=True,  # 启用缓存
            )
            if summary:
                enhanced_metadata["deepseek_enhanced"] = True
            enhanced_metadata["chunk_summary"] = summary or ""  # 存入摘要

        if config["enable_deepseek_tags"]:
            tags_str = deepseek_process_note(
                chunk_content,
                task="tags",
                model=config.get("deepseek_chat_model", "deepseek-chat"),
                use_cache=True,  # 启用缓存
            )
            if tags_str:
                enhanced_metadata["deepseek_enhanced"] = True
            deepseek_tags = [t.strip() for t in tags_str.split(",")] if tags_str else []
            # 合并本地标签与DeepSeek标签（去重）
            original_tags = [t.strip() for t in note_tags.split(",")] if note_tags else []
            enhanced_tags = list(set(original_tags + deepseek_tags))
            enhanced_metadata["tags"] = ",".join(enhanced_tags)

        return enhanced_metadata

# %% [markdown]
# ## _is_valid_chunk(self, text: str, min_length: int = 10) -> bool

    # %%
    def _is_valid_chunk(self, text: str, min_length: int = 10) -> bool:
        """检查文本块是否有效（非空、非极短、非纯符号）"""
        if not text or len(text.strip()) < min_length:
            return False
        # 可选：检查是否仅为符号、数字或空格
        if re.match(r"^[\s\\d\\W]+$", text):
            return False
        return True

# %% [markdown]
# ## _preprocess_text_for_embedding(self, text: str) -> str

    # %%
    def _preprocess_text_for_embedding(self, text: str) -> str:
        """专门为嵌入模型设计的文本预处理"""
        import re

        # 1. 移除或替换可能引起问题的特殊字符和模式
        # 例如：过多的换行符合并
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 例如：将多个连续相同字符缩减（针对“哈哈哈...”）
        text = re.sub(r"(.)\1{4,}", r"\1\1\1", text)  # 5个以上相同字符保留3个

        # 2. 移除或替换无语义的极端口语词（可选，根据需求调整）
        # 这里示例移除一些可能无实际语义的强感叹词，避免影响核心语义
        strong_interjections = [r"我操\s*", r"他妈\s*的", r"哈哈哈\s*"]
        for pattern in strong_interjections:
            text = re.sub(pattern, "", text)

        # 3. 确保文本两端无多余空格
        text = text.strip()

        return text

# %% [markdown]
# ## _convert_health_data_to_text(self, raw_content: str) -> str

    # %%
    def _convert_health_data_to_text(self, raw_content: str) -> str:
        """
        将健康笔记中的结构化数据行转换为自然语言描述。
        示例输入: "110，4：14" -> "今日步数110步，睡眠时长4小时14分钟。"
        示例输入: "799，7：44，1" -> "今日步数799步，睡眠时长7小时44分钟，喝啤酒1瓶。"
        """
        lines = [line for line in raw_content.strip().split("\n") if line]
        converted_lines = []

        for line in lines:
            line = line.strip()
            # 匹配数字模式：如 "110，4：14" 或 "11033，4：7，4"
            if re.match(r"^\d+[，,]\s*\d+[:：]\d+([，,]\s*\d+)?$", line):
                parts = re.split(r"[，,]\s*", line)
                if len(parts) >= 2:
                    # 解析步数
                    steps = parts[0]
                    desc = f"今日步数{steps}步，"

                    # 解析睡眠时间（格式如 4:14 或 4：14）
                    sleep_time = parts[1].replace("：", ":")
                    if ":" in sleep_time:
                        sleep_parts = sleep_time.split(":")
                        if len(sleep_parts) == 2:
                            desc += f"睡眠时长{sleep_parts[0]}小时{sleep_parts[1]}分钟"

                    # 解析啤酒数量（如果有）
                    if len(parts) >= 3 and parts[2].isdigit():
                        beer_count = parts[2]
                        desc += f"，喝啤酒{beer_count}瓶"

                    line = desc + "。"
            converted_lines.append(line)

        return "\n".join(converted_lines)

# %% [markdown]
# ## _condense_dense_lists(self, text: str) -> str

    # %%
    def _condense_dense_lists(self, text: str) -> str:
        """
        尝试浓缩密集的列表文本，减少token数量但保留关键信息。
        例如：将“A、B、C、D、E”浓缩为“A等5人”。
        """
        import re

        # 匹配中文顿号分隔的列表模式，如“张三、李四、王五”
        pattern = r"([\u4e00-\u9fa5]{2,4}、){3,}[\u4e00-\u9fa5]{2,4}"
        matches = re.findall(pattern, text)

        for match in matches:
            original = match
            # 提取所有人名
            names = original.split("、")
            if len(names) > 4:  # 仅对较长的列表进行浓缩
                condensed = f"{names[0]}、{name[1]}等{len(names)}人"
                text = text.replace(original, condensed)
                log.debug(f"浓缩密集列表: {original[:20]}... -> {condensed}")
        return text

# %% [markdown]
# ## _aggressive_text_reduction(self, text: str) -> str

    # %%
    def _aggressive_text_reduction(self, text: str) -> str:
        """当模型明确报告超长时，采取更激进的文本缩减策略"""
        # 1. 直接截取前400个字符（保证极短）
        safe_length = 400
        len_text = len(text)
        if len_text > safe_length:
            text = text[:safe_length]
            log.debug(f"激进缩减，从{len_text}缩减至{safe_length}个字符")

        # 2. 【新增】针对“人名列表”的特殊处理
        # 检测模式：包含大量顿号、冒号，且无明显段落
        if "：" in text and "、" in text and len(text.splitlines()) < 5:
            log.debug(
                f"检测到密集人名列表，进行针对性清理。该文本块头为：{text[:200]} ……"
            )
            # 移除所有空格和换行，简化格式
            text = text.replace("\n", "").replace(" ", "")
            # 将顿号替换为逗号（可能token更少）
            text = text.replace("、", ",")
            # 如果仍然过长，只保留前N个人名
            if len(text) > safe_length:
                # 尝试按逗号分割，保留前一部分
                parts = text.split(",")
                if len(parts) > 10:
                    text = ",".join(parts[:10]) + "...（名单截断）"

        # 3. 进一步移除所有重复字符模式
        import re

        text = re.sub(r"(.)\1{2,}", r"\1", text)  # 3个以上相同字符保留1个

        return text

# %% [markdown]
# ## _reduce_text_length(self, text: str, max_chars: int = 400) -> str

    # %%
    def _reduce_text_length(self, text: str, max_chars: int = 400) -> str:
        """智能缩减文本长度，优先保留信息密度高的部分。"""
        if len(text) <= max_chars:
            return text

        original_len = len(text)
        log.warning(f"文本过长({original_len}字符)，启动智能缩减。")

        # 策略1: 移除纯格式性、低信息量的行（如空行、纯分隔符）
        lines = text.splitlines()
        filtered_lines = []
        for line in lines:
            stripped = line.strip()
            # 保留非空行，且不是纯符号或数字编号的行
            if stripped and not re.match(r"^[\s\-*=•·●○◆◇■□▣▢▤▥▦▧▨▩▱▰]*$", stripped):
                filtered_lines.append(line)
        text = "\n".join(filtered_lines)

        # 策略2: 如果仍是长列表，尝试浓缩（调用上述新方法）
        text = self._condense_dense_lists(text)

        # 策略3: 若仍超长，进行关键句提取（简易版）
        if len(text) > max_chars:
            # 优先保留包含日期、数字、关键动词（如“总结”、“认为”、“记录”）的句子
            sentences = re.split(r"(?<=[。！？；\n])", text)
            important_sentences = []
            for sent in sentences:
                # 简单的关键词启发式规则
                if (
                    re.search(r"\d{4}年\d{1,2}月\d{1,2}日", sent)
                    or re.search(r"\b(总计|合计|主要|关键|总结|认为|记录|建议)\b", sent)
                    or re.search(r"[A-Za-z\u4e00-\u9fa5]{2,}：[^。]+", sent)
                ):  # 包含冒号定义的项
                    important_sentences.append(sent)
            if important_sentences:
                text = "".join(important_sentences)
                log.debug(f"通过关键句提取缩减文本。")

        # 策略4: 最后防线，按段落截断但添加标记
        if len(text) > max_chars:
            # 不是粗暴截断，而是找到最近的段落结束处
            truncated = text[:max_chars]
            # 尝试在段落边界处截断
            last_para_break = truncated.rfind("\n\n")
            if last_para_break > max_chars * 0.5:  # 如果能找到合理的段落边界
                text = (
                    truncated[:last_para_break]
                    + f"\n\n【注：因长度限制，后续内容已省略。原始文本共{original_len}字符。】"
                )
            else:
                text = truncated + f"...【文本截断，原始长度{original_len}字符】"
            log.warning(f"文本经智能缩减后仍超长，已进行截断并添加标记。")

        log.info(f"智能缩减完成: {original_len} -> {len(text)} 字符")
        return text

# %% [markdown]
# ## _normalize_date_string(self, date_str: str) -> str

    # %%
    def _normalize_date_string(self, date_str: str) -> str:
        """
        将各种格式的日期字符串规范化为统一的“YYYY年M月D日”格式。
        支持格式：
            - “2026年04月14日” 或 “2026年4月14日”
            - “2026年04月14号” 或 “2026年4月14号”
            - “2026-04-14” 或 “2026-4-14”
            - “2026/04/14” 或 “2026/4/14”
        输出：统一为“2026年4月14日”（数字不补零）。
        """
        if not date_str or not isinstance(date_str, str):
            return ""

        # 定义多种匹配模式并提取年、月、日数字
        patterns = [
            # 匹配“2026年04月14日”或“2026年4月14号”等
            r'(?P<year>\d{4})年(?P<month>\d{1,2})月(?P<day>\d{1,2})[日号]',
            # 匹配“2026-04-14”或“2026-4-14”
            r'(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})',
            # 匹配“2026/04/14”或“2026/4/14”
            r'(?P<year>\d{4})/(?P<month>\d{1,2})/(?P<day>\d{1,2})',
        ]

        for pattern in patterns:
            match = re.search(pattern, date_str)
            if match:
                year = match.group('year')
                # 去除月份和日期的前导零，例如“04” -> “4”
                month = str(int(match.group('month')))
                day = str(int(match.group('day')))
                # 统一格式化为“YYYY年M月D日”
                return f"{year}年{month}月{day}日"

        # 如果没有匹配到任何已知格式，返回原字符串或空字符串（根据需求）
        return date_str

# %% [markdown]
# ## _normalize_single_date_unit(self, raw_text: str, captured_date: str) -> str

    # %%
    def _normalize_single_date_unit(self, raw_text: str, captured_date: str) -> str:
        """
        规范化单个日期单元的文本。
        """
        lines = raw_text.splitlines()
        if not lines:
            return ""

        # 1. 规范化标题行：统一为 “### YYYY年M月D日” 格式
        # 【核心调用】使用新的通用日期规范化函数
        normalized_captured_date = self._normalize_date_string(captured_date)
        normalized_header = f"### {normalized_captured_date}"

        # 2. 处理主体内容：去除标题行之后的连续空行，以及单元末尾的连续空行
        body_lines = []
        for line in lines[1:]:  # 从标题行之后开始
            # 跳过开头的连续空行，直到遇到第一个非空行
            if body_lines or line.strip():
                body_lines.append(line.rstrip())  # 同时去除每行右侧空格

        # 去除末尾的连续空行
        while body_lines and not body_lines[-1].strip():
            body_lines.pop()

        # 3. 重新组装
        # 如果主体为空，只返回标题行；否则用两个换行符连接标题和主体（常见格式）
        if not body_lines:
            return normalized_header
        else:
            # 注意：这里保持主体内部原有的换行结构，只规范边界
            normalized_body = "\n".join(body_lines)
            return f"{normalized_header}\n\n{normalized_body}"

# %% [markdown]
# ## _iterative_chunking(self, text, note_title, source_date)

# %%
    def _iterative_chunking(
        self, text: str, note_title: str, source_date: str
    ) -> list:
        """
        统一迭代分块策略（自适应探测 or 固定安全大小）。
        逐段提取安全可嵌入的块 → 注入上下文 → 重叠 → 继续，
        直到全部处理完毕。完全消除二次拆分路径。
        启用自适应时逐块探测，否则用 chunk_size*0.9 作固定安全大小。
        返回：已注入上下文的文本块列表。
        """
        can_adaptive = (
            self.enable_adaptive_chunking and self.adaptive_optimizer.enabled
        )

        chunks = []
        pos = 0
        text_len = len(text)
        MIN_SIZE = 100
        last_safe_len = None  # 温启动：用上一次的安全长度
        sentence_end_pat = re.compile(r"[。！？.!?]+")
        ctx_splitter = ContextAwareSplitter(max_chunk_size=self.chunk_size)

        while pos < text_len:
            remaining = text[pos:]
            remaining_len = len(remaining)

            # 快速路径：保守阈值以下，无需探测
            if remaining_len <= self._safe_net_chars:
                chunk_ctx = ctx_splitter._inject_context(
                    remaining, note_title, source_date
                )
                chunks.append(chunk_ctx)
                self._try_embed_chunk(chunk_ctx)
                log.debug(
                    f"迭代分块: 剩余文本({remaining_len}字符)≤保守安全上限"
                    f"({self._safe_net_chars}字符)，直接作为末块"
                )
                break

            # 1. 确定安全长度：探测 or 固定值
            if can_adaptive:
                safe_len = self.adaptive_optimizer.probe_max_safe_length(
                    remaining, self.model_name, start_len=last_safe_len
                )
                last_safe_len = safe_len
            else:
                safe_len = int(self.chunk_size * 0.85)  # 固定安全大小

            # 2. 预留头部空间(上下文注入约增加30~50字符)，并在窗口内找句子边界
            margin = int(safe_len * 0.9)  # 留10%余量

            # 灰色地带：探测后可能发现剩余文本仍在margin内
            if remaining_len <= margin:
                chunk_ctx = ctx_splitter._inject_context(
                    remaining, note_title, source_date
                )
                chunks.append(chunk_ctx)
                self._try_embed_chunk(chunk_ctx)
                log.debug(
                    f"迭代分块: 剩余文本({remaining_len}字符)≤探测margin"
                    f"({margin}字符)，直接作为末块"
                )
                break
            window_text = remaining[:margin]

            # 复用现有标点感知切分逻辑找最佳切分位置
            best_split = ctx_splitter._find_best_split_position(window_text)

            # 最小块尺寸保护：句子边界太靠前时用 margin 截断，避免过短块
            MIN_FRACTION = 0.4
            if best_split >= max(MIN_SIZE, int(margin * MIN_FRACTION)):
                actual_chunk = remaining[:best_split]
                chunk_end = pos + best_split
            else:
                log.debug(
                    f"  句子边界({best_split}字符)过短(阈值{max(MIN_SIZE, int(margin * MIN_FRACTION))}字符)"
                    f"，改用 margin={margin}字符 截断"
                )
                actual_chunk = window_text
                chunk_end = pos + margin

            # 3. 注入上下文
            chunk_ctx = ctx_splitter._inject_context(
                actual_chunk, note_title, source_date
            )
            chunks.append(chunk_ctx)
            self._try_embed_chunk(chunk_ctx)
            log.debug(
                f"迭代分块: pos={pos}字符, 剩余={remaining_len}字符, "
                f"安全块大小({'探测' if can_adaptive else '固定'})={safe_len}字符, "
                f"实际取={len(actual_chunk)}字符"
            )

            if chunk_end >= text_len:
                break

            # 4. 计算重叠: 从 chunk_end 往回找句子边界（1句重叠）
            lookback_start = max(pos, chunk_end - 300)
            lookback_text = text[lookback_start:chunk_end]
            sentence_ends = [
                (lookback_start + m.end()) for m in sentence_end_pat.finditer(lookback_text)
            ]

            if sentence_ends:
                # 用倒数第二个句子结束位置（实现1个完整句子的重叠）
                next_pos = sentence_ends[-2] if len(sentence_ends) >= 2 else sentence_ends[-1]
            else:
                # 无句子边界，回退到100字符重叠
                next_pos = chunk_end - 100

            # 防卡死：确保至少有 MIN_SIZE 的进展
            next_pos = max(pos + MIN_SIZE, min(next_pos, chunk_end - 1))
            pos = next_pos

        log.info(
            f"笔记《{note_title}》迭代分块完成"
            f"（{'自适应探测' if can_adaptive else '固定大小'}），"
            f"共 {len(chunks)} 块，各块字符数: {[len(c) for c in chunks]}"
        )
        return chunks

# %% [markdown]
# ## _extract_author_from_note(self, note_title: str, note_tags: str, source_notebook_title: str = "") -> str

    # %%
    def _extract_author_from_note(self, note_title: str, note_tags: str, source_notebook_title: str = "") -> str:
        """
        从笔记标题、标签和笔记本信息中提取作者信息。
        返回格式："XXA"、"XXB"、"团队_共同维护"。
        """
        import re
        split_ptn = re.compile(r"[,，]")
        default_personal_author = getinivaluefromcloud("joplinai", "default_personal_author")

        # 初始化
        metadata = {'note_author': default_personal_author, 'note_type': '个人笔记'}

        # —— 配置：定义共享笔记本列表和同事列表（可从配置读取）——
        if (shared_nb_titles_str := getinivaluefromcloud("joplinai", "shared_notebook_titles")):
            shared_nb_titles = [title.strip() for title in split_ptn.split(shared_nb_titles_str)]
        else:
            shared_nb_titles = ["运营管理", "经销商", "平台商（合作商）"]

        if (colleague_str := getinivaluefromcloud("joplinai", "colleague")):
            colleague = [title.strip() for title in split_ptn.split(colleague_str)]
        else:
            colleague= ["XXA", "XXB"]

        # --- 第0层：收藏判定（最高优先级）---
        collection_kws = ['收藏', '好文', '摘抄', '转载']
        tag_list = [t.strip() for t in split_ptn.split(note_tags)] if note_tags else []
        title_lower = note_title.lower()
        if any(kw in tag_list for kw in collection_kws) or any(kw in title_lower for kw in ['[收藏]', '【收藏】']):
            metadata.update({'note_author': '收藏', 'note_type': '收藏'})
            return metadata  # 直接返回，不再判断其他类型

        # --- 第1层：记录类型判定（会议、谈话）---
        if any(t in tag_list for t in ['会议记录', '会议']) or '会议纪要' in note_title:
            metadata['note_type'] = '会议记录'
        elif any(t in tag_list for t in ['谈话记录', '谈话', '沟通']) or '谈话记录' in note_title:
            metadata['note_type'] = '谈话记录'

        # --- 第2层：作者判定（复用并优化原有三层策略）---
        # 2.1 标签 author_
        if note_tags:
            tag_list = [tag.strip() for tag in split_ptn.split(note_tags)]
            for tag in tag_list:
                if tag.startswith("author_"):
                    author_candidate = tag[7:]  # 移除 "author_"
                    if author_candidate and author_candidate in colleague:
                        metadata['note_author'] = f'{author_candidate}'
                    break

        # 2.2 标题解析 (姓名) 或 [姓名] （如果作者还未被标签确定）
        if metadata['note_author'] == default_personal_author:  # 默认状态才解析标题
            bracket_pattern = re.compile(r"[（【\(\[]([\u4e00-\u9fa5]{2,4})[）】\)\]]")
            match = bracket_pattern.search(note_title)
            if match:
                author_candidate = match.group(1)
                if author_candidate in colleague:
                    metadata['note_author'] = f'{author_candidate}'

        # 2.3 笔记本上下文判定（如果作者仍为默认，且笔记本是共享的）
        if metadata['note_author'] == default_personal_author and source_notebook_title in shared_nb_titles:
            metadata.update({'note_author': '团队_共同维护', 'note_type': '团队协作'})

        return metadata

# %% [markdown]
# ## split_into_semantic_chunks(self, text: str, note_title: str = "", note_tags: str = "", source_notebook_title: str = "", twice_probe: bool = True) -> List[Dict]

    # %%
    def split_into_semantic_chunks(
        self,
        text: str,
        note_title: str = "",
        note_tags: str = "",
        source_notebook_title: str = "",
        twice_probe: bool = True
    ) -> List[Dict]:
        """将文本分割成语义块，并返回块字典列表。
        【增强】使用统一的正则表达式处理带`###`或不带的日期行，确保日期保留在块头部并被提取。
        """
        if not text:
            return []

        chunks = []
        # ========== 第一：统一按日期行进行一级分割 ==========
        # 使用类常量 UNIFIED_DATE_PATTERN
        date_matches = list(self.UNIFIED_DATE_PATTERN.finditer(text))

        if not date_matches:
            log.debug(
                f"笔记《{note_title}》未检测到任何日期标题行，采用通用章节分块策略。"
                "优先章节，其次各级标题。"
            )
            # 使用类常量 GENERAL_SECTION_PATTERN
            major_sections = self.GENERAL_SECTION_PATTERN.split(text)
            major_sections = [s.strip() for s in major_sections if s.strip()]
            chunks = major_sections
            log.debug(
                f"笔记《{note_title}》完成章节分块，共得到 {len(chunks)} 个块。"
            )
        else:
            log.debug(
                f"笔记《{note_title}》检测到 {len(date_matches)} 个日期标题行（含###或不含），"
                "将按此分割。"
            )
            for i, match in enumerate(date_matches):
                date_line_start = match.start()
                captured_date = match.group(1)  # 正则捕获的纯日期，如“2026年4月3日”
                next_start = (
                    date_matches[i + 1].start()
                    if i + 1 < len(date_matches)
                    else len(text)
                )

                # 1. 提取原始日期单元字符串
                raw_day_unit = text[date_line_start:next_start]

                # 2. 【核心】规范化此日期单元
                # 目标：统一格式，消除因文本全局位置变化带来的边界差异
                normalized_unit = self._normalize_single_date_unit(
                    raw_day_unit, captured_date
                )

                # 3. 将规范化后的单元加入列表
                chunks.append(normalized_unit)
            # 处理第一个日期行之前可能存在的文本（如笔记开头的说明）
            if date_matches[0].start() > 0:
                preface = text[: date_matches[0].start()].strip()
                if preface:
                    chunks.insert(0, preface)  # 将前言作为第一个块
            # 利用“日期倒序更新”特征，反转后，最早的日期块索引永远为0
            chunks.reverse()
            log.debug(
                f"笔记《{note_title}》完成日期单元规范化与列表反转，共得到 {len(chunks)} 个块。"
            )
        # ========== 第一步结束 ==========

        final_chunks = []  # 存储所有最终文本块
        for idx, raw_chunk in enumerate(chunks, 1):
            # 在分割逻辑中，对每个 raw_chunk 进行转换，当下仅针对《健康运动笔记》
            converted_chunk = self._convert_health_data_to_text(raw_chunk)
            converted_chunk = self._condense_dense_lists(converted_chunk)
            converted_chunk = converted_chunk.strip()
            # 1. 提取该日期单元的日期 (仍使用 UNIFIED_DATE_PATTERN 搜索当前块)
            date_match = self.UNIFIED_DATE_PATTERN.search(converted_chunk)
            unit_date = date_match.group(1) if date_match else ""

            context_splitter = ContextAwareSplitter(
                max_chunk_size=self.chunk_size,
                min_chunk_size=100,  # 保持原有参数
                target_overlap_sentences=2,
                fallback_overlap_chars=100,
            )
            # 2. 三档阈值判断是否需要迭代分块
            if len(converted_chunk) <= self._safe_net_chars:
                # 第1档：保守阈值以下 → 直接注入（快路径，无探测开销）
                chunk_with_context = context_splitter._inject_context(
                    converted_chunk, note_title, unit_date
                )
                final_chunks.append(chunk_with_context)
                self._try_embed_chunk(chunk_with_context)
            elif (self.enable_adaptive_chunking
                  and len(converted_chunk) <= int(self.chunk_size * 0.9)):
                # 第2档：灰色地带 → 用探测结果争取更大直通块
                safe_len = self.adaptive_optimizer.probe_max_safe_length(
                    converted_chunk, self.model_name
                )
                if len(converted_chunk) <= int(safe_len * 0.9):
                    chunk_with_context = context_splitter._inject_context(
                        converted_chunk, note_title, unit_date
                    )
                    final_chunks.append(chunk_with_context)
                    self._try_embed_chunk(chunk_with_context)
                else:
                    sub_chunks = self._iterative_chunking(
                        converted_chunk, note_title, unit_date
                    )
                    final_chunks.extend(sub_chunks)
            else:
                # 第3档：确定超长 → 统一迭代分块
                sub_chunks = self._iterative_chunking(
                    converted_chunk, note_title, unit_date
                )
                final_chunks.extend(sub_chunks)

        # 如果经过上述步骤，分块结果仍然不理想，启用最终回退
        if not final_chunks or (
            len(final_chunks) == 1 and len(final_chunks[0]) > self.chunk_size * 1.1
        ):
            log.debug(f"按照语义拆分不太合格：{[len(chunk) for chunk in final_chunks]}")
            final_chunks = self._split_into_paragraphs_chunks(text)
            log.debug(
                f"回退用段落甚至字符拆分后：{[len(chunk) for chunk in final_chunks]}"
            )

        # ========== 构建块字典和元数据 ==========
        chunk_dicts = []
        block_number = 1  # 【新增】用于为有效块生成连续索引，从1开始
        # 使用类常量 DATE_IN_CONTEXT_PATTERN 提取最终块的日期
        for idx, chunk_content in enumerate(final_chunks, 1):
            estimated_date = ""
            date_in_chunk = self.DATE_IN_CONTEXT_PATTERN.search(chunk_content.strip())
            if date_in_chunk:
                extracted_date = date_in_chunk.group(1)
                # 【核心调用】使用相同的通用日期规范化函数
                estimated_date = self._normalize_date_string(extracted_date)
            # 提取作者信息
            note_meta = self._extract_author_from_note(
                note_title,
                note_tags,
                source_notebook_title
            )

            # 清理内容格式
            content = self.clean_text(chunk_content)
            if not self._is_valid_chunk(content):
                log.info(
                    f"跳过无效文本块（索引【{idx}/{len(final_chunks)}】，"
                    f"该文本块清理后长度{len(content)}字符）。"
                    f"内容预览: '{content[:50]}'"
                )
                continue  # 跳过当前循环，不处理此块

            # === 计算此块的内容哈希 ===
            chunk_hash = compute_content_hash(chunk_content)
            chunk_metadata = {
                "chunk_index": block_number,
                "source_notebook_title": source_notebook_title,
                "source_note_title": note_title,
                "source_note_tags": note_tags,
                "estimated_date": estimated_date,
                "word_count": len(chunk_content),
                "content_hash": chunk_hash,
                "note_author": note_meta['note_author'],
                "note_type": note_meta['note_type'],
            }
            # DeepSeek 增强生成摘要和标签
            enhanced_metadata = {}
            try:
                enhanced_metadata = self.enhance_by_deepseek_for_summary_tags(
                    chunk_content, note_tags, self.config,
                )
            except Exception as e:
                log.error(
                    f"对笔记《{note_title}》的块 {block_number} "
                    f"（长度：{len(chunk_content)}）进行DeepSeek增强时失败: {e}",
                    exc_info=True,
                )
            # tags = list(set([t.strip() for t in note_tags] + [t.strip() for t in enhanced_metadata.get("tags", "")]))
            tags = [t.strip() for t in note_tags.split(",") if t.strip()]
            tags_str = ",".join(sorted(tags)) if tags else ""  # 排序保证一致性
            meta_hash = compute_content_hash(f"{tags_str}{source_notebook_title}")
            enhanced_metadata["meta_hash"] = meta_hash

            metadata = {**chunk_metadata, **enhanced_metadata}
            chunk_key = hashlib.md5(chunk_content.encode("utf-8")).hexdigest()
            chunk_emb = self._chunk_embedding_cache.get(chunk_key)
            chunk_dicts.append({"content": chunk_content, "metadata": metadata, "embedding": chunk_emb})
            block_number += 1  # 只有有效块才递增

        log.info(f"已将笔记《{note_title}》文本分割成 {len(chunk_dicts)} 个有效的语义块。")
        # print(chunk_dicts)
        return chunk_dicts

# %% [markdown]
# ## _split_into_paragraphs_chunks(self, text: str, target_size: int = None) -> List[str]

    # %%
    def _split_into_paragraphs_chunks(
        self, text: str, target_size: int = None
    ) -> List[str]:
        """按段落和句子分割文本，确保块大小更均匀、安全。"""
        if not text:
            return []
        if target_size is None:
            target_size = self.chunk_size
        # 1. 按双换行符分割段落
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current_chunk = ""

        SAFETY_FACTOR = 0.9  # 更保守的阈值，避免块接近上限
        target_size = int(target_size * SAFETY_FACTOR)

        for para in paragraphs:
            # 如果单个段落就超过安全阈值，需要按句子进一步分割
            if len(para) > target_size:
                # 先保存已积累的块
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                # 对长段落进行句子级分割
                sentences = re.split(r"(?<=[。！？；\n])", para)
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    if len(current_chunk) + len(sent) <= target_size:
                        current_chunk += sent if not current_chunk else " " + sent
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sent
            # 如果段落可以安全加入当前块
            elif len(current_chunk) + len(para) <= target_size:
                current_chunk += para if not current_chunk else "\n\n" + para
            else:
                # 当前块已满，保存并开始新块
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para

        # 添加最后一个块
        if current_chunk:
            chunks.append(current_chunk.strip())

        # 最终检查：如果仍有块过大，启用最终回退（但应尽量避免走到这一步）
        if any(len(ch) > target_size for ch in chunks):
            log.warning("段落分割后仍存在过大块，启用字符级回退。")
            return self._split_text_into_chunks_fallback(text, target_size)

        return chunks

# %% [markdown]
# ## _split_text_into_chunks_fallback(self, text: str, target_size = None) -> List[str]

    # %%
    def _split_text_into_chunks_fallback(
        self, text: str, target_size=None
    ) -> List[str]:
        """原有的基于字符的智能分块逻辑（作为回退）
        基于字符位置分块，保留完整单词"""
        if target_size is None:
            target_size = self.chunk_size
        if len(text) <= target_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            # 计算本块的结束位置
            end = start + target_size

            # 如果已经到文本末尾
            if end >= len(text):
                chunks.append(text[start:])
                break

            # 寻找合适的断点（空格或标点）
            # 向前找最近的空格
            while end > start and text[end] not in " \t\n.,;!?":
                end -= 1

            # 如果没找到空格，强制在chunk_size处截断
            if end == start:
                end = start + target_size

            # 添加块
            chunk = text[start:end].strip()
            if chunk:  # 避免添加空块
                chunks.append(chunk)

            # 更新起始位置
            start = end

        return chunks

# %% [markdown]
# ## get_ollama_embedding_other(self, text: str) -> List[float]
    # %%
    def get_ollama_embedding_other(self, text: str) -> List[float]:
        """
        调用 Ollama 生成文本嵌入。
        注意：传入的 text 应是已分块的适当长度文本。
        直接调用模型，不加try包裹，方便探测，错误处理由调用方负责
        """
        response = ollama.embeddings(model=self.model_name, prompt=text)
        return response["embedding"]

# %% [markdown]
# ## get_ollama_embedding(self, text: str, host: str = "10.9.0.2", port: int = 11434)

    # %%
    def get_ollama_embedding(
        self, text: str, host: str = "10.9.0.2", port: int = 11034
    ):
        """调用远程恒创云Ollama生成嵌入"""
        host = self.config.get("ollama_host", host)
        port = self.config.get("ollama_port", port)
        url = f"http://{host}:{port}/api/embeddings"
        model = self.model_name
        # print(host, port, model)
        payload = {"model": model, "prompt": text}
        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()["embedding"]
        except Exception as e:
            # 从响应体中提取 Ollama 详细错误信息（如"input length exceeds context length"）
            error_msg = str(e)
            if hasattr(e, "response") and e.response is not None:
                try:
                    body = e.response.text
                    if body:
                        error_msg = f"{error_msg} | {body}"
                except Exception:
                    pass
            log.error(f"远程Ollama嵌入调用失败[API-token超限]: {error_msg}")
            raise Exception(error_msg) from e

# %% [markdown]
# ## get_cached_embedding(self, text_hash: str) -> Optional[List[float]]

    # %%
    @lru_cache(maxsize=200)
    def get_cached_embedding(self, text_hash: str) -> Optional[List[float]]:
        """缓存嵌入结果"""
        return self.embedding_cache.get(text_hash)

# %% [markdown]
# ## get_merged_embedding(self, text: str,) -> List[float]
    # %%
    def get_merged_embedding(
        self,
        chunk_dict: Dict,
    ) -> List[float]:
        text = chunk_dict["content"]
        text_hash = chunk_dict["base_metadata"]["content_hash"]
        source_note_title = chunk_dict["base_metadata"]["source_note_title"]
        chunk_index = chunk_dict["base_metadata"]["chunk_index"]

        # 分块阶段已预生成嵌入，直接取用
        chunk_emb = chunk_dict.get("embedding")
        if chunk_emb is not None:
            self.embedding_cache[text_hash] = chunk_emb  # 同步至主缓存
            log.info(
                f"笔记《{source_note_title}》的文本块【{chunk_index}】分块缓存击中"
            )
            return chunk_emb

        # 检查缓存
        cached = self.get_cached_embedding(text_hash)
        if cached:
            log.info(
                f"笔记《{source_note_title}》的文本块【{chunk_index}】缓存击中: {text_hash[:12]}"
            )
            return cached

        # 在尝试本地Ollama嵌入前，对文本进行一次嵌入预处理
        processed_text = self._preprocess_text_for_embedding(text)
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                # 调用安全的嵌入生成
                # embedding = self.get_ollama_embedding_safe(processed_text)
                embedding = self.get_ollama_embedding(processed_text)
                if embedding:
                    break
            except Exception as e:
                if attempt == max_retries:
                    # 多次重试仍失败，判断是否为长度错误
                    error_msg = str(e).lower()
                    is_length_error = any(
                        kw in error_msg
                        for kw in ["context length", "input length", "too long", "exceed", "500"]
                    )
                    if is_length_error:
                        # 句子边界感知截断：取安全上限内完整句子，而非 crude [:N]
                        safe_chars = int(self.chunk_size * 0.9)
                        truncated = processed_text[:safe_chars]
                        for i in range(len(truncated) - 1, safe_chars // 2, -1):
                            if truncated[i] in '。！？.!?\n':
                                truncated = truncated[:i+1]
                                break
                        log.warning(
                            f"笔记《{source_note_title}》的文本块【{chunk_index}】"
                            f"长度({len(text)}字符)5次重试均token超限，"
                            f"句子边界截断至{len(truncated)}字符（安全上限={safe_chars}字符）"
                        )
                        try:
                            embedding = self.get_ollama_embedding(truncated)
                        except Exception as e2:
                            log.error(f"截断回退嵌入仍失败: {e2}")
                            return []
                    else:
                        # 非长度错误（如Ollama不可用），不触发重分块
                        log.error(
                            f"笔记《{source_note_title}》的文本块【{chunk_index}】"
                            f"嵌入最终失败（非长度错误）: {e}"
                        )
                        return []
                    break
                else:
                    log.warning(f"获取嵌入失败(第{attempt}次): {e}")
                continue

        if not embedding:
            log.error(f"为文本生成嵌入最终失败，将返回空列表。")
            return []
        self.embedding_cache[text_hash] = embedding

        return embedding
