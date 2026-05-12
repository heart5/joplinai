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
# # 文本分块器
# PunctuationAwareSplitter + ContextAwareSplitter — 标点与结构感知的文本分块。

# %%
import logging
import re
from typing import List

# %%
import pathmagic

with pathmagic.Context():
    try:
        from func.logme import log
    except ImportError as e:
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger(__name__)
        log.error(f"导入项目模块失败: {e}")


# %%
__all__ = ["PunctuationAwareSplitter", "ContextAwareSplitter"]

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
                # 这保证了在宏观结构内，依然遵循"语义->句子->硬切"的保底逻辑。
                sub_chunks = self._split_recursively(primary_chunk)
                final_chunks.extend(sub_chunks)

        log.info(
            f"[语义切割] 文本【{repr(text[:30])}……】总长{len(text)}字符，"
            f"切割为{len(final_chunks)}块，"
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
