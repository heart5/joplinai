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
# # 导入库

# %%
import hashlib
import logging
import re
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import ollama
import requests

try:
    from func.datatools import compute_content_hash
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

    def __init__(self, embedding_generator, enabled=False, cache_size=100):
        """
        初始化优化器。
        :param embedding_generator: EmbeddingGenerator 实例，用于实际调用嵌入接口。
        :param enabled: 是否启用自适应分块。
        :param cache_size: 缓存大小，避免重复探测。
        """
        self.embedding_generator = embedding_generator
        self.enabled = enabled
        # 缓存：键为 (模型名, 文本特征哈希)，值为探测出的最佳块大小（字符数）
        self.cache = {}
        self.cache_size = cache_size

    def _get_text_signature(self, text: str) -> str:
        """
        生成文本的简易特征签名，用于缓存键。
        基于文本长度、开头和结尾部分，平衡识别精度与计算开销。
        """
        import hashlib

        # 取文本长度、前200字符和后200字符（如果存在）的组合
        prefix = text[:200] if len(text) > 200 else text
        suffix = text[-200:] if len(text) > 200 else text
        # 生成一个简短的哈希作为签名
        sig_content = f"{len(text)}|{prefix}|{suffix}"
        signature = hashlib.md5(sig_content.encode("utf-8")).hexdigest()[
            :12
        ]  # 取前12位
        return signature

    def probe_max_safe_length(self, text: str, model_name: str) -> int:
        """
        探测针对此文本，模型能安全处理的最大长度（字符数）。
        采用“指数增长”法：从保守长度开始，每次翻倍，直到嵌入调用失败（抛出长度相关错误）。
        返回最后一次成功的长度。
        """
        if not self.enabled:
            # 功能未启用，直接返回默认 chunk_size
            return self.embedding_generator.chunk_size

        cache_key = (model_name, self._get_text_signature(text))
        if cache_key in self.cache:
            log.debug(f"自适应分块缓存命中: {cache_key} -> {self.cache[cache_key]}")
            return self.cache[cache_key]

        log.info(f"开始自适应分块探测: 模型={model_name}, 文本长度={len(text)}字符")

        # 1. 定义探测参数
        # start_len = 256  # 起始探测长度，一个比较安全的保守值
        start_len = int(self.embedding_generator.chunk_size * 0.8)
        max_len = len(text)  # 理论最大值
        growth_factor = 1.1  # 每次增长倍数
        current_safe_len = start_len

        # 2. 指数增长阶段：快速找到失败边界
        test_len = start_len
        while test_len <= max_len:
            test_text = text[:test_len]  # 取文本前 test_len 个字符进行测试
            try:
                # 该方法内部已有重试和容错机制，足以用于探测。
                _ = self.embedding_generator.get_ollama_embedding(test_text)
                # 调用成功，更新安全长度记录
                current_safe_len = test_len
                log.debug(f"  探测通过: {test_len} 字符")
                # 指数增长
                test_len = int(test_len * growth_factor)
            except Exception as e:
                error_msg = str(e).lower()
                # 判断是否为长度相关错误（模型上下文长度限制）
                if any(
                    keyword in error_msg
                    for keyword in [
                        "context length",
                        "input length",
                        "too long",
                        "exceed",
                    ]
                ):
                    log.debug(
                        f"  探测失败（长度限制）: {test_len} 字符, 错误: {error_msg[:100]}"
                    )
                    # 遇到长度错误，停止增长，current_safe_len 即为最后一次成功的长度
                    break
                else:
                    # 其他类型的错误（如网络问题、参数错误），保守处理，停止探测并使用默认 chunk_size
                    log.warning(f"  探测过程中遇到错误，停止探测并使用默认值: {e}")
                    current_safe_len = (
                        self.embedding_generator.chunk_size
                    )  # 修正：失败时直接回退到全局默认值
                    break

        # 3. 确保结果在合理范围内
        # 最终安全长度不应小于起始长度，也不应大于默认 chunk_size 的某个倍数（如2倍），避免极端值。
        final_safe_len = max(
            start_len,
            min(
                current_safe_len,
                self.embedding_generator.chunk_size * 2,
            ),
            self.embedding_generator.chunk_size,
        )
        log.info(
            f"自适应分块探测完成: 建议块大小={final_safe_len} 字符 (原始默认={self.embedding_generator.chunk_size})"
        )

        # 4. 缓存结果
        if len(self.cache) >= self.cache_size:
            # 简单淘汰最早插入的一项
            self.cache.pop(next(iter(self.cache)))
        self.cache[cache_key] = final_safe_len

        return final_safe_len

    def get_adaptive_chunk_size(self, text: str) -> int:
        """
        主接口：获取针对当前文本的自适应分块大小。
        :param text: 待分块的完整文本。
        :return: 建议的分块大小（字符数）。
        """
        return self.probe_max_safe_length(text, self.embedding_generator.model_name)


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
        max_chunk_size: int = 768,
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

        log.info(
            f"文本【{repr(text[:20])}……】长度为({len(text)})，被增强型语义切割工具切割为({len(chunks)})块"
        )
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
# # EmbeddingGenerator类


# %%
class EmbeddingGenerator:
    """嵌入生成器，支持长文本分块处理"""

# %% [markdown]
# ## \_\_init__(self, model_name: str, chunk_size: int = 1024)

    # %%
    def __init__(
        self,
        model_name: str,
        chunk_size: int = 1024,
        enable_adaptive_chunking=False,  # 【新增】是否启用自适应分块
        chunk_overlap=50,
        adaptive_cache_size=100,  # 【新增】自适应探测缓存大小
    ):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_dim = self._get_model_dimension()
        self.embedding_cache = {}  # 简单缓存
        self._set_chunk_size()
        # 【新增】初始化自适应分块优化器
        self.adaptive_optimizer = AdaptiveChunkOptimizer(
            embedding_generator=self,
            enabled=enable_adaptive_chunking,
            cache_size=adaptive_cache_size,
        )
        # 保存状态，便于其他地方判断
        self.enable_adaptive_chunking = enable_adaptive_chunking

# %% [markdown]
# ## _get_model_dimension(self)

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
# ## _set_chunk_size(self) -> None
    # %%
    def _set_chunk_size(self) -> None:
        """精确获取模型上下文限制（token→字符转换）"""
        # 特殊处理已知模型
        if self.model_name == "nomic-embed-text":
            self.chunk_size = 1850  # 768 token × 3字符/token × 0.8余量 ≈ 1850
            return
        elif self.model_name == "dengcao/bge-large-zh-v1.5":
            self.chunk_size = (
                512  # 512 token，中文是一对一，应为是一对三或者四，实际中文为主
            )
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
                f"模型 {self.model_name} 上下文长度: {num_ctx} tokens, 分块大小: {self.chunk_size} 字符"
            )

        except Exception as e:
            log.warning(
                f"获取模型上下文失败({self.model_name})，使用默认值1024字符: {e}"
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
# ## _normalize_single_date_unit(self, raw_text: str, captured_date: str) -> str

    # %%
    def _normalize_single_date_unit(self, raw_text: str, captured_date: str) -> str:
        """
        规范化单个日期单元的文本。
        输入: 原始切片字符串 (如 "### 2026年4月3日\n16938，5：46，3\n...")
        输出: 格式统一的字符串，确保相同日期的输出绝对一致。
        """
        lines = raw_text.splitlines()
        if not lines:
            return ""

        # 1. 规范化标题行：统一为 “### YYYY年MM月DD日” 格式
        # 使用正则捕获的纯日期，避免原文本中“号”或空格的差异
        normalized_header = f"### {captured_date}"

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
# ## get_adaptive_chunk_size(self, text, note_title, idx)

    # %%
    def get_adaptive_chunk_size(self, converted_chunk, note_title, idx):
        # === 【新增】自适应分块决策 ===
        # 决定本次分块使用的目标大小
        current_target_chunk_size = self.chunk_size  # 默认使用全局配置

        try:
            adaptive_size = self.adaptive_optimizer.get_adaptive_chunk_size(
                converted_chunk
            )
            # 使用探测到的值，但限制其在一个合理范围：
            # 下限：不能小于默认值的75%（避免过小）
            # 上限：不能大于默认值的200%（避免过大导致其他问题）
            lower_bound = int(self.chunk_size * 0.75)
            upper_bound = int(self.chunk_size * 2.0)
            clamped_size = max(lower_bound, min(adaptive_size, upper_bound))

            if clamped_size != self.chunk_size:
                current_target_chunk_size = clamped_size
                log.info(
                    f"笔记《{note_title}》的文本块【{idx}】启用自适应分块: "
                    f"文本长度={len(converted_chunk)}, "
                    f"默认大小={self.chunk_size}, 自适应大小={current_target_chunk_size}"
                )
        except Exception as e:
            log.warning(
                f"笔记《{note_title}》的文本块【{idx}】自适应分块决策失败，将使用默认大小: {e}"
            )
            # 失败时，回退到默认大小

        log.debug(
            f"笔记《{note_title}》的文本块【{idx}】"
            f"最终确定的分块目标大小为: {current_target_chunk_size} 字符"
        )
        return current_target_chunk_size

# %% [markdown]
# ## split_into_semantic_chunks(self, text: str, note_title: str = "", note_tags: str = "") -> List[Dict]

    # %%
    def split_into_semantic_chunks(
        self, text: str, note_title: str = "", note_tags: str = ""
    ) -> List[Dict]:
        """将文本分割成语义块，并返回块字典列表。
        【增强】使用统一的正则表达式处理带`###`或不带的日期行，确保日期保留在块头部并被提取。
        """
        if not text:
            return []

        chunks = []  # 存储初步分割的文本块

        # ========== 第一步：统一按日期行进行一级分割 ==========
        # 核心优化：使用一个正则，同时匹配“### 日期”和“日期”两种格式，并捕获日期部分
        # 正则解释：
        # ^                    : 行的开始（配合 re.MULTILINE）
        # (?:###\s*)?          : 非捕获组，匹配可选的“###”及可能跟随的空格
        # (\d{4}[-年/]\d{1,2}[-月/]\d{1,2}日?) : 捕获组1，匹配核心日期格式
        # \s*$                 : 行尾可能有的空格
        # 支持格式示例：
        #   “### 2024年1月1日”
        #   “2024年1月1日”
        #   “2024-01-01”
        #   “2024/01/01”
        unified_date_pattern = re.compile(
            r"^(?:###\s*)?(\d{4}[-年/]\d{1,2}[-月/]\d{1,2}[日号])\s*$", re.MULTILINE
        )
        date_matches = list(unified_date_pattern.finditer(text))

        if not date_matches:
            # 如果没有找到任何日期行，回退到原有的按章节分割逻辑
            log.debug("笔记未检测到任何日期标题行，回退至通用章节分块。")
            major_sections = re.split(r"\n(?:#{1,3}\s+.*?|\-{3,})\n", text)
            major_sections = [s.strip() for s in major_sections if s.strip()]
            chunks = major_sections
        else:
            # 找到日期行，统一按日期行分割，并确保日期行保留在块内
            log.debug(
                f"检测到 {len(date_matches)} 个日期标题行（含###或不含），将按此分割。"
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

        # 后续逻辑保持不变：对每个初步分割出的块，检查大小，如果过大则进行二次分割。
        final_chunks = []
        for idx, raw_chunk in enumerate(chunks, 1):
            # 在分割逻辑中，对每个 raw_chunk 进行转换，当下仅针对《健康运动笔记》
            converted_chunk = self._convert_health_data_to_text(raw_chunk)
            converted_chunk = self._condense_dense_lists(converted_chunk)
            if len(converted_chunk) <= int(self.chunk_size * 1.1):
                # 如果块大小合理，直接使用
                final_chunks.append(converted_chunk)
            else:
                if (
                    self.enable_adaptive_chunking
                    and len(converted_chunk) > self.chunk_size * 1.1
                ):
                    # 仅当文本长度显著超过默认大小时，才启动自适应探测，避免对小文本的开销
                    current_target_chunk_size = self.get_adaptive_chunk_size(
                        converted_chunk, note_title, idx
                    )
                    # if current_target_chunk_size > self.chunk_size:
                    #     current_target_chunk_size = self.chunk_size
                    # sub_chunks = self._split_into_paragraphs_chunks(converted_chunk, current_target_chunk_size)
                    log.info(
                        f"笔记《{note_title}》的文本块【{idx}】"
                        f"初步块过长({len(converted_chunk)}字符)，"
                        f"通过自适应获取合适的chunk_size为({current_target_chunk_size})，"
                        f"通过滑动窗口分块法进行二次语义分割。"
                    )
                    sub_chunks = PunctuationAwareSplitter(
                        max_chunk_size=current_target_chunk_size
                    ).split(converted_chunk)
                else:
                    # 如果块过大，则调用原有的段落/句子级分割函数进行细化
                    log.debug(
                        f"笔记《{note_title}》的文本块【{idx}】"
                        f"初步块过长({len(converted_chunk)}字符)，进行二次语义分割。"
                    )
                    sub_chunks = PunctuationAwareSplitter(
                        max_chunk_size=self.chunk_size
                    ).split(converted_chunk)
                    # sub_chunks = self._split_into_paragraphs_chunks(converted_chunk)
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
        valid_chunk_counter = 0  # 【新增】用于为有效块生成连续索引
        # 复用第一步的统一正则进行日期提取，确保一致性
        for idx, chunk_content in enumerate(final_chunks):
            estimated_date = ""
            # 优先从块的开头匹配日期（因为分割逻辑已保证日期行在头部）
            # 再次使用相同的 unified_date_pattern，但用 match 从开头搜索
            date_at_start = unified_date_pattern.match(chunk_content.strip())
            if date_at_start:
                estimated_date = date_at_start.group(1)
            else:
                # 如果开头没有（例如是前言块），则在块内搜索第一个日期作为估算
                date_in_chunk = unified_date_pattern.search(chunk_content)
                if date_in_chunk:
                    estimated_date = date_in_chunk.group(1)

            # 清理内容格式
            content = self.clean_text(chunk_content)
            if not self._is_valid_chunk(content):
                log.info(
                    f"跳过无效文本块（索引【{idx}/{len(final_chunks)}】，该文本块清理后长度{len(content)}字符）。内容预览: '{content[:50]}'"
                )
                continue  # 跳过当前循环，不处理此块

            # === 计算此块的内容哈希 ===
            chunk_hash = compute_content_hash(chunk_content)
            chunk_metadata = {
                "chunk_index": valid_chunk_counter,
                "source_note_title": note_title,
                "source_note_tags": note_tags,
                "estimated_date": estimated_date,
                "word_count": len(chunk_content),
                # === 将哈希存入元数据 ===
                "content_hash": chunk_hash,
            }
            chunk_dicts.append({"content": content, "metadata": chunk_metadata})
            valid_chunk_counter += 1  # 只有有效块才递增

        log.info(f"将文本分割成 {len(chunk_dicts)} 个语义块。")
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
# ## get_ollama_embedding(self, text: str) -> List[float]
    # %%
    def get_ollama_embedding(self, text: str) -> List[float]:
        """
        调用 Ollama 生成文本嵌入。
        注意：传入的 text 应是已分块的适当长度文本。
        直接调用模型，不加try包裹，方便探测，错误处理由调用方负责
        """
        response = ollama.embeddings(model=self.model_name, prompt=text)
        return response["embedding"]

# %% [markdown]
# ## get_cached_embedding(self, text_hash: str) -> Optional[List[float]]

    # %%
    @lru_cache(maxsize=200)
    def get_cached_embedding(self, text_hash: str) -> Optional[List[float]]:
        """缓存嵌入结果"""
        return self.embedding_cache.get(text_hash)

# %% [markdown]
# ## _get_rechunked_embedding(self, original_text: str, safe_subchunk_size: int) -> List[float]

    # %%
    def _get_rechunked_embedding(
        self, original_text: str, safe_subchunk_size: int
    ) -> List[float]:
        """
        核心重分块逻辑。
        原理：将超长文本按安全大小重新分割为多个语义子块，分别嵌入后合并。
        """
        import numpy as np

        # 1. 使用安全大小进行语义重分块
        sub_chunks = self._split_with_custom_size(
            text=original_text, target_chunk_size=safe_subchunk_size
        )
        if not sub_chunks:
            log.error("重分块未能产生有效子块")
            return []

        log.info(
            f"📊 重分块详情 | 原块: {len(original_text)}字符 | 子块数: {len(sub_chunks)} | 目标大小: {safe_subchunk_size}字符"
        )

        # 2. 为每个子块生成嵌入
        sub_embeddings = []
        for i, chunk_content in enumerate(sub_chunks):
            try:
                # 递归调用 get_ollama_embedding，由于子块已很小，通常不会再次触发重分块
                emb = self.get_ollama_embedding(chunk_content)
                if emb:
                    sub_embeddings.append(emb)
                    log.debug(f"子块 {i + 1}/{len(sub_chunks)} 嵌入成功")
                else:
                    log.warning(f"子块 {i + 1} 嵌入返回空，跳过")
            except Exception as e:
                log.warning(f"子块 {i + 1} 嵌入异常: {str(e)[:50]}，跳过")
                continue

        # 3. 合并嵌入向量 (平均池化)
        if not sub_embeddings:
            log.error("所有子块嵌入均失败")
            return []
        if len(sub_embeddings) == 1:
            return sub_embeddings

        merged_embedding = np.mean(sub_embeddings, axis=0)
        log.info(f"合并 {len(sub_embeddings)} 个子块嵌入成功")
        return merged_embedding.tolist()

# %% [markdown]
# ## _split_with_custom_size(self, text: str, target_chunk_size: int) -> List[str]

    # %%
    def _split_with_custom_size(self, text: str, target_chunk_size: int) -> List[str]:
        """
        使用自定义目标大小进行语义分割。
        优先复用现有分块逻辑，临时调整 chunk_size 参数。
        """
        original_chunk_size = self.chunk_size
        try:
            self.chunk_size = target_chunk_size
            # 调用您现有的语义分块方法，它会使用临时的 chunk_size
            chunk_dicts = self.split_into_semantic_chunks(
                text=text, note_title="[重分块]", note_tags=""
            )
            sub_chunks = [chunk["content"] for chunk in chunk_dicts]

            # 二次检查：如果语义分块后仍有块过大，进行句子级分割
            final_chunks = []
            for chunk in sub_chunks:
                if len(chunk) > target_chunk_size * 1.2:
                    log.debug(f"子块仍过大({len(chunk)}字符)，进行句子级分割")
                    sentences = self._split_into_sentences(chunk)
                    current_chunk = ""
                    for sent in sentences:
                        if len(current_chunk) + len(sent) > target_chunk_size:
                            if current_chunk:
                                final_chunks.append(current_chunk.strip())
                            current_chunk = sent
                        else:
                            current_chunk += "" + sent if current_chunk else sent
                    if current_chunk:
                        final_chunks.append(current_chunk.strip())
                else:
                    final_chunks.append(chunk)
            return final_chunks
        finally:
            self.chunk_size = original_chunk_size  # 恢复原状

# %% [markdown]
# ## _split_into_sentences(self, text: str) -> List[str]

    # %%
    def _split_into_sentences(self, text: str) -> List[str]:
        """简单的中文句子分割"""
        import re

        sentence_endings = r"[。！？；\n]"
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]

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

        # 检查缓存
        cached = self.get_cached_embedding(text_hash)
        if cached:
            log.info(
                f"笔记《{source_note_title}》的文本块【{chunk_index}】缓存击中: {text_hash[:12]}"
            )
            return cached

        # 2. 长度安全校验（防御性编程）
        # safe_limit = int(self.chunk_size * 0.9) # 安全阈值
        # if len(text) > safe_limit:
        #     # 立即触发二次分块，而非缩减
        #     log.info(
        #         f"笔记《{chunk_dict["base_metadata"]["source_note_title"]}》的"
        #         f"文本块【{chunk_dict["base_metadata"]["chunk_index"]}】"
        #         f"长度为({len(text)}，超过安全长度({safe_limit}))，开始执行二次分块嵌入……"
        #     )
        #     return self._get_rechunked_embedding(text, safe_subchunk_size=int(self.chunk_size*0.5))

        # 在尝试本地Ollama嵌入前，对文本进行一次嵌入预处理
        processed_text = self._preprocess_text_for_embedding(text)
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                # 调用安全的嵌入生成
                # embedding = self.get_ollama_embedding_safe(processed_text)
                embedding = self.get_ollama_embedding(processed_text)
                if embedding:
                    break
            except Exception as e:
                if attempt == max_retries:
                    # 尝试到最后一次，则进行二次分块，兜底
                    log.info(
                        f"笔记《{source_note_title}》的文本块【{chunk_index}】"
                        f"长度为({len(text)})，尝试嵌入操作到了第({attempt})次，仍然失败"
                        "，开始执行二次分块嵌入……"
                    )
                    embedding = self._get_rechunked_embedding(
                        processed_text, safe_subchunk_size=int(self.chunk_size * 0.5)
                    )
                    break
                else:
                    log.warning(f"获取嵌入失败(第{attempt}次): {e}")
                continue

        if not embedding:
            log.error(f"为文本生成嵌入最终失败，将返回空列表。")
            return []
        self.embedding_cache[text_hash] = embedding

        return embedding
