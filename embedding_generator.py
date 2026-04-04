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
# ### clean_text(text: str) -> str
    # %%
    def clean_text(self, text: str) -> str:
        """清理笔记文本：移除图片、格式符号、多余换行"""
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
# ### _convert_health_data_to_text(self, raw_content: str) -> str

    # %%
    def _convert_health_data_to_text(self, raw_content: str) -> str:
        """
        将健康笔记中的结构化数据行转换为自然语言描述。
        示例输入: "110，4：14" -> "今日步数110步，睡眠时长4小时14分钟。"
        示例输入: "799，7：44，1" -> "今日步数799步，睡眠时长7小时44分钟，喝啤酒1瓶。"
        """
        lines = raw_content.strip().split('\n')
        converted_lines = []
        
        for line in lines:
            line = line.strip()
            # 匹配数字模式：如 "110，4：14" 或 "11033，4：7，4"
            if re.match(r'^\d+[，,]\s*\d+[:：]\d+([，,]\s*\d+)?$', line):
                parts = re.split(r'[，,]\s*', line)
                if len(parts) >= 2:
                    # 解析步数
                    steps = parts
                    desc = f"今日步数{steps}步，"
                    
                    # 解析睡眠时间（格式如 4:14 或 4：14）
                    sleep_time = parts[1].replace('：', ':')
                    if ':' in sleep_time:
                        sleep_parts = sleep_time.split(':')
                        if len(sleep_parts) == 2:
                            desc += f"睡眠时长{sleep_parts[0]}小时{sleep_parts[1]}分钟"
                    
                    # 解析啤酒数量（如果有）
                    if len(parts) >= 3 and parts[2].isdigit():
                        beer_count = parts
                        desc += f"，喝啤酒{beer_count}瓶"
                    
                    line = desc + "。"
            converted_lines.append(line)
        
        return '\n'.join(converted_lines)

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
        # 1. 直接截取前400个字符（保证极短）
        safe_length = 400
        len_text = len(text)
        if len_text > safe_length:
            text = text[:safe_length]
            log.debug(f"激进缩减，从{len_text}缩减至{safe_length}个字符")

        # 2. 【新增】针对“人名列表”的特殊处理
        # 检测模式：包含大量顿号、冒号，且无明显段落
        if '：' in text and '、' in text and len(text.splitlines()) < 5:
            log.debug(f"检测到密集人名列表，进行针对性清理。该文本块头为：{text[:200]} ……")
            # 移除所有空格和换行，简化格式
            text = text.replace('\n', '').replace(' ', '')
            # 将顿号替换为逗号（可能token更少）
            text = text.replace('、', ',')
            # 如果仍然过长，只保留前N个人名
            if len(text) > safe_length:
                # 尝试按逗号分割，保留前一部分
                parts = text.split(',')
                if len(parts) > 10:
                    text = ','.join(parts[:10]) + '...（名单截断）'

        # 3. 进一步移除所有重复字符模式
        import re
        text = re.sub(r'(.)\1{2,}', r'\1', text)  # 3个以上相同字符保留1个

        return text

# %% [markdown]
# ### _normalize_single_date_unit(self, raw_text: str, captured_date: str) -> str

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
            normalized_body = '\n'.join(body_lines)
            return f"{normalized_header}\n\n{normalized_body}"

# %% [markdown]
# ### split_into_semantic_chunks(self, text: str, note_title: str = "", note_tags: str = "") -> List[Dict]

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
                next_start = date_matches[i + 1].start() if i + 1 < len(date_matches) else len(text)

                # 1. 提取原始日期单元字符串
                raw_day_unit = text[date_line_start:next_start]

                # 2. 【核心】规范化此日期单元
                # 目标：统一格式，消除因文本全局位置变化带来的边界差异
                normalized_unit = self._normalize_single_date_unit(raw_day_unit, captured_date)

                # 3. 将规范化后的单元加入列表
                chunks.append(normalized_unit)
            # 处理第一个日期行之前可能存在的文本（如笔记开头的说明）
            if date_matches[0].start() > 0:
                preface = text[: date_matches[0].start()].strip()
                if preface:
                    chunks.insert(0, preface)  # 将前言作为第一个块
        # 利用“日期倒序更新”特征，反转后，最早的日期块索引永远为0
        chunks.reverse()
        log.debug(f"笔记《{note_title}》完成日期单元规范化与列表反转，共得到 {len(chunks)} 个块。")
        # ========== 第一步结束 ==========

        # 后续逻辑保持不变：对每个初步分割出的块，检查大小，如果过大则进行二次分割。
        final_chunks = []
        for raw_chunk in chunks:
            # 在分割逻辑中，对每个 raw_chunk 进行转换，当下仅针对《健康运动笔记》
            converted_chunk = self._convert_health_data_to_text(raw_chunk)
            if len(converted_chunk) <= self.chunk_size * 1.1:
                # 如果块大小合理，直接使用
                final_chunks.append(converted_chunk)
            else:
                # 如果块过大，则调用原有的段落/句子级分割函数进行细化
                log.debug(f"初步块过长({len(converted_chunk)}字符)，进行二次语义分割。")
                sub_chunks = self._split_into_paragraphs_chunks(converted_chunk)
                final_chunks.extend(sub_chunks)

        # 如果经过上述步骤，分块结果仍然不理想，启用最终回退
        if not final_chunks or (
            len(final_chunks) == 1 and len(final_chunks[0]) >= self.chunk_size * 1.1
        ):
            log.debug(f"按照语义拆分不太合格：{[len(chunk) for chunk in final_chunks]}")
            final_chunks = self._split_into_paragraphs_chunks(text)
            log.debug(f"回退用段落甚至字符拆分后：{[len(chunk) for chunk in final_chunks]}")

        # ========== 构建块字典和元数据 ==========
        chunk_dicts = []
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
            if len(content) < 10:
                content = note_title
            # === 计算此块的内容哈希 ===
            chunk_hash = hashlib.md5(chunk_content.encode('utf-8')).hexdigest()
            chunk_metadata = {
                "chunk_index": idx,
                "source_note_title": note_title,
                "source_note_tags": note_tags,
                "estimated_date": estimated_date,  # 现在能正确提取
                "word_count": len(chunk_content),
                # === 将哈希存入元数据 ===
                "content_hash": chunk_hash,
            }
            chunk_dicts.append({"content": content, "metadata": chunk_metadata})

        log.info(f"将文本分割成 {len(chunk_dicts)} 个语义块。")
        # print(chunk_dicts)
        return chunk_dicts

# %% [markdown]
# ### split_into_semantic_chunks_other3(self, text: str, note_title: str = "", note_tags: str = "") -> List[Dict]

    # %%
    def split_into_semantic_chunks_other3(
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
            start_pos = 0
            for i, match in enumerate(date_matches):
                date_line_start = match.start()
                date_line_end = match.end()
                # match.group(1) 是捕获的纯日期字符串，如“2024年1月1日”
                current_date = match.group(1)

                # 确定这个日期单元的内容结束位置（下一个日期行开始，或文本末尾）
                next_start = (
                    date_matches[i + 1].start() if i + 1 < len(date_matches) else len(text)
                )

                # 提取从当前日期行开始，到下一个日期行之前的所有内容作为一个“天单元”
                # 这保证了日期行（无论有无###）是此单元的第一行。
                day_unit = text[
                    date_line_start:next_start
                ]  # 注意：这里不strip()，保留原始格式和换行

                # 将这个“天单元”添加到待处理块列表
                chunks.append(day_unit)
                start_pos = next_start

            # 处理第一个日期行之前可能存在的文本（如笔记开头的说明）
            if date_matches[0].start() > 0:
                preface = text[: date_matches[0].start()].strip()
                if preface:
                    chunks.insert(0, preface)  # 将前言作为第一个块
        # ========== 第一步结束 ==========

        # 后续逻辑保持不变：对每个初步分割出的块，检查大小，如果过大则进行二次分割。
        final_chunks = []
        for raw_chunk in chunks:
            # 在分割逻辑中，对每个 raw_chunk 进行转换，当下仅针对《健康运动笔记》
            converted_chunk = self._convert_health_data_to_text(raw_chunk)
            if len(converted_chunk) <= self.chunk_size * 1.1:
                # 如果块大小合理，直接使用
                final_chunks.append(converted_chunk)
            else:
                # 如果块过大，则调用原有的段落/句子级分割函数进行细化
                log.debug(f"初步块过长({len(converted_chunk)}字符)，进行二次语义分割。")
                sub_chunks = self._split_into_paragraphs_chunks(converted_chunk)
                final_chunks.extend(sub_chunks)

        # 如果经过上述步骤，分块结果仍然不理想，启用最终回退
        if not final_chunks or (
            len(final_chunks) == 1 and len(final_chunks[0]) >= self.chunk_size * 1.1
        ):
            log.debug(f"按照语义拆分不太合格：{[len(chunk) for chunk in final_chunks]}")
            final_chunks = self._split_into_paragraphs_chunks(text)
            log.debug(f"回退用段落甚至字符拆分后：{[len(chunk) for chunk in final_chunks]}")

        # ========== 构建块字典和元数据 ==========
        final_chunks.reverse()  # 核心操作：反转列表，确保最早日期的idx稳定
        chunk_dicts = []
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
            if len(content) < 10:
                content = note_title
            # === 计算此块的内容哈希 ===
            chunk_hash = hashlib.md5(chunk_content.encode('utf-8')).hexdigest()
            chunk_metadata = {
                "chunk_index": idx,
                "source_note_title": note_title,
                "source_note_tags": note_tags,
                "estimated_date": estimated_date,  # 现在能正确提取
                "word_count": len(chunk_content),
                # === 将哈希存入元数据 ===
                "content_hash": chunk_hash,
            }
            chunk_dicts.append({"content": content, "metadata": chunk_metadata})

        log.info(f"将文本分割成 {len(chunk_dicts)} 个语义块。")
        return chunk_dicts

# %% [markdown]
# ### split_into_semantic_chunks_other2(self, text: str, note_title: str = "", note_tags: str = "") -> List[Dict]

    # %%
    def split_into_semantic_chunks_other2(self, text: str, note_title: str = "", note_tags: str = "") -> List[Dict]:
        """
        将文本分割成语义块，并返回块字典列表。
        每个字典包含 'content' 和初步的 'metadata'。
        【增强】专门处理以 “### YYYY年MM月DD日” 格式开头的日记体笔记。
        """
        if not text:
            return []

        chunks = []  # 最终存储所有文本块的列表

        # ========== 第一步：核心修改 - 按日期标题行进行一级分割 ==========
        # 匹配 “### 2026年3月22日” 或 “### 2026年3月22号” 及其变体（如可能有多余空格）
        # 正则解释：r‘^###\s*(\d{4}年\d{1,2}月\d{1,2}日(?:号)?)\s*$’
        # ^### : 以###开头
        # \s* : 可能有的空格
        # (\d{4}年\d{1,2}月\d{1,2}日(?:号)?) : 核心日期格式，日或号
        # \s*$ : 可能有的空格，然后行结束
        # 使用 re.MULTILINE 模式让 ^ 和 $ 匹配每行的开头结尾
        import re
        date_section_pattern = re.compile(r'^###\s*(\d{4}年\d{1,2}月\d{1,2}日(?:号)?)\s*$', re.MULTILINE)

        # 找到所有日期标题行的位置
        date_matches = list(date_section_pattern.finditer(text))

        if not date_matches:
            # 如果没有找到这种格式的日期行，则回退到原有的语义分割逻辑
            log.debug("笔记未检测到‘### 日期’格式，回退至通用语义分块。")
            major_sections = re.split(r'\n(?:#{1,3}\s+.*?|\-{3,}|\d{4}年\d{1,2}月\d{1,2}日.*?)\n', text)
            major_sections = [s.strip() for s in major_sections if s.strip()]
            chunks = major_sections # 后续会对这些块进行大小判断和二次分割
        else:
            # 有日期行，按日期行分割成“天单元”
            log.debug(f"检测到 {len(date_matches)} 个日期标题行，将按此分割。")
            start_pos = 0
            for i, match in enumerate(date_matches):
                date_line_start = match.start()
                date_line_end = match.end()
                current_date = match.group(1)  # 提取到的日期字符串，如“2026年3月22日”

                # 确定这个日期单元的内容结束位置（下一个日期行开始，或文本末尾）
                next_start = date_matches[i + 1].start() if i + 1 < len(date_matches) else len(text)

                # 提取从当前日期行开始，到下一个日期行之前的所有内容作为一个“天单元”
                # 这保证了日期行（### 2026年3月22日）是此单元的第一行。
                day_unit = text[date_line_start:next_start].strip()

                # 将这个“天单元”添加到待处理块列表
                # 注意：此时‘day_unit’可能还很长（包含多个段落），后续会判断是否需要进一步分割。
                chunks.append(day_unit)
                start_pos = next_start

            # 处理第一个日期行之前可能存在的文本（如笔记开头的说明）
            if date_matches[0].start() > 0:
                preface = text[:date_matches[0].start()].strip()
                if preface:
                    chunks.insert(0, preface)  # 将前言作为第一个块
        # ========== 第一步结束 ==========

        # 后续逻辑：对每个初步分割出的块（可能是“天单元”或前言），检查大小，如果过大则进行二次分割。
        final_chunks = []
        for raw_chunk in chunks:
            if len(raw_chunk) <= self.chunk_size * 1.1:
                # 如果块大小合理，直接使用
                final_chunks.append(raw_chunk)
            else:
                # 如果块过大（例如某一天的内容特别长），则调用原有的段落/句子级分割函数进行细化
                # 注意：此时 raw_chunk 可能是一个以日期行开头的“天单元”
                log.debug(f"初步块过长({len(raw_chunk)}字符)，进行二次语义分割。")
                sub_chunks = self._split_into_paragraphs_chunks(raw_chunk)
                final_chunks.extend(sub_chunks)

        # 如果经过上述步骤，分块结果仍然不理想（比如块数太少或太大），启用最终回退
        if not final_chunks or (len(final_chunks) == 1 and len(final_chunks[0]) >= self.chunk_size * 1.1):
            log.debug(f"按照语义拆分不太合格：{[len(chunk) for chunk in final_chunks]}")
            final_chunks = self._split_into_paragraphs_chunks(text)  # 回退到全局段落分割
            log.debug(f"回退用段落甚至字符拆分后：{[len(chunk) for chunk in final_chunks]}")

        # ========== 构建块字典和元数据 ==========
        chunk_dicts = []
        for idx, chunk_content in enumerate(final_chunks):
            # --- 提取该块的日期 ---
            # 由于我们确保了日期行在块内，现在可以直接从块内容开头匹配
            date_in_chunk = date_section_pattern.match(chunk_content) # 使用match从开头匹配
            if not date_in_chunk:
                # 如果不是以日期行开头，则在内容中搜索第一个日期（用于前言块或回退分割的块）
                date_in_chunk = date_section_pattern.search(chunk_content)
            estimated_date = date_in_chunk.group(1) if date_in_chunk else ""
            # ---------------------

            chunk_metadata = {
                "chunk_index": idx,
                "source_note_title": note_title,
                "source_note_tags": note_tags,
                "estimated_date": estimated_date,  # 现在有很大概率能获取到值
                "word_count": len(chunk_content),
            }
            # 清理内容格式
            content = self.clean_text(chunk_content)
            if len(content) < 10:
                content = note_title
            chunk_dicts.append({
                "content": content,
                "metadata": chunk_metadata
            })

        log.info(f"将文本分割成 {len(chunk_dicts)} 个语义块。")
        return chunk_dicts

# %% [markdown]
# ### split_into_semantic_chunks_other(self, text: str, note_title: str = "", note_tags: str = "") -> List[Dict]

    # %%
    def split_into_semantic_chunks_other(self, text: str, note_title: str = "", note_tags: str = "") -> List[Dict]:
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
            log.debug(f"按照语义拆分不太合格：{[len(chunk) for chunk in chunks]}")
            chunks = self._split_into_paragraphs_chunks(text)
            log.debug(f"回退用段落甚至字符拆分后：{[len(chunk) for chunk in chunks]}")

        # 4. 为每个块构建初步元数据
        chunk_dicts = []
        for idx, chunk_content in enumerate(chunks):
            # 提取块内可能的关键词或日期（简单示例）
            import datetime
            date_match = re.search(r'(\d{4}年\d{1,2}月\d{1,2}日)', chunk_content)
            chunk_metadata = {
                "chunk_index": idx,
                # "content": chunk_content,
                "source_note_title": note_title,
                "source_note_tags": note_tags,
                "estimated_date": date_match.group(1) if date_match else "",
                "word_count": len(chunk_content),
                # 后续可在 joplinai.py 中补充 DeepSeek 生成的摘要或关键词
            }
            # 到了这里再开始清理内容格式和无用文本
            content = self.clean_text(chunk_content)
            if len(content) < 10:
                content = note_title
            chunk_dicts.append({
                "content": content,
                "metadata": chunk_metadata
            })

        log.info(f"将文本分割成 {len(chunk_dicts)} 个语义块。")
        return chunk_dicts

# %% [markdown]
# ### _split_into_paragraphs_chunks(self, text: str) -> List[str]

    # %%
    def _split_into_paragraphs_chunks(self, text: str) -> List[str]:
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
        # 1. 关键预处理：对文本进行清洗和规范化
        processed_text = self._preprocess_text_for_embedding(text)

        if not processed_text or len(processed_text.strip()) < 6:
            # log.warning("输入文本为空或过短，返回零向量")
            # return [0.0] * self.embedding_dim
            log.warning("输入文本为空或过短，返回空列表")
            return []

        # 2. (可选但推荐) 估算token长度并主动截断
        # 假设模型最大上下文为512 tokens，我们设定一个安全阈值（如450 tokens）
        estimated_tokens = self._estimate_token_count(processed_text)
        safe_token_limit = int(self.chunk_size * 0.8)  # 根据模型调整，可设为配置项
        if estimated_tokens > safe_token_limit:
            log.warning(
                f"文本预估token数({estimated_tokens})超过安全阈值({safe_token_limit})，将进行智能截断"
            )
            processed_text = self._smart_truncate(processed_text, safe_token_limit)
            log.debug(f"截断后文本预览: {processed_text[:50]}...")

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
        # 返回一个零向量或随机向量，确保流程不中断
        # return [0.0] * self.embedding_dim
        log.error(f"为文本生成嵌入最终失败，将返回空列表，该文本如下: '{text[:50]}...'")
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
            log.info(f"使用本地Ollama嵌入，模型为：{self.model_name}")
            embedding = self.get_ollama_embedding(text)
            # 如果文本块中有嵌入量化操作失败，返回空列表，方便后面捕捉，避免写入残值导致误判
            if not embedding:
                log.debug(f"问题文本块如下：\n{text}\n再次用更保守又是更安全的方式处理…………")
                embedding = self.get_ollama_embedding_safe(text)
                if not embedding:
                    log.error(f"多次尝试处理问题文本块仍然失败，返回空向量列表。问题文本块如下：\n{text}")
                    return []

            # 存储到缓存
            if embedding:
                self.embedding_cache[text_hash] = embedding
                log.info(f"生成缓存嵌入: {text_hash[:8]}，{len(embedding)}")
            return embedding
