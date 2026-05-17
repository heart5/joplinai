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
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import ollama
import pathmagic
import requests


# %%
with pathmagic.Context():
    try:
        from aimod.chunk_optimizer import AdaptiveChunkOptimizer
        from aimod.note_enhancer import enhance_note
        from aimod.text_preprocessor import TextPreprocessor
        from aimod.text_splitter import ContextAwareSplitter
        from func.datatools import compute_content_hash
        from func.jpfuncs import getinivaluefromcloud
        from func.logme import log
    except ImportError as e:
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger(__name__)
        log.error(f"导入项目模块失败: {e}")


# %% [markdown]
# # EmbeddingGenerator类


# %%
__all__ = ["EmbeddingGenerator"]

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
        self.text_prep = TextPreprocessor(chunk_size=self.chunk_size)
        self._set_chunk_size()

        # 初始化探测缓存客户端（远程优先，失败不影响运行）
        probe_client = None
        try:
            remote_url = getinivaluefromcloud("joplinai", "joplinai_center_url")
            if not remote_url:
                remote_url = "http://127.0.0.1:5003"
            api_key = getinivaluefromcloud("joplinai", "joplinai_center_api_key")
            if remote_url and api_key:
                from aimod.probe_client import ProbeCacheClient
                probe_client = ProbeCacheClient(remote_url, api_key)
                log.info("[自适应探测] 远程缓存客户端已连接")
        except Exception:
            pass

        self.adaptive_optimizer = AdaptiveChunkOptimizer(
            embedding_generator=self,
            enabled=enable_adaptive_chunking,
            probe_client=probe_client,
        )
        self.enable_adaptive_chunking = enable_adaptive_chunking

    def __repr__(self):
        return f"EmbeddingGenerator(model={self.model_name!r}, dim={self.embedding_dim}, chunk_size={self.chunk_size})"

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
            processed = self.text_prep.preprocess_for_embedding(chunk_text)
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
            test_response = ollama.embed(model=self.model_name, input="test")
            dim = len(test_response["embeddings"][0])
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
            self.chunk_size = 1850
            self.text_prep.chunk_size = self.chunk_size
            return
        elif self.model_name == "dengcao/bge-large-zh-v1.5":
            self.chunk_size = 512
            self.text_prep.chunk_size = self.chunk_size
            return
        elif self.model_name == "qwen:1.8b":
            self.chunk_size = 4000
            self.text_prep.chunk_size = self.chunk_size
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
                f"[模型上下文] {self.model_name}: num_ctx={num_ctx}token, "
                f"chunk_size={self.chunk_size}字符 (={num_ctx}token × 3字符/token × 0.8)"
            )

        except Exception as e:
            log.warning(
                f"[模型上下文] 获取失败({self.model_name}), "
                f"回退默认chunk_size=1024字符: {e}"
            )
            self.chunk_size = 1024

        self.text_prep.chunk_size = self.chunk_size

# %% [markdown]
# ## enhance_chunk_metadata(chunk_content: str, note_tags: str, config: Dict)

    # %%
    _enhance_model_logged = False

    def enhance_chunk_metadata(self, chunk_content: str, note_tags: str, config: Dict):
        """增强生成小结和标签（provider-agnostic: cloud/ollama/none）

        Returns:
            enhanced_metadata: 增强后的元数据字典，包含 enhanced 标记
        """
        enhanced_metadata = {"enhanced": False}

        cloud_model = config.get("cloud_model", "deepseek-chat")
        ollama_chat_model = config.get("ollama_chat_model", "qwen2.5:1.5b")
        summary_provider = config.get("summary_model", "cloud")
        tags_provider = config.get("tags_model", "cloud")

        ollama_host = config.get("ollama_host") or ""
        ollama_port = config.get("ollama_port") or "11434"
        ollama_url = f"http://{ollama_host}:{ollama_port}" if ollama_host else ""

        # 首次调用时输出增强模型策略
        if not self._enhance_model_logged:
            log.info(
                f"AI增强策略：摘要={summary_provider}"
                f"({cloud_model if summary_provider == 'cloud' else ollama_chat_model})"
                f"，标签={tags_provider}"
                f"({cloud_model if tags_provider == 'cloud' else ollama_chat_model})"
            )
            self._enhance_model_logged = True

        # 摘要增强
        summary = enhance_note(
            chunk_content, task="summary", provider=summary_provider,
            model=cloud_model if summary_provider == "cloud" else ollama_chat_model,
            ollama_host=ollama_url,
        )
        if summary:
            enhanced_metadata["enhanced"] = True
            enhanced_metadata["chunk_summary"] = summary
        else:
            enhanced_metadata["chunk_summary"] = ""

        # 标签增强
        tags_str = enhance_note(
            chunk_content, task="tags", provider=tags_provider,
            model=cloud_model if tags_provider == "cloud" else ollama_chat_model,
            ollama_host=ollama_url,
        )
        if tags_str:
            enhanced_metadata["enhanced"] = True

        if tags_str:
            extracted_tags = self._clean_tags(tags_str)
            original_tags = [t.strip() for t in note_tags.split(",") if t.strip()] if note_tags else []
            enhanced_tags = list(set(original_tags + extracted_tags))
            enhanced_metadata["tags"] = ",".join(enhanced_tags)

        return enhanced_metadata

    @staticmethod
    def _clean_tags(raw_tags: str) -> list:
        """清洗模型输出的标签，返回合规关键词列表。

        统一分隔符、去编号前缀后分三类处理：
        - 含中文：保留，长度限制2-12字符
        - 纯ASCII短词(≤12)：专有名词，直接保留
        - 纯ASCII长词(>12)：尝试拆分为英文单词，拆不开则丢弃
        """
        import re

        text = raw_tags.replace("\n", ",").replace("\r", ",").replace("，", ",")
        text = re.sub(r",\s*,+", ",", text)
        candidates = [t.strip() for t in text.split(",") if t.strip()]

        cleaned = []
        for tag in candidates:
            tag = re.sub(r"^[\d]+[\.\、\)\s]\s*", "", tag).strip()
            tag = re.sub(r"^[-*]\s*", "", tag).strip()
            if not tag:
                continue

            has_cjk = bool(re.search(r"[一-鿿]", tag))
            if has_cjk:
                if 2 <= len(tag) <= 12:
                    cleaned.append(tag)
            elif len(tag) <= 12:
                cleaned.append(tag)
            else:
                # 超长纯ASCII拼接词，尝试拆分
                parts = EmbeddingGenerator._split_compound(tag)
                cleaned.extend(parts)

        seen = set()
        result = []
        for tag in cleaned:
            if tag not in seen:
                seen.add(tag)
                result.append(tag)
        return result

    # 用于拆分英文拼接词的小词表（业务/技术/通用高频词）
    _COMPOUND_WORDS: set = None

    @classmethod
    def _get_compound_words(cls) -> set:
        if cls._COMPOUND_WORDS is not None:
            return cls._COMPOUND_WORDS
        cls._COMPOUND_WORDS = {
            # 业务
            "sale", "sales", "market", "marketing", "price", "network",
            "confirm", "confirmation", "valid", "validation", "train", "training",
            "develop", "development", "place", "manage", "management",
            "customer", "product", "service", "business", "finance",
            "relation", "relationship", "process", "processing",
            "pay", "payment", "ship", "shipping", "deliver", "delivery",
            "support", "operation", "brand", "store", "order", "stock",
            "supply", "chain", "retail", "trade", "client", "partner",
            "contract", "invoice", "budget", "revenue", "profit", "cost",
            # 营销
            "promote", "promotion", "strategy", "campaign", "channel",
            "content", "target", "launch", "growth", "social", "media",
            # 运营
            "company", "warehouse", "logistics", "inventory", "quality",
            "standard", "policy", "report", "review", "account",
            "region", "district", "area", "office", "meeting",
            "weekly", "monthly", "annual", "terminal", "dealer",
            # 团队
            "leader", "member", "staff", "hire", "recruit", "coach",
            "perform", "performance", "goal", "objective", "bonus",
            "salary", "feedback", "agenda",
            # 生活
            "health", "sleep", "walk", "drink", "smoke", "study",
            "learn", "exercise", "food", "meal", "travel", "trip",
            "family", "child", "school", "doctor", "hospital",
            # 技术
            "data", "code", "server", "cloud", "file", "system", "user",
            "admin", "login", "cache", "config", "model", "token",
            "query", "search", "index", "base", "node", "test", "debug",
            "build", "deploy", "web", "app", "api", "key", "log",
            "core", "type", "script", "python", "java", "linux",
            "docker", "git", "json", "html", "http", "sql", "ssh",
            # 通用
            "time", "line", "note", "team", "plan", "task", "work",
            "home", "list", "view", "link", "page", "post", "text",
            "name", "date", "info", "check", "back", "call", "chat",
            "open", "read", "send", "auto", "soft", "hard", "life",
            "book", "card", "gold", "blue", "fast", "plus", "mini",
            "free", "pro", "max", "lite", "tool", "pack", "unit",
            # 较长的词，本身可作为拆分结果
            "technology", "presentation", "information", "organization",
            "communication", "application",
        }
        return cls._COMPOUND_WORDS

    @classmethod
    def _split_compound(cls, word: str) -> list:
        """拆解英文拼接词(如 salesvalidation → [sales, validation])。

        用词表做动态规划：找到覆盖整词的最少拆分。
        拆分失败返回空列表。
        """
        word_lower = word.lower()
        n = len(word_lower)
        vocab = cls._get_compound_words()

        # dp[i] = 从位置i到末尾的最少段数，无法拆分则为 -1
        dp = [-1] * (n + 1)
        dp[n] = 0
        split_from = [-1] * (n + 1)

        for i in range(n - 1, -1, -1):
            best = float('inf')
            best_j = -1
            for j in range(i + 1, min(i + 16, n + 1)):
                if word_lower[i:j] in vocab:
                    if dp[j] != -1 and dp[j] + 1 < best:
                        best = dp[j] + 1
                        best_j = j
            if best != float('inf'):
                dp[i] = best
                split_from[i] = best_j

        if dp[0] == -1 or dp[0] == 1:
            # 拆不开，或整词命中（说明词表里有这个词，但作为标签太长）
            return []

        # 回溯，保留原词的大小写
        parts = []
        pos = 0
        while pos < n:
            nxt = split_from[pos]
            parts.append(word[pos:nxt])
            pos = nxt
        return [p for p in parts if len(p) >= 3]


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
                    f"[迭代分块] 剩余文本({remaining_len}字符)≤安全净字符上限"
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
                    f"[迭代分块] 剩余文本({remaining_len}字符)≤探测margin"
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
                    f"  [迭代分块] 句子边界({best_split}字符)过短"
                    f"(阈值{max(MIN_SIZE, int(margin * MIN_FRACTION))}字符)，"
                    f"改用 margin={margin}字符 硬截断"
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
                f"[迭代分块] pos={pos}字符, 剩余={remaining_len}字符, "
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
            f"[迭代分块完成] 笔记《{note_title}》"
            f"（{'自适应探测' if can_adaptive else '固定大小'}），"
            f"共{len(chunks)}块，各块字符数: {[len(c) for c in chunks]}"
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
        twice_probe: bool = True,
        image_descriptions: Optional[str] = None,
    ) -> List[Dict]:
        """将文本分割成语义块，并返回块字典列表。
        【增强】使用统一的正则表达式处理带`###`或不带的日期行，确保日期保留在块头部并被提取。

        Args:
            image_descriptions: 图片视觉描述文本，若提供则前置到原文，
                               使图片信息融入分块上下文。
        """
        if not text:
            return []

        # 图片描述前置到文本开头，随分块自然分布
        if image_descriptions:
            text = f"【图片内容描述】\n{image_descriptions}\n\n{text}"

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
                normalized_unit = self.text_prep.normalize_single_date_unit(
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
            converted_chunk = self.text_prep.convert_health_data_to_text(raw_chunk)
            converted_chunk = self.text_prep.condense_dense_lists(converted_chunk)
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
            log.debug(f"[分块回退] 语义拆分不理想，各块字符数：{[len(chunk) for chunk in final_chunks]}")
            final_chunks = self._split_into_paragraphs_chunks(text)
            log.debug(
                f"[分块回退] 段落/字符拆分后各块字符数：{[len(chunk) for chunk in final_chunks]}"
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
                estimated_date = self.text_prep.normalize_date_string(extracted_date)
            # 提取作者信息
            note_meta = self._extract_author_from_note(
                note_title,
                note_tags,
                source_notebook_title
            )

            # 清理内容格式
            content = self.text_prep.clean_text(chunk_content)
            if not self.text_prep.is_valid_chunk(content):
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
            # AI增强生成摘要和标签
            enhanced_metadata = {}
            try:
                enhanced_metadata = self.enhance_chunk_metadata(
                    chunk_content, note_tags, self.config,
                )
            except Exception as e:
                log.error(
                    f"[AI增强失败] 笔记《{note_title}》块{block_number} "
                    f"（长度：{len(chunk_content)}字符）: {e}",
                    exc_info=True,
                )
            tags = [t.strip() for t in note_tags.split(",") if t.strip()]
            tags_str = ",".join(sorted(tags)) if tags else ""  # 排序保证一致性
            stored_tags = enhanced_metadata.get("tags", tags_str)
            stored_summary = enhanced_metadata.get("chunk_summary", "")
            meta_hash = compute_content_hash(f"{stored_tags}{source_notebook_title}{stored_summary}")
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
            log.warning("[段落分割] 仍存在过大块，启用字符级回退")
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
# ## get_ollama_embedding(self, text: str, host: str = "10.9.0.2", port: int = 11434)

    # %%
    def get_ollama_embedding(
        self, text: str, host: str = "10.9.0.2", port: int = 11034
    ):
        """调用远程恒创云Ollama生成嵌入"""
        host = self.config.get("ollama_host", host)
        port = self.config.get("ollama_port", port)
        url = f"http://{host}:{port}/api/embed"
        model = self.model_name
        # print(host, port, model)
        payload = {"model": model, "input": text}
        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            return resp.json()["embeddings"][0]
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
            log.error(f"[远程Ollama] 嵌入调用失败(token超限): {error_msg}")
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
                f"[嵌入缓存] 笔记《{source_note_title}》块【{chunk_index}】分块阶段缓存击中"
            )
            return chunk_emb

        # 检查缓存
        cached = self.get_cached_embedding(text_hash)
        if cached:
            log.info(
                f"[嵌入缓存] 笔记《{source_note_title}》块【{chunk_index}】缓存击中: {text_hash[:12]}"
            )
            return cached

        # 在尝试本地Ollama嵌入前，对文本进行一次嵌入预处理
        processed_text = self.text_prep.preprocess_for_embedding(text)
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
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
                            f"[嵌入重试耗尽] 笔记《{source_note_title}》块【{chunk_index}】"
                            f"长度({len(text)}字符)经5次重试均token超限，"
                            f"句子边界截断至{len(truncated)}字符（安全上限={safe_chars}字符）"
                        )
                        try:
                            embedding = self.get_ollama_embedding(truncated)
                        except Exception as e2:
                            log.error(f"[嵌入失败] 截断回退嵌入仍失败: {e2}")
                            return []
                    else:
                        # 非长度错误（如Ollama不可用），不触发重分块
                        log.error(
                            f"[嵌入失败] 笔记《{source_note_title}》块【{chunk_index}】"
                            f"嵌入最终失败（非token长度错误）: {e}"
                        )
                        return []
                    break
                else:
                    log.warning(f"[嵌入重试] 第{attempt}次获取嵌入失败: {e}")
                continue

        if not embedding:
            log.error(f"[嵌入失败] 为文本生成嵌入最终失败，将返回空列表")
            return []
        self.embedding_cache[text_hash] = embedding

        return embedding
