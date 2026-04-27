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
# # Joplin笔记智能问答系统

# %% [markdown]
# ## 导入库

# %% [markdown]
# ### 导入核心库

# %%
import argparse
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import ollama
from chromadb.config import Settings

# %% [markdown]
# ### 项目模块导入

# %%
try:
    from aimod.cache_manager import SQLiteCacheManager
    from aimod.vector_db_manager import VectorDBManager
    from func.configpr import (
        findvaluebykeyinsection,
        getcfpoptionvalue,
        setcfpoptionvalue,
    )
    from func.datatools import compute_content_hash
    from func.first import dirmainpath, getdirmain
    from func.getid import getdeviceid, getdevicename, gethostuser
    from func.jpfuncs import (
        getinivaluefromcloud,
        getnote,
    )
    from func.logme import log
    from func.sysfunc import execcmd, not_IPython
    from func.wrapfuncs import timethis
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    log.error(f"导入项目模块失败: {e}")

# %% [markdown]
# ## 配置设置

# %%
CONFIG = {
    "embedding_model": "dengcao/bge-large-zh-v1.5",  # 嵌入模型（与joplinai.py保持一致）
    # "embedding_model": "qwen:1.8b",  # 嵌入模型（与joplinai.py保持一致）
    "chat_model": chat_model
    if (chat_model := getinivaluefromcloud("joplinai", "chat_model"))
    else "qwen:1.8b",  # 聊天模型（用于问答）
    "db_path": getdirmain() / "data" / "joplin_vector_db",  # ChromaDB存储路径
    "max_retrieved_notes": 10,  # 最大检索笔记数量
    # 最大检索文本块数，默认20
    "max_retrieved_chunks": max_retrieved_chunks
    if (
        max_retrieved_chunks := getinivaluefromcloud("joplinai", "max_retrieved_chunks")
    )
    else 20,
    "similarity_threshold": 0.5,  # 相似度阈值
    # 是否使用DeepSeek进行问答，默认为False
    "enable_deepseek": enable_deepseek
    if (enable_deepseek := getinivaluefromcloud("joplinai", "enable_deepseek"))
    else False,
    "deepseek_api_key": getinivaluefromcloud("joplinai", "deepseek_token"),
    "deepseek_chat_model": "deepseek-chat",
    # 上下文最大长度（字符），默认为4000
    "context_max_length": context_max_length
    if (context_max_length := getinivaluefromcloud("joplinai", "context_max_length"))
    else 4000,
    "min_answer_length": 50,  # 添加最小答案长度要求
}

# %%
# 根据嵌入模型生成状态文件路径
model_name = (
    CONFIG.get("embedding_model").replace(":", "_").replace("/", "_").replace("-", "_")
)
CONFIG["collection_name"] = f"joplin_{model_name}"

# %% [markdown]
# ## 全局变量

# %%
# 确保缓存目录存在
cache_dir = getdirmain() / "data" / ".deepseek_cache"
os.makedirs(cache_dir, exist_ok=True)
cache_db_path = cache_dir / "deepseek_cache.db"

# 创建全局缓存管理器实例
global_cache_manager = SQLiteCacheManager(db_path=str(cache_db_path))

# %% [markdown]
# ## 问答系统核心


# %%
class JoplinQASystem:
    """Joplin笔记问答系统"""

# %% [markdown]
# ### __init__(self, config: Dict)

    # %%
    def __init__(self, config: Dict):
        self.config = config
        # for_creation=False表示用于查询
        self.vector_db = VectorDBManager(
            config["db_path"], config["embedding_model"], for_creation=False
        )
        self.conversation_history = []  # 对话历史

# %% [markdown]
# ### ask(self, question: str, use_history: bool = True) -> Dict

    # %%
    def ask(self, question: str, use_history: bool = True) -> Dict:
        """
        回答用户问题

        Args:
            question: 用户问题
            use_history: 是否使用对话历史

        Returns:
            包含答案和相关上下文的字典
        """
        log.info(f"处理问题: {question}")

        # 1. 检索相关笔记
        similar_notes = self.vector_db.search_similar_notes(
            question, n_results=self.config["max_retrieved_notes"]
        )

        # 过滤低相似度的笔记
        filtered_notes = [
            note
            for note in similar_notes
            if note["similarity"] >= self.config["similarity_threshold"]
        ]

        if not filtered_notes:
            log.warning("未找到相关笔记，将使用通用回答")
            answer = self._generate_generic_answer(question)
            return {
                "question": question,
                "answer": answer,
                "relevant_notes": [],
                "sources": [],
                "is_based_on_notes": False,
                "context_length": 0,  # 添加context_length字段
            }

        # 2. 构建上下文
        context = self._build_context(filtered_notes, question)

        # 3. 生成答案
        if self.config["enable_deepseek"] and self.config["deepseek_api_key"]:
            answer = self._generate_answer_with_deepseek(question, context)
        else:
            answer = self._generate_answer_with_ollama(question, context)

        # 4. 更新对话历史
        if use_history:
            self.conversation_history.append(
                {
                    "question": question,
                    "answer": answer,
                    "timestamp": datetime.now().isoformat(),
                    "relevant_note_ids": [note["note_id"] for note in filtered_notes],
                }
            )
            # 保持历史记录长度
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

        # 5. 提取来源信息
        sources = self._extract_sources(filtered_notes)

        return {
            "question": question,
            "answer": answer,
            "relevant_notes": filtered_notes,
            "sources": sources,
            "is_based_on_notes": True,
            "context_length": len(context),
        }

# %% [markdown]
# ### _build_context(self, notes: List[Dict], question: str) -> str

    # %%
    def _build_context(self, notes: List[Dict], question: str) -> str:
        """构建问答上下文"""
        context_parts = []

        # 添加系统提示
        system_prompt = """你是我个人的笔记助手，请基于我的笔记内容回答问题。
    我的笔记特点：
    1. 记录工作、学习、生活的点滴
    2. 包含具体项目、人名、日期
    3. 可能有零散的想法和计划
    
    回答要求：
    1. 基于笔记事实，不要编造
    2. 引用具体的笔记内容
    3. 如果笔记中没有相关信息，请说明
    4. 用第一人称视角回答（因为这是我的笔记）
    """
        context_parts.append(system_prompt)

        # 添加相关笔记内容
        context_parts.append("\n相关笔记内容：")
        for i, note in enumerate(notes, 1):
            note_content = note["content"][:800]  # 限制每个笔记长度
            metadata = note.get("metadata", {})
            tags = metadata.get("tags", "")
            summary = metadata.get("summary", "")

            note_context = f"\n【笔记{i} - 相似度:{note['similarity']:.2f}】"
            if tags:
                note_context += f" 标签:{tags}"
            if summary:
                note_context += f"\n摘要: {summary}"
            note_context += f"\n内容: {note_content}"

            context_parts.append(note_context)

        # 添加对话历史（如果存在）
        if self.conversation_history:
            context_parts.append("\n对话历史：")
            for hist in self.conversation_history[-3:]:  # 最近3条历史
                context_parts.append(f"问: {hist['question']}")
                context_parts.append(f"答: {hist['answer'][:200]}...")

        # 添加当前问题
        context_parts.append(f"\n用户问题: {question}")
        context_parts.append("\n请基于以上笔记内容回答问题，并注明信息来源。")

        full_context = "\n".join(context_parts)

        # 限制上下文长度
        if len(full_context) > self.config["context_max_length"]:
            full_context = full_context[: self.config["context_max_length"]] + "..."

        log.debug(f"构建上下文长度: {len(full_context)}字符")
        return full_context

# %% [markdown]
# ### _generate_answer_with_ollama(self, question: str, context: str) -> str

    # %%
    def _generate_answer_with_ollama(self, question: str, context: str) -> str:
        """使用本地Ollama生成答案"""
        try:
            model_name = self.config["chat_model"]

            # 构建消息
            messages = [
                {
                    "role": "system",
                    "content": "你是一个专业的笔记助手，基于用户笔记回答问题。",
                },
                {"role": "user", "content": context},
            ]

            response = ollama.chat(
                model=model_name,
                messages=messages,
                options={"temperature": 0.3, "num_predict": 2000},
            )

            answer = response["message"]["content"]
            log.info(f"使用Ollama模型 {model_name} 生成答案，长度: {len(answer)}")
            return answer

        except Exception as e:
            log.error(f"Ollama生成答案失败: {e}")
            return f"抱歉，生成答案时出现错误: {str(e)}"

# %% [markdown]
# ### _generate_answer_with_deepseek(self, question: str, context: str) -> str

    # %%
    def _generate_answer_with_deepseek(self, question: str, context: str) -> str:
        """使用DeepSeek API生成答案"""
        try:
            import requests

            headers = {
                "Authorization": f"Bearer {self.config['deepseek_api_key']}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.config["deepseek_chat_model"],
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个专业的笔记助手，基于用户笔记回答问题。",
                    },
                    {"role": "user", "content": context},
                ],
                "temperature": 0.3,
                "max_tokens": 1000,
            }

            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()

            answer = response.json()["choices"][0]["message"]["content"]
            log.info(f"使用DeepSeek生成答案，长度: {len(answer)}")
            return answer

        except Exception as e:
            log.error(f"DeepSeek生成答案失败: {e}")
            # 回退到Ollama
            return self._generate_answer_with_ollama(question, context)

# %% [markdown]
# ### _generate_generic_answer(self, question: str) -> str

    # %%
    def _generate_generic_answer(self, question: str) -> str:
        """生成通用回答（当没有相关笔记时）"""
        generic_responses = [
            "我在您的笔记中没有找到相关信息。您可以尝试：\n1. 添加相关笔记\n2. 重新表述问题\n3. 检查笔记是否已正确向量化",
            "目前笔记库中没有与此问题直接相关的内容。建议您完善相关主题的笔记。",
            "未找到相关笔记信息。请确保相关笔记已导入并完成向量化处理。",
        ]

        import random

        return random.choice(generic_responses)

# %% [markdown]
# ### _extract_sources(self, notes: List[Dict]) -> List[Dict]

    # %%
    def _extract_sources(self, notes: List[Dict]) -> List[Dict]:
        """提取来源信息"""
        sources = []
        for note in notes:
            metadata = note.get("metadata", {})
            source = {
                "note_id": note["note_id"],
                "title": note["title"],
                "similarity": note["similarity"],
                "tags": metadata.get("tags", "").split(",")
                if metadata.get("tags")
                else [],
                "has_summary": bool(metadata.get("summary")),
            }
            sources.append(source)
        return sources

# %% [markdown]
# ### clear_history(self)

    # %%
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        log.info("对话历史已清空")

# %% [markdown]
# ### get_statistics(self) -> Dict

    # %%
    def get_statistics(self) -> Dict:
        """获取系统统计信息"""
        try:
            collection_info = (
                self.vector_db.collection.count() if self.vector_db.collection else 0
            )
            return {
                "total_notes_in_db": collection_info,
                "conversation_history_count": len(self.conversation_history),
                "config": {
                    "embedding_model": self.config["embedding_model"],
                    "chat_model": self.config["chat_model"],
                    "using_deepseek": self.config["enable_deepseek"],
                },
            }
        except:
            return {
                "total_notes_in_db": 0,
                "conversation_history_count": len(self.conversation_history),
            }


# %% [markdown]
# ## PromptManager类

# %%
class PromptManager:
    """集中管理从云端获取的系统提示词，杜绝硬编码。"""

    @staticmethod
    def get_sys_prompt_for_role(user_identity: Optional[Dict]) -> str:
        """
        根据用户身份，从云端获取对应的系统提示词。
        如果云端未配置，则返回一个极简的、安全的通用提示。
        """
        from func.jpfuncs import getinivaluefromcloud

        if not user_identity:
            # 如果没有身份信息，使用最基础的提示
            base_prompt = getinivaluefromcloud("joplinai", "sys_prompt_base")
            if base_prompt:
                return base_prompt
            else:
                # 终极后备：一个完全不涉及具体业务的中性提示
                return "请根据提供的笔记内容回答问题。如果笔记中没有相关信息，请说明。"

        user_role = user_identity.get("role")
        user_display_name = user_identity.get("display_name", "")

        # 从云端获取默认个人作者和同事列表（这些也是配置）
        default_personal_author = (
            getinivaluefromcloud("joplinai", "default_personal_author") or "用户"
        )

        import re

        split_ptn = re.compile(r"[,，]")
        if colleague_str := getinivaluefromcloud("joplinai", "colleague"):
            colleagues = [title.strip() for title in split_ptn.split(colleague_str)]
        else:
            colleagues = []
        colleague_str_for_prompt = "，".join([f"“{person}”" for person in colleagues])

        # 根据角色获取对应的提示词配置键
        prompt_key = ""
        if user_role == "admin":
            prompt_key = "sys_prompt"
        elif user_role == "colleague":
            prompt_key = "sys_colleague_prompt"
        else:
            # 未知角色，使用基础提示
            prompt_key = "sys_prompt_base"

        # 核心：从云端获取提示词模板
        prompt_template = getinivaluefromcloud("joplinai", prompt_key)

        if prompt_template:
            # 如果模板中存在占位符，进行动态替换
            # 例如，模板中可能有 {default_personal_author}、{colleague_str}、{user_display_name} 等占位符
            prompt = prompt_template.replace(
                "{default_personal_author}", default_personal_author
            )
            prompt = prompt.replace("{colleague_str}", colleague_str_for_prompt)
            prompt = prompt.replace("{user_display_name}", user_display_name)
            return prompt
        else:
            # 云端未配置对应提示词，根据角色返回一个结构化的默认提示
            log.warning(f"云端未配置提示词键: {prompt_key}，将使用内置通用模板。")
            if user_role == "admin":
                return f"""你是我（{default_personal_author}）的笔记助手，基于笔记回答问题。笔记可能包含个人记录、团队共享信息或收藏文章。请根据笔记片段的【类型】和【作者】元数据，客观、准确地回答。如果笔记中没有相关信息，请说明。"""
            elif user_role == "colleague":
                return f"""你是{user_display_name}的笔记助手，基于共享笔记库回答问题。你只能访问作者为“团队_共同维护”或“同事_{user_display_name}”的笔记片段。请基于这些内容回答，如果无相关信息，请说明。"""
            else:
                return "请根据提供的笔记内容回答问题。如果笔记中没有相关信息，请说明。"

    @staticmethod
    def get_common_prompt_variants():
        """获取其他可能需要从云端获取的常用提示词变体（如摘要、标签生成提示）。"""
        from func.jpfuncs import getinivaluefromcloud

        variants = {}
        variant_keys = [
            "prompt_for_summarization",
            "prompt_for_tag_extraction",
            "prompt_for_answer_refinement",
        ]
        for key in variant_keys:
            variants[key] = getinivaluefromcloud("joplinai", key)
        return variants

# %% [markdown]
# ## 优化的问答系统核心OptimizedJoplinQASystem(JoplinQASystem)


# %%
class OptimizedJoplinQASystem(JoplinQASystem):
    """优化版的问答系统，针对个人笔记"""

# %% [markdown]
# ### __init__(self, config: Dict)

    # %%
    def __init__(self, config: Dict):
        # 首先，调用父类的初始化（如果继承了 JoplinQASystem）
        super().__init__(config)  # 如果 OptimizedJoplinQASystem 继承自 JoplinQASystem
        # 或者手动初始化父类的必要属性：
        # self.config = config
        # self.vector_db = VectorDBManager(...)
        # self.conversation_history = []

        # 关键修复：初始化 embedding_generator
        # 需要导入或创建 EmbeddingGenerator 类（参考 joplinai.py 中的使用）
        from aimod.embedding_generator import EmbeddingGenerator  # 请根据实际模块名调整

        self.embedding_generator = EmbeddingGenerator(
            config,
            model_name=config["embedding_model"],
            cache_manager=global_cache_manager,  # 传入统一的缓存管理器
        )
        log.info(
            f"OptimizedJoplinQASystem 初始化完成，已加载 embedding_generator，"
            f"嵌入模型为：{config['embedding_model']}"
        )

# %% [markdown]
# ### ask(self, question: str, use_history: bool = True, user_identity: Optional[Dict] = None) -> Dict

    # %%
    def ask(self, question: str, use_history: bool = True, user_identity: Optional[Dict] = None) -> Dict:
        """提问入口，现在基于块进行检索和回答。
        用户身份示例：
        user_identity = {
            'username': 'chenzhiwei',
            'display_name': '陈志伟',
            'role': 'colleague'  # 'admin' 或 'colleague'
        }
        """
        # 1. 预处理问题
        processed_question = self._preprocess_question(question)

        # 2. 获取问题嵌入
        chunk_dict_for_question = {
            "content": processed_question,
            "base_metadata": {
                "content_hash": compute_content_hash(processed_question),
                "source_note_title": f"{processed_question[:20]}",
                "chunk_index": 0,
            }
        }
        query_embedding = self.embedding_generator.get_merged_embedding(chunk_dict_for_question)
        if not query_embedding:
            return {"answer": "无法生成问题嵌入，请检查配置。", "relevant_chunks": []}

        # 3. 搜索相似块（注意：调用的是 search_similar_chunks）
        similar_chunks = self.vector_db.search_similar_chunks(
            query_embedding,
            limit=self.config.get("max_retrieved_chunks", 15),  # 可配置
            user_identity=user_identity  # 新增参数
        )

        # 4. 过滤和重排序块
        filtered_chunks = self._filter_and_rank_chunks(similar_chunks, question)

        # 5. 构建优化上下文（基于块）
        context = self._build_optimized_context_from_chunks(
            filtered_chunks,
            question,
            user_identity=user_identity  # 新增参数
        )
        log.debug(f"过滤后块数: {len(filtered_chunks)}")
        log.debug(f"构建的上下文长度: {len(context)}")
        # log.debug(f"上下文预览 (前500字符): {context[:500]}")

        # 6. 生成答案
        answer = self._generate_optimized_answer(question, context)
        final_answer = self._postprocess_answer(answer)

        # 构建与原始JoplinQASystem兼容的返回字典
        # 关键：将 `filtered_chunks` 转换为 `relevant_notes` 格式
        relevant_notes_for_return = self._get_relevant_notes_for_return(filtered_chunks)
        # 4. 更新对话历史
        if use_history:
            self.conversation_history.append(
                {
                    "question": question,
                    "answer": answer,
                    "timestamp": datetime.now().isoformat(),
                    "relevant_note_ids": [note["note_id"] for note in relevant_notes_for_return],
                }
            )
            # 保持历史记录长度
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

        # 构建来源信息（sources）
        sources = self._extract_sources(relevant_notes_for_return)

        return {
            "question": question,
            "answer": final_answer,
            "relevant_notes": relevant_notes_for_return,  # <-- 必须包含此键
            "sources": sources,
            "relevant_chunks": filtered_chunks,  # 返回块信息
            "context_length": len(context),
            "is_based_on_notes": len(filtered_chunks) > 0,  # 如果有相关块，则基于笔记
        }

# %% [markdown]
# ### _preprocess_question(self, question: str) -> str

    # %%
    def _preprocess_question(self, question: str) -> str:
        """预处理问题，提高检索效果"""
        # 移除常见疑问词
        question = question.lower()
        stop_words = ["请问", "请", "帮我", "我想知道", "什么是", "怎么", "如何"]
        for word in stop_words:
            question = question.replace(word, "")

        # 提取关键词
        keywords = self._extract_keywords(question)

        # 如果是关于"我"的问题，添加个人化标记
        if "我" in question or "我的" in question:
            question = "个人笔记 " + question

        return question.strip()

# %% [markdown]
# ### _extract_keywords(self, text: str) -> List[str]

    # %%
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词（简单实现）"""
        # 移除标点
        import re

        text = re.sub(r"[^\w\s]", "", text)

        # 按词频提取（这里可以更复杂）
        words = text.split()
        from collections import Counter

        word_counts = Counter(words)

        # 返回前3个高频词（排除停用词）
        stop_words = {"的", "了", "在", "是", "我", "你", "他", "她", "它", "这", "那"}
        keywords = [
            word
            for word, count in word_counts.most_common(10)
            if word not in stop_words and len(word) > 1
        ]

        return keywords[:3]

# %% [markdown]
# ### _filter_and_rank_chunks(self, chunks: List[Dict], question: str) -> List[Dict]

    # %%
    def _filter_and_rank_chunks(self, chunks: List[Dict], question: str) -> List[Dict]:
        """过滤和重排序检索到的文本块。"""
        if not chunks:
            return []
        log.debug(
            f"原始检索到 {len(chunks)} 个块，相似度样例: {[c.get('similarity') for c in chunks[:3]]}"
        )

        # 1. 按相似度排序
        chunks.sort(key=lambda x: x["similarity"], reverse=True)

        # 2. 应用阈值过滤
        threshold = self.config.get("similarity_threshold", 0.6)
        filtered = [chunk for chunk in chunks if chunk["similarity"] >= threshold]

        # 3. 如果过滤后太少，放宽条件
        if len(filtered) < 3 and len(chunks) > 0:
            filtered = chunks[:3]

        # 4. 基于问题关键词进一步过滤（可选）
        keywords = self._extract_keywords(question)
        if keywords:
            scored_chunks = []
            for chunk in filtered:
                score = chunk["similarity"]
                content = chunk["content"].lower()
                for keyword in keywords:
                    if keyword in content:
                        score += 0.1
                scored_chunks.append((score, chunk))
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            filtered = [chunk for _, chunk in scored_chunks]

        return filtered

# %% [markdown]
# ### _build_optimized_context_from_chunks(self, chunks: List[Dict], question: str, user_identity: Optional[Dict] = None) -> str

    # %%
    def _build_optimized_context_from_chunks(
        self, chunks: List[Dict],
        question: str,
        user_identity: Optional[Dict] = None,
    ) -> str:
        """基于检索到的块构建问答上下文。"""
        if not chunks:
            return "没有找到相关笔记内容。"

        # 按原始笔记对块进行分组，便于组织
        from collections import defaultdict

        notes_dict = defaultdict(list)
        for chunk in chunks:
            note_id = chunk["source_note_id"]
            notes_dict[note_id].append(chunk)

        # 构建系统提示
        if user_identity:
            user_role = user_identity.get('role')
            user_display_name = user_identity.get('display_name')
            # === 【新增】调试日志：记录角色和显示名 ===
            log.debug(f"[权限过滤] 解析身份 -> role: '{user_role}', display_name: '{user_display_name}'")

            default_personal_author = getinivaluefromcloud("joplinai", "default_personal_author")

            import re
            split_ptn = re.compile(r"[,，]")
            if (colleague_str := getinivaluefromcloud("joplinai", "colleague")):
                colleague = [title.strip() for title in split_ptn.split(colleague_str)]
            else:
                colleague= ["XXA", "XXB"]
            colleague_str = "，".join([f"“{person}”" for person in colleague])

            # 使用 PromptManager 获取动态提示词
            sys_prompt = PromptManager.get_sys_prompt_for_role(user_identity)
            log.debug(f"[提示词管理] 为用户角色 '{user_identity.get('role') if user_identity else None}' 获取到系统提示词，长度: {len(sys_prompt)}")

        else:
            log.warning("[权限过滤] user_identity 为 None！将不应用任何过滤。这可能是个安全问题！")

        # 为每个原始笔记构建上下文部分
        note_contexts = []
        for note_id, chunk_list in notes_dict.items():
            # 取该笔记的第一个块获取标题等信息（假设所有块metadata一致）
            sample_chunk = chunk_list[0]
            note_title = sample_chunk["metadata"].get("source_note_title", "未知标题")
            default_author = user_display_name if user_role != 'admin' else default_personal_author
            # log.info(default_author)
            note_author = sample_chunk.get("metadata", {}).get("note_author", default_author)
            note_type = sample_chunk.get("metadata", {}).get("note_type", "个人笔记")

            # 合并该笔记所有相关块的标签字符串，拆分合并，并用集合去重
            all_tags = set()
            for chunk in chunk_list:
                tags_str = chunk.get("metadata", {}).get("tags", "")
                if tags_str:
                    all_tags.update(
                        tag.strip() for tag in tags_str.split(",") if tag.strip()
                    )
            note_tags = ",".join(list(all_tags))

            # 合并该笔记的所有相关块内容
            combined_content = "\n---\n".join([c["content"] for c in chunk_list])

            note_context = f"""【笔记：{note_title} | 类型：{note_type} | 作者：{note_author}】
    标签：{note_tags}
    相关内容：
    {combined_content}
    """
            note_contexts.append(note_context)

        history_parts = []
        # 添加对话历史（如果存在）
        if self.conversation_history:
            for hist in self.conversation_history[-3:]:  # 最近3条历史
                history_parts.append(f"问: {hist['question']}")
                history_parts.append(f"答: {hist['answer'][:200]}...")
        # 组合完整上下文
        context = f"""{sys_prompt}

    相关笔记内容：
    {chr(10).join(note_contexts)}

    对话历史：
    {chr(10).join(history_parts)}

    我的问题：{question}

    请基于以上笔记内容回答，如果笔记中没有相关信息，请说明。"""

        # 限制长度
        max_len = self.config.get(
            "context_max_length", 2000
        )  # 可适当提高，因为现在信息更密集
        # print(context)
        log.info(f"设置的最大上下文长度为：{max_len}，本次提交的上下文长度为：{len(context)}")
        if len(context) > max_len:
            context = context[:max_len] + "..."

        return context

# %% [markdown]
# ### _build_optimized_context(self, notes: List[Dict], question: str) -> str

    # %%
    def _build_optimized_context(self, notes: List[Dict], question: str) -> str:
        """构建优化上下文"""
        if not notes:
            return "没有找到相关笔记。"

        # 构建系统提示
        system_prompt = """你是我个人的笔记助手，基于Joplin笔记回答问题。
笔记记录了工作（轻行动功能饮料、习龙酱酒等）、学习、生活、想法等各种内容，请注意，因为有共享笔记本，这意味着有些笔记是同事或者朋友记录的。
如果涉及到轻行动品牌（主要是功能饮料等系列产品），注意人物关系，我（白晔峰）是轻行动运营公司的负责人，陈志伟（志伟）和张永是领导班子成员，白磊（磊帅）是轻行动商业模式的创始人
请基于以下相关笔记片段。
如果笔记中没有相关信息，请如实告知。"""

        # 构建笔记上下文
        note_contexts = []
        for i, note in enumerate(notes[:3]):  # 最多3条
            metadata = note.get("metadata", {})
            tags = metadata.get("tags", "无标签")
            summary = metadata.get("summary", "")

            # 提取最相关的片段（基于问题关键词）
            content_snippet = self._extract_relevant_snippet(
                note["content"], question, max_length=300
            )

            note_context = f"""【笔记{i + 1}】
标签：{tags}
摘要：{summary}
相关内容：{content_snippet}"""

            note_contexts.append(note_context)

        # 组合完整上下文
        context = f"""{system_prompt}

相关笔记：
{chr(10).join(note_contexts)}

我的问题：{question}

请基于以上笔记回答，如果笔记中没有相关信息，请说明。"""

        # 限制长度
        max_len = self.config.get("context_max_length", 1500)
        if len(context) > max_len:
            context = context[:max_len] + "..."

        return context

# %% [markdown]
# ### _get_relevant_notes_for_return(filtered_chunks: List)

    # %%
    def _get_relevant_notes_for_return(self, filtered_chunks: List):
        # 1. 按原始笔记ID (original_note_id) 对块进行分组
        from collections import defaultdict

        notes_dict = defaultdict(list)

        for chunk in filtered_chunks:
            metadata = chunk.get("metadata", {})
            original_note_id = metadata.get("source_note_id")

            # 如果 original_note_id 不存在，尝试用其他逻辑获取（如从chunk_id解析）
            if not original_note_id:
                # 备选方案：假设 chunk_id 格式为 “{note_id}_chunk_{index}”
                chunk_id = chunk.get("chunk_id", "")
                if "_chunk_" in chunk_id:
                    # 【修正】取分割后的第一部分作为笔记ID
                    original_note_id = chunk_id.split("_chunk_")[0]
                else:
                    original_note_id = chunk_id  # 降级处理

            notes_dict[original_note_id].append(chunk)

        # 2. 为每个原始笔记构建一条返回记录
        relevant_notes_for_return = []
        for source_note_id, chunk_list in notes_dict.items():
            if not chunk_list:
                continue

            # 取第一个块获取笔记标题（假设同笔记的所有块metadata一致）
            sample_chunk = chunk_list[0]
            sample_metadata = sample_chunk.get("metadata", {})

            # 确定笔记标题：优先用 source_note_title
            # 【优化】此处逻辑有冗余，直接获取并设置默认值即可
            note_title = sample_metadata.get("source_note_title", "未知标题")

            # 计算该笔记下所有块中的最高相似度作为此笔记的关联度
            max_similarity = max(chunk.get("similarity", 0.0) for chunk in chunk_list)

            # 聚合所有块的标签等信息
            all_tags = set()
            for chunk in chunk_list:
                tags_str = chunk.get("metadata", {}).get("tags", "")
                if tags_str:
                    all_tags.update(
                        tag.strip() for tag in tags_str.split(",") if tag.strip()
                    )

            note_entry = {
                "note_id": source_note_id,  # 返回原始笔记ID
                "title": note_title,  # 正确的来源笔记标题
                "similarity": max_similarity,  # 笔记级别的关联度
                "metadata": {
                    "aggregated_from_chunks": len(chunk_list),  # 包含几个相关块
                    "tags": ",".join(list(all_tags)),  # 聚合后的标签
                    "source_note_title": note_title,
                },
                # 如果需要，可以附加关联的块ID列表
                "related_chunk_ids": [
                    chunk.get("chunk_id") for chunk in chunk_list if chunk.get("chunk_id")
                ],
            }
            relevant_notes_for_return.append(note_entry)

        # 3. 按相似度排序
        relevant_notes_for_return.sort(key=lambda x: x["similarity"], reverse=True)

        return relevant_notes_for_return

# %% [markdown]
# ### _extract_relevant_snippet(self, content: str, question: str, max_length: int = 300) -> str

    # %%
    def _extract_relevant_snippet(
        self, content: str, question: str, max_length: int = 300
    ) -> str:
        """提取内容中最相关的片段"""
        keywords = self._extract_keywords(question)

        if not keywords:
            # 没有关键词，取开头部分
            return content[:max_length] + ("..." if len(content) > max_length else "")

        # 查找包含关键词的句子
        import re

        sentences = re.split(r"[。！？；\n]", content)

        relevant_sentences = []
        for sentence in sentences:
            for keyword in keywords:
                if keyword in sentence:
                    relevant_sentences.append(sentence.strip())
                    break

        if relevant_sentences:
            # 合并相关句子
            snippet = "。".join(relevant_sentences[:5]) + "。"
            if len(snippet) > max_length:
                snippet = snippet[:max_length] + "..."
            return snippet
        else:
            # 没有找到关键词，取开头
            return content[:max_length] + ("..." if len(content) > max_length else "")

# %% [markdown]
# ### _generate_optimized_answer(self, question: str, context: str) -> str

    # %%
    def _generate_optimized_answer(self, question: str, context: str) -> str:
        """生成优化答案"""
        # 使用DeepSeek
        if self.config["enable_deepseek"] and self.config["deepseek_api_key"]:
            log.info(f"启用deepseek聊天模式")
            return self._generate_answer_with_deepseek_optimized(question, context)
        else:
            log.info(f"启用本地ollama调用的聊天大模型{self.config['chat_model']}")
            return self._generate_answer_with_ollama(question, context)

# %% [markdown]
# ### _generate_answer_with_deepseek_optimized(self, question: str, context: str) -> str

    # %%
    def _generate_answer_with_deepseek_optimized(
        self, question: str, context: str
    ) -> str:
        """优化版的DeepSeek答案生成"""
        try:
            import requests

            headers = {
                "Authorization": f"Bearer {self.config['deepseek_api_key']}",
                "Content-Type": "application/json",
            }

            # 优化提示词
#             prompt = f"""{context}

# 回答要求：
# 1. 基于笔记事实，不要编造
# 2. 引用具体的笔记内容（如：根据笔记《1》提到...）
# 3. 如果笔记信息不完整，可以合理推断但需说明
# 4. 回答要具体、实用
# 5. 语言自然，像在对话"""

            prompt = f"""{context}

回答要求：
1. 基于笔记事实，不要编造
2. 如果笔记信息不完整，可以合理推断但需说明
3. 回答要具体、实用
4. 语言自然，像在对话"""

            payload = {
                "model": self.config["deepseek_chat_model"],
                "messages": [
                    {
                        "role": "system",
                        "content": "你是我个人的笔记助手，帮助我回忆和整理笔记内容。",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 1800,
                "top_p": 0.9,
            }

            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()

            answer = response.json()["choices"][0]["message"]["content"].strip()

            # 检查答案质量
            if len(answer) < self.config.get("min_answer_length", 30):
                log.warning(f"答案过短: {len(answer)}字符")
                # 尝试重新生成
                answer = self._regenerate_answer(question, context)

            return answer

        except Exception as e:
            log.error(f"DeepSeek生成答案失败: {e}")
            return f"抱歉，生成答案时出错: {str(e)[:100]}"

# %% [markdown]
# ### _regenerate_answer(self, question: str, context: str) -> str

    # %%
    def _regenerate_answer(self, question: str, context: str) -> str:
        """重新生成答案（当答案质量不高时）"""
        # 简化上下文，重新生成
        simplified_context = f"问题：{question}\n\n笔记摘要：{context[:500]}..."

        try:
            import requests

            headers = {
                "Authorization": f"Bearer {self.config['deepseek_api_key']}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.config["deepseek_chat_model"],
                "messages": [
                    {"role": "user", "content": f"请回答：{question}"},
                ],
                "temperature": 0.7,
                "max_tokens": 400,
            }

            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()

            answer = response.json()["choices"]["message"]["content"].strip()
            return answer

        except Exception as e:
            return "根据我的笔记，我找到了一些相关信息，但无法生成完整的回答。建议您直接查看相关笔记。"

# %% [markdown]
# ### _postprocess_answer(self, answer: str) -> str

    # %%
    def _postprocess_answer(self, answer: str) -> str:
        """后处理答案"""
        # 移除多余的空白
        import re

        answer = re.sub(r"\n\s*\n", "\n\n", answer)

        # 确保以句号结束
        if answer and answer[-1] not in [".", "。", "!", "！", "?", "？"]:
            answer += "。"

        return answer.strip()


# %% [markdown]
# ## 命令行界面


# %%
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="Joplin笔记智能问答系统")

    parser.add_argument("--question", "-q", type=str, help="直接提问的问题")

    parser.add_argument("--interactive", "-i", action="store_true", help="进入交互模式")

    parser.add_argument(
        "--model",
        type=str,
        default=CONFIG["chat_model"],
        help=f"聊天模型名称（默认：{CONFIG['chat_model']}）",
    )

    parser.add_argument(
        "--use-deepseek",
        action="store_true",
        # default=CONFIG["enable_deepseek"],
        default=False,
        help=f"使用DeepSeek API（默认：{CONFIG['enable_deepseek']}）",
    )

    parser.add_argument(
        "--max-notes",
        type=int,
        default=CONFIG["max_retrieved_notes"],
        help=f"最大检索笔记数（默认：{CONFIG['max_retrieved_notes']}）",
    )

    parser.add_argument("--clear-history", action="store_true", help="清空对话历史")

    parser.add_argument("--stats", action="store_true", help="显示系统统计信息")

    return parser.parse_args()


# %% [markdown]
# ## 交互式界面


# %%
def interactive_mode(qa_system: JoplinQASystem):
    """交互式问答模式"""
    print("\n" + "=" * 60)
    print("Joplin笔记智能问答系统 - 交互模式")
    print("=" * 60)
    print("命令说明:")
    print("  /quit 或 /exit - 退出")
    print("  /clear - 清空对话历史")
    print("  /stats - 显示统计信息")
    print("  /help - 显示帮助")
    print("=" * 60)

    while True:
        try:
            question = input("\n问: ").strip()

            if not question:
                continue

            # 处理命令
            if question.lower() in ["/quit", "/exit"]:
                print("再见！")
                break
            elif question.lower() == "/clear":
                qa_system.clear_history()
                print("对话历史已清空")
                continue
            elif question.lower() == "/stats":
                stats = qa_system.get_statistics()
                print(f"\n系统统计:")
                print(f"  数据库笔记数: {stats['total_notes_in_db']}")
                print(f"  对话历史数: {stats['conversation_history_count']}")
                print(f"  嵌入模型: {stats['config']['embedding_model']}")
                print(f"  聊天模型: {stats['config']['chat_model']}")
                print(f"  使用DeepSeek: {stats['config']['using_deepseek']}")
                continue
            elif question.lower() == "/help":
                print("\n可用命令:")
                print("  /quit, /exit - 退出程序")
                print("  /clear - 清空对话历史")
                print("  /stats - 显示系统统计")
                print("  /help - 显示此帮助")
                continue

            # 处理问题
            print("\n思考中...", end="", flush=True)

            start_time = time.time()
            result = qa_system.ask(question)
            elapsed_time = time.time() - start_time

            print(f"\r答: {result['answer']}")

            # 显示来源信息
            num_for_show = 3
            if result["is_based_on_notes"] and result["relevant_notes"]:
                print(
                    f"\n来源 ({len(result['relevant_notes'])}块相关文本块)，仅显示前{num_for_show}:"
                )
                for i, note in enumerate(result["relevant_notes"][:num_for_show], 1):
                    similarity = note["similarity"]
                    tags = note["metadata"].get("tags", "无标签")
                    print(f"  {i}. 相似度: {similarity:.2f} | 标签: {tags}")
            if result["sources"]:
                print(
                    f"\n以上文本块来源 ({len(result['sources'])}条相关笔记)，仅显示前{num_for_show}:"
                )
                for i, note in enumerate(result["sources"][:num_for_show], 1):
                    print(f"  {i}. 《{note['title']}》")

            print(
                f"\n[处理时间: {elapsed_time:.2f}秒 | 上下文长度: {result.get('context_length', 0)}字符]"
            )

        except KeyboardInterrupt:
            print("\n\n退出交互模式")
            break
        except Exception as e:
            print(f"\n错误: {e}")


# %% [markdown]
# ## 主函数


# %%
@timethis
def main():
    """主函数"""
    args = parse_args()

    # 更新配置
    from joplinai import CONFIG as config

    dynamic_config = {**CONFIG.copy(), **config}
    dynamic_config["chat_model"] = args.model
    dynamic_config["enable_deepseek"] = args.use_deepseek
    dynamic_config["max_retrieved_notes"] = args.max_notes

    # 初始化问答系统
    log.info("初始化Joplin问答系统...")
    # qa_system = JoplinQASystem(dynamic_config)
    qa_system = OptimizedJoplinQASystem(dynamic_config)

    # 检查向量数据库
    if not qa_system.vector_db.collection:
        log.error("向量数据库未找到或无法加载！")
        log.error("请先运行 joplinai.py 进行笔记向量化")
        return

    # 处理命令
    if args.clear_history:
        qa_system.clear_history()
        print("对话历史已清空")
        return

    if args.stats:
        stats = qa_system.get_statistics()
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return

    # 处理直接提问
    if args.question:
        print(f"问题: {args.question}")
        print("\n思考中...")

        result = qa_system.ask(args.question)

        print(f"\n答案: {result['answer']}")

        if result["is_based_on_notes"] and result["relevant_notes"]:
            print(f"\n基于 {len(result['relevant_notes'])} 条相关文本块:")
            for note in result["relevant_notes"]:
                print(f"  - 相似度: {note['similarity']:.2f}")

        return

    # 进入交互模式
    if args.interactive or (
        not args.question and not args.stats and not args.clear_history
    ):
        interactive_mode(qa_system)
        return


# %% [markdown]
# ## 主程序入口

# %%
if __name__ == "__main__":
    main()
