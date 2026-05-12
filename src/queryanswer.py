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
# ## 问答系统核心


# %%
class JoplinQASystem:
    """Joplin笔记问答系统"""

# %% [markdown]
# ### __init__(self, config: Dict = None)

    # %%
    def __init__(self, config: Dict = None):
        from joplinai import CONFIG as CONFIG_JA
        from queryanswer import CONFIG as CONFIG_QA
        config_all = {**CONFIG_JA, **CONFIG_QA}
        if config:
            config_all.update(config)
        self.config = config_all
        self.vector_db = VectorDBManager(
            self.config["db_path"], self.config["embedding_model"], for_creation=False
        )
        self.conversation_history = []

        from aimod.embedding_generator import EmbeddingGenerator
        self.embedding_generator = EmbeddingGenerator(
            self.config,
            model_name=self.config["embedding_model"],
        )
        log.info(
            f"JoplinQASystem 初始化完成，已加载 embedding_generator，"
            f"嵌入模型为：{self.config['embedding_model']}"
        )

# %% [markdown]
# ### ask(self, question: str, use_history: bool = True) -> Dict

    # %%
    def ask(self, question: str, use_history: bool = True, user_identity: Optional[Dict] = None) -> Dict:
        """提问入口，基于块进行检索和回答。
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

        # 3. 搜索相似块
        similar_chunks = self.vector_db.search_similar_chunks(
            query_embedding,
            limit=self.config.get("max_retrieved_chunks", 15),
            user_identity=user_identity
        )

        # 4. 过滤和重排序块
        filtered_chunks = self._filter_and_rank_chunks(similar_chunks, question)

        # 5. 构建优化上下文（基于块）
        context = self._build_optimized_context_from_chunks(
            filtered_chunks,
            question,
            user_identity=user_identity
        )
        log.debug(f"过滤后块数: {len(filtered_chunks)}")
        log.debug(f"构建的上下文长度: {len(context)}")

        # 6. 生成答案
        answer = self._generate_optimized_answer(question, context)
        final_answer = self._postprocess_answer(answer)

        # 构建兼容的返回字典：将 filtered_chunks 转换为 relevant_notes 格式
        relevant_notes_for_return = self._get_relevant_notes_for_return(filtered_chunks)
        # 更新对话历史
        if use_history:
            self.conversation_history.append(
                {
                    "question": question,
                    "answer": answer,
                    "timestamp": datetime.now().isoformat(),
                    "relevant_note_ids": [note["note_id"] for note in relevant_notes_for_return],
                }
            )
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]

        sources = self._extract_sources(relevant_notes_for_return)

        return {
            "question": question,
            "answer": final_answer,
            "relevant_notes": relevant_notes_for_return,
            "sources": sources,
            "relevant_chunks": filtered_chunks,
            "context_length": len(context),
            "is_based_on_notes": len(filtered_chunks) > 0,
        }

# %% [markdown]
# ### _preprocess_question(self, question: str) -> str

    # %%
    def _preprocess_question(self, question: str) -> str:
        """预处理问题，提高检索效果"""
        question = question.lower()
        stop_words = ["请问", "请", "帮我", "我想知道", "什么是", "怎么", "如何"]
        for word in stop_words:
            question = question.replace(word, "")

        if "我" in question or "我的" in question:
            question = "个人笔记 " + question

        return question.strip()

# %% [markdown]
# ### _extract_keywords(self, text: str) -> List[str]

    # %%
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词（简单实现）"""
        import re
        text = re.sub(r"[^\w\s]", "", text)
        words = text.split()
        from collections import Counter
        word_counts = Counter(words)
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

        chunks.sort(key=lambda x: x["similarity"], reverse=True)

        threshold = self.config.get("similarity_threshold", 0.6)
        filtered = [chunk for chunk in chunks if chunk["similarity"] >= threshold]

        if len(filtered) < 3 and len(chunks) > 0:
            filtered = chunks[:3]

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
# ### _build_optimized_context_from_chunks(self, chunks, question, user_identity)

    # %%
    def _build_optimized_context_from_chunks(
        self, chunks: List[Dict],
        question: str,
        user_identity: Optional[Dict] = None,
    ) -> str:
        """基于检索到的块构建问答上下文。"""
        if not chunks:
            return "没有找到相关笔记内容。"

        from collections import defaultdict

        notes_dict = defaultdict(list)
        for chunk in chunks:
            note_id = chunk["source_note_id"]
            notes_dict[note_id].append(chunk)

        if user_identity:
            user_role = user_identity.get('role')
            user_display_name = user_identity.get('display_name')
            log.debug(f"[权限过滤] 解析身份 -> role: '{user_role}', display_name: '{user_display_name}'")

            default_personal_author = getinivaluefromcloud("joplinai", "default_personal_author")

            import re
            split_ptn = re.compile(r"[,，]")
            if (colleague_str := getinivaluefromcloud("joplinai", "colleague")):
                colleague = [title.strip() for title in split_ptn.split(colleague_str)]
            else:
                colleague = ["XXA", "XXB"]
            colleague_str = "，".join([f"“{person}”" for person in colleague])

            sys_prompt = PromptManager.get_sys_prompt_for_role(user_identity)
            log.debug(f"[提示词管理] 为用户角色 '{user_identity.get('role') if user_identity else None}' 获取到系统提示词，长度: {len(sys_prompt)}")

        else:
            log.warning("[权限过滤] user_identity 为 None！将不应用任何过滤。这可能是个安全问题！")

        note_contexts = []
        for note_id, chunk_list in notes_dict.items():
            sample_chunk = chunk_list[0]
            note_title = sample_chunk["metadata"].get("source_note_title", "未知标题")
            default_author = user_display_name if user_role != 'admin' else default_personal_author
            note_author = sample_chunk.get("metadata", {}).get("note_author", default_author)
            note_type = sample_chunk.get("metadata", {}).get("note_type", "个人笔记")

            all_tags = set()
            for chunk in chunk_list:
                tags_str = chunk.get("metadata", {}).get("tags", "")
                if tags_str:
                    all_tags.update(
                        tag.strip() for tag in tags_str.split(",") if tag.strip()
                    )
            note_tags = ",".join(list(all_tags))

            combined_content = "\n---\n".join([c["content"] for c in chunk_list])

            note_context = f"""【笔记：{note_title} | 类型：{note_type} | 作者：{note_author}】
标签：{note_tags}
相关内容：
{combined_content}
"""
            note_contexts.append(note_context)

        history_parts = []
        if self.conversation_history:
            history_parts.append("对话历史：")
            for hist in self.conversation_history[-3:]:
                history_parts.append(f"问: {hist['question']}")
                history_parts.append(f"答: {hist['answer'][:200]}...")

        context = f"""{sys_prompt}

相关笔记内容：
{chr(10).join(note_contexts)}

{chr(10).join(history_parts)}

我的问题：{question}

请基于以上笔记内容回答，如果笔记中没有相关信息，请说明。"""

        max_len = self.config.get("context_max_length", 2000)
        log.info(f"设置的最大上下文长度为：{max_len}，本次提交的上下文长度为：{len(context)}")
        if len(context) > max_len:
            context = context[:max_len] + "..."

        return context

# %% [markdown]
# ### _get_relevant_notes_for_return(filtered_chunks: List)

    # %%
    def _get_relevant_notes_for_return(self, filtered_chunks: List):
        """将过滤后的块按原始笔记聚合，构建兼容的返回格式"""
        from collections import defaultdict

        notes_dict = defaultdict(list)

        for chunk in filtered_chunks:
            metadata = chunk.get("metadata", {})
            original_note_id = metadata.get("source_note_id")

            if not original_note_id:
                chunk_id = chunk.get("chunk_id", "")
                if "_chunk_" in chunk_id:
                    original_note_id = chunk_id.split("_chunk_")[0]
                else:
                    original_note_id = chunk_id

            notes_dict[original_note_id].append(chunk)

        relevant_notes_for_return = []
        for source_note_id, chunk_list in notes_dict.items():
            if not chunk_list:
                continue

            sample_chunk = chunk_list[0]
            sample_metadata = sample_chunk.get("metadata", {})

            note_title = sample_metadata.get("source_note_title", "未知标题")

            max_similarity = max(chunk.get("similarity", 0.0) for chunk in chunk_list)

            all_tags = set()
            for chunk in chunk_list:
                tags_str = chunk.get("metadata", {}).get("tags", "")
                if tags_str:
                    all_tags.update(
                        tag.strip() for tag in tags_str.split(",") if tag.strip()
                    )

            note_entry = {
                "note_id": source_note_id,
                "title": note_title,
                "similarity": max_similarity,
                "metadata": {
                    "aggregated_from_chunks": len(chunk_list),
                    "tags": ",".join(list(all_tags)),
                    "source_note_title": note_title,
                },
                "related_chunk_ids": [
                    chunk.get("chunk_id") for chunk in chunk_list if chunk.get("chunk_id")
                ],
            }
            relevant_notes_for_return.append(note_entry)

        relevant_notes_for_return.sort(key=lambda x: x["similarity"], reverse=True)

        return relevant_notes_for_return

# %% [markdown]
# ### _generate_optimized_answer(self, question: str, context: str) -> str

    # %%
    def _generate_optimized_answer(self, question: str, context: str) -> str:
        """生成优化答案"""
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

            if len(answer) < self.config.get("min_answer_length", 30):
                log.warning(f"答案过短: {len(answer)}字符")
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

            answer = response.json()["choices"][0]["message"]["content"].strip()
            return answer

        except Exception as e:
            return "根据我的笔记，我找到了一些相关信息，但无法生成完整的回答。建议您直接查看相关笔记。"

# %% [markdown]
# ### _postprocess_answer(self, answer: str) -> str

    # %%
    def _postprocess_answer(self, answer: str) -> str:
        """后处理答案"""
        import re
        answer = re.sub(r"\n\s*\n", "\n\n", answer)
        if answer and answer[-1] not in [".", "。", "!", "！", "?", "？"]:
            answer += "。"
        return answer.strip()

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
    qa_system = JoplinQASystem(dynamic_config)

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
