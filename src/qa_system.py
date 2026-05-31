# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     split_at_heading: true
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
import json
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional

import jieba
import ollama

# %% [markdown]
# ### 项目模块导入

# %%
import pathmagic

with pathmagic.Context():
    try:
        from aimod.vector_db_manager import VectorDBManager
        from func.datatools import compute_content_hash
        from func.jpfuncs import getinivaluefromcloud
        from func.logme import log
        from src.prompt_manager import PromptManager
        from src.qa_config import CONFIG
    except ImportError as e:
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger(__name__)
        log.error(f"导入项目模块失败: {e}")

# %% [markdown]
# ## 问答系统核心


# %%
__all__ = ["QASystem"]

class QASystem:
    """Joplin笔记问答系统"""

# %% [markdown]
# ### __init__(self, config: Dict = None)

    # %%
    def __init__(self, config: Dict = None):
        from joplinai import CONFIG as CONFIG_JA
        from src.qa_config import CONFIG as CONFIG_QA
        config_all = {**CONFIG_JA, **CONFIG_QA}
        if config:
            config_all.update(config)
        self.config = config_all
        self.vector_db = VectorDBManager(
            self.config["db_path"], self.config["ollama_embedding_model"], for_creation=False
        )
        self.conversation_history = []

        from aimod.embedding_generator import EmbeddingGenerator
        self.embedding_generator = EmbeddingGenerator(
            self.config,
            model_name=self.config["ollama_embedding_model"],
        )
        log.info(
            f"QASystem 初始化完成，已加载 embedding_generator，"
            f"嵌入模型为：{self.config['ollama_embedding_model']}"
        )

    def __repr__(self):
        return f"QASystem(embed={self.config.get('ollama_embedding_model', '?')}, chat={self.config.get('qa_ollama_chat_model', '?')})"

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
        if not user_identity:
            return {"answer": "未授权：缺少用户身份，请先登录。"}

        # 1. 预处理问题
        processed_question = self._preprocess_question(question)

        # 2. 获取问题嵌入（HyDE 优先，跳过独立嵌入）
        if self.config.get("hyde_enabled", True):
            hyde = self._generate_hyde(question)
            if hyde:
                query_embedding = self._fuse_hyde_embedding(processed_question, hyde)
            else:
                query_embedding = None

            if not query_embedding:
                # HyDE 失败→回退到独立问题嵌入
                log.info("[HyDE] 失败，回退到独立问题嵌入")
                query_embedding = self._embed_single(processed_question)
        else:
            query_embedding = self._embed_single(processed_question)

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

        # 4.5 LLM 精排
        if self.config.get("rerank_enabled", True):
            before = [c.get("metadata", {}).get("source_note_title", "") for c in filtered_chunks[:5]]
            filtered_chunks = self._rerank_by_llm(filtered_chunks, question)
            after = [c.get("metadata", {}).get("source_note_title", "") for c in filtered_chunks[:5]]
            if before != after:
                log.info(f"[精排] 排名变化: {before[:3]} → {after[:3]}")

        # 5. 构建优化上下文（基于块）
        context = self._build_optimized_context_from_chunks(
            filtered_chunks,
            question,
            user_identity=user_identity
        )
        log.debug(f"过滤后块数: {len(filtered_chunks)}")
        log.debug(f"构建的上下文长度: {len(context)}")

        # 6. 生成答案
        answer, gen_meta = self._generate_optimized_answer(question, context)
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
            "gen_meta": gen_meta,
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
        """提取关键词"""
        try:
            words = jieba.lcut(text)
        except Exception:
            words = text.split()
        words = [w.strip() for w in words if re.search(r"\w", w) and len(w.strip()) > 1]
        from collections import Counter
        word_counts = Counter(words)
        stop_words = {"的", "了", "在", "是", "我", "你", "他", "她", "它", "这", "那"}
        keywords = [
            word
            for word, count in word_counts.most_common(10)
            if word not in stop_words
        ]
        return keywords[:5]

# %% [markdown]
# ### HyDE (Hypothetical Document Embedding)

    # %%
    def _generate_hyde(self, question: str) -> dict:
        """调用云端 LLM 生成 search_query + 假设答案，失败返回 None"""
        if self.config.get("cloud_model", "none") == "none" or not self.config.get("cloud_api_key"):
            return None
        try:
            import requests
            prompt = (
                "你是个人笔记助手。用户提问后，请帮助优化向量检索。\n"
                f"用户问题：{question}\n\n"
                "请生成两个字段：\n"
                "1. search_query：3-8个关键词，用于向量检索\n"
                "2. hypothetical_answer：假设用户笔记中可能如何记录这个主题，"
                "生成一段100-200字的假设笔记\n\n"
                '严格返回JSON：{"search_query": "...", "hypothetical_answer": "..."}'
            )
            resp = requests.post(
                self.config.get("cloud_api_url", "https://api.deepseek.com/v1/chat/completions"),
                headers={
                    "Authorization": f"Bearer {self.config['cloud_api_key']}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config["cloud_model"],
                    "messages": [
                        {"role": "system", "content": "你是检索优化助手，只返回JSON。"},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 500,
                },
                timeout=30,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            # 提取JSON
            import re as _re
            m = _re.search(r"\{[^{}]*\}", text, _re.DOTALL)
            if m:
                result = json.loads(m.group())
                if result.get("search_query") and result.get("hypothetical_answer"):
                    log.info(f"[HyDE] search_query={result['search_query'][:60]}")
                    return result
            log.warning(f"[HyDE] 解析失败: {text[:200]}")
            return None
        except Exception as e:
            log.warning(f"[HyDE] 生成失败: {e}")
            return None

    # %%
    def _embed_single(self, text: str) -> Optional[List[float]]:
        """嵌入单个文本（HyDE 回退 / HyDE 关闭时使用）。"""
        chunk_dict = {
            "content": text,
            "base_metadata": {
                "content_hash": compute_content_hash(text),
                "source_note_title": text[:20],
                "chunk_index": 0,
            },
        }
        return self.embedding_generator.get_merged_embedding(chunk_dict)

    # %%
    def _fuse_hyde_embedding(self, question: str, hyde: dict) -> Optional[List[float]]:
        """融合原始问题 + search_query + 假设答案的嵌入向量（批量嵌入优化）。

        将3个文本批量发送给Ollama，一次请求替代3次串行调用，省约6-9秒。
        """
        texts = [
            (question, 0.3),
            (hyde["search_query"], 0.4),
            (hyde["hypothetical_answer"], 0.3),
        ]

        # 预处理所有文本后批量嵌入
        prepped = [
            self.embedding_generator.text_prep.preprocess_for_embedding(t)
            for t, _ in texts
        ]

        try:
            vectors = self.embedding_generator.get_ollama_embeddings_batch(prepped)
        except Exception as e:
            log.warning(f"[HyDE] 批量嵌入失败: {e}，降级为原始问题嵌入")
            return None

        if not vectors or len(vectors) != len(texts):
            log.warning(
                f"[HyDE] 批量嵌入返回数量异常: {len(vectors) if vectors else 0}"
            )
            return None

        dim = len(vectors[0])
        fused = [0.0] * dim
        for i, (_, weight) in enumerate(texts):
            for j in range(dim):
                fused[j] += vectors[i][j] * weight
        log.info(f"[HyDE] 三向量融合完成 dim={dim}")
        return fused

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
                tags = (chunk.get("metadata", {}).get("tags") or "").lower()
                summary = (chunk.get("metadata", {}).get("summary") or "").lower()
                for keyword in keywords:
                    if keyword in content:
                        score += 0.10
                    if keyword in tags:
                        score += 0.15
                    if keyword in summary:
                        score += 0.15
                scored_chunks.append((score, chunk))
            scored_chunks.sort(key=lambda x: x[0], reverse=True)
            filtered = [chunk for _, chunk in scored_chunks]

        return filtered

# %% [markdown]
# ### LLM 精排 (RankGPT 风格)

    # %%
    def _rerank_by_llm(self, chunks: List[Dict], question: str) -> List[Dict]:
        """LLM 直接对候选块打分排序，失败返回原序"""
        if len(chunks) <= 3:
            return chunks
        if self.config.get("cloud_model", "none") == "none" or not self.config.get("cloud_api_key"):
            return chunks
        try:
            import requests

            candidates = []
            for i, c in enumerate(chunks[:10]):
                meta = c.get("metadata", {})
                title = meta.get("source_note_title", "?")
                tags = meta.get("tags", "N/A")
                content_preview = c["content"][:200].replace("\n", " ")
                candidates.append(
                    f"[{i}] 【{title}】\n标签：{tags}\n内容：{content_preview}..."
                )

            prompt = (
                f"请根据问题对以下笔记片段进行相关性排序，返回最相关片段的编号列表。\n\n"
                f"问题：{question}\n\n"
                + "\n\n".join(candidates)
                + "\n\n按相关性从高到低列出编号（逗号分隔，如 0,3,1,5,2,...）："
            )

            resp = requests.post(
                self.config.get("cloud_api_url", "https://api.deepseek.com/v1/chat/completions"),
                headers={
                    "Authorization": f"Bearer {self.config['cloud_api_key']}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config["cloud_model"],
                    "messages": [
                        {"role": "system", "content": "你是排序助手。只返回逗号分隔的数字编号，不要解释。"},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 4000,
                },
                timeout=60,
            )
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            # 提取数字
            import re as _re
            indices = [int(x) for x in _re.findall(r"\d+", text)]
            if len(indices) < 3:
                log.warning(f"[精排] 解析结果不足3个: {text[:100]}")
                return chunks

            # 按LLM排序重建列表
            ranked = []
            seen = set()
            for idx in indices:
                if 0 <= idx < len(chunks) and idx not in seen:
                    ranked.append(chunks[idx])
                    seen.add(idx)
            # 未排到的追加在后面
            for i, c in enumerate(chunks):
                if i not in seen:
                    ranked.append(c)
            log.info(f"[精排] LLM重排完成 top-3: {indices[:3]}")
            return ranked
        except Exception as e:
            log.warning(f"[精排] 失败，保留原序: {e}")
            return chunks

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

            parts = []
            for c in chunk_list:
                s = c.get("metadata", {}).get("summary", "")
                if s:
                    parts.append(f"[摘要] {s}\n{c['content']}")
                else:
                    parts.append(c["content"])
            combined_content = "\n---\n".join(parts)

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

        max_len = int(self.config.get("max_context_chars", 2000))
        log.info(f"设置的最大上下文长度为：{max_len}，本次提交的上下文长度为：{len(context)}")
        if len(context) > max_len:
            tail = f"{chr(10).join(history_parts)}\n\n我的问题：{question}\n\n请基于以上笔记内容回答，如果笔记中没有相关信息，请说明。"
            head = f"{sys_prompt}\n\n相关笔记内容：\n"
            fixed_len = len(head) + len(tail)
            note_budget = max_len - fixed_len - 20
            if note_budget > 500:
                truncated_notes = []
                used = 0
                for nc in note_contexts:
                    if used + len(nc) <= note_budget:
                        truncated_notes.append(nc)
                        used += len(nc)
                    else:
                        remaining = note_budget - used
                        if remaining > 200:
                            truncated_notes.append(nc[:remaining] + "\n...")
                        truncated_notes.append("（后续笔记因上下文长度限制已省略）")
                        break
                context = head + chr(10).join(truncated_notes) + "\n" + tail
            else:
                context = context[:max_len] + "..."
            log.info(f"上下文截断: {len(context)}/{max_len} 字符")

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
    def _generate_optimized_answer(self, question: str, context: str):
        """生成优化答案，返回 (answer, gen_meta)"""
        if self.config["cloud_model"] != "none" and self.config["cloud_api_key"]:
            log.info(f"启用云端聊天模式")
            return self._generate_answer_with_cloud(question, context)
        else:
            log.info(f"启用本地ollama调用的聊天大模型{self.config['qa_ollama_chat_model']}")
            return self._generate_answer_with_ollama(question, context)

# %% [markdown]
# ### _generate_answer_with_cloud(self, question: str, context: str) -> str

    # %%
    def _generate_answer_with_cloud(
        self, question: str, context: str
    ):
        """优化版的云端答案生成，返回 (answer, gen_meta)"""
        try:
            import requests

            headers = {
                "Authorization": f"Bearer {self.config['cloud_api_key']}",
                "Content-Type": "application/json",
            }

            prompt = f"""{context}

回答要求：
1. 基于笔记事实，不要编造
2. 如果笔记信息不完整，可以合理推断但需说明
3. 回答要具体、实用
4. 语言自然，像在对话"""

            max_output_tokens = self.config.get("max_output_tokens", 4096)
            model = self.config["cloud_model"]
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "你是我个人的笔记助手，帮助我回忆和整理笔记内容。",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
                "max_tokens": max_output_tokens,
                "top_p": 0.9,
            }

            response = requests.post(
                self.config.get("cloud_api_url", "https://api.deepseek.com/v1/chat/completions"),
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()

            answer = response.json()["choices"][0]["message"]["content"].strip()
            finish = response.json()["choices"][0].get("finish_reason", "?")

            gen_meta = {
                "provider": "cloud",
                "model": model,
                "max_output_tokens": max_output_tokens,
                "finish_reason": finish,
                "answer_chars": len(answer),
            }
            log.info(f"云端生成答案: {len(answer)}字, finish={finish}, max_tokens={max_output_tokens}")

            if len(answer) < self.config.get("min_answer_length", 30):
                log.warning(f"答案过短: {len(answer)}字符")
                answer = self._regenerate_answer(question, context)
                gen_meta["regenerated"] = True
                gen_meta["answer_chars"] = len(answer)

            return answer, gen_meta

        except Exception as e:
            resp_body = ""
            try:
                resp_body = e.response.text[:500]
            except Exception:
                pass
            log.error(f"云端模型生成答案失败: {e} | body: {resp_body}")
            return f"抱歉，生成答案时出错: {str(e)[:100]}", {
                "provider": "cloud",
                "model": self.config.get("cloud_model", "?"),
                "error": str(e)[:200],
                "answer_chars": 0,
            }

# %% [markdown]
# ### _regenerate_answer(self, question: str, context: str) -> str

    # %%
    def _regenerate_answer(self, question: str, context: str) -> str:
        """重新生成答案（当答案质量不高时）"""
        simplified_context = f"问题：{question}\n\n笔记摘要：{context[:500]}..."

        try:
            import requests

            headers = {
                "Authorization": f"Bearer {self.config['cloud_api_key']}",
                "Content-Type": "application/json",
            }

            payload = {
                "model": self.config["cloud_model"],
                "messages": [
                    {"role": "user", "content": f"请回答：{question}"},
                ],
                "temperature": 0.7,
                "max_tokens": 400,
            }

            response = requests.post(
                self.config.get("cloud_api_url", "https://api.deepseek.com/v1/chat/completions"),
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()

            answer = response.json()["choices"][0]["message"]["content"].strip()
            finish = response.json()["choices"][0].get("finish_reason", "?")

            log.info(f"重新生成答案: {len(answer)}字, finish={finish}, max_tokens={payload['max_tokens']}")
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
    def _generate_answer_with_ollama(self, question: str, context: str):
        """使用本地Ollama生成答案，返回 (answer, gen_meta)"""
        try:
            model_name = self.config["qa_ollama_chat_model"]

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
            gen_meta = {
                "provider": "ollama",
                "model": model_name,
                "max_output_tokens": 2000,
                "finish_reason": response.get("done_reason", "stop"),
                "answer_chars": len(answer),
            }
            log.info(f"使用Ollama模型 {model_name} 生成答案，长度: {len(answer)}")
            return answer, gen_meta

        except Exception as e:
            log.error(f"Ollama生成答案失败: {e}")
            return f"抱歉，生成答案时出现错误: {str(e)}", {
                "provider": "ollama",
                "model": self.config.get("qa_ollama_chat_model", "?"),
                "error": str(e)[:200],
                "answer_chars": 0,
            }

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
                    "ollama_embedding_model": self.config["ollama_embedding_model"],
                    "qa_ollama_chat_model": self.config["qa_ollama_chat_model"],
                    "using_cloud": self.config["cloud_model"] != "none",
                },
            }
        except:
            return {
                "total_notes_in_db": 0,
                "conversation_history_count": len(self.conversation_history),
            }


if __name__ == "__main__":
    from src.qa_cli import main
    main()
