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
# ## 导入核心库

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
    from func.configpr import (
        findvaluebykeyinsection,
        getcfpoptionvalue,
        setcfpoptionvalue,
    )
    from func.first import dirmainpath, getdirmain
    from func.getid import getdeviceid, getdevicename, gethostuser
    from func.jpfuncs import (
        getinivaluefromcloud,
        getnote,
    )
    from func.logme import log
    from func.sysfunc import execcmd, not_IPython
    from func.wrapfuncs import timethis
    from vector_db_manager import VectorDBManager
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
    "chat_model": "qwen:1.8b",  # 聊天模型（用于问答）
    "db_path": getdirmain() / "data" / "joplin_vector_db",  # ChromaDB存储路径
    "max_retrieved_notes": 5,  # 最大检索笔记数量
    "similarity_threshold": 0.3,  # 相似度阈值
    "enable_deepseek": False,  # 是否使用DeepSeek进行问答
    "deepseek_api_key": getinivaluefromcloud("joplinai", "deepseek_token"),
    "deepseek_chat_model": "deepseek-chat",
    "context_max_length": 4000,  # 上下文最大长度（字符）
}

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

    def __init__(self, config: Dict):
        self.config = config
        # for_creation=False表示用于查询
        self.vector_db = VectorDBManager(
            config["db_path"], config["embedding_model"], for_creation=False
        )
        self.conversation_history = []  # 对话历史

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

    def _build_context(self, notes: List[Dict], question: str) -> str:
        """构建问答上下文"""
        context_parts = []

        # 添加系统提示
        system_prompt = """你是一个基于Joplin笔记的智能助手。请基于以下笔记内容回答用户的问题。
如果笔记中没有相关信息，请如实告知。回答时请引用相关笔记的内容。"""
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
                options={"temperature": 0.3, "num_predict": 1000},
            )

            answer = response["message"]["content"]
            log.info(f"使用Ollama模型 {model_name} 生成答案，长度: {len(answer)}")
            return answer

        except Exception as e:
            log.error(f"Ollama生成答案失败: {e}")
            return f"抱歉，生成答案时出现错误: {str(e)}"

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

    def _generate_generic_answer(self, question: str) -> str:
        """生成通用回答（当没有相关笔记时）"""
        generic_responses = [
            "我在您的笔记中没有找到相关信息。您可以尝试：\n1. 添加相关笔记\n2. 重新表述问题\n3. 检查笔记是否已正确向量化",
            "目前笔记库中没有与此问题直接相关的内容。建议您完善相关主题的笔记。",
            "未找到相关笔记信息。请确保相关笔记已导入并完成向量化处理。",
        ]

        import random

        return random.choice(generic_responses)

    def _extract_sources(self, notes: List[Dict]) -> List[Dict]:
        """提取来源信息"""
        sources = []
        for note in notes:
            metadata = note.get("metadata", {})
            source = {
                "note_id": note["note_id"],
                "similarity": note["similarity"],
                "tags": metadata.get("tags", "").split(",")
                if metadata.get("tags")
                else [],
                "has_summary": bool(metadata.get("summary")),
            }
            sources.append(source)
        return sources

    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        log.info("对话历史已清空")

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
        default=CONFIG["enable_deepseek"],
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
            if result["is_based_on_notes"] and result["relevant_notes"]:
                print(f"\n来源 ({len(result['relevant_notes'])}条相关笔记):")
                for i, note in enumerate(result["relevant_notes"], 1):
                    similarity = note["similarity"]
                    tags = note["metadata"].get("tags", "无标签")
                    print(f"  {i}. 相似度: {similarity:.2f} | 标签: {tags}")

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
    dynamic_config = CONFIG.copy()
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
            print(f"\n基于 {len(result['relevant_notes'])} 条相关笔记:")
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
