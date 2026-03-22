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

# %%
"""
joplinai.py | 优化版：AI增强Joplin笔记管理（向量化+增量更新+并发加速）
核心功能：将Joplin笔记向量化存储到ChromaDB，支持语义检索；仅增量处理更新笔记。
"""

# %%
# -------------------------- 1. 配置区（集中管理参数） -------------------------
import hashlib
import json
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
import ollama
from chromadb.config import Settings

# 项目模块导入（根据实际路径调整）
try:
    from func.configpr import (
        findvaluebykeyinsection,
        getcfpoptionvalue,
        setcfpoptionvalue,
    )
    from func.first import dirmainpath, getdirmain
    from func.getid import getdeviceid, getdevicename, gethostuser
    from func.jpfuncs import (
        createnote,
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
    from func.sysfunc import execcmd, not_IPython
    from func.wrapfuncs import timethis
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    log.error(f"导入项目模块失败: {e}")

# %%
# 核心配置（根据需求调整）
CONFIG = {
    "notebook_title": "顺风顺水",  # 目标笔记本名称
    # "embedding_model": "nomic-embed-text",  # 嵌入模型（Ollama本地模型，优先选nomic-embed-text）
    "embedding_model": "qwen:1.8b",  # 嵌入模型（Ollama本地模型，优先选nomic-embed-text）
    "chunk_size": 4000,  # 文本分块大小（字符数，根据模型上下文调整，nomic支持8192）
    "max_context": 8000,  # 模型最大上下文（字符数，nomic-embed-text为8192）
    "concurrency_type": "thread",  # 并发类型："thread"（多线程，I/O密集型）/"process"（多进程，CPU密集型）
    "max_workers": 4,  # 并发数（线程/进程数）
    "db_path": getdirmain() / "data" / "joplin_vector_db",  # ChromaDB存储路径
    "state_path": getdirmain()
    / "data"
    / "joplin_process_state.json",  # 处理状态文件路径
    "log_level": logging.INFO,  # 日志级别
}


# %%
def clean_text(text: str) -> str:
    """清理笔记文本：移除图片、格式符号、多余换行"""
    if not text:
        return ""
    # 移除图片链接：![alt](url)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # 移除Markdown格式符号：#、`、*、>、~、-等
    text = re.sub(r"[#*`>~-]", "", text)
    # 合并多余换行（3个以上换行→2个）
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# %%
def compute_content_hash(title: str, body: str) -> str:
    """计算笔记内容哈希（用于增量更新判断）"""
    content = f"{title}{body}"
    return hashlib.md5(content.encode("utf-8")).hexdigest()


# %%
def load_process_state(state_path: Path) -> Dict[str, Dict]:
    """加载处理状态（笔记ID→{更新时间, 哈希, 处理时间}）"""
    if state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.error(f"加载状态文件失败: {e}，将重置状态")
    return {}


# %%
def save_process_state(state: Dict, state_path: Path):
    """保存处理状态到JSON文件"""
    try:
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error(f"保存状态文件失败: {e}")


# %%
# -------------------------- 3. 向量数据库操作（ChromaDB） -------------------------
class VectorDBManager:
    """ChromaDB向量数据库管理器（封装集合操作）"""

    def __init__(self, db_path: Path, collection_name: str = "joplin_notes"):
        self.client = chromadb.PersistentClient(
            path=str(db_path), settings=Settings(anonymized_telemetry=False)
        )
        self.collection_name = collection_name
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """获取或创建集合（支持余弦相似度）"""
        try:
            return self.client.get_collection(name=self.collection_name)
        except Exception:
            log.info(f"集合 {self.collection_name} 不存在，创建新集合")
            return self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},  # 余弦相似度
            )

    def upsert_note(
        self, note_id: str, text: str, embedding: List[float], tags: List[str]
    ):
        """插入/更新笔记向量数据"""
        self.collection.upsert(
            ids=[note_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[{"note_id": note_id, "tags": ",".join(tags)}],
        )

    def delete_note(self, note_id: str):
        """删除笔记向量数据"""
        self.collection.delete(ids=[note_id])


# %%
# -------------------------- 4. 嵌入生成（Ollama API调用+并发） -------------------------
def get_ollama_embedding(text: str, model_name: str, max_context: int) -> List[float]:
    """调用Ollama模型生成单文本嵌入（自动截断超长文本）"""
    # 截断文本（保留前max_context字符，避免超过模型上下文）
    truncated_text = text[:max_context] if len(text) > max_context else text
    if len(text) > max_context:
        log.warning(f"文本过长（{len(text)}字符），已截断至{max_context}字符")

    try:
        response = ollama.embeddings(model=model_name, prompt=truncated_text)
        return response["embedding"]
    except Exception as e:
        log.error(
            f"Ollama嵌入生成失败（模型: {model_name}，文本长度: {len(truncated_text)}）: {e}"
        )
        return []  # 返回空嵌入（后续可跳过该笔记）


# %%
def split_text(text: str, chunk_size: int) -> List[str]:
    """将长文本按chunk_size分块（最后一块可能不足）"""
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


# %%
def get_merged_embedding(
    text: str, model_name: str, chunk_size: int, max_context: int
) -> List[float]:
    """生成合并嵌入（短文本直接嵌入，长文本分块平均合并）"""
    total_len = len(text)
    if total_len <= max_context:
        # 短文本：直接生成嵌入
        return get_ollama_embedding(text, model_name, max_context)
    else:
        # 长文本：分块生成嵌入→平均合并
        chunks = split_text(text, chunk_size)
        log.info(f"长文本分块: {len(chunks)}块（每块{chunk_size}字符）")
        # 并发生成各块嵌入（根据配置选线程/进程）
        chunk_embeddings = concurrent_generate_embeddings(
            chunks,
            model_name,
            max_context,
            CONFIG["concurrency_type"],
            CONFIG["max_workers"],
        )
        if not chunk_embeddings or any(not emb for emb in chunk_embeddings):
            log.error("部分块嵌入生成失败，跳过该笔记")
            return []
        # 按维度平均合并（假设所有嵌入向量维度相同）
        return [sum(dim) / len(chunk_embeddings) for dim in zip(*chunk_embeddings)]


# %%
def concurrent_generate_embeddings(
    texts: List[str],
    model_name: str,
    max_context: int,
    concurrency_type: str,
    max_workers: int,
) -> List[List[float]]:
    """并发生成多个文本的嵌入（支持多线程/多进程）"""
    if concurrency_type == "thread":
        executor_class = ThreadPoolExecutor
    elif concurrency_type == "process":
        executor_class = ProcessPoolExecutor
    else:
        raise ValueError(f"不支持的并发类型: {concurrency_type}（可选thread/process）")

    results = []
    with executor_class(max_workers=max_workers) as executor:
        # 提交任务（注意：多进程需确保函数可序列化，避免lambda）
        future_to_text = {
            executor.submit(get_ollama_embedding, text, model_name, max_context): text
            for text in texts
        }
        for future in as_completed(future_to_text):
            try:
                emb = future.result()
                results.append(emb)
            except Exception as e:
                log.error(f"并发嵌入生成失败: {e}")
                results.append([])  # 占位空嵌入
    return results


# %%
# -------------------------- 5. 笔记处理核心逻辑（增量更新+入库） -------------------------
def process_single_note(
    note, vector_db: VectorDBManager, model_name: str, chunk_size: int, max_context: int
) -> bool:
    """处理单条笔记（清理→嵌入→入库），返回是否成功"""
    try:
        # 获取笔记完整信息（含更新时间）
        note_detail = getnote(note.id)
        if not note_detail:
            log.warning(f"笔记 {note.id} 获取失败，跳过")
            return False

        # 计算内容哈希（标题+正文）
        current_hash = compute_content_hash(note.title, note.body)
        current_update_time = (
            note_detail.updated_time
        )  # Joplin笔记的更新时间（Unix时间戳）

        # 清理文本（标题+正文）
        cleaned_body = clean_text(note.body)
        text = f"{note.title}\n{cleaned_body}"

        # 生成嵌入
        embedding = get_merged_embedding(text, model_name, chunk_size, max_context)
        if not embedding:  # 嵌入生成失败
            log.error(f"笔记 {note.id} 嵌入生成失败，跳过")
            return False

        # 获取标签
        tags = get_tag_titles(note.id)  # 项目函数：获取笔记标签列表

        # 入库（Upsert：存在则更新，不存在则新增）
        vector_db.upsert_note(note.id, text, embedding, tags)
        log.info(f"笔记 {note.id} 处理完成（标题: {note.title[:20]}...）")
        return True

    except Exception as e:
        log.error(f"处理笔记 {note.id} 失败: {e}", exc_info=True)
        return False


# %%
def process_notes_incremental(notebook_title: str, config: Dict):
    """增量处理笔记本笔记（仅处理新增/更新笔记）"""
    # 初始化向量数据库
    vector_db = VectorDBManager(config["db_path"])
    # 加载处理状态
    process_state = load_process_state(config["state_path"])
    # 获取笔记本所有笔记
    notes = get_notes_in_notebook_by_title(notebook_title=notebook_title)
    if not notes:
        log.info(f"笔记本 '{notebook_title}' 无笔记，跳过处理")
        return

    log.info(f"开始增量处理笔记本 '{notebook_title}'，共 {len(notes)} 条笔记")
    updated_count = 0

    for note in notes:
        note_id = note.id
        # 获取笔记当前状态（更新时间、哈希）
        note_detail = getnote(note_id)
        if not note_detail:
            continue
        current_update_time = note_detail.updated_time
        current_hash = compute_content_hash(note.title, note.body)

        # 判断是否需要处理：未记录 或 更新时间/哈希变化
        last_state = process_state.get(note_id, {})
        if (
            last_state.get("update_time") == current_update_time
            and last_state.get("hash") == current_hash
        ):
            log.debug(f"笔记 {note_id} 未更新，跳过")
            continue

        # 处理笔记（单线程处理，如需批量并发可调整此处）
        success = process_single_note(
            note=note,
            vector_db=vector_db,
            model_name=config["embedding_model"],
            chunk_size=config["chunk_size"],
            max_context=config["max_context"],
        )
        if success:
            # 更新处理状态
            process_state[note_id] = {
                "update_time": current_update_time,
                "hash": current_hash,
                "processed_time": datetime.now().timestamp(),
            }
            updated_count += 1

    # 保存最新状态
    save_process_state(process_state, config["state_path"])
    log.info(f"增量处理完成：共处理 {updated_count} 条笔记（总计 {len(notes)} 条）")


# %%
# -------------------------- 6. 主流程入口 -------------------------
def main():
    """主函数：执行增量处理"""
    log.info("===== 启动Joplin笔记向量化处理 =====")
    try:
        process_notes_incremental(
            notebook_title=CONFIG["notebook_title"], config=CONFIG
        )
        log.info("===== 处理完成 =====")
    except Exception as e:
        log.critical(f"主流程执行失败: {e}", exc_info=True)


# %%
if __name__ == "__main__":
    main()
