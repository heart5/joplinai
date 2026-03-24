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
# # joplinai AI增强Joplin笔记管理（向量化+增强更新+并发加速）

# %% [markdown]
# """
# joplinai.py | 优化版：AI增强Joplin笔记管理（向量化+增量更新+并发加速）
# 核心功能：将Joplin笔记向量化存储到ChromaDB，支持语义检索；仅增量处理更新笔记。
# """

# %% [markdown]
# ## 导入核心库

# %% [markdown]
# ### 导入系统库

# %%
import argparse
import hashlib
import json
import logging
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
import ollama
from chromadb.config import Settings

# %% [markdown]
# ### 项目模块导入（根据实际路径调整）

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

# %% [markdown]
# ## 核心配置（根据需求调整）

# %%
CONFIG = {
    "notebook_titles": ["顺风顺水", "日新月异", "运营管理"],  # 改为笔记本名称列表
    # "embedding_model": "nomic-embed-text",  # 嵌入模型（Ollama本地模型，优先选nomic-embed-text）
    "embedding_model": "qwen:1.8b",  # 嵌入模型（Ollama本地模型，优先选nomic-embed-text）
    "chunk_size": 2000,  # 文本分块大小（字符数，根据模型上下文调整，nomic支持8192）
    "max_context": 4000,  # 模型最大上下文（字符数，nomic-embed-text为8192）
    "concurrency_type": "thread",  # 固定使用多线程，移除 process 选项
    "max_workers": min(
        16, (os.cpu_count() or 1) * 2
    ),  # 动态设置最大工作者数：CPU核心数 * 2，上限为16
    "db_path": getdirmain() / "data" / "joplin_vector_db",  # ChromaDB存储路径
    "state_path": getdirmain()
    / "data"
    / "joplin_process_state.json",  # 处理状态文件路径
    # "log_level": logging.INFO,  # 日志级别
    "enable_deepseek_embed": False,  # 是否用DeepSeek嵌入替代本地嵌入（增强向量质量）
    "enable_deepseek_summary": False,  # 是否用DeepSeek生成摘要（增强笔记元数据）
    "enable_deepseek_tags": False,  # 是否用DeepSeek提取标签（增强笔记标签）
    "deepseek_api_key": getcfpoptionvalue("joplinai", "deepseek", "token"),
    "deepseek_chat_model": "deepseek-chat",  # 修正模型名称
    "deepseek_embed_model": "deepseek-embedding",
    "force_update": False,  # 新增：强制更新开关，默认关闭
}


# %% [markdown]
# ## 功能函数集
# %% [markdown]
# ### 工具小集合
# %% [markdown]
# #### clean_text(text: str) -> str
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


# %% [markdown]
# #### compute_content_hash(title: str, body: str) -> str
# %%
def compute_content_hash(title: str, body: str) -> str:
    """计算笔记内容哈希（用于增量更新判断）"""
    content = f"{title}{body}"
    return hashlib.md5(content.encode("utf-8")).hexdigest()


# %% [markdown]
# #### load_process_state(state_path: Path) -> Dict[str, Dict]
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


# %% [markdown]
# #### save_process_state(state: Dict, state_path: Path)
# %%
def save_process_state(state: Dict, state_path: Path):
    """保存处理状态（增强序列化）"""

    def serialize(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        if isinstance(obj, (list, tuple)):
            return [serialize(item) for item in obj]
        if isinstance(obj, dict):
            return {k: serialize(v) for k, v in obj.items()}
        return str(obj)

    try:
        serialized_state = serialize(state)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(serialized_state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.error(f"保存状态文件失败: {e}")


# %% [markdown]
# ### 向量数据库操作（ChromaDB）
# %%
class VectorDBManager:
    """ChromaDB向量数据库管理器（封装集合操作）"""

    def __init__(self, db_path: Path, model_name: str):
        self.client = chromadb.PersistentClient(path=str(db_path))
        self.collection_name = f"joplin_{model_name.replace(':', '_')}"
        self._ensure_collection(model_name)

    def _ensure_collection(self, model_name: str):
        """确保集合存在且维度匹配"""
        try:
            self.collection = self.client.get_collection(self.collection_name)
            # 检查维度是否匹配
            sample = self.collection.get(limit=1)
            if sample and "embeddings" in sample:
                existing_dim = len(sample["embeddings"][0])
                current_dim = self._get_model_dimension(model_name)
                if existing_dim != current_dim:
                    raise ValueError(
                        f"维度不匹配: 现有{existing_dim}D, 需要{current_dim}D"
                    )
        except:
            # 维度不匹配或集合不存在时重建
            if hasattr(self, "collection"):
                self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "dimension": self._get_model_dimension(model_name),
                },
            )

    def _get_model_dimension(self, model_name: str) -> int:
        """获取模型嵌入维度"""
        dimensions = {
            "nomic-embed-text": 768,
            "qwen:1.8b": 768,
            "text2vec-large-chinese": 1024,
        }
        return dimensions.get(model_name, 768)  # 默认768维

    def upsert_note(
        self,
        note_id: str,
        text: str,
        embedding: List[float],
        tags: List[str],
        metadata: Dict,
    ):
        """插入/更新笔记向量数据"""
        self.collection.upsert(
            ids=[note_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[
                {
                    "note_id": note_id,
                    "tags": ",".join(tags),
                    "summary": metadata.get("summary"),
                }
            ],
        )

    def delete_note(self, note_id: str):
        """删除笔记向量数据"""
        self.collection.delete(ids=[note_id])


# %% [markdown]
# ### 嵌入生成（Ollama API调用+并发）
# %% [markdown]
# #### get_model_max_context(model_name: str) -> int
# %%
def get_model_max_context(model_name: str) -> int:
    """精确获取模型上下文限制（token→字符转换）"""
    try:
        # 特殊处理已知模型
        if model_name == "nomic-embed-text":
            return 24000  # 8192 token × 3字符/token × 0.9余量 ≈ 22000，取24000更安全
        if model_name == "qwen:1.8b":
            return 5529  # 2048 token × 3×0.9

        # 通用模型处理
        model_info = ollama.show(model=model_name)
        num_ctx = model_info.get("parameters", {}).get("num_ctx", 2048)
        return int(num_ctx * 3 * 0.9)  # 通用转换公式
    except Exception as e:
        log.warning(f"获取模型上下文失败({model_name})，使用默认值8192字符: {e}")
        return 8192  # 安全默认值


# %% [markdown]
# #### get_ollama_embedding(text: str, model_name: str) -> List[float]
# %%
def get_ollama_embedding(text: str, model_name: str) -> List[float]:
    """调用 Ollama 生成文本嵌入。注意：传入的 text 应是已分块的适当长度文本。"""
    for attempt in range(3):
        try:
            response = ollama.embeddings(model=model_name, prompt=text)
            return response["embedding"]
        except Exception as e:
            log.warning(f"嵌入失败({attempt + 1}/3): {str(e)[:100]}")
            time.sleep(2**attempt)

    log.error(f"嵌入生成最终失败: {model_name}, 文本长度{len(text)}")
    return []


# %% [markdown]
# #### get_merged_embedding(text: str, model_name: str, config: Dict) -> List[float]
# %%
def get_merged_embedding(text: str, model_name: str, config: Dict) -> List[float]:
    """合并嵌入生成（本地嵌入为主，DeepSeek嵌入为增强选项）"""
    if config["enable_deepseek_embed"] and config["deepseek_api_key"]:
        # 优先用DeepSeek增强嵌入
        log.info("使用DeepSeek增强嵌入")
        from deepseek_enhancer import get_deepseek_embedding

        return get_deepseek_embedding(
            text, model=config.get("deepseek_embed_model", "deepseek-embedding")
        )
    else:
        # 默认用本地Ollama嵌入（保留您已优化的分块逻辑）
        log.info("使用本地Ollama嵌入")
        CHUNK_SIZE = get_model_max_context(model_name)
        chunks = [text[i : i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

        if not chunks:
            return []

        log.info(
            f"文本分块: {len(chunks)}块（每块{CHUNK_SIZE}字符），实际每块字符数量{[len(chunk) for chunk in chunks]}"
        )

        # 顺序处理（避免并发问题）
        embeddings = []
        for chunk in chunks:
            emb = get_ollama_embedding(chunk, model_name)
            if emb:
                embeddings.append(emb)

        if not embeddings:
            return []

        # 简单平均合并
        return [sum(dim) / len(embeddings) for dim in zip(*embeddings)]


# %% [markdown]
# ### 笔记处理核心逻辑（增量更新+入库）
# %% [markdown]
# #### process_single_note(note, vector_db: VectorDBManager, model_name: str, config: Dict) -> bool
# %%
def process_single_note(
    note, vector_db: VectorDBManager, model_name: str, config: Dict
) -> bool:
    """处理单条笔记（清理→嵌入→入库），返回是否成功"""
    try:
        # 获取笔记完整信息（含更新时间）
        note_detail = getnote(note.id)
        if not note_detail:
            log.warning(f"笔记《{note.title}》（{note.id}） 获取失败，跳过")
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
        embedding = get_merged_embedding(text, model_name, config)
        if not embedding:  # 嵌入生成失败
            log.error(f"笔记《{note.title}》（{note.id}） 获取失败，跳过")
            return False

        # 获取标签
        local_tags = get_tag_titles(note.id)  # 项目函数：获取笔记标签列表
        # -------------------------- DeepSeek增强加工（可选） --------------------------
        enhanced_metadata = {}
        # enhanced_metadata["note_id"] = note.id
        if config["enable_deepseek_summary"] and config["deepseek_api_key"]:
            from deepseek_enhancer import deepseek_process_note

            summary = deepseek_process_note(
                text,
                task="summary",
                model=config.get("deepseek_chat_model", "deepseek-chat"),
            )
            enhanced_metadata["summary"] = summary or ""  # 存入摘要

        if config["enable_deepseek_tags"] and config["deepseek_api_key"]:
            tags_str = deepseek_process_note(
                text,
                task="tags",
                model=config.get("deepseek_chat_model", "deepseek-chat"),
            )
            deepseek_tags = [t.strip() for t in tags_str.split(",")] if tags_str else []
            # 合并本地标签与DeepSeek标签（去重）
            enhanced_tags = list(set(local_tags + deepseek_tags))
        else:
            enhanced_tags = local_tags

        enhanced_metadata["tags"] = ",".join(enhanced_tags)
        # 将调试信息改为日志记录，移除 print 语句
        log.debug(f"笔记《{note.title}》（{note.id}）增强元数据: {enhanced_metadata}")

        log.info(
            f"笔记《{note.title}》（{note.id}）准备入库，嵌入维度：{len(embedding)}"
        )
        # ------ 入库（本地向量库+增强元数据），存在则更新，不存在则新增 ------
        vector_db.upsert_note(
            note_id=note.id,
            text=text,
            embedding=embedding,
            tags=enhanced_tags,
            metadata=enhanced_metadata,
        )
        log.info(f"笔记《{note.title}》（{note.id}）向量化处理完成！")
        return True

    except Exception as e:
        log.error(
            f"向量化处理笔记《{note.title}》（{note.id}）失败: {e}", exc_info=True
        )
        return False


# %% [markdown]
# #### process_notes_incremental(notebook_title: str, config: Dict)
# %%
def process_notes_incremental(notebook_title: str, config: Dict):
    """增量处理笔记本笔记（修复时间处理问题）"""
    # 动态获取模型最大上下文
    model_name = config["embedding_model"]
    max_context = get_model_max_context(model_name)
    chunk_size = min(config["chunk_size"], max_context // 2)
    log.info(
        f"使用模型“{model_name}”，动态分块配置：chunk_size={chunk_size}，max_context={max_context}"
    )

    # 初始化向量数据库
    vector_db = VectorDBManager(config["db_path"], model_name)

    # 加载处理状态
    process_state = load_process_state(config["state_path"])
    # 获取强制更新配置
    force_update = config.get("force_update", False)

    # 获取笔记本所有笔记
    notes = get_notes_in_notebook_by_title(notebook_title=notebook_title)
    if not notes:
        log.info(f"笔记本 '{notebook_title}' 无笔记，跳过处理")
        return

    log.info(f"开始增量处理笔记本 '{notebook_title}'，共 {len(notes)} 条笔记")
    updated_count = 0
    new_time_notes = []
    failed_notes = []

    with ThreadPoolExecutor(max_workers=config["max_workers"]) as executor:
        # 提交所有笔记处理任务
        future_to_note = {}
        for note in notes:
            # 检查是否需要处理（增量更新判断）
            note_id = note.id
            note_detail = getnote(note_id)
            if not note_detail:
                continue

            # ... 时间格式转换 关键修复：统一时间格式为时间戳
            current_update_time = note_detail.updated_time
            if isinstance(current_update_time, datetime):
                current_update_time = current_update_time.timestamp()
            elif isinstance(current_update_time, str):
                current_update_time = datetime.fromisoformat(
                    current_update_time
                ).timestamp()

            current_hash = compute_content_hash(note.title, note.body)
            last_state = process_state.get(note_id, {})

            # 只有需要更新的笔记才提交处理，为了方便调试，增加了云端配置的强制更新选项
            if force_update or not (
                last_state.get("update_time") == current_update_time
                and last_state.get("hash") == current_hash
            ):
                # 提交任务到线程池/进程池
                future = executor.submit(
                    process_single_note,
                    note,
                    vector_db,
                    model_name,
                    config,
                )
                future_to_note[future] = (note_id, current_update_time, current_hash)
                new_time_notes.append(note.title)

        # 收集处理结果
        for future in as_completed(future_to_note):
            note_id, update_time, content_hash = future_to_note[future]
            try:
                success = future.result()
                if success:
                    # 更新处理状态
                    process_state[note_id] = {
                        "update_time": float(update_time),
                        "hash": content_hash,
                        "processed_time": datetime.now().timestamp(),
                    }
                    updated_count += 1
                else:
                    failed_notes.append(note.title)
            except Exception as e:
                log.error(f"并发处理笔记 {note_id} 异常: {e}")
                failed_notes.append(note.title)

    # 保存状态
    save_process_state(process_state, config["state_path"])
    log.info(
        f"增量处理完成：新日期需要更新 {len(new_time_notes)} 条，成功 {updated_count} 条，失败 {len(failed_notes)} 条（总计 {len(notes)} 条）"
    )
    if failed_notes:
        log.warning(f"失败笔记ID: {failed_notes}")


# %% [markdown]
# ### 主流程入口

# %% [markdown]
# #### parse_args()


# %%
def parse_args():
    parser = argparse.ArgumentParser(description="Joplin笔记向量化处理工具")
    parser.add_argument(
        "--model",
        type=str,
        default=CONFIG["embedding_model"],
        help=f"Ollama嵌入模型名称（默认：{CONFIG['embedding_model']}）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=CONFIG["max_workers"],
        help=f"并发数（默认：{CONFIG['max_workers']}）",
    )
    parser.add_argument(
        "--enable_deepseek_embed",
        action="store_true",
        default=CONFIG["enable_deepseek_embed"],
        help=f"开启deepseek嵌入支持（默认：{CONFIG['enable_deepseek_embed']}）",
    )
    parser.add_argument(
        "--enable_deepseek_summary",
        action="store_true",
        default=CONFIG["enable_deepseek_summary"],
        help=f"开启deepseek摘要支持（默认：{CONFIG['enable_deepseek_summary']}）",
    )
    parser.add_argument(
        "--enable_deepseek_tags",
        action="store_true",
        default=CONFIG["enable_deepseek_tags"],
        help=f"开启deepseek标签支持（默认：{CONFIG['enable_deepseek_tags']}）",
    )
    return parser.parse_args()


# %% [markdown]
# #### main()


# %%
@timethis
def main():
    """主函数：执行增量处理"""
    args = parse_args()
    # 动态覆盖配置
    dynamic_config = CONFIG.copy()
    dynamic_config["embedding_model"] = args.model
    notebook_titles = [
        title.strip()
        for title in getinivaluefromcloud("joplinai", "imp_nbs").split(",")
    ]
    if notebook_titles:
        dynamic_config["notebook_titles"] = notebook_titles
    dynamic_config["max_workers"] = args.workers
    dynamic_config["enable_deepseek_embed"] = args.enable_deepseek_embed
    dynamic_config["enable_deepseek_summary"] = args.enable_deepseek_summary
    dynamic_config["enable_deepseek_tags"] = args.enable_deepseek_tags
    dynamic_config["force_update"] = getinivaluefromcloud("joplinai", "force_update")

    log.info(
        f"动态配置：模型={dynamic_config['embedding_model']}, \
        笔记本={dynamic_config['notebook_titles']}, \
        并发数={dynamic_config['max_workers']}， \
        使能deepseek嵌入模型为{dynamic_config['enable_deepseek_embed']}， \
        使能deepseek摘要功能为{dynamic_config['enable_deepseek_summary']}， \
        使能deepseek标签功能为{dynamic_config['enable_deepseek_tags']}， \
        强制更新为{dynamic_config['force_update']} \
        "
    )
    log.info("===== 启动Joplin笔记向量化处理 =====")
    try:
        for notebook_title in notebook_titles:
            log.info(f"开始处理笔记本: {notebook_title}")
            process_notes_incremental(
                notebook_title=notebook_title, config=dynamic_config
            )
        log.info("===== 所有笔记本处理完成 =====")
    except Exception as e:
        log.critical(f"主流程执行失败: {e}", exc_info=True)


# %% [markdown]
# ## 主函数

# %%
if __name__ == "__main__":
    main()
