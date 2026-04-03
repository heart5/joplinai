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
# # deepseek_enhancer.py（新增增强模块）

# %% [markdown]
# # 导入库

# %%
import hashlib
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import requests
from cache_manager import SQLiteCacheManager

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
# # DeepSeek配置

# %%
# DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")  # 从环境变量读取（安全）
DEEPSEEK_API_KEY = getinivaluefromcloud("joplinai", "deepseek_token")
DEEPSEEK_EMBED_URL = "https://api.deepseek.com/v1/embeddings"  # 嵌入API端点
DEEPSEEK_CHAT_URL = "https://api.deepseek.com/v1/chat/completions"  # 大模型API端点
DEFAULT_EMBED_MODEL = "deepseek-embedding"  # DeepSeek嵌入模型（示例）
DEFAULT_CHAT_MODEL = "deepseek-chat"  # DeepSeek大模型（示例）

# %% [markdown]
# # 缓存支持

# %% [markdown]
# ## 初始化全局缓存管理器

# %%
# 初始化全局缓存管理器（单例模式）
_CACHE_MANAGER = None
CACHE_DIR = getdirmain() / "data" / ".deepseek_cache"
CACHE_FILE = CACHE_DIR / "deepseek_cache.db"


# %% [markdown]
# ## _ensure_cache_dir()

# %%
def _ensure_cache_dir():
    """确保缓存目录存在"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)


# %% [markdown]
# ## get_cache_manager()

# %%
def get_cache_manager():
    global _CACHE_MANAGER
    _ensure_cache_dir()
    if _CACHE_MANAGER is None:
        _CACHE_MANAGER = SQLiteCacheManager(db_path=CACHE_FILE)
    return _CACHE_MANAGER


# %% [markdown]
# ## _get_content_hash(text: str) -> str

# %%
def _get_content_hash(text: str) -> str:
    """生成文本内容哈希"""
    return hashlib.md5(text.encode()).hexdigest()


# %% [markdown]
# ## get_cached_result(content_hash: str, task: str) -> Optional[str]

# %%
def get_cached_result(content_hash: str, task: str) -> Optional[str]:
    """获取缓存结果"""
    return get_cache_manager().get(content_hash, task)


# %% [markdown]
# ## save_to_cache(content_hash: str, task: str, result: str)

# %%
def save_to_cache(content_hash: str, task: str, result: str):
    """保存结果到缓存"""
    get_cache_manager().set(content_hash, task, result)


# %% [markdown]
# # 增强功能1：DeepSeek嵌入（提升向量质量）

# %%
@timethis
def get_deepseek_embedding(
    text: str, model: str = DEFAULT_EMBED_MODEL, max_retries: int = 3
) -> Optional[List[float]]:
    """DeepSeek嵌入API调用"""
    if not DEEPSEEK_API_KEY:
        log.warning("未配置DeepSeek API Key")
        return None

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    MAX_TEXT_LEN = 12000
    chunks = (
        [text[i : i + MAX_TEXT_LEN] for i in range(0, len(text), MAX_TEXT_LEN)]
        if len(text) > MAX_TEXT_LEN
        else [text]
    )

    all_embeddings = []
    for chunk in chunks:
        payload = {"model": model, "input": chunk, "encoding_format": "float"}
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    DEEPSEEK_EMBED_URL, headers=headers, json=payload, timeout=30
                )
                if response.status_code == 400:
                    log.error(f"嵌入API 400错误: {response.text}")
                    return None
                response.raise_for_status()
                embedding = response.json()["data"]["embedding"]
                all_embeddings.append(embedding)
                break
            except Exception as e:
                log.warning(f"嵌入失败({attempt + 1}/{max_retries}): {str(e)[:100]}")
                time.sleep(2**attempt)
        else:
            return None

    return (
        [sum(dim) / len(all_embeddings) for dim in zip(*all_embeddings)]
        if all_embeddings
        else None
    )


# %% [markdown]
# # 增强功能2：DeepSeek大模型（笔记智能加工）

# %%
@timethis
def deepseek_process_note(
    text: str,
    task: str = "summary",
    model: str = DEFAULT_CHAT_MODEL,
    max_retries: int = 3,
    use_cache: bool = True,  # 新增参数，控制是否使用缓存
) -> Optional[str]:
    """DeepSeek大模型处理"""
    if not DEEPSEEK_API_KEY:
        log.warning("未配置DeepSeek API Key")
        return None

    # 添加个人笔记QA提示词
    prompts = {
        "summary": "用1-3句话总结以下笔记核心内容，突出主题和结论：\n%s",
        "tags": """请从以下文本中提取3-5个核心关键词作为标签。
    要求：
    1. 每个标签必须是2-6个字的简短名词、专业术语或关键地名
    2. 避免使用长短语和完整的句子
    3. 用英文逗号分隔，不要编号
    4. 优先提取文本中反复出现的关键概念、实体和**关键地点**
    5. 对于地名，仅提取在文中作为核心要素出现的地点（如事件发生地、主要研究对象所在地、重要机构所在地等）
    
    文本内容：
    %s
    
    请直接输出标签，不要有其他说明。""",
    }

    # 缓存逻辑
    if use_cache:
        content_hash = _get_content_hash(text[:8000])  # 只哈希前8000字符，兼顾效率
        cached_result = get_cached_result(content_hash, task)
        if cached_result:
            log.info(f"deepseek增强使用缓存{task}: {content_hash[:8]}")
            return cached_result

    if task not in prompts:
        log.error(f"不支持的任务类型: {task}")
        return None

    # 使用 % 格式化，避免花括号问题
    safe_text = text[:8000]
    prompt = prompts[task] % safe_text
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 500,
    }
    # print(payload)

    for attempt in range(max_retries):
        try:
            response = requests.post(
                DEEPSEEK_CHAT_URL, headers=headers, json=payload, timeout=30
            )
            if response.status_code == 400:
                log.error(f"聊天API 400错误: {response.text}")
                return None
            response.raise_for_status()
            # print(response.json())
            result = response.json()["choices"][0]["message"]["content"].strip()

            # 保存到缓存
            if use_cache:
                log.info(
                    f"deepseek增强生成并缓存至文件《{CACHE_FILE}》。{task}: {content_hash[:8]}"
                )
                save_to_cache(content_hash, task, result)

            return result
        except Exception as e:
            log.warning(f"大模型调用失败({attempt + 1}/{max_retries}): {str(e)[:100]}")
            time.sleep(2**attempt)
    return None


# %% [markdown]
# # 主函数

# %%
if __name__ == "__main__":
    result = deepseek_process_note(
        "请在100字之内介绍joplin笔记软件", task="summary", model="deepseek-chat"
    )
    print(result)
