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
    # _CACHE_MANAGER.import_from_json_directory(CACHE_DIR, clear_existing=True)
    _CACHE_MANAGER.import_from_json_directory(CACHE_DIR)
    return _CACHE_MANAGER


# %% [markdown]
# ## _get_content_hash(text: str) -> str

# %%
def _get_content_hash(text: str) -> str:
    """生成文本内容哈希"""
    return hashlib.md5(text.encode()).hexdigest()


# %% [markdown]
# # 增强功能1：DeepSeek嵌入（提升向量质量）

# %% [markdown]
# ## get_deepseek_embedding(text: str, model: str = DEFAULT_EMBED_MODEL, max_retries: int = 3) -> Optional[List[float]]

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

# %% [markdown]
# ## deepseek_process_note(text: str, task: str = "summary", model: str = DEFAULT_CHAT_MODEL, max_retries: int = 3, use_cache: bool = True,) -> Optional[str]

# %%
@timethis
def deepseek_process_note(
    text: str,
    task: str = "summary",
    model: str = DEFAULT_CHAT_MODEL,
    max_retries: int = 3,
    use_cache: bool = True,
) -> Optional[str]:
    """DeepSeek大模型处理（集成智能验证机制）"""
    if not DEEPSEEK_API_KEY:
        log.warning("未配置DeepSeek API Key")
        return None

    if not use_cache:
        # 不使用缓存，直接调用API
        return _call_deepseek_api_directly(text, task, model, max_retries)

    # 使用缓存流程
    content_hash = _get_content_hash(text[:8000])
    cache_manager = get_cache_manager()

    # 1. 查询缓存（此时cache_manager只返回数据，不调用API）
    cache_result = cache_manager.get(content_hash, task)

    if cache_result.content is not None:
        # 缓存命中
        log.info(f"缓存命中 {task}: {content_hash[:12]}")

        # 2. 检查是否需要验证
        if cache_result.requires_validation:
            log.info(f"缓存条目达到验证阈值，启动异步验证: {content_hash[:12]}")
            # 重要：这里可以同步验证，但为了不阻塞当前请求，建议异步或后台执行。
            # 以下是同步验证的示例（可能会增加本次请求的延迟）：
            _validate_cache_entry_in_background(
                original_text=text[:8000],
                task=task,
                cache_key=cache_result.cache_key,
                cached_content=cache_result.content,
                model=model,
                max_retries=max_retries,
            )
            # 注意：_validate_cache_entry_in_background 函数应立即返回，实际验证在后台线程进行。

        # 3. 无论是否需要验证，都先返回当前缓存的内容
        return cache_result.content

    # 缓存未命中，调用API获取新结果
    log.info(f"缓存未命中，调用API: {task} for {content_hash[:12]}")
    new_result = _call_deepseek_api_directly(text, task, model, max_retries)

    if new_result:
        # 将新结果保存到缓存
        cache_manager.set(content_hash, task, new_result)
        log.info(f"新结果已缓存: {content_hash[:12]}")

    return new_result


# %% [markdown]
# ## _validate_cache_entry_in_background(original_text: str, task: str, cache_key: str, cached_content: str, model: str, max_retries: int)

# %%
def _validate_cache_entry_in_background(
    original_text: str,
    task: str,
    cache_key: str,
    cached_content: str,
    model: str,
    max_retries: int,
):
    """
    在后台执行验证的逻辑。
    实际部署时，可以将此函数放入线程池、任务队列或异步框架中。
    """
    # 这里为了简化，展示同步逻辑。生产环境应使用 threading.Thread 或 asyncio。
    try:
        # 发起一次新的、不经过缓存的API调用
        new_result = _call_deepseek_api_directly(
            original_text, task, model, max_retries, use_cache=False
        )

        if new_result is None:
            # API调用失败
            get_cache_manager().update_on_validation(
                cache_key, None, validation_successful=False
            )
            return

        # 对比结果
        if new_result.strip() == cached_content.strip():
            # 内容未变
            get_cache_manager().update_on_validation(
                cache_key, None, validation_successful=True
            )
        else:
            # 内容已变，更新缓存
            log.warning(f"验证发现内容变化，更新缓存: {cache_key[:12]}...")
            get_cache_manager().update_on_validation(
                cache_key, new_result, validation_successful=True
            )
    except Exception as e:
        log.error(f"后台验证过程异常: {e}")
        get_cache_manager().update_on_validation(
            cache_key, None, validation_successful=False
        )


# %% [markdown]
# ## _call_deepseek_api_directly(text: str, task: str, model: str, max_retries: int, use_cache: bool = False) -> Optional[str]

# %%
@timethis
def _call_deepseek_api_directly(
    text: str, task: str, model: str, max_retries: int, use_cache: bool = False
) -> Optional[str]:
    """DeepSeek大模型处理
    直接调用DeepSeek API的核心函数。
    注意：这个函数内部不应该再调用缓存逻辑，避免循环。
    """
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

# %%
_CACHE_MANAGER.get_stats()
