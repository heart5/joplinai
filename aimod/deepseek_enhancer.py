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
# # deepseek_enhancer.py（新增增强模块）

# %% [markdown]
# # 导入库

# %%
import os
import time
from typing import Optional

import requests

# %%
import pathmagic

with pathmagic.Context():
    try:
        from aimod.cache_manager import SQLiteCacheManager
        from func.datatools import compute_content_hash
        from func.first import getdirmain
        from func.jpfuncs import getinivaluefromcloud
        from func.logme import log
    except ImportError as e:
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger(__name__)
        log.error(f"导入项目模块失败: {e}")

# %% [markdown]
# # DeepSeek配置

# %%
DEEPSEEK_API_KEY = getinivaluefromcloud("joplinai", "deepseek_token")
DEEPSEEK_EMBED_URL = "https://api.deepseek.com/v1/embeddings"  # 嵌入API端点
DEEPSEEK_CHAT_URL = "https://api.deepseek.com/v1/chat/completions"  # 大模型API端点
DEFAULT_EMBED_MODEL = "deepseek-embedding"  # DeepSeek嵌入模型（示例）
DEFAULT_CHAT_MODEL = "deepseek-chat"  # DeepSeek大模型（示例）
DEFAULT_VISION_MODEL = "deepseek-v4-pro"  # DeepSeek Vision 多模态模型
DEFAULT_OLLAMA_VISION_MODEL = "minicpm-v"  # 本地 Ollama Vision 模型

# %% [markdown]
# # 缓存支持

# %% [markdown]
# ## 初始化全局缓存管理器

# %%
# 初始化全局缓存管理器（单例模式）
_CACHE_MANAGER = None
CACHE_DIR = getdirmain() / "data"
CACHE_FILE = CACHE_DIR / "joplinai_center.db"


# %% [markdown]
# ## _ensure_cache_dir()

# %%
def _ensure_cache_dir() -> None:
    """确保缓存目录存在"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)


# %% [markdown]
# ## get_cache_manager()

# %%
__all__ = [
    "get_cache_manager",
    "deepseek_process_note",
    "deepseek_describe_images",
    "deepseek_process_note_vision",
    "ollama_vision_describe",
]

def get_cache_manager():
    """获取缓存管理器 — 云端未配 URL 则本机为生产主机走本地"""
    global _CACHE_MANAGER
    if _CACHE_MANAGER is not None:
        return _CACHE_MANAGER

    remote_url = getinivaluefromcloud("joplinai", "joplinai_center_url")
    if not remote_url:
        remote_url = "http://127.0.0.1:5003"
    api_key = getinivaluefromcloud("joplinai", "joplinai_center_api_key")
    if remote_url and api_key:
        from aimod.deepseek_client import DeepSeekCacheClient

        _CACHE_MANAGER = DeepSeekCacheClient(remote_url, api_key)
        return _CACHE_MANAGER

    _ensure_cache_dir()
    _CACHE_MANAGER = SQLiteCacheManager(db_path=CACHE_FILE)
    return _CACHE_MANAGER


# %% [markdown]
#
# %% [markdown]
# # DeepSeek大模型（笔记智能加工）

# %% [markdown]
# ## deepseek_process_note(text: str, task: str = "summary", model: str = DEFAULT_CHAT_MODEL, max_retries: int = 3, use_cache: bool = True,) -> Optional[str]

# %%
# @timethis
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
    content_hash = compute_content_hash(text[:8000])
    cache_manager = get_cache_manager()

    # 1. 查询缓存（此时cache_manager只返回数据，不调用API）
    cache_result = cache_manager.get(content_hash, task)

    if cache_result.content is not None:
        # 缓存命中
        # log.info(f"deepseek增强缓存命中 {task}: {content_hash[:12]}")

        # 2. 检查是否需要验证
        if cache_result.requires_validation:
            log.info(
                f"deepseek增强缓存条目达到验证阈值，启动异步验证: {content_hash[:12]}"
            )
            # 重要：这里可以同步验证，但为了不阻塞当前请求，建议异步或后台执行。
            # 以下是同步验证的示例（可能会增加本次请求的延迟）：
            _validate_cache_entry_async(
                original_text=text[:8000],
                task=task,
                cache_key=cache_result.cache_key,
                cached_content=cache_result.content,
                model=model,
                max_retries=max_retries,
            )
            # 注意：_validate_cache_entry_async 函数应立即返回，实际验证在后台线程进行。

        # 3. 无论是否需要验证，都先返回当前缓存的内容
        return cache_result.content

    # 缓存未命中，调用API获取新结果
    log.info(f"deepseek增强缓存未命中，调用API: {task} for {content_hash[:12]}")
    new_result = _call_deepseek_api_directly(text, task, model, max_retries)

    if new_result:
        # 将新结果保存到缓存
        cache_manager.set(content_hash, task, new_result)
        # log.info(f"deepseek增强新结果已缓存: {content_hash[:12]}")

    return new_result


# %% [markdown]
# ## _validate_cache_entry_async(original_text: str, task: str, cache_key: str, cached_content: str, model: str, max_retries: int)

# %%
def _validate_cache_entry_async(
    original_text: str,
    task: str,
    cache_key: str,
    cached_content: str,
    model: str,
    max_retries: int,
) -> None:
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
            log.warning(f"deepseek增强验证发现内容变化，更新缓存: {cache_key[:12]}...")
            get_cache_manager().update_on_validation(
                cache_key, new_result, validation_successful=True
            )
    except Exception as e:
        log.error(f"deepseek增强后台验证过程异常: {e}")
        get_cache_manager().update_on_validation(
            cache_key, None, validation_successful=False
        )


# %% [markdown]
# ## _call_deepseek_api_directly(text: str, task: str, model: str, max_retries: int, use_cache: bool = False) -> Optional[str]

# %%
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
        log.error(f"deepseek增强不支持的任务类型: {task}")
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
                log.error(f"deepseek增强聊天API 400错误: {response.text}")
                return None
            response.raise_for_status()
            # print(response.json())
            result = response.json()["choices"][0]["message"]["content"].strip()

            return result
        except Exception as e:
            log.warning(
                f"deepseek增强大模型调用失败({attempt + 1}/{max_retries}): {str(e)[:100]}"
            )
            time.sleep(2**attempt)
    return None


# %% [markdown]
# # DeepSeek Vision API（图片处理）

# %% [markdown]
# ## _call_deepseek_vision_api(messages: list[dict], model: str, max_retries: int)

# %%
def _call_deepseek_vision_api(
    messages: list[dict],
    model: str = DEFAULT_VISION_MODEL,
    max_retries: int = 2,
) -> Optional[str]:
    """Call DeepSeek V4 vision API with native multimodal format.

    Each message dict has top-level 'image_data' (pure base64, no prefix)
    or 'image_url' field alongside 'role' and 'content':
    {"role": "user", "content": "描述图片", "image_data": "base64..."}
    """
    if not DEEPSEEK_API_KEY:
        log.warning("未配置DeepSeek API Key")
        return None

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 800,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                DEEPSEEK_CHAT_URL, headers=headers, json=payload, timeout=60
            )
            if response.status_code == 400:
                log.error(f"DeepSeek Vision API 400错误: {response.text[:200]}")
                return None
            response.raise_for_status()
            result = response.json()["choices"][0]["message"]["content"].strip()
            return result
        except Exception as e:
            log.warning(
                f"DeepSeek Vision API调用失败({attempt + 1}/{max_retries}): {str(e)[:100]}"
            )
            time.sleep(2 ** attempt)
    return None


# %% [markdown]
# ## deepseek_describe_images(images: dict[str, dict], context: str, model: str) -> Optional[str]

# %%
def deepseek_describe_images(
    images: dict[str, dict],
    context: str = "",
    model: str = DEFAULT_VISION_MODEL,
) -> Optional[str]:
    """Describe a set of images using DeepSeek V4 Vision API (native format).

    Args:
        images: {resource_id: {'b64': str, 'mime': str}} dict
        context: surrounding text context for better descriptions
        model: DeepSeek model name

    Returns concatenated image description text, or None if all fail.
    """
    if not images:
        return None

    # One message per image (DeepSeek V4: image_data is top-level message field)
    image_items = list(images.items())
    descriptions = []

    for rid, img_data in image_items:
        prompt = "请用中文描述这张图片的内容，重点说明图片传达的关键信息。"
        if context:
            prompt = (
                f"以下是笔记的部分文本上下文，请结合上下文描述图片内容：\n"
                f"{context[:3000]}\n\n"
                f"请用中文描述这张图片的内容，重点说明图片与上下文的关系。"
            )
        message = {
            "role": "user",
            "content": prompt,
            "image_data": img_data["b64"],
        }
        result = _call_deepseek_vision_api([message], model=model)
        if result:
            descriptions.append(result)
        else:
            log.warning(f"图片 {rid} 描述失败")

    if not descriptions:
        return None
    return "\n".join(descriptions)


# %% [markdown]
# ## deepseek_process_note_vision(note_content, images, context, model) -> Optional[str]

# %%
def deepseek_process_note_vision(
    note_content: str,
    images: dict[str, dict],
    context: str = "",
    model: str = DEFAULT_VISION_MODEL,
) -> Optional[str]:
    """Generate a comprehensive note enhancement using vision + text.

    Combines note text and embedded images, sends to DeepSeek V4,
    returns a description that enriches the text-only understanding.

    Use cache-friendly: wrap with deepseek_process_note for caching.
    """
    return deepseek_describe_images(images, context=context or note_content, model=model)


# %% [markdown]
# # Ollama 本地 Vision API

# %% [markdown]
# ## ollama_vision_describe(images, context, model, ollama_host) -> Optional[str]

# %%
def ollama_vision_describe(
    images: dict[str, dict],
    context: str = "",
    model: str = DEFAULT_OLLAMA_VISION_MODEL,
    ollama_host: str = "http://127.0.0.1:11434",
) -> Optional[str]:
    """Describe images using a local Ollama vision model.

    Args:
        images: {resource_id: {'b64': str, 'mime': str}} dict
        context: surrounding text context for better descriptions
        model: Ollama model name
        ollama_host: Ollama API base URL

    Returns concatenated image description text, or None if all fail.
    """
    if not images:
        return None

    prompt = "请用中文描述这张图片的内容，重点说明图片中的文字信息和关键数据。"
    if context:
        prompt = (
            f"以下是笔记的部分文本上下文：\n{context[:2000]}\n\n"
            f"请结合上下文用中文描述这张图片的内容，"
            f"重点说明图片中的文字信息和关键数据，以及图片与上下文的关系。"
        )

    descriptions = []
    for rid, img_data in images.items():
        try:
            response = requests.post(
                f"{ollama_host}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "images": [img_data["b64"]],
                    "stream": False,
                },
                timeout=120,
            )
            if response.status_code == 200:
                content = response.json()["message"]["content"].strip()
                descriptions.append(content)
                log.info(f"Ollama Vision 描述成功: {rid[:12]}... ({len(content)}字符)")
            else:
                log.warning(
                    f"Ollama Vision 失败({response.status_code}): {rid[:12]}..."
                    f" {response.text[:100]}"
                )
        except Exception as e:
            log.warning(f"Ollama Vision 异常: {rid[:12]}... {e}")

    if not descriptions:
        return None
    return "\n".join(descriptions)


# %% [markdown]
# # 主函数

# %%
if __name__ == "__main__":
    result = deepseek_process_note(
        "请在100字之内介绍joplin笔记软件", task="summary", model="deepseek-chat"
    )
    print(result)
