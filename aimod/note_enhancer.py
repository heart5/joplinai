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
# # note_enhancer.py（AI增强模块）

# %% [markdown]
# # 导入库

# %%
import time
from typing import Optional

import ollama
import requests

# %%
import pathmagic

with pathmagic.Context():
    try:
        from func.datatools import compute_content_hash
        from func.jpfuncs import getinivaluefromcloud
        from func.logme import log
    except ImportError as e:
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger(__name__)
        log.error(f"导入项目模块失败: {e}")

# %% [markdown]
# # 模型配置

# %%
_DEFAULT_CLOUD_API_URL = "https://api.deepseek.com/v1/chat/completions"
_DEFAULT_CLOUD_MODEL = "deepseek-v4-flash"
_DEFAULT_VISION_MODEL = "deepseek-v4-pro"
_SILICONFLOW_VISION_URL = "https://api.siliconflow.cn/v1/chat/completions"
_DEFAULT_SF_VISION_MODEL = "Qwen/Qwen3-VL-32B-Instruct"


def _get_cloud_api_url() -> str:
    """获取云端大模型 API URL，默认 DeepSeek"""
    return getinivaluefromcloud("joplinai", "cloud_api_url") or _DEFAULT_CLOUD_API_URL


def _get_cloud_api_key() -> str:
    """获取云端大模型 API Key，优先 cloud_api_key，回退 deepseek_token（向后兼容）"""
    return getinivaluefromcloud("joplinai", "cloud_api_key") or getinivaluefromcloud("joplinai", "deepseek_token")

# %% [markdown]
# # 缓存支持

# %% [markdown]
# ## 初始化全局缓存管理器

# %%
# 初始化全局缓存管理器（单例模式）
_CACHE_MANAGER = None


# %% [markdown]
# ## get_cache_manager()

# %%
__all__ = [
    "get_cache_manager",
    "cloud_process_note",
    "describe_images",
    "process_note_vision",
    "ollama_process_note",
    "enhance_note",
    "get_call_stats",
    "reset_call_stats",
    "get_ollama_call_stats",
    "reset_ollama_call_stats",
    "_VisionClient",
    "_SiliconFlowVisionClient",
]

def get_cache_manager():
    """获取缓存管理器 — 本机为数据中心时走 localhost，否则走云端 URL"""
    global _CACHE_MANAGER
    if _CACHE_MANAGER is not None:
        return _CACHE_MANAGER

    center_deviceid = getinivaluefromcloud("joplinai", "center_host_deviceid")
    if center_deviceid:
        try:
            from func.getid import getdeviceid
            local_id = getdeviceid()
            if local_id and str(local_id) == str(center_deviceid):
                remote_url = "http://127.0.0.1:5003"
            else:
                remote_url = getinivaluefromcloud("joplinai", "joplinai_center_url")
        except ImportError:
            remote_url = getinivaluefromcloud("joplinai", "joplinai_center_url")
    else:
        remote_url = getinivaluefromcloud("joplinai", "joplinai_center_url")
    if not remote_url:
        remote_url = "http://127.0.0.1:5003"
    api_key = getinivaluefromcloud("joplinai", "joplinai_center_api_key")
    if remote_url and api_key:
        from aimod.cache_client import CacheClient

        _CACHE_MANAGER = CacheClient(remote_url, api_key)
        return _CACHE_MANAGER

    log.error("缓存管理器初始化失败：缺少 joplinai_center_url 或 joplinai_center_api_key")
    return None


# %% [markdown]
#
# %% [markdown]
# # 大模型（笔记智能加工）

# %% [markdown]
# ## cloud_process_note(text: str, task: str = "summary", model: str = DEFAULT_CHAT_MODEL, max_retries: int = 3, use_cache: bool = True,) -> Optional[str]

# %%
# @timethis
def cloud_process_note(
    text: str,
    task: str = "summary",
    model: str = "",
    max_retries: int = 3,
    use_cache: bool = True,
) -> Optional[str]:
    """大模型处理（集成智能验证机制）"""
    api_key = _get_cloud_api_key()
    if not api_key:
        log.warning("未配置云端 API Key")
        return None

    if not model:
        model = getinivaluefromcloud("joplinai", "cloud_model") or _DEFAULT_CLOUD_MODEL

    if not use_cache:
        # 不使用缓存，直接调用API
        return _call_cloud_api(text, task, model, max_retries)

    # 使用缓存流程
    content_hash = compute_content_hash(text[:8000])
    cache_manager = get_cache_manager()

    # 1. 查询缓存（此时cache_manager只返回数据，不调用API）
    cache_result = cache_manager.get(content_hash, task, model)

    if cache_result.content is not None:
        # 缓存命中
        # log.info(f"增强缓存命中 {task}: {content_hash[:12]}")

        # 2. 检查是否需要验证
        if cache_result.requires_validation:
            log.info(
                f"增强缓存条目达到验证阈值，启动异步验证: {content_hash[:12]}"
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
    log.info(f"增强缓存未命中，调用API: {task} for {content_hash[:12]}")
    new_result = _call_cloud_api(text, task, model, max_retries)

    if new_result:
        # 将新结果保存到缓存
        cache_manager.set(content_hash, task, new_result, model)
        # log.info(f"增强新结果已缓存: {content_hash[:12]}")

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
        new_result = _call_cloud_api(
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
            log.warning(f"增强验证发现内容变化，更新缓存: {cache_key[:12]}...")
            get_cache_manager().update_on_validation(
                cache_key, new_result, validation_successful=True
            )
    except Exception as e:
        log.error(f"增强后台验证过程异常: {e}")
        get_cache_manager().update_on_validation(
            cache_key, None, validation_successful=False
        )


# %% [markdown]
# ## _call_cloud_api(text: str, task: str, model: str, max_retries: int, use_cache: bool = False) -> Optional[str]

# %%
def _call_cloud_api(
    text: str, task: str, model: str, max_retries: int, use_cache: bool = False
) -> Optional[str]:
    """云端大模型处理
    直接调用云端 API 的核心函数。
    注意：这个函数内部不应该再调用缓存逻辑，避免循环。
    """
    api_key = _get_cloud_api_key()
    if not api_key:
        log.warning("未配置云端 API Key")
        return None

    # 添加个人笔记QA提示词
    prompts = {
        "summary": "用1-3句话总结以下笔记核心内容，突出主题和结论：\n%s",
        "tags": """请从以下文本中提取3-5个核心关键词作为标签。
    要求：
    1. 请用中文关键词
    2. 每个标签必须是2-6个字的简短名词、专业术语或关键地名
    3. 避免使用长短语和完整的句子
    4. 用英文逗号分隔，不要编号
    5. 优先提取文本中反复出现的关键概念、实体和**关键地点**
    6. 对于地名，仅提取在文中作为核心要素出现的地点（如事件发生地、主要研究对象所在地、重要机构所在地等）

    文本内容：
    %s

    请直接输出标签，不要有其他说明。""",
    }

    if task not in prompts:
        log.error(f"增强不支持的任务类型: {task}")
        return None

    # 使用 % 格式化，避免花括号问题
    safe_text = text[:8000]
    prompt = prompts[task] % safe_text
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 500,
    }

    api_url = _get_cloud_api_url()
    for attempt in range(max_retries):
        try:
            response = requests.post(
                api_url, headers=headers, json=payload, timeout=30
            )
            if response.status_code == 400:
                log.error(f"云端API 400错误: {response.text}")
                return None
            response.raise_for_status()
            result = response.json()["choices"][0]["message"]["content"].strip()

            return result
        except Exception as e:
            log.warning(
                f"云端大模型调用失败({attempt + 1}/{max_retries}): {str(e)[:100]}"
            )
            time.sleep(2**attempt)
    return None


# %% [markdown]
# ## _call_ollama(text: str, task: str, model: str) -> Optional[str]

# %%
def _call_ollama(text: str, task: str, model: str, ollama_host: str) -> Optional[str]:
    """Ollama 模型处理标签/摘要，复用与云端模型相同的 prompt"""
    prompts = {
        "summary": "用1-3句话总结以下笔记核心内容，突出主题和结论：\n%s",
        "tags": """请从以下文本中提取3-5个核心关键词作为标签。
    要求：
    1. 请用中文关键词
    2. 每个标签必须是2-6个字的简短名词、专业术语或关键地名
    3. 避免使用长短语和完整的句子
    4. 用英文逗号分隔，不要编号
    5. 优先提取文本中反复出现的关键概念、实体和**关键地点**
    6. 对于地名，仅提取在文中作为核心要素出现的地点（如事件发生地、主要研究对象所在地、重要机构所在地等）

    文本内容：
    %s

    请直接输出标签，不要有其他说明。""",
    }

    if task not in prompts:
        log.error(f"Ollama 模型不支持的任务类型: {task}")
        return None

    prompt = prompts[task] % text[:8000]

    for attempt in range(2):
        try:
            response = ollama.Client(host=ollama_host).chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3, "num_predict": 500},
            )
            result = response["message"]["content"].strip()
            log.info(f"Ollama 模型 {model}/{task} 完成，输出 {len(result)} 字符")
            return result
        except Exception as e:
            if attempt == 0:
                log.warning(f"Ollama处理失败({model}/{task}), 重试中: {e}")
                time.sleep(3)
            else:
                log.error(f"Ollama处理失败({model}/{task}): {e}")
                return None


# %% [markdown]
# ## ollama_process_note(text, task, model, use_cache) -> Optional[str]

# %%
# Ollama/Cloud 模型调用统计（供向量化运行结束汇总）
_call_stats = {"summary": {"ollama": 0, "cloud": 0}, "tags": {"ollama": 0, "cloud": 0}}


def get_call_stats() -> dict:
    """返回当前运行的 Ollama/Cloud 模型调用统计"""
    return {
        "summary": dict(_call_stats["summary"]),
        "tags": dict(_call_stats["tags"]),
    }


def reset_call_stats():
    """重置调用统计（新运行开始时调用）"""
    _call_stats["summary"]["ollama"] = 0
    _call_stats["summary"]["cloud"] = 0
    _call_stats["tags"]["ollama"] = 0
    _call_stats["tags"]["cloud"] = 0


def get_ollama_call_stats() -> dict:
    """返回当前运行的 Ollama 模型调用统计"""
    return {task: _call_stats[task]["ollama"] for task in _call_stats}


def reset_ollama_call_stats():
    """重置 Ollama 调用统计"""
    for task in _call_stats:
        _call_stats[task]["ollama"] = 0

# %%
def ollama_process_note(
    text: str,
    task: str = "summary",
    model: str = "qwen2.5:1.5b",
    use_cache: bool = True,
    ollama_host: str = "",
) -> Optional[str]:
    """Ollama 模型处理笔记（标签/摘要），支持缓存。

    与 cloud_process_note 相同的接口，使用 Ollama 模型。
    ollama_host 由调用方从设备级云配置获取，确保跨主机路由正确。
    """
    if not use_cache:
        return _call_ollama(text, task, model, ollama_host)

    content_hash = compute_content_hash(text[:8000])
    cache_manager = get_cache_manager()

    cache_result = cache_manager.get(content_hash, task, model)
    if cache_result.content is not None:
        return cache_result.content

    result = _call_ollama(text, task, model, ollama_host)
    if result:
        cache_manager.set(content_hash, task, result, model)
        if task in _call_stats:
            _call_stats[task]["ollama"] += 1
    return result


# %% [markdown]
# ## enhance_note(text, task, provider, model, use_cache) -> Optional[str]

# %%
_enhance_provider_logged = set()


def enhance_note(
    text: str,
    task: str = "summary",
    provider: str = "cloud",
    model: str = "",
    use_cache: bool = True,
    ollama_host: str = "",
) -> Optional[str]:
    """统一增强入口：根据 provider 路由到 Ollama 或云端模型。

    Args:
        text: 笔记文本
        task: "summary" 或 "tags"
        provider: "ollama" / "cloud" / "none"
        model: 模型名（cloud 默认 deepseek-v4-flash，ollama 默认 qwen2.5:1.5b）
        use_cache: 是否使用缓存
        ollama_host: provider=ollama 时的 Ollama 主机地址（由设备级云配置提供）
    """
    if provider == "none":
        return None

    if provider == "cloud":
        if not model:
            model = getinivaluefromcloud("joplinai", "cloud_model") or _DEFAULT_CLOUD_MODEL
        key = (task, provider)
        if key not in _enhance_provider_logged:
            log.info(f"增强 [{task}] 路由: provider=cloud, model={model}")
            _enhance_provider_logged.add(key)
        result = cloud_process_note(text, task=task, model=model, use_cache=use_cache)
        if result and task in _call_stats:
            _call_stats[task]["cloud"] += 1
        return result

    if provider == "ollama":
        if not model:
            model = "qwen2.5:1.5b"
        key = (task, provider)
        if key not in _enhance_provider_logged:
            log.info(f"增强 [{task}] 路由: provider=ollama, model={model}, host={ollama_host}")
            _enhance_provider_logged.add(key)
        return ollama_process_note(text, task=task, model=model, use_cache=use_cache, ollama_host=ollama_host)

    log.error(f"enhance_note 不支持的 provider: {provider}")
    return None


# %% [markdown]
# # Vision API（硅基流动云端视觉模型）

# %% [markdown]
# ## _VisionClient — 视觉模型策略模式

# %%
class _VisionClient:
    """视觉模型统一接口"""
    def describe(self, image_b64: str, mime: str, prompt: str) -> Optional[str]:
        raise NotImplementedError


class _SiliconFlowVisionClient(_VisionClient):
    """硅基流动视觉模型（OpenAI兼容格式）"""

    URL = _SILICONFLOW_VISION_URL

    def __init__(self, api_key: str, model_id: str):
        self.api_key = api_key
        self.model_id = model_id

    def describe(self, image_b64: str, mime: str, prompt: str) -> Optional[str]:
        data_url = f"data:{mime};base64,{image_b64}"
        payload = {
            "model": self.model_id,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ]
            }],
            "temperature": 0.3,
            "max_tokens": 800,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        for attempt in range(2):
            try:
                resp = requests.post(
                    self.URL, json=payload, headers=headers, timeout=60
                )
                if resp.status_code == 400:
                    log.error(f"硅基流动 Vision API 400错误: {resp.text[:200]}")
                    return None
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                log.warning(
                    f"硅基流动 Vision API调用失败({attempt + 1}/2): {str(e)[:100]}"
                )
                time.sleep(2 ** attempt)
        return None


# %% [markdown]
# ## _get_vision_api_key() -> str

# %%
def _get_vision_api_key() -> str:
    """获取硅基流动 API Key（复用 siliconflow_api_key）"""
    return getinivaluefromcloud("joplinai", "siliconflow_api_key") or ""


# %% [markdown]
# ## _is_valid_vision_result(content: str) -> bool

# %%
def _is_valid_vision_result(content: str, min_length: int = 30) -> bool:
    """Check if a vision model response is a valid description (not a refusal/hallucination)."""
    if not content or len(content) < min_length:
        return False

    refusal_patterns = [
        "很抱歉，我无法",
        "对不起，我不能",
        "抱歉，无法提供",
        "由于我无法访问",
        "我无法查看",
        "无法提供对您提到的",
        "错误的链接或损坏",
        "I cannot provide",
        "I'm unable to",
        "cannot access the image",
    ]
    for pattern in refusal_patterns:
        if pattern in content:
            return False

    import re
    meta_only_patterns = [
        r"^这是一[张个幅].*(?:图片|照片|截图|图像)[，。]?$",
        r"^该.*(?:图片|照片|截图|图像).*(?:显示|展示|包含).*[。，]?$",
    ]
    for pattern in meta_only_patterns:
        if re.match(pattern, content.strip()):
            return False

    return True


# %% [markdown]
# ## describe_images(images, context, model) -> Optional[str]

# %%
def describe_images(
    images: dict[str, dict],
    context: str = "",
    model: str = "",
) -> Optional[str]:
    """Describe images using SiliconFlow vision model with resource-id based caching.

    缓存键: (resource_id, "vision_desc", model)
    同一图片+同一模型不会重复调用API。

    Args:
        images: {resource_id: {'b64': str, 'mime': str}} dict
        context: surrounding text context for better descriptions
        model: vision model ID (default Qwen/Qwen2.5-VL-32B-Instruct)

    Returns concatenated image description text, or None if all fail.
    """
    if not images:
        return None

    if not model:
        model = _DEFAULT_SF_VISION_MODEL

    api_key = _get_vision_api_key()
    if not api_key:
        log.warning("未配置 siliconflow_api_key，无法调用视觉模型")
        return None

    client = _SiliconFlowVisionClient(api_key, model)
    cache_manager = get_cache_manager()

    task = "vision_desc"
    descriptions = []

    for rid, img_data in images.items():
        # 按 resource_id + model 查缓存
        if cache_manager:
            try:
                cache_result = cache_manager.get(rid, task, model)
                if cache_result.content is not None and _is_valid_vision_result(cache_result.content):
                    descriptions.append(cache_result.content)
                    log.info(
                        f"Vision 缓存命中: {rid[:12]}... "
                        f"model={model} ({len(cache_result.content)}字符)"
                    )
                    continue
            except Exception:
                pass

        prompt = "请用中文描述这张图片的内容，重点说明图片传达的关键信息。"
        if context:
            prompt = (
                f"以下是笔记的部分文本上下文，请结合上下文描述图片内容：\n"
                f"{context[:3000]}\n\n"
                f"请用中文描述这张图片的内容，重点说明图片与上下文的关系。"
            )

        result = client.describe(img_data["b64"], img_data["mime"], prompt)
        if result and _is_valid_vision_result(result):
            descriptions.append(result)
            log.info(
                f"硅基流动 Vision 描述成功: {rid[:12]}... "
                f"model={model} ({len(result)}字符)"
            )
            if cache_manager:
                try:
                    cache_manager.set(rid, task, result, model)
                except Exception:
                    pass
        else:
            log.warning(f"图片 {rid} 描述失败 (model={model})")

    if not descriptions:
        return None
    return "\n".join(descriptions)


# %% [markdown]
# ## process_note_vision(note_content, images, context, model) -> Optional[str]

# %%
def process_note_vision(
    note_content: str,
    images: dict[str, dict],
    context: str = "",
    model: str = "",
) -> Optional[str]:
    """Generate a comprehensive note enhancement using vision + text.

    Combines note text and embedded images, sends to SiliconFlow vision model,
    returns a description that enriches the text-only understanding.

    Results are cached by resource_id + model to avoid repeat API calls.
    """
    return describe_images(images, context=context or note_content, model=model)


# %% [markdown]
# # 主函数

# %%
if __name__ == "__main__":
    result = cloud_process_note(
        "请在100字之内介绍joplin笔记软件", task="summary", model="deepseek-v4-flash"
    )
    print(result)
