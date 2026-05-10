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
# # Joplinai 统一缓存客户端
# DeepSeek 摘要/标签缓存 + 自适应探测结果缓存，远程优先 + 本地回退。

# %%
import logging
from typing import Any, Dict, Optional

import requests

# %%
import pathmagic

with pathmagic.context():
    try:
        from aimod.cache_manager import CacheResult, SQLiteCacheManager
        from func.logme import log
    except ImportError as e:
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger(__name__)
        log.error(f"导入项目模块失败: {e}")


# %% [markdown]
# # DeepSeekCacheClient

# %%
class DeepSeekCacheClient:
    """DeepSeek 摘要/标签缓存客户端 — 远程优先 + 本地 SQLite 回退"""

    def __init__(self, remote_url: str, api_key: str):
        self.remote_url = remote_url.rstrip("/")
        self.auth_headers = {"X-API-Key": api_key}
        self.local = SQLiteCacheManager(
            db_path="data/.deepseek_cache/deepseek_cache.db"
        )

    def _request(self, method: str, path: str, **kwargs) -> Optional[requests.Response]:
        try:
            resp = requests.request(
                method,
                f"{self.remote_url}{path}",
                headers=self.auth_headers,
                timeout=5,
                **kwargs,
            )
            if resp.ok:
                return resp
            log.warning(f"远程缓存 {method} {path} 返回 {resp.status_code}")
        except Exception as e:
            log.warning(f"远程缓存 {method} {path} 失败: {e}")
        return None

    def get(self, content_hash: str, task: str) -> CacheResult:
        resp = self._request(
            "POST", "/cache/deepseek/get", json={"content_hash": content_hash, "task": task}
        )
        if resp is not None:
            data = resp.json()
            return CacheResult(
                content=data["content"],
                requires_validation=data["requires_validation"],
                cache_key=data["cache_key"],
                current_hit_count=data["current_hit_count"],
                total_hits=data["total_hits"],
            )
        return self.local.get(content_hash, task)

    def set(self, content_hash: str, task: str, result: str):
        if self._request(
            "POST",
            "/cache/deepseek/set",
            json={"content_hash": content_hash, "task": task, "result": result},
        ):
            return
        self.local.set(content_hash, task, result)

    def update_on_validation(
        self, cache_key: str, new_result: Optional[str], validation_successful: bool
    ):
        if self._request(
            "POST",
            "/cache/deepseek/validate",
            json={
                "cache_key": cache_key,
                "new_result": new_result,
                "validation_successful": validation_successful,
            },
        ):
            return
        self.local.update_on_validation(cache_key, new_result, validation_successful)

    def get_stats(self, cache_key: str = None) -> Dict[str, Any]:
        params = {"cache_key": cache_key} if cache_key else {}
        resp = self._request("GET", "/cache/deepseek/stats", params=params)
        if resp is not None:
            return resp.json()
        return self.local.get_stats(cache_key=cache_key)


# %% [markdown]
# # ProbeCacheClient

# %%
class ProbeCacheClient:
    """自适应探测结果缓存客户端 — 远程优先，失败降级不报错"""

    def __init__(self, remote_url: str, api_key: str):
        self.remote_url = remote_url.rstrip("/")
        self.auth_headers = {"X-API-Key": api_key}
        self._memory = {}  # {text_md5: safe_len}

    def get(self, text_md5: str) -> Optional[int]:
        """查远程缓存。返回 safe_len 或 None。"""
        if text_md5 in self._memory:
            return self._memory[text_md5]
        try:
            resp = requests.get(
                f"{self.remote_url}/cache/probe/get/{text_md5}",
                headers=self.auth_headers,
                timeout=5,
            )
            if resp.status_code == 200:
                data = resp.json()
                safe_len = data["safe_len"]
                self._memory[text_md5] = safe_len
                log.debug(f"探测缓存命中: {text_md5[:8]}... → {safe_len}字符")
                return safe_len
        except Exception:
            pass
        return None

    def set(self, text_md5: str, safe_len: int, snippet: str,
            model_name: str, chunk_size: int):
        """写入远程缓存。fire-and-forget，异常忽略。"""
        self._memory[text_md5] = safe_len
        try:
            requests.post(
                f"{self.remote_url}/cache/probe/set",
                json={
                    "text_md5": text_md5,
                    "safe_len": safe_len,
                    "snippet": snippet,
                    "model_name": model_name,
                    "chunk_size": chunk_size,
                },
                headers=self.auth_headers,
                timeout=5,
            )
        except Exception:
            pass
