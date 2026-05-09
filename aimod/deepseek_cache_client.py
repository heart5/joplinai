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
# # DeepSeek 缓存远程客户端

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


class RemoteCacheClient:
    """远程缓存客户端 — 与 SQLiteCacheManager 相同接口，远程优先 + 本地回退"""

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
            "POST", "/cache/get", json={"content_hash": content_hash, "task": task}
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
            "/cache/set",
            json={"content_hash": content_hash, "task": task, "result": result},
        ):
            return
        self.local.set(content_hash, task, result)

    def update_on_validation(
        self, cache_key: str, new_result: Optional[str], validation_successful: bool
    ):
        if self._request(
            "POST",
            "/cache/validate",
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
        resp = self._request("GET", "/cache/stats", params=params)
        if resp is not None:
            return resp.json()
        return self.local.get_stats(cache_key=cache_key)
