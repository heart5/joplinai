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
# # CacheClient
# AI增强缓存客户端 — 纯远程，无本地回退

# %%
import logging
import time
from typing import Any, Dict, Optional

import requests

# %%
import pathmagic

with pathmagic.Context():
    try:
        from aimod.cache_manager import CacheResult
        from func.logme import log
    except ImportError as e:
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger(__name__)
        log.error(f"导入项目模块失败: {e}")


# %%
__all__ = ["CacheClient"]

class CacheClient:
    """AI增强缓存客户端 — 纯远程，无本地回退"""

    _RETRIES = 3
    _RETRY_DELAY = 2

    def __init__(self, remote_url: str, api_key: str):
        self.remote_url = remote_url.rstrip("/")
        self.auth_headers = {"X-API-Key": api_key}

    def _request(self, method: str, path: str, **kwargs) -> Optional[requests.Response]:
        last_err = ""
        for attempt in range(1, self._RETRIES + 1):
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
                log.warning(f"远程数据中心 {method} {path} 返回 {resp.status_code} (第{attempt}次)")
            except Exception as e:
                last_err = str(e)
                log.warning(f"远程数据中心 {method} {path} 失败: {e} (第{attempt}次)")
            if attempt < self._RETRIES:
                time.sleep(self._RETRY_DELAY * attempt)
        log.error(f"远程数据中心 {method} {path} 重试{self._RETRIES}次均失败: {last_err}")
        return None

    def get(self, content_hash: str, task: str, model: str = "") -> CacheResult:
        resp = self._request(
            "POST", "/cache/enhance/get",
            json={"content_hash": content_hash, "task": task, "model": model},
        )
        if resp is not None:
            data = resp.json()
            if data.get("found"):
                return CacheResult(
                    content=data["content"],
                    requires_validation=data["requires_validation"],
                    cache_key=data["cache_key"],
                    current_hit_count=data["current_hit_count"],
                    total_hits=data["total_hits"],
                )
        return CacheResult(
            content=None,
            requires_validation=False,
            cache_key=f"{content_hash}_{task}",
            current_hit_count=0,
            total_hits=0,
        )

    def set(self, content_hash: str, task: str, result: str, model: str = ""):
        if self._request(
            "POST",
            "/cache/enhance/set",
            json={"content_hash": content_hash, "task": task, "result": result, "model": model},
        ):
            return
        log.error(f"远程缓存写入失败: {content_hash[:12]}_{task}")

    def update_on_validation(
        self, cache_key: str, new_result: Optional[str], validation_successful: bool
    ):
        if self._request(
            "POST",
            "/cache/enhance/validate",
            json={
                "cache_key": cache_key,
                "new_result": new_result,
                "validation_successful": validation_successful,
            },
        ):
            return
        log.error(f"远程缓存验证更新失败: {cache_key}")

    def get_stats(self, cache_key: str = None) -> Dict[str, Any]:
        params = {"cache_key": cache_key} if cache_key else {}
        resp = self._request("GET", "/cache/enhance/stats", params=params)
        if resp is not None:
            return resp.json()
        log.warning("远程获取缓存统计失败")
        return {}

    def get_report(self) -> Dict[str, Any]:
        resp = self._request("GET", "/cache/enhance/report")
        if resp is not None:
            return resp.json()
        log.warning("远程获取缓存报告失败，返回空报告")
        return {}
