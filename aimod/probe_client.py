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
# # ProbeCacheClient
# 自适应探测结果缓存客户端 — 远程优先，失败降级不报错

# %%
import logging
from typing import Any, Dict, Optional

import requests

# %%
import pathmagic

with pathmagic.Context():
    try:
        from func.logme import log
    except ImportError as e:
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger(__name__)
        log.error(f"导入项目模块失败: {e}")


# %%
__all__ = ["ProbeCacheClient"]

class ProbeCacheClient:
    """自适应探测结果缓存客户端 — 远程优先，失败降级不报错"""

    def __init__(self, remote_url: str, api_key: str):
        self.remote_url = remote_url.rstrip("/")
        self.auth_headers = {"X-API-Key": api_key}
        self._memory = {}  # {text_md5: safe_len}

    def get(self, text_md5: str) -> Optional[int]:
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

    def get_report(self) -> Dict[str, Any]:
        try:
            resp = requests.get(
                f"{self.remote_url}/cache/probe/report",
                headers=self.auth_headers,
                timeout=10,
            )
            if resp.ok:
                return resp.json()
        except Exception as e:
            log.warning(f"远程获取探测缓存报告失败: {e}")
        return {}
