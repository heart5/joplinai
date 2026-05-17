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
# # ProcessStateClient
# 笔记处理状态客户端 — 纯远程，无本地 fallback

# %%
import logging
from typing import Optional

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
__all__ = ["ProcessStateClient"]


class ProcessStateClient:
    """笔记处理状态客户端 — 纯远程，center_api 不可用时报错"""

    def __init__(self, remote_url: str, api_key: str):
        self.remote_url = remote_url.rstrip("/")
        self.auth_headers = {"X-API-Key": api_key}

    def _request(self, method: str, path: str, **kwargs) -> Optional[requests.Response]:
        try:
            resp = requests.request(
                method,
                f"{self.remote_url}{path}",
                headers=self.auth_headers,
                timeout=10,
                **kwargs,
            )
            if resp.ok:
                return resp
            log.warning(f"远程状态 {method} {path} 返回 {resp.status_code}")
        except Exception as e:
            log.warning(f"远程状态 {method} {path} 失败: {e}")
        return None

    # ---- 笔记处理状态 ----

    def batch_load(self, model_name: str) -> dict:
        resp = self._request("POST", "/state/batch_load", json={"model_name": model_name})
        if resp is not None:
            data = resp.json()
            result = dict(data.get("states", {}))
            if data.get("virtual_collections"):
                result["_virtual_collections"] = data["virtual_collections"]
            return result
        log.error(f"batch_load 远程调用失败 (model={model_name})，状态无法加载，视为全新运行")
        return {}

    def batch_save(self, model_name: str, state: dict) -> bool:
        states = {k: v for k, v in state.items() if k != "_virtual_collections"}
        virtual_collections = state.get("_virtual_collections", {})
        resp = self._request("POST", "/state/batch_save", json={
            "model_name": model_name,
            "states": states,
            "virtual_collections": virtual_collections,
        })
        if resp is not None:
            return True
        log.error(f"batch_save 远程调用失败 (model={model_name})，状态未持久化")
        return False

    # ---- 运行时标记 (checkpoint / batch_progress) ----

    def load_run_state(self, model_name: str, key: str) -> Optional[dict]:
        resp = self._request("POST", "/state/run_state/load", json={
            "model_name": model_name,
            "key": key,
        })
        if resp is not None:
            data = resp.json()
            if data.get("found"):
                return data["value"]
        return None

    def save_run_state(self, model_name: str, key: str, value: dict) -> bool:
        resp = self._request("POST", "/state/run_state/save", json={
            "model_name": model_name,
            "key": key,
            "value": value,
        })
        return resp is not None

    def delete_run_state(self, model_name: str, key: str) -> bool:
        resp = self._request("POST", "/state/run_state/delete", json={
            "model_name": model_name,
            "key": key,
        })
        return resp is not None
