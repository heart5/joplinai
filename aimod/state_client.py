# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     split_at_heading: true
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
import time
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
__all__ = ["ProcessStateClient", "CenterAPIUnreachableError"]


class CenterAPIUnreachableError(RuntimeError):
    """center_api 不可达，且重试耗尽。调用方应据此决定是否退出。"""


class ProcessStateClient:
    """笔记处理状态客户端 — 纯远程，center_api 不可用时报错"""

    _RETRIES = 3
    _RETRY_DELAY = 3

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
                    timeout=10,
                    **kwargs,
                )
                if resp.ok:
                    return resp
                log.warning(f"远程状态 {method} {path} 返回 {resp.status_code} (第{attempt}次)")
            except Exception as e:
                last_err = str(e)
                log.warning(f"远程状态 {method} {path} 失败: {e} (第{attempt}次)")
            if attempt < self._RETRIES:
                time.sleep(self._RETRY_DELAY * attempt)
        log.error(f"远程状态 {method} {path} 重试{self._RETRIES}次均失败: {last_err}")
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
        raise CenterAPIUnreachableError(
            f"batch_load 远程调用失败 (model={model_name})，"
            f"center_api 不可达。为避免全量重处理事故，同步已中止。"
            f"请确认 center_api 正常后再运行，或用 --enable_force_update 强制运行。"
        )

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
        log.error(f"batch_save 远程调用失败 (model={model_name})，状态未持久化——下次运行可能重复处理")
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
