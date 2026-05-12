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
# 笔记处理状态客户端 — 远程优先 + 本地 JSON 回退

# %%
import json
import logging
from pathlib import Path
from typing import Dict, Optional

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
    """笔记处理状态客户端 — 远程优先 + 本地 JSON 回退"""

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

    def batch_load(self, model_name: str, local_path: Path) -> Dict[str, Dict]:
        resp = self._request("POST", "/state/batch_load", json={"model_name": model_name})
        if resp is not None:
            data = resp.json()
            result = dict(data.get("states", {}))
            if "virtual_collections" in data and data["virtual_collections"]:
                result["_virtual_collections"] = data["virtual_collections"]
            return result
        return self._local_load(local_path)

    def batch_save(self, model_name: str, state: Dict, local_path: Path):
        states = {k: v for k, v in state.items() if k != "_virtual_collections"}
        virtual_collections = state.get("_virtual_collections", {})
        resp = self._request("POST", "/state/batch_save", json={
            "model_name": model_name,
            "states": states,
            "virtual_collections": virtual_collections,
        })
        if resp is not None:
            return
        self._local_save(state, local_path)

    @staticmethod
    def _local_load(path: Path) -> Dict:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    @staticmethod
    def _local_save(state: Dict, path: Path):
        def serialize(obj):
            from datetime import datetime as dt
            if isinstance(obj, dt):
                return obj.isoformat()
            if isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            if isinstance(obj, (list, tuple)):
                return [serialize(item) for item in obj]
            if isinstance(obj, dict):
                return {k: serialize(v) for k, v in obj.items()}
            return str(obj)

        try:
            serialized_state = serialize(state)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(serialized_state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.error(f"保存状态文件{path}失败: {e}")
