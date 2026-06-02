# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %% [markdown]
# # ChromaDB HTTP 薄封装 — 免编译纯 Python 实现

# %%
"""当 `pip install chromadb` 在 Termux/Android 等环境编译失败时的替代方案。

只依赖 requests，直接调 ChromaDB REST API（v1）。
接口完全模拟 chromadb.HttpClient + Collection 的核心方法，
覆盖 vector_db_manager.py 实际用到的所有调用。

TC Docker: chromadb/chroma, 端口映射 8009→8000。
"""

# %%
from typing import Dict, List, Optional

import requests

# %% [markdown]
# ## ChromaDBHttpClient


# %%
class ChromaDBHttpClient:
    """模拟 chromadb.HttpClient，只实现 vector_db_manager 需要的部分。"""

    def __init__(self, host: str = "127.0.0.1", port: int = 8000, **kwargs):
        self._base = f"http://{host}:{port}/api/v1"
        self._session = requests.Session()
        self._session.headers["Content-Type"] = "application/json"

    # ------------------------------------------------------------------
    # 集合管理
    # ------------------------------------------------------------------

    def _get_collection_id(self, name: str) -> Optional[str]:
        """按名称查 collection id。"""
        resp = self._session.get(f"{self._base}/collections", timeout=10)
        if resp.ok:
            for coll in resp.json():
                if coll.get("name") == name:
                    return coll["id"]
        return None

    def get_collection(self, name: str):
        """获取已存在的集合，不存在则抛异常。"""
        cid = self._get_collection_id(name)
        if cid is None:
            raise ValueError(f"Collection {name} does not exist.")
        return _ChromaCollection(self._base, self._session, cid, name)

    def create_collection(self, name: str, metadata: Optional[Dict] = None, **kwargs):
        """创建新集合。"""
        body = {"name": name}
        if metadata:
            body["metadata"] = metadata
        resp = self._session.post(f"{self._base}/collections", json=body, timeout=10)
        if resp.ok:
            data = resp.json()
            return _ChromaCollection(self._base, self._session, data["id"], name)
        raise RuntimeError(f"创建集合失败: {resp.status_code} {resp.text}")

    def delete_collection(self, name: str):
        """删除集合。"""
        cid = self._get_collection_id(name)
        if cid:
            resp = self._session.delete(f"{self._base}/collections/{cid}", timeout=10)
            if not resp.ok:
                raise RuntimeError(f"删除集合失败: {resp.status_code} {resp.text}")

    def list_collections(self):
        """列出所有集合。"""
        resp = self._session.get(f"{self._base}/collections", timeout=10)
        if resp.ok:
            return resp.json()
        return []


# %% [markdown]
# ## _ChromaCollection — 集合操作


# %%
class _ChromaCollection:
    """模拟 chromadb.Collection 的核心方法。"""

    def __init__(self, base_url: str, session: requests.Session, cid: str, name: str):
        self._base = base_url
        self._session = session
        self.id = cid
        self.name = name

    # ------------------------------------------------------------------
    # count / get
    # ------------------------------------------------------------------

    def count(self) -> int:
        resp = self._session.get(f"{self._base}/collections/{self.id}/count", timeout=10)
        if resp.ok:
            return int(resp.text)
        return 0

    def get(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict:
        """获取集合中的条目。"""
        body: Dict = {}
        if ids:
            body["ids"] = list(ids)
        if where:
            body["where"] = where
        if limit is not None:
            body["limit"] = limit
        if offset is not None:
            body["offset"] = offset
        if include:
            body["include"] = include
        resp = self._session.post(
            f"{self._base}/collections/{self.id}/get", json=body, timeout=30
        )
        if resp.ok:
            return resp.json()
        return {}

    # ------------------------------------------------------------------
    # query
    # ------------------------------------------------------------------

    def query(
        self,
        query_embeddings: Optional[List] = None,
        query_texts: Optional[List[str]] = None,
        n_results: int = 10,
        where: Optional[Dict] = None,
        include: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict:
        body: Dict = {"n_results": n_results}
        if query_embeddings:
            body["query_embeddings"] = query_embeddings
        if query_texts:
            body["query_texts"] = query_texts
        if where:
            body["where"] = where
        if include:
            body["include"] = include
        resp = self._session.post(
            f"{self._base}/collections/{self.id}/query", json=body, timeout=30
        )
        if resp.ok:
            return resp.json()
        return {}

    # ------------------------------------------------------------------
    # upsert / update / delete
    # ------------------------------------------------------------------

    def upsert(
        self,
        ids: List[str],
        embeddings: Optional[List] = None,
        metadatas: Optional[List[Dict]] = None,
        documents: Optional[List[str]] = None,
        **kwargs,
    ):
        body: Dict = {"ids": ids}
        if embeddings is not None:
            body["embeddings"] = embeddings
        if metadatas is not None:
            body["metadatas"] = metadatas
        if documents is not None:
            body["documents"] = documents
        resp = self._session.post(
            f"{self._base}/collections/{self.id}/upsert", json=body, timeout=60
        )
        if not resp.ok:
            raise RuntimeError(f"upsert 失败: {resp.status_code} {resp.text}")

    def update(
        self,
        ids: List[str],
        embeddings: Optional[List] = None,
        metadatas: Optional[List[Dict]] = None,
        documents: Optional[List[str]] = None,
        **kwargs,
    ):
        body: Dict = {"ids": ids}
        if embeddings is not None:
            body["embeddings"] = embeddings
        if metadatas is not None:
            body["metadatas"] = metadatas
        if documents is not None:
            body["documents"] = documents
        resp = self._session.post(
            f"{self._base}/collections/{self.id}/update", json=body, timeout=60
        )
        if not resp.ok:
            raise RuntimeError(f"update 失败: {resp.status_code} {resp.text}")

    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict] = None,
        **kwargs,
    ):
        body: Dict = {}
        if ids:
            body["ids"] = list(ids)
        if where:
            body["where"] = where
        resp = self._session.post(
            f"{self._base}/collections/{self.id}/delete", json=body, timeout=60
        )
        if not resp.ok:
            raise RuntimeError(f"delete 失败: {resp.status_code} {resp.text}")
