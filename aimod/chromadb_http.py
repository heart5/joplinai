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
# ---

# %% [markdown]
# # ChromaDB HTTP 薄封装 — 免编译纯 Python 实现 (v2 API)

# %%
"""当 `pip install chromadb` 在 Termux/Android 等环境编译失败时的替代方案。

只依赖 requests，直接调 ChromaDB REST API（v2，含 tenant/database）。
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
_V2_BASE = "/api/v2/tenants/default_tenant/databases/default_database"


class ChromaDBHttpClient:
    """模拟 chromadb.HttpClient，只实现 vector_db_manager 需要的部分。"""

    def __init__(self, host: str = "127.0.0.1", port: int = 8000, **kwargs):
        self._base = f"http://{host}:{port}"
        self._session = requests.Session()
        self._session.headers["Content-Type"] = "application/json"

    # ------------------------------------------------------------------
    # 集合管理
    # ------------------------------------------------------------------

    def _list_collections(self) -> List[Dict]:
        resp = self._session.get(f"{self._base}{_V2_BASE}/collections", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _get_collection_id(self, name: str) -> Optional[str]:
        for coll in self._list_collections():
            if coll.get("name") == name:
                return coll["id"]
        return None

    def get_collection(self, name: str):
        cid = self._get_collection_id(name)
        if cid is None:
            raise ValueError(f"Collection {name} does not exist.")
        return _ChromaCollection(self._base, self._session, cid, name)

    def create_collection(self, name: str, metadata: Optional[Dict] = None, **kwargs):
        body = {"name": name}
        if metadata:
            body["metadata"] = metadata
        resp = self._session.post(
            f"{self._base}{_V2_BASE}/collections", json=body, timeout=10
        )
        if resp.ok:
            data = resp.json()
            return _ChromaCollection(self._base, self._session, data["id"], name)
        raise RuntimeError(f"创建集合失败: {resp.status_code} {resp.text}")

    def delete_collection(self, name: str):
        cid = self._get_collection_id(name)
        if cid:
            resp = self._session.delete(
                f"{self._base}{_V2_BASE}/collections/{cid}", timeout=10
            )
            if not resp.ok:
                raise RuntimeError(f"删除集合失败: {resp.status_code} {resp.text}")

    def list_collections(self):
        return self._list_collections()


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

    @property
    def _coll_url(self):
        return f"{self._base}{_V2_BASE}/collections/{self.id}"

    # ------------------------------------------------------------------
    # count / get
    # ------------------------------------------------------------------

    def count(self) -> int:
        resp = self._session.get(f"{self._coll_url}/count", timeout=10)
        if resp.ok:
            try:
                return int(resp.text)
            except (ValueError, TypeError):
                pass
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
        resp = self._session.post(f"{self._coll_url}/get", json=body, timeout=30)
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
        resp = self._session.post(f"{self._coll_url}/query", json=body, timeout=30)
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
        resp = self._session.post(f"{self._coll_url}/upsert", json=body, timeout=60)
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
        resp = self._session.post(f"{self._coll_url}/update", json=body, timeout=60)
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
        resp = self._session.post(f"{self._coll_url}/delete", json=body, timeout=60)
        if not resp.ok:
            raise RuntimeError(f"delete 失败: {resp.status_code} {resp.text}")
