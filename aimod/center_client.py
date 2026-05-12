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
# # Joplinai 数据中心客户端
# DeepSeek 摘要/标签缓存 + 自适应探测结果缓存 + 历史数据库，远程优先 + 本地回退。

# %%
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# %%
import pathmagic

with pathmagic.Context():
    try:
        from aimod.cache_manager import CacheResult, SQLiteCacheManager
        from func.first import getdirmain
        from func.logme import log
    except ImportError as e:
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger(__name__)
        log.error(f"导入项目模块失败: {e}")


# %% [markdown]
# # 公共：URL 发现

# %%
CENTER_DB_PATH = getdirmain() / "data" / "joplinai_center.db"


# %% [markdown]
# # DeepSeekCacheClient

# %%
class DeepSeekCacheClient:
    """DeepSeek 摘要/标签缓存客户端 — 远程优先 + 本地 SQLite 回退"""

    def __init__(self, remote_url: str, api_key: str):
        self.remote_url = remote_url.rstrip("/")
        self.auth_headers = {"X-API-Key": api_key}
        self.local = SQLiteCacheManager(db_path=str(CENTER_DB_PATH))

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
            log.warning(f"远程数据中心 {method} {path} 返回 {resp.status_code}")
        except Exception as e:
            log.warning(f"远程数据中心 {method} {path} 失败: {e}")
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

    def get_report(self) -> Dict[str, Any]:
        resp = self._request("GET", "/cache/deepseek/report")
        if resp is not None:
            return resp.json()
        log.warning("远程获取缓存报告失败，返回空报告")
        return {}


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


# %% [markdown]
# # HistoryClient

# %%
class HistoryClient:
    """历史数据库客户端 — 远程优先 + 本地 SQLite 回退"""

    def __init__(self, remote_url: str, api_key: str):
        self.remote_url = remote_url.rstrip("/")
        self.auth_headers = {"X-API-Key": api_key}
        self._local_db = str(CENTER_DB_PATH)

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
            log.warning(f"远程历史 {method} {path} 返回 {resp.status_code}")
        except Exception as e:
            log.warning(f"远程历史 {method} {path} 失败: {e}")
        return None

    # ---- 写入 ----

    def add_notebook_record(self, notebook_title: str, stats: Dict,
                            run_id: str, timestamp: str):
        chunk_stats = stats.get("chunk_stats", {})
        payload = {
            "run_id": run_id,
            "notebook_title": notebook_title,
            "timestamp": timestamp,
            "total_notes": stats.get("total_notes", 0),
            "updated_count": stats.get("updated_count", 0),
            "failed_count": len(stats.get("failed_notes", [])),
            "notes_added_count": len(stats.get("notes_added", [])),
            "notes_removed_count": len(stats.get("notes_removed", [])),
            "chunk_stats": {
                "total_chunks": chunk_stats.get("total_chunks", 0),
                "upserted": chunk_stats.get("upserted", 0),
                "skipped": chunk_stats.get("skipped", 0),
                "orphans_cleaned": chunk_stats.get("orphans_cleaned", 0),
            },
            "notes_added_list": stats.get("notes_added", []),
            "notes_removed_list": stats.get("notes_removed", []),
            "failed_notes_list": stats.get("failed_notes", []),
        }
        if self._request("POST", "/history/notebook_record", json=payload):
            log.debug(f"远程历史写入: {notebook_title}")
            return
        self._local_add_notebook_record(notebook_title, stats, run_id, timestamp)

    def finalize_run(self, run_id: str, timestamp: str, embedding_model: str,
                     notebook_count: int, total_notes_processed: int,
                     total_chunks_processed: int, total_notes_added: int,
                     total_notes_removed: int, success: bool = True,
                     error_message: str = None):
        payload = {
            "run_id": run_id,
            "timestamp": timestamp,
            "embedding_model": embedding_model,
            "notebook_count": notebook_count,
            "total_notes_processed": total_notes_processed,
            "total_chunks_processed": total_chunks_processed,
            "total_notes_added": total_notes_added,
            "total_notes_removed": total_notes_removed,
            "success": success,
            "error_message": error_message,
        }
        if self._request("POST", "/history/finalize_run", json=payload):
            log.info(f"远程历史 finalize: {run_id}")
            return
        self._local_finalize_run(run_id, timestamp, embedding_model, notebook_count,
                                 total_notes_processed, total_chunks_processed,
                                 total_notes_added, total_notes_removed, success, error_message)

    # ---- 查询 ----

    def get_cumulative_stats(self, days: int = None) -> Dict:
        params = {}
        if days:
            params["days"] = days
        resp = self._request("GET", "/history/cumulative_stats", params=params)
        if resp is not None:
            return resp.json()
        return self._local_get_cumulative_stats(days)

    def get_change_analysis(self, notebook_title: str = None, days: int = 30) -> Dict:
        params = {"days": days}
        if notebook_title:
            params["notebook"] = notebook_title
        resp = self._request("GET", "/history/change_analysis", params=params)
        if resp is not None:
            return resp.json()
        return self._local_get_change_analysis(notebook_title, days)

    def get_efficiency_metrics(self, days: int = 30) -> Dict:
        resp = self._request("GET", "/history/efficiency_metrics", params={"days": days})
        if resp is not None:
            return resp.json()
        return self._local_get_efficiency_metrics(days)

    # ---- 本地回退 ----

    def _local_add_notebook_record(self, notebook_title: str, stats: Dict,
                                   run_id: str, timestamp: str):
        try:
            chunk_stats = stats.get("chunk_stats", {})
            conn = sqlite3.connect(self._local_db)
            conn.execute("""INSERT INTO notebook_history (
                run_id, notebook_title, timestamp,
                total_notes, updated_count, failed_count,
                notes_added_count, notes_removed_count,
                total_chunks, chunks_upserted, chunks_skipped, chunks_orphans_cleaned,
                notes_added_list, notes_removed_list, failed_notes_list
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                run_id, notebook_title, timestamp,
                stats.get("total_notes", 0), stats.get("updated_count", 0),
                len(stats.get("failed_notes", [])),
                len(stats.get("notes_added", [])), len(stats.get("notes_removed", [])),
                chunk_stats.get("total_chunks", 0), chunk_stats.get("upserted", 0),
                chunk_stats.get("skipped", 0), chunk_stats.get("orphans_cleaned", 0),
                json.dumps(stats.get("notes_added", []), ensure_ascii=False),
                json.dumps(stats.get("notes_removed", []), ensure_ascii=False),
                json.dumps(stats.get("failed_notes", []), ensure_ascii=False),
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            log.error(f"本地历史写入失败: {e}")

    def _local_finalize_run(self, run_id: str, timestamp: str, embedding_model: str,
                            notebook_count: int, total_notes_processed: int,
                            total_chunks_processed: int, total_notes_added: int,
                            total_notes_removed: int, success: bool, error_message: str):
        try:
            conn = sqlite3.connect(self._local_db)
            conn.execute("""INSERT OR REPLACE INTO global_run_history (
                run_id, timestamp, embedding_model, notebook_count,
                total_notes_processed, total_chunks_processed,
                total_notes_added, total_notes_removed, success, error_message
            ) VALUES (?,?,?,?,?,?,?,?,?,?)""", (
                run_id, timestamp, embedding_model, notebook_count,
                total_notes_processed, total_chunks_processed,
                total_notes_added, total_notes_removed, success, error_message,
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            log.error(f"本地历史 finalize 失败: {e}")

    def _local_get_cumulative_stats(self, days: int = None) -> Dict:
        try:
            conn = sqlite3.connect(self._local_db)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            time_condition = ""
            params = []
            if days:
                time_condition = "WHERE timestamp >= datetime('now', ?)"
                params.append(f"-{days} days")
            cursor.execute(f"""SELECT
                COUNT(DISTINCT run_id) as total_runs,
                COUNT(DISTINCT notebook_title) as total_notebooks_touched,
                SUM(total_notes) as total_notes_processed_all_time,
                SUM(total_chunks) as total_chunks_processed_all_time,
                SUM(notes_added_count) as total_notes_added_all_time,
                SUM(notes_removed_count) as total_notes_removed_all_time,
                SUM(chunks_upserted) as total_chunks_updated_all_time,
                SUM(chunks_orphans_cleaned) as total_orphans_cleaned_all_time
            FROM notebook_history {time_condition}""", params)
            cumulative = dict(cursor.fetchone())
            cursor.execute("""SELECT strftime('%Y-%W', timestamp) as week,
                COUNT(DISTINCT run_id) as runs_count, SUM(total_notes) as notes_processed,
                SUM(total_chunks) as chunks_processed, SUM(notes_added_count) as notes_added,
                SUM(notes_removed_count) as notes_removed
            FROM notebook_history WHERE timestamp >= datetime('now', '-90 days')
            GROUP BY week ORDER BY week DESC LIMIT 12""")
            weekly_trends = [dict(row) for row in cursor.fetchall()]
            cursor.execute(f"""SELECT notebook_title, COUNT(*) as process_count,
                SUM(total_notes) as total_notes, SUM(total_chunks) as total_chunks,
                MAX(timestamp) as last_processed
            FROM notebook_history {time_condition}
            GROUP BY notebook_title ORDER BY process_count DESC LIMIT 10""", params)
            top_notebooks = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return {"cumulative": cumulative, "weekly_trends": weekly_trends,
                    "top_notebooks": top_notebooks,
                    "analysis_period": f"最近{days}天" if days else "全部历史"}
        except Exception as e:
            log.error(f"本地累积统计失败: {e}")
            return {}

    def _local_get_change_analysis(self, notebook_title: str = None, days: int = 30) -> Dict:
        try:
            conn = sqlite3.connect(self._local_db)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            where = ["timestamp >= datetime('now', ?)"]
            params = [f"-{days} days"]
            if notebook_title:
                where.append("notebook_title = ?")
                params.append(notebook_title)
            cursor.execute(
                f"SELECT notes_added_list, notes_removed_list FROM notebook_history WHERE {' AND '.join(where)}",
                params)
            all_added, all_removed = [], []
            for row in cursor.fetchall():
                if row["notes_added_list"]:
                    all_added.extend(json.loads(row["notes_added_list"]))
                if row["notes_removed_list"]:
                    all_removed.extend(json.loads(row["notes_removed_list"]))
            unique_added = list(set(all_added))
            unique_removed = list(set(all_removed))
            conn.close()
            return {
                "analysis_period": f"最近{days}天",
                "notebook": notebook_title or "全局",
                "unique_notes_added": unique_added,
                "unique_notes_removed": unique_removed,
                "added_count": len(unique_added),
                "removed_count": len(unique_removed),
                "net_growth": len(unique_added) - len(unique_removed),
                "frequently_changed_notes": list(set(unique_added) & set(unique_removed)),
                "frequently_changed_count": len(set(unique_added) & set(unique_removed)),
            }
        except Exception as e:
            log.error(f"本地变动分析失败: {e}")
            return {}

    def _local_get_efficiency_metrics(self, days: int = 30) -> Dict:
        try:
            conn = sqlite3.connect(self._local_db)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""SELECT
                AVG(total_notes) as avg_notes_per_run, AVG(total_chunks) as avg_chunks_per_run,
                SUM(chunks_upserted)*100.0/NULLIF(SUM(total_chunks),0) as avg_update_rate_percent,
                SUM(chunks_skipped)*100.0/NULLIF(SUM(total_chunks),0) as avg_skip_rate_percent,
                SUM(notes_added_count)*100.0/NULLIF(SUM(total_notes),0) as avg_addition_rate_percent,
                SUM(notes_removed_count)*100.0/NULLIF(SUM(total_notes),0) as avg_removal_rate_percent,
                COUNT(DISTINCT DATE(timestamp)) as active_days,
                COUNT(DISTINCT run_id) as total_runs,
                COUNT(DISTINCT run_id)*1.0/NULLIF(COUNT(DISTINCT DATE(timestamp)),1) as avg_runs_per_day
            FROM notebook_history WHERE timestamp >= datetime('now', ?)""", [f"-{days} days"])
            metrics = dict(cursor.fetchone())
            cursor.execute("""SELECT COUNT(*) as total_runs,
                SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) as successful_runs
            FROM global_run_history WHERE timestamp >= datetime('now', ?)""", [f"-{days} days"])
            rs = cursor.fetchone()
            metrics["success_rate_percent"] = (rs["successful_runs"]*100.0/rs["total_runs"]) if rs and rs["total_runs"]>0 else 0.0
            conn.close()
            for k in list(metrics.keys()):
                if metrics[k] is None:
                    metrics[k] = 0.0 if ("percent" in k or "rate" in k or "avg" in k) else 0
                if "percent" in k or "rate" in k:
                    metrics[k] = round(float(metrics[k]), 2)
                elif isinstance(metrics[k], (int, float)):
                    metrics[k] = round(float(metrics[k]), 2)
            return metrics
        except Exception as e:
            log.error(f"本地效率指标失败: {e}")
            return {
                "avg_notes_per_run": 0.0, "avg_chunks_per_run": 0.0,
                "avg_update_rate_percent": 0.0, "avg_skip_rate_percent": 0.0,
                "avg_addition_rate_percent": 0.0, "avg_removal_rate_percent": 0.0,
                "active_days": 0, "total_runs": 0, "avg_runs_per_day": 0.0,
                "success_rate_percent": 0.0,
            }


# %% [markdown]
# # ProcessStateClient


# %%
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


# %% [markdown]
# # UserManagerClient


# %%
class UserManagerClient:
    """用户管理客户端 — 远程优先 + 本地 SQLite 回退

    实现与 UserManager 完全相同的公开接口，joplin_web_app.py 无需改动。
    """

    def __init__(self, remote_url: str, api_key: str, local_db_path: str):
        self.remote_url = remote_url.rstrip("/")
        self.auth_headers = {"X-API-Key": api_key}
        self._local_db_path = local_db_path
        self._local = None

    @property
    def local(self):
        if self._local is None:
            from src.user_manager import UserManager
            self._local = UserManager(self._local_db_path)
        return self._local

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
            log.warning(f"远程用户管理 {method} {path} 返回 {resp.status_code}")
        except Exception as e:
            log.warning(f"远程用户管理 {method} {path} 失败: {e}")
        return None

    # ---- 认证 ----

    def verify_user(self, username: str, password: str) -> Optional[Dict]:
        password_hash = self.local._hash_password(password)
        resp = self._request("POST", "/auth/verify", json={
            "username": username, "password_hash": password_hash,
        })
        if resp is not None:
            data = resp.json()
            return data["user"] if data.get("found") else None
        return self.local.verify_user(username, password)

    def create_session(self, user_id: int, duration_hours: int = 24) -> str:
        resp = self._request("POST", "/auth/create_session", json={
            "user_id": user_id, "duration_hours": duration_hours,
        })
        if resp is not None:
            return resp.json()["session_id"]
        return self.local.create_session(user_id, duration_hours)

    def validate_session(self, session_id: str) -> Optional[Dict]:
        resp = self._request("POST", "/auth/validate_session", json={"session_id": session_id})
        if resp is not None:
            data = resp.json()
            return data["user"] if data.get("valid") else None
        return self.local.validate_session(session_id)

    def delete_session(self, session_id: str):
        resp = self._request("POST", "/auth/delete_session", json={"session_id": session_id})
        if resp is not None:
            return
        self.local.delete_session(session_id)

    # ---- 用户 CRUD ----

    def get_all_users(self) -> List[Dict]:
        resp = self._request("GET", "/users")
        if resp is not None:
            return resp.json()["users"]
        return self.local.get_all_users()

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        resp = self._request("GET", f"/users/{username}")
        if resp is not None:
            data = resp.json()
            return data["user"] if data.get("found") else None
        return self.local.get_user_by_username(username)

    def get_user_with_notebooks(self, username: str) -> Optional[Dict]:
        resp = self._request("GET", f"/users/{username}")
        if resp is not None:
            data = resp.json()
            return data["user"] if data.get("found") else None
        return self.local.get_user_with_notebooks(username)

    def create_user(self, username: str, password: str, display_name: str,
                    role: str = "team_member", allowed_notebooks: list = None) -> bool:
        password_hash = self.local._hash_password(password)
        resp = self._request("POST", "/users/create", json={
            "username": username, "password_hash": password_hash,
            "display_name": display_name, "role": role,
            "allowed_notebooks": allowed_notebooks or [],
        })
        if resp is not None:
            return resp.json().get("ok", False)
        return self.local.create_user(username, password, display_name, role, allowed_notebooks)

    def delete_user(self, target_username: str, admin_username: str) -> bool:
        resp = self._request("POST", "/users/delete", json={
            "target_username": target_username, "admin_username": admin_username,
        })
        if resp is not None:
            return resp.json().get("ok", False)
        return self.local.delete_user(target_username, admin_username)

    def update_user_role(self, target_username: str, new_role: str, admin_username: str) -> bool:
        resp = self._request("POST", "/users/update_role", json={
            "target_username": target_username, "new_role": new_role,
            "admin_username": admin_username,
        })
        if resp is not None:
            return resp.json().get("ok", False)
        return self.local.update_user_role(target_username, new_role, admin_username)

    def update_user_permissions(self, username: str, role: str = None,
                                allowed_notebooks: list = None) -> bool:
        payload = {"username": username}
        if role is not None:
            payload["role"] = role
        if allowed_notebooks is not None:
            payload["allowed_notebooks"] = allowed_notebooks
        resp = self._request("POST", "/users/update_permissions", json=payload)
        if resp is not None:
            return resp.json().get("ok", False)
        return self.local.update_user_permissions(username, role, allowed_notebooks)

    def reset_user_password(self, target_username: str, new_password: str,
                            admin_username: str) -> bool:
        new_password_hash = self.local._hash_password(new_password)
        resp = self._request("POST", "/users/reset_password", json={
            "target_username": target_username,
            "new_password_hash": new_password_hash,
            "admin_username": admin_username,
        })
        if resp is not None:
            return resp.json().get("ok", False)
        return self.local.reset_user_password(target_username, new_password, admin_username)

    def update_user_active_status(self, target_username: str, is_active: bool,
                                  admin_username: str) -> bool:
        resp = self._request("POST", "/users/toggle_active", json={
            "target_username": target_username, "is_active": is_active,
            "admin_username": admin_username,
        })
        if resp is not None:
            return resp.json().get("ok", False)
        return self.local.update_user_active_status(target_username, is_active, admin_username)

    def update_user_display_name(self, target_username: str, new_display_name: str,
                                 admin_username: str) -> bool:
        resp = self._request("POST", "/users/update_display_name", json={
            "target_username": target_username,
            "new_display_name": new_display_name,
            "admin_username": admin_username,
        })
        if resp is not None:
            return resp.json().get("ok", False)
        return self.local.update_user_display_name(target_username, new_display_name, admin_username)

    # ---- 聊天会话 ----

    def get_user_chat_sessions(self, user_id: int) -> List[Dict]:
        resp = self._request("GET", f"/chat_sessions/{user_id}")
        if resp is not None:
            return resp.json()["sessions"]
        return self.local.get_user_chat_sessions(user_id)

    def create_chat_session(self, user_id: int, name: str = "新对话") -> str:
        resp = self._request("POST", "/chat_sessions/create", json={
            "user_id": user_id, "name": name,
        })
        if resp is not None:
            return resp.json()["session_id"]
        return self.local.create_chat_session(user_id, name)

    def rename_chat_session(self, session_id: str, new_name: str) -> bool:
        resp = self._request("POST", "/chat_sessions/rename", json={
            "session_id": session_id, "new_name": new_name,
        })
        if resp is not None:
            return resp.json().get("ok", False)
        return self.local.rename_chat_session(session_id, new_name)

    def delete_chat_session(self, session_id: str) -> bool:
        resp = self._request("POST", "/chat_sessions/delete", json={"session_id": session_id})
        if resp is not None:
            return resp.json().get("ok", False)
        return self.local.delete_chat_session(session_id)

    def set_active_chat_session(self, user_id: int, session_id: str):
        resp = self._request("POST", "/chat_sessions/set_active", json={
            "user_id": user_id, "session_id": session_id,
        })
        if resp is not None:
            return
        self.local.set_active_chat_session(user_id, session_id)

    def get_active_chat_session(self, user_id: int) -> Optional[str]:
        resp = self._request("GET", f"/chat_sessions/{user_id}/active")
        if resp is not None:
            data = resp.json()
            return data["session_id"] if data.get("found") else None
        return self.local.get_active_chat_session(user_id)

    def _create_chat_session_with_id(self, user_id: int, session_id: str, name: str):
        resp = self._request("POST", "/chat_sessions/create_with_id", json={
            "user_id": user_id, "session_id": session_id, "name": name,
        })
        if resp is not None:
            return
        self.local._create_chat_session_with_id(user_id, session_id, name)

    # ---- 问答历史 ----

    def save_qa_history(self, user_id: int, session_id: str, question: str,
                        answer: str, metadata: dict = None):
        resp = self._request("POST", "/qa/save", json={
            "user_id": user_id, "session_id": session_id,
            "question": question, "answer": answer,
            "metadata": metadata,
        })
        if resp is not None:
            return
        self.local.save_qa_history(user_id, session_id, question, answer, metadata)

    def get_qa_history(self, user_id: int, limit: int = 50, offset: int = 0,
                       session_id: Optional[str] = None):
        params = {"limit": limit, "offset": offset}
        if session_id:
            params["session_id"] = session_id
            resp = self._request("GET", f"/qa/{user_id}", params=params)
        else:
            resp = self._request("GET", f"/qa/{user_id}", params=params)
        if resp is not None:
            return resp.json()["history"]
        return self.local.get_qa_history(user_id, limit, offset, session_id)

    def get_qa_history_by_session(self, session_id: str) -> List[Dict]:
        """按 session_id 查询历史（用于 restore_history）"""
        resp = self._request("GET", f"/qa/by_session/{session_id}")
        if resp is not None:
            return resp.json()["history"]
        return self.local.get_qa_history(0, limit=100, session_id=session_id)

    # ---- 审计日志 ----

    def log_audit(self, user_id: Optional[int], action: str, details: str = "",
                  ip_address: str = ""):
        resp = self._request("POST", "/audit/log", json={
            "user_id": user_id, "action": action, "details": details,
            "ip_address": ip_address,
        })
        if resp is not None:
            return
        self.local.log_audit(user_id, action, details, ip_address)

    def get_audit_logs(self, page: int = 1, per_page: int = 20,
                       username: Optional[str] = None, action: Optional[str] = None,
                       start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict:
        params = {"page": page, "per_page": per_page}
        if username:
            params["username"] = username
        if action:
            params["action"] = action
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        resp = self._request("GET", "/audit/logs", params=params)
        if resp is not None:
            return resp.json()
        return self.local.get_audit_logs(page, per_page, username, action, start_date, end_date)

    def get_audit_actions(self) -> List[str]:
        resp = self._request("GET", "/audit/actions")
        if resp is not None:
            return resp.json()["actions"]
        return self.local.get_audit_actions()

    def clear_audit_logs(self, before_days: int = 90) -> int:
        resp = self._request("POST", "/audit/clear", json={"before_days": before_days})
        if resp is not None:
            return resp.json().get("deleted", 0)
        return self.local.clear_audit_logs(before_days)
