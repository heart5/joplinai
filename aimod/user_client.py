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
# # UserManagerClient
# 用户管理客户端 — 远程优先 + 本地 SQLite 回退
# 实现与 UserManager 完全相同的公开接口。

# %%
import logging
from typing import Any, Dict, List, Optional

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
__all__ = ["UserManagerClient"]

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

    def create_share(self, user_id: int, question: str, answer: str):
        """创建公开分享链接（本地存储，3天自动过期）。"""
        return self.local.create_share(user_id, question, answer)

    def revoke_share(self, share_id: str) -> bool:
        """撤销公开分享链接。"""
        return self.local.revoke_share(share_id)

    def get_shared_qa(self, share_id: str):
        """获取公开分享的QA内容。"""
        return self.local.get_shared_qa(share_id)
