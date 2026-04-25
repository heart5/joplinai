# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 用户管理

# %% [markdown]
# # 引入库

# %%
import hashlib
import json
import logging
import secrets
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pathmagic

with pathmagic.context():
    from func.first import getdirmain
    from func.jpfuncs import getinivaluefromcloud
    from func.logme import log


# %% [markdown]
# # UserManager类

# %%
class UserManager:
    """基于SQLite的用户、会话及权限管理器"""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        """初始化数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 用户表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                display_name TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('admin', 'colleague')),
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        """)

        # 会话表 (用于持久化登录)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        """)

        # 操作日志表 (可选，用于审计)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                action TEXT NOT NULL,
                details TEXT,
                ip_address TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)

        # 问答历史表 (用于永久保存用户的问答记录)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS qa_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_id TEXT NOT NULL, -- 可用于区分不同设备或浏览器会话
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                metadata TEXT, -- 存储JSON格式的元数据，如来源笔记、是否基于笔记等
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        """)

        conn.commit()
        conn.close()
        log.info(f"用户数据库初始化完成: {self.db_path}")

    def _hash_password(self, password: str) -> str:
        """使用SHA-256加盐哈希密码（生产环境建议使用bcrypt）"""
        if not (salt := getinivaluefromcloud("joplinai", "salt")):
            salt = "joplinai_salt_1232024"  # 应改为从配置文件读取
        return hashlib.sha256((salt + password).encode()).hexdigest()

    def create_user(
        self, username: str, password: str, display_name: str, role: str = "colleague"
    ) -> bool:
        """创建新用户"""
        if role not in ["admin", "colleague"]:
            return False

        password_hash = self._hash_password(password)
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO users (username, password_hash, display_name, role)
                VALUES (?, ?, ?, ?)
            """,
                (username, password_hash, display_name, role),
            )
            conn.commit()
            conn.close()
            log.info(f"用户创建成功: {username}({display_name}), 角色: {role}")
            return True
        except sqlite3.IntegrityError:
            log.warning(f"用户名已存在: {username}")
            return False

    def verify_user(self, username: str, password: str) -> Optional[Dict]:
        """验证用户凭据，返回用户信息字典"""
        password_hash = self._hash_password(password)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 返回字典样式的行
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, username, display_name, role FROM users
            WHERE username = ? AND password_hash = ? AND is_active = 1
        """,
            (username, password_hash),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            user = dict(row)
            self._update_last_login(user["id"])
            return user
        return None

    def _update_last_login(self, user_id: int):
        """更新用户最后登录时间"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET last_login = ? WHERE id = ?", (datetime.now(), user_id)
        )
        conn.commit()
        conn.close()

    def create_session(self, user_id: int, duration_hours: int = 24) -> str:
        """为用户创建会话，返回session_id"""
        session_id = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(hours=duration_hours)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO sessions (session_id, user_id, expires_at)
            VALUES (?, ?, ?)
        """,
            (session_id, user_id, expires_at),
        )
        conn.commit()
        conn.close()
        return session_id

    def validate_session(self, session_id: str) -> Optional[Dict]:
        """验证session_id是否有效并返回关联的用户信息"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT u.id, u.username, u.display_name, u.role
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.session_id = ? AND s.expires_at > ? AND u.is_active = 1
        """,
            (session_id, datetime.now()),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return dict(row)
        return None

    def delete_session(self, session_id: str):
        """注销会话"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()

    def get_all_users(self) -> List[Dict]:
        """获取所有用户列表（管理员功能）"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, display_name, role, is_active, last_login FROM users ORDER BY id"
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def update_user_role(
        self, target_username: str, new_role: str, admin_username: str
    ):
        """更新用户角色（仅管理员）并记录审计日志"""
        # 实现略，包含事务和审计日志插入
        pass

    def log_audit(
        self,
        user_id: Optional[int],
        action: str,
        details: str = "",
        ip_address: str = "",
    ):
        """记录审计日志"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO audit_log (user_id, action, details, ip_address)
            VALUES (?, ?, ?, ?)
        """,
            (user_id, action, details, ip_address),
        )
        conn.commit()
        conn.close()

    def save_qa_history(
        self,
        user_id: int,
        session_id: str,
        question: str,
        answer: str,
        metadata: dict = None,
    ):
        """保存一次用户问答记录到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        metadata_json = json.dumps(metadata) if metadata else None
        cursor.execute(
            """
            INSERT INTO qa_history (user_id, session_id, question, answer, metadata)
            VALUES (?, ?, ?, ?, ?)
        """,
            (user_id, session_id, question, answer, metadata_json),
        )
        conn.commit()
        conn.close()

    def get_qa_history(self, user_id: int, limit: int = 50, offset: int = 0):
        """获取指定用户的问答历史，按时间倒序排列（最新的在前）"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 返回字典形式的行
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT session_id, question, answer, metadata, created_at
            FROM qa_history
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        """,
            (user_id, limit, offset),
        )
        rows = cursor.fetchall()
        conn.close()

        history = []
        for row in rows:
            item = dict(row)
            # 将metadata JSON字符串解析回字典
            if item["metadata"]:
                try:
                    item["metadata"] = json.loads(item["metadata"])
                except:
                    item["metadata"] = {}
            history.append(item)
        return history


# %% [markdown]
# # 全局实例化

# %%
# 全局实例化
USER_DB_PATH = getdirmain() / "data" / "joplinai_users.db"
USER_MANAGER = UserManager(USER_DB_PATH)

# 初始化默认管理员用户（如果不存在）
if not (admin_pw := getinivaluefromcloud("joplinai", "admin_pw")):
    admin_pw = "your_default_admin_password"
if not USER_MANAGER.verify_user("baiyefeng", admin_pw):
    USER_MANAGER.create_user("baiyefeng", admin_pw, "白晔峰", "admin")
