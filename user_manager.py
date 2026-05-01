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


# %% [markdown]
# ## __init__(self, db_path: str)

    # %%
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self._init_db()

# %% [markdown]
# ## _init_db(self)

    # %%
    def _init_db(self):
        """初始化数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                display_name TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('admin', 'team_leader', 'team_member')),
                allowed_notebooks TEXT, -- JSON数组格式：["笔记本A", "笔记本B"]
                is_active BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        """)
        # 创建索引
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)"
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)")
        log.info("用户数据库表结构已初始化（支持三级权限）")

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

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_id TEXT NOT NULL UNIQUE,
                name TEXT DEFAULT '新对话',
                is_active INTEGER DEFAULT 0,       -- 0否1是，标记当前活动会话
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        """)

        conn.commit()
        conn.close()
        log.info(f"用户数据库初始化完成: {self.db_path}")

# %% [markdown]
# ## _hash_password(self, password: str) -> str

    # %%
    def _hash_password(self, password: str) -> str:
        """使用SHA-256加盐哈希密码（生产环境建议使用bcrypt）"""
        if not (salt := getinivaluefromcloud("joplinai", "salt")):
            salt = "joplinai_salt_1232024"  # 应改为从配置文件读取
        return hashlib.sha256((salt + password).encode()).hexdigest()

# %% [markdown]
# ## create_user(self, username: str, password: str, display_name: str, role: str = "colleague") -> bool

    # %%
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

# %% [markdown]
# ## verify_user(self, username: str, password: str) -> Optional[Dict]

    # %%
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

# %% [markdown]
# ## _update_last_login(self, user_id: int)

    # %%
    def _update_last_login(self, user_id: int):
        """更新用户最后登录时间"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE users SET last_login = ? WHERE id = ?", (datetime.now(), user_id)
        )
        conn.commit()
        conn.close()

# %% [markdown]
# ## create_session(self, user_id: int, duration_hours: int = 24) -> str

    # %%
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

# %% [markdown]
# ## validate_session(self, session_id: str) -> Optional[Dict]

    # %%
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

# %% [markdown]
# ## delete_session(self, session_id: str)

    # %%
    def delete_session(self, session_id: str):
        """注销会话"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()

# %% [markdown]
# ## get_all_users(self) -> List[Dict]

    # %%
    def get_all_users(self) -> List[Dict]:
        """获取所有用户列表（管理员功能）"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, display_name, role, is_active, created_at, last_login FROM users ORDER BY id"
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

# %% [markdown]
# ## update_user_role(self, target_username: str, new_role: str, admin_username: str)

    # %%
    def update_user_role(
        self, target_username: str, new_role: str, admin_username: str
    ):
        """更新用户角色（仅管理员）并记录审计日志"""
        # 实现略，包含事务和审计日志插入
        pass

# %% [markdown]
# ## log_audit(self,user_id: Optional[int],action: str,details: str = "",ip_address: str = "",)

    # %%
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

# %% [markdown]
# ## save_qa_history(self, user_id: int, session_id: str, question: str, answer: str, metadata: dict = None,)

    # %%
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

# %% [markdown]
# ## get_qa_history(self, user_id: int, limit: int = 50, offset: int = 0, session_id: Optional[str] = None,)

    # %%
    def get_qa_history(
        self,
        user_id: int,
        limit: int = 50,
        offset: int = 0,
        session_id: Optional[str] = None,
    ):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if session_id:
            cursor.execute(
                """SELECT session_id, question, answer, metadata, created_at
                FROM qa_history
                WHERE session_id = ?
                ORDER BY created_at DESC LIMIT ? OFFSET ?""",
                (session_id, limit, offset),
            )
        else:
            cursor.execute(
                """SELECT session_id, question, answer, metadata, created_at
                FROM qa_history
                WHERE user_id = ?
                ORDER BY created_at DESC LIMIT ? OFFSET ?""",
                (user_id, limit, offset),
            )
        rows = cursor.fetchall()
        conn.close()
        history = []
        for row in rows:
            item = dict(row)
            if item["metadata"]:
                try:
                    item["metadata"] = json.loads(item["metadata"])
                except:
                    item["metadata"] = {}
            history.append(item)
        return history

    # 在 UserManager 类中添加以下方法

# %% [markdown]
# ## reset_user_password(self, target_username: str, new_password: str, admin_username: str ) -> bool

    # %%
    def reset_user_password(
        self, target_username: str, new_password: str, admin_username: str
    ) -> bool:
        """
        管理员重置用户密码
        Args:
            target_username: 被操作用户名
            new_password: 新密码（明文）
            admin_username: 执行操作的管理员用户名（用于审计）
        Returns:
            成功与否
        """
        password_hash = self._hash_password(new_password)
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE users SET password_hash = ? WHERE username = ?",
                (password_hash, target_username),
            )
            if cursor.rowcount == 0:
                conn.close()
                return False
            conn.commit()
            conn.close()
            # 记录审计日志（需要先查出admin_user_id，这里简化，可根据实际情况调整）
            admin_user = self._get_user_by_username(admin_username)
            if admin_user:
                self.log_audit(
                    admin_user["id"],
                    "RESET_PASSWORD",
                    details=f"重置用户 [{target_username}] 的密码",
                    ip_address="",  # IP可在web层补充
                )
            log.info(f"管理员 {admin_username} 重置了用户 {target_username} 的密码")
            return True
        except Exception as e:
            log.error(f"重置密码失败: {e}")
            return False

# %% [markdown]
# ## update_user_active_status(self, target_username: str, is_active: bool, admin_username: str) -> bool

    # %%
    def update_user_active_status(
        self, target_username: str, is_active: bool, admin_username: str
    ) -> bool:
        """
        启用/禁用用户账户
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE users SET is_active = ? WHERE username = ?",
                (1 if is_active else 0, target_username),
            )
            if cursor.rowcount == 0:
                conn.close()
                return False
            conn.commit()
            conn.close()
            admin_user = self._get_user_by_username(admin_username)
            if admin_user:
                self.log_audit(
                    admin_user["id"],
                    "UPDATE_USER_STATUS",
                    details=f"将用户 [{target_username}] 状态设为 {'活跃' if is_active else '禁用'}",
                    ip_address="",
                )
            log.info(
                f"管理员 {admin_username} 将用户 {target_username} 状态设为 {'活跃' if is_active else '禁用'}"
            )
            return True
        except Exception as e:
            log.error(f"更新用户状态失败: {e}")
            return False

# %% [markdown]
# ## update_user_role(self, target_username: str, new_role: str, admin_username: str) -> bool

    # %%
    def update_user_role(
        self, target_username: str, new_role: str, admin_username: str
    ) -> bool:
        """
        更新用户角色（admin/colleague）
        """
        if new_role not in ["admin", "colleague"]:
            return False
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE users SET role = ? WHERE username = ?",
                (new_role, target_username),
            )
            if cursor.rowcount == 0:
                conn.close()
                return False
            conn.commit()
            conn.close()
            admin_user = self._get_user_by_username(admin_username)
            if admin_user:
                self.log_audit(
                    admin_user["id"],
                    "UPDATE_USER_ROLE",
                    details=f"将用户 [{target_username}] 角色改为 {new_role}",
                    ip_address="",
                )
            log.info(
                f"管理员 {admin_username} 将用户 {target_username} 角色改为 {new_role}"
            )
            return True
        except Exception as e:
            log.error(f"更新用户角色失败: {e}")
            return False

# %% [markdown]
# ## change_user_display_name(self, target_username: str, new_display_name: str, admin_username: str) -> bool

    # %%
    def change_user_display_name(
        self, target_username: str, new_display_name: str, admin_username: str
    ) -> bool:
        """
        修改用户的显示名称
        """
        if not new_display_name.strip():
            return False
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE users SET display_name = ? WHERE username = ?",
                (new_display_name.strip(), target_username),
            )
            if cursor.rowcount == 0:
                conn.close()
                return False
            conn.commit()
            conn.close()
            admin_user = self._get_user_by_username(admin_username)
            if admin_user:
                self.log_audit(
                    admin_user["id"],
                    "UPDATE_DISPLAY_NAME",
                    details=f"将用户 [{target_username}] 显示名改为 {new_display_name}",
                    ip_address="",
                )
            log.info(f"管理员 {admin_username} 修改了用户 {target_username} 的显示名")
            return True
        except Exception as e:
            log.error(f"更新用户显示名失败: {e}")
            return False

# %% [markdown]
# ## _get_user_by_username(self, username: str) -> Optional[Dict]

    # %%
    # 辅助方法：根据用户名获取用户ID等信息（内部使用）
    def _get_user_by_username(self, username: str) -> Optional[Dict]:
        """根据用户名获取用户信息（简化版，用于审计）"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, username, display_name, role, allowed_notebooks FROM users WHERE username = ?",
            (username,),
        )
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

# %% [markdown]
# ## create_user(self, username: str, password: str, display_name: str, role: str = "team_member", allowed_notebooks: list = None,) -> bool

    # %%
    def create_user(
        self,
        username: str,
        password: str,
        display_name: str,
        role: str = "team_member",
        allowed_notebooks: list = None,
    ) -> bool:
        """
        创建新用户（支持笔记本白名单）

        Args:
            username: 用户名
            password: 密码
            display_name: 显示名称
            role: 角色（admin/team_leader/team_member）
            allowed_notebooks: 允许访问的笔记本列表
        """
        if allowed_notebooks is None:
            allowed_notebooks = []

        # 验证角色
        valid_roles = ["admin", "team_leader", "team_member"]
        if role not in valid_roles:
            log.error(f"无效的角色: {role}")
            return False

        # 将笔记本列表转为JSON字符串
        notebooks_json = json.dumps(allowed_notebooks, ensure_ascii=False)

        password_hash = self._hash_password(password)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO users (username, password_hash, display_name, role, allowed_notebooks) VALUES (?, ?, ?, ?, ?)",
                    (username, password_hash, display_name, role, notebooks_json),
                )
                conn.commit()

                log.info(
                    f"创建用户成功: {username} (角色: {role}, 授权笔记本: {len(allowed_notebooks)}个)"
                )
                return True
        except sqlite3.IntegrityError:
            log.error(f"用户名已存在: {username}")
            return False

# %% [markdown]
# ## update_user_permissions(self, username: str, role: str = None, allowed_notebooks: list = None) -> bool

    # %%
    def update_user_permissions(
        self, username: str, role: str = None, allowed_notebooks: list = None
    ) -> bool:
        """
        更新用户权限配置

        Args:
            username: 用户名
            role: 新角色（可选）
            allowed_notebooks: 新笔记本白名单（可选）
        """
        updates = []
        params = []

        if role is not None:
            updates.append("role = ?")
            params.append(role)

        if allowed_notebooks is not None:
            updates.append("allowed_notebooks = ?")
            params.append(json.dumps(allowed_notebooks, ensure_ascii=False))

        if not updates:
            return True  # 无更新

        params.append(username)

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                update_sql = f"UPDATE users SET {', '.join(updates)} WHERE username = ?"
                cursor.execute(update_sql, params)
                conn.commit()

                log.info(
                    f"更新用户权限: {username} - 角色: {role}, 笔记本: {allowed_notebooks}"
                )
                return True
        except Exception as e:
            log.error(f"更新用户权限失败: {e}")
            return False

# %% [markdown]
# ## get_user_with_notebooks(self, username: str) -> Optional[Dict]

    # %%
    def get_user_with_notebooks(self, username: str) -> Optional[Dict]:
        """
        获取用户信息（包含解析后的笔记本列表）
        """
        user = self._get_user_by_username(username)
        if not user:
            return None

        # 解析笔记本白名单
        notebooks_json = user.get("allowed_notebooks", "[]")
        try:
            allowed_notebooks = json.loads(notebooks_json)
        except:
            allowed_notebooks = []

        user["allowed_notebooks"] = allowed_notebooks
        user["allowed_notebooks_str"] = (
            ", ".join(allowed_notebooks) if allowed_notebooks else "无"
        )

        return user

# %% [markdown]
# ## create_chat_session(self, user_id: int, name: str = "新对话") -> str

    # %%
    def create_chat_session(self, user_id: int, name: str = "新对话") -> str:
        """创建新的问答会话，返回 session_id"""
        session_id = f"chat_{user_id}_{secrets.token_urlsafe(16)}"
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_sessions (user_id, session_id, name) VALUES (?, ?, ?)",
            (user_id, session_id, name),
        )
        conn.commit()
        conn.close()
        return session_id

# %% [markdown]
# ## get_user_chat_sessions(self, user_id: int) -> List[Dict]

    # %%
    def get_user_chat_sessions(self, user_id: int) -> List[Dict]:
        """获取用户所有问答会话，按更新时间倒序，并附带消息数量"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT cs.id, cs.session_id, cs.name, cs.created_at, cs.updated_at,
                   (SELECT COUNT(*) FROM qa_history WHERE session_id = cs.session_id) as message_count
            FROM chat_sessions cs
            WHERE cs.user_id = ?
            ORDER BY cs.updated_at DESC
        """,
            (user_id,),
        )
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

# %% [markdown]
# ## rename_chat_session(self, session_id: str, new_name: str) -> bool

    # %%
    def rename_chat_session(self, session_id: str, new_name: str) -> bool:
        """重命名会话"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE chat_sessions SET name = ?, updated_at = ? WHERE session_id = ?",
            (new_name, datetime.now(), session_id),
        )
        affected = cursor.rowcount
        conn.commit()
        conn.close()
        return affected > 0

# %% [markdown]
# ## delete_chat_session(self, session_id: str) -> bool

    # %%
    def delete_chat_session(self, session_id: str) -> bool:
        """删除会话（同时级联删除 qa_history 中的相关记录）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # 先删问答历史（外键 ON DELETE CASCADE 已设置）
        cursor.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
        affected = cursor.rowcount
        conn.commit()
        conn.close()
        return affected > 0

# %% [markdown]
# ## set_active_chat_session(self, user_id: int, session_id: str)

    # %%
    def set_active_chat_session(self, user_id: int, session_id: str):
        """设置当前活动会话（将用户其他会话的 is_active 置0，本会话置1）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE chat_sessions SET is_active = 0 WHERE user_id = ?", (user_id,)
        )
        cursor.execute(
            "UPDATE chat_sessions SET is_active = 1, updated_at = ? WHERE session_id = ?",
            (datetime.now(), session_id),
        )
        conn.commit()
        conn.close()

# %% [markdown]
# ## get_active_chat_session(self, user_id: int) -> Optional[str]

    # %%
    def get_active_chat_session(self, user_id: int) -> Optional[str]:
        """获取当前用户的最新活动会话，没有则自动迁移旧数据或创建默认会话"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # 先查活动会话
        cursor.execute(
            "SELECT session_id FROM chat_sessions WHERE user_id = ? AND is_active = 1",
            (user_id,),
        )
        row = cursor.fetchone()
        if row:
            conn.close()
            return row["session_id"]

        # 查最近更新的会话
        cursor.execute(
            "SELECT session_id FROM chat_sessions WHERE user_id = ? ORDER BY updated_at DESC LIMIT 1",
            (user_id,),
        )
        row = cursor.fetchone()
        if row:
            conn.close()
            return row["session_id"]

        conn.close()

        # ===== 全新：尝试迁移旧数据 =====
        # 获取用户的用户名
        qa_conn = sqlite3.connect(self.db_path)  # 需要将 DB_PATH 导入
        qa_cursor = qa_conn.cursor()
        qa_cursor.execute("SELECT username FROM users WHERE id = ?", (user_id,))
        username = qa_cursor.fetchone()[0]
        if username:
            old_session_id = f"web_{username}"
            # 检查 qa_history 表中是否有旧数据
            qa_cursor.execute(
                "SELECT COUNT(*) FROM qa_history WHERE session_id = ?",
                (old_session_id,),
            )
            count = qa_cursor.fetchone()[0]

            if count > 0:
                # 为旧数据创建对应的 chat_sessions 记录
                self._create_chat_session_with_id(user_id, old_session_id, "默认对话")
                self.set_active_chat_session(user_id, old_session_id)
                return old_session_id

        qa_conn.close()
        # 完全没有数据，创建全新默认会话
        return self.create_chat_session(user_id, "默认对话")

# %% [markdown]
# ## _create_chat_session_with_id(self, user_id: int, session_id: str, name: str)

    # %%
    def _create_chat_session_with_id(self, user_id: int, session_id: str, name: str):
        """使用指定的 session_id 创建会话（用于迁移旧数据）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO chat_sessions (user_id, session_id, name) VALUES (?, ?, ?)",
            (user_id, session_id, name),
        )
        conn.commit()
        conn.close()

# %% [markdown]
# ## delete_user(self, target_username: str, admin_username: str) -> bool

    # %%
    def delete_user(self, target_username: str, admin_username: str) -> bool:
        """
        删除指定用户（管理员操作）。
        1. 删除该用户的所有会话（chat_sessions）及其问答历史（通过外键级联处理，但为保险手动删除）
        2. 删除该用户的登录会话（sessions）
        3. 删除用户本身
        4. 记录审计日志
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 先获取用户ID，用于后续删除
            cursor.execute("SELECT id FROM users WHERE username = ?", (target_username,))
            row = cursor.fetchone()
            if not row:
                conn.close()
                log.warning(f"尝试删除不存在的用户: {target_username}")
                return False
            
            user_id = row
            
            # 1. 删除该用户的所有问答会话（chat_sessions 会级联删除 qa_history，但显式删除更可靠）
            cursor.execute("DELETE FROM qa_history WHERE session_id IN (SELECT session_id FROM chat_sessions WHERE user_id = ?)", (user_id,))
            cursor.execute("DELETE FROM chat_sessions WHERE user_id = ?", (user_id,))
            
            # 2. 删除登录会话
            cursor.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            
            # 3. 删除用户
            cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
            
            conn.commit()
            conn.close()
            
            # 4. 记录审计日志
            admin_user = self._get_user_by_username(admin_username)
            if admin_user:
                self.log_audit(
                    admin_user["id"],
                    "DELETE_USER",
                    details=f"管理员删除了用户 [{target_username}] (ID: {user_id})",
                    ip_address="",
                )
            log.info(f"管理员 {admin_username} 删除了用户 {target_username}")
            return True
        except Exception as e:
            log.error(f"删除用户失败: {e}")
            return False

# %% [markdown]
# ## 审计相关功能函数

    # %%
    def get_audit_logs(
        self,
        page: int = 1,
        per_page: int = 20,
        username: Optional[str] = None,
        action: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict:
        """
        分页查询审计日志（管理员功能）
        Args:
            page: 页码（从1开始）
            per_page: 每页条数
            username: 按用户名筛选（模糊匹配）
            action: 按操作类型筛选
            start_date: 起始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
        Returns:
            {"total": 总数, "logs": 日志列表, "page": 当前页码, "per_page": 每页条数}
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
    
        # 构建查询条件
        conditions = []
        params = []
    
        if username:
            # 通过子查询关联用户名（audit_log.user_id -> users.username）
            conditions.append("a.user_id IN (SELECT id FROM users WHERE username LIKE ?)")
            params.append(f"%{username}%")
    
        if action:
            conditions.append("a.action = ?")
            params.append(action)
    
        if start_date:
            conditions.append("a.timestamp >= ?")
            params.append(start_date)
    
        if end_date:
            # 包含end_date当天
            conditions.append("a.timestamp <= ?")
            params.append(f"{end_date} 23:59:59")
    
        where_clause = " AND ".join(conditions) if conditions else "1=1"
    
        # 查询总数
        cursor.execute(
            f"""
            SELECT COUNT(*) as total
            FROM audit_log a
            WHERE {where_clause}
            """,
            params,
        )
        total = cursor.fetchone()["total"]
    
        # 查询当前页数据
        offset = (page - 1) * per_page
        cursor.execute(
            f"""
            SELECT
                a.id,
                a.user_id,
                COALESCE(u.username, '(已删除用户)') as username,
                COALESCE(u.display_name, '未知') as display_name,
                a.action,
                a.details,
                a.ip_address,
                a.timestamp
            FROM audit_log a
            LEFT JOIN users u ON a.user_id = u.id
            WHERE {where_clause}
            ORDER BY a.timestamp DESC
            LIMIT ? OFFSET ?
            """,
            params + [per_page, offset],
        )
        rows = cursor.fetchall()
        conn.close()
    
        return {
            "total": total,
            "logs": [dict(row) for row in rows],
            "page": page,
            "per_page": per_page,
            "total_pages": max(1, (total + per_page - 1) // per_page),
        }
    
    
    def clear_audit_logs(self, before_days: int = 90) -> int:
        """
        清理指定天数之前的审计日志
        Args:
            before_days: 保留最近N天的日志，之前的将被删除
        Returns:
            删除的记录数
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cutoff = datetime.now() - timedelta(days=before_days)
        cursor.execute(
            "DELETE FROM audit_log WHERE timestamp < ?",
            (cutoff.strftime("%Y-%m-%d %H:%M:%S"),),
        )
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        log.info(f"清理了 {deleted} 条 {before_days} 天前的审计日志")
        return deleted
    
    
    def get_audit_actions(self) -> List[str]:
        """获取所有不同的操作类型（用于筛选下拉框）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT action FROM audit_log ORDER BY action")
        rows = cursor.fetchall()
        conn.close()
        return [row[0] for row in rows]


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
