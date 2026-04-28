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
            "SELECT id, username, display_name, role, is_active, created_at, last_login FROM users ORDER BY id"
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

    # 在 UserManager 类中添加以下方法

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
