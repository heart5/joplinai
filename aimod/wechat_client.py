"""微信聊天记录数据访问客户端，支持本地 SQLite 直连和远程 HTTP 两种模式。"""

import logging
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests

log = logging.getLogger("wechat_client")

__all__ = ["WeChatClient", "WeChatError"]

_TABLE_NAME_RE = re.compile(r"^wc_\w+$")
_ACCOUNT_RE = re.compile(r"^[\w一-鿿]+$")


class WeChatError(Exception):
    """微信数据访问错误"""


class WeChatClient:
    """微信数据访问客户端。

    支持两种模式：
    - local：直连本地 SQLite 数据库文件
    - remote：通过 HTTP API 查询远程 HCX 服务

    模式切换：
        WeChatClient(mode="local", db_path="/path/to/wcitemsall_merged.db")
        WeChatClient(mode="remote", api_url="https://ollama.qingxd.com/wechat")
        WeChatClient(mode="auto", ...)  # 先试 local，不通退 remote
    """

    def __init__(
        self,
        mode: str = "auto",
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        db_path: Optional[str] = None,
    ):
        self.mode = mode
        self.api_url = api_url.rstrip("/") if api_url else None
        self.api_key = api_key
        self.db_path = Path(db_path) if db_path else None
        self._conn: Optional[sqlite3.Connection] = None

        if mode == "local" or (mode == "auto" and self.db_path and self.db_path.exists()):
            self.mode = "local"
        elif mode == "remote" or (mode == "auto" and self.api_url):
            self.mode = "remote"
        else:
            # 自动选择：本地优先
            if self.db_path and self.db_path.exists():
                self.mode = "local"
            elif self.api_url:
                self.mode = "remote"
            else:
                raise WeChatError("无法确定数据访问模式：请提供 db_path 或 api_url")

    # ── 内部工具 ──

    @staticmethod
    def _validate_account(account: str):
        """校验账号名：只允许中文、字母、数字、下划线。"""
        if not _ACCOUNT_RE.match(account):
            raise WeChatError(f"非法的账号名: {account!r}")

    def _table_name(self, account: str) -> str:
        self._validate_account(account)
        return f"wc_{account}"

    @staticmethod
    def _normalize_time(val) -> str:
        """统一 time 为 YYYY-MM-DD HH:MM:SS 格式。"""
        if val is None or val == "":
            return ""
        if isinstance(val, (int, float)):
            try:
                return datetime.fromtimestamp(val).strftime("%Y-%m-%d %H:%M:%S")
            except (OSError, ValueError):
                return str(val)
        s = str(val).strip()
        if re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", s):
            return s
        if re.match(r"^\d{4}-\d{2}-\d{2}$", s):
            return s + " 00:00:00"
        try:
            return datetime.fromtimestamp(int(float(s))).strftime("%Y-%m-%d %H:%M:%S")
        except (OSError, ValueError, TypeError):
            return s

    # ── 本地模式 ──

    def _local_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            if not self.db_path or not self.db_path.exists():
                raise WeChatError(f"数据库文件不存在: {self.db_path}")
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            log.info(f"本地连接已打开: {self.db_path}")
        return self._conn

    def _local_query(self, account: str, *, date_from=None, date_to=None,
                     sender=None, keyword=None, type_filter=None,
                     after_id=None, limit: int = 1000) -> List[Dict]:
        conn = self._local_conn()
        table = self._table_name(account)

        conditions = []
        params = []

        if after_id is not None:
            conditions.append("id > ?")
            params.append(int(after_id))

        if date_from:
            conditions.append("time >= ?")
            params.append(date_from)

        if date_to:
            conditions.append("time <= ?")
            params.append(date_to)

        if sender:
            conditions.append("sender LIKE ?")
            params.append(f"%{sender}%")

        if type_filter:
            if isinstance(type_filter, str):
                type_filter = [type_filter]
            placeholders = ",".join("?" for _ in type_filter)
            conditions.append(f"type IN ({placeholders})")
            params.extend(type_filter)

        if keyword:
            conditions.append("content LIKE ?")
            params.append(f"%{keyword}%")

        where = " AND ".join(conditions) if conditions else "1"
        sql = f"SELECT id, time, send, sender, type, content, source FROM [{table}] WHERE {where} ORDER BY id ASC LIMIT ?"
        params.append(limit)

        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as e:
            raise WeChatError(f"查询失败: {e}") from e

        results = []
        for r in rows:
            results.append({
                "id": r["id"],
                "time": self._normalize_time(r["time"]),
                "send": bool(r["send"]),
                "sender": r["sender"],
                "type": r["type"],
                "content": r["content"],
                "source": r["source"],
            })
        return results

    def _local_get_contacts(self, account: str, days: int = 30) -> List[Dict]:
        conn = self._local_conn()
        table = self._table_name(account)
        from datetime import datetime as dt_mod, timedelta
        since = (dt_mod.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
        rows = conn.execute(
            f"""SELECT sender, COUNT(*) as msg_count,
                       SUM(CASE WHEN send=1 THEN 1 ELSE 0 END) as sent_count,
                       MAX(time) as last_time
                FROM [{table}]
                WHERE time >= ?
                GROUP BY sender
                ORDER BY msg_count DESC
                LIMIT 200""",
            (since,),
        ).fetchall()
        return [dict(r) for r in rows]

    def _local_get_stats(self, account: str, date_from: str, date_to: str) -> Dict:
        conn = self._local_conn()
        table = self._table_name(account)

        total = conn.execute(
            f"SELECT COUNT(*) FROM [{table}] WHERE time >= ? AND time <= ?",
            (date_from, date_to)
        ).fetchone()[0]

        type_dist = conn.execute(
            f"SELECT type, COUNT(*) as cnt FROM [{table}] WHERE time >= ? AND time <= ? GROUP BY type ORDER BY cnt DESC",
            (date_from, date_to)
        ).fetchall()
        type_dist = [{"type": r[0], "count": r[1]} for r in type_dist]

        daily = conn.execute(
            f"SELECT substr(time,1,10) as d, COUNT(*) as cnt FROM [{table}] WHERE time >= ? AND time <= ? GROUP BY d ORDER BY d",
            (date_from, date_to)
        ).fetchall()
        daily = [{"date": r[0], "count": r[1]} for r in daily]

        return {"total": total, "type_distribution": type_dist, "daily": daily}

    def _local_get_conversation(self, account: str, sender: str,
                                date_from=None, date_to=None, limit=100) -> List[Dict]:
        conn = self._local_conn()
        table = self._table_name(account)

        conditions = ["sender LIKE ?"]
        params = [f"%{sender}%"]

        if date_from:
            conditions.append("time >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("time <= ?")
            params.append(date_to)

        where = " AND ".join(conditions)
        rows = conn.execute(
            f"SELECT id, time, send, sender, type, content, source FROM [{table}] WHERE {where} ORDER BY id ASC LIMIT ?",
            params + [limit]
        ).fetchall()

        return [{
            "id": r["id"],
            "time": self._normalize_time(r["time"]),
            "send": bool(r["send"]),
            "sender": r["sender"],
            "type": r["type"],
            "content": r["content"],
            "source": r["source"],
        } for r in rows]

    def _local_get_finance_records(self, account: str, date_from: str, date_to: str) -> List[Dict]:
        """获取财务相关消息。"""
        conn = self._local_conn()
        table = self._table_name(account)

        # sender 含支付/银行关键词
        rows = conn.execute(
            f"""SELECT id, time, send, sender, type, content, source FROM [{table}]
                WHERE time >= ? AND time <= ?
                  AND (sender LIKE '%支付%' OR sender LIKE '%银行%' OR sender LIKE '%信用卡%'
                       OR content LIKE '%￥%' OR content LIKE '%消费%'
                       OR content LIKE '%转账%' OR content LIKE '%红包%')
                ORDER BY id ASC""",
            (date_from, date_to)
        ).fetchall()

        return [{
            "id": r["id"],
            "time": self._normalize_time(r["time"]),
            "send": bool(r["send"]),
            "sender": r["sender"],
            "type": r["type"],
            "content": r["content"],
            "source": r["source"],
        } for r in rows]

    # ── 远程模式 ──

    def _remote_get(self, path: str, params: Optional[Dict] = None) -> Dict:
        url = f"{self.api_url}{path}"
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            raise WeChatError(f"远程请求失败: {e}") from e

    # ── 统一接口 ──

    def query(self, account: str, *, date_from=None, date_to=None,
              sender=None, keyword=None, type_filter=None,
              after_id=None, limit=1000) -> List[Dict]:
        if self.mode == "local":
            return self._local_query(account, date_from=date_from, date_to=date_to,
                                     sender=sender, keyword=keyword, type_filter=type_filter,
                                     after_id=after_id, limit=limit)
        result = self._remote_get("/wechat/query", {
            "account": account, "date_from": date_from, "date_to": date_to,
            "sender": sender, "keyword": keyword, "type": type_filter,
            "after_id": after_id, "limit": limit,
        })
        return result.get("records", [])

    def get_contacts(self, account: str, days: int = 30) -> List[Dict]:
        if self.mode == "local":
            return self._local_get_contacts(account, days)
        result = self._remote_get("/wechat/contacts", {"account": account, "days": days})
        return result.get("contacts", [])

    def get_stats(self, account: str, date_from: str, date_to: str) -> Dict:
        if self.mode == "local":
            return self._local_get_stats(account, date_from, date_to)
        return self._remote_get("/wechat/stats", {
            "account": account, "date_from": date_from, "date_to": date_to,
        })

    def get_conversation(self, account: str, sender: str,
                         date_from=None, date_to=None, limit=100) -> List[Dict]:
        if self.mode == "local":
            return self._local_get_conversation(account, sender, date_from, date_to, limit)
        result = self._remote_get("/wechat/conversation", {
            "account": account, "sender": sender,
            "date_from": date_from, "date_to": date_to, "limit": limit,
        })
        return result.get("records", [])

    def get_finance_records(self, account: str, date_from: str, date_to: str) -> List[Dict]:
        if self.mode == "local":
            return self._local_get_finance_records(account, date_from, date_to)
        result = self._remote_get("/wechat/finance", {
            "account": account, "date_from": date_from, "date_to": date_to,
        })
        return result.get("records", [])

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
