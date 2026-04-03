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
# # 缓存管理器

# %% [markdown]
# ## 导入库

# %%
# cache_manager.py
import hashlib
import json
import logging
import sqlite3
from datetime import datetime
from typing import Optional

# %%
try:
    from embedding_generator import embeddinggenerator
    from func.configpr import (
        findvaluebykeyinsection,
        getcfpoptionvalue,
        setcfpoptionvalue,
    )
    from func.first import dirmainpath, getdirmain
    from func.jpfuncs import (
        getinivaluefromcloud,
    )
    from func.logme import log
    from func.sysfunc import execcmd, not_IPython
    from func.wrapfuncs import timethis
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    log.error(f"导入项目模块失败: {e}")


# %% [markdown]
# ## SQLiteCacheManager类

# %%
class SQLiteCacheManager:
    """基于SQLite的高性能缓存管理器，用于DeepSeek增强结果。"""


# %% [markdown]
# ### __init__(self, db_path: str = "data/deepseek_cache.db")

    # %%
    def __init__(self, db_path: str = "data/deepseek_cache.db"):
        self.db_path = db_path
        self._init_db()

# %% [markdown]
# ### _init_db(self)

    # %%
    def _init_db(self):
        """初始化数据库和表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_cache (
                cache_key TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                task TEXT NOT NULL,
                result TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                last_accessed DATETIME NOT NULL
            )
        """)
        # 创建索引以加速查询和清理
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON processing_cache (timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hash_task ON processing_cache (content_hash, task)")
        conn.commit()
        conn.close()

# %% [markdown]
# ### get(self, content_hash: str, task: str) -> Optional[str]

    # %%
    def get(self, content_hash: str, task: str) -> Optional[str]:
        """获取缓存结果，并更新最后访问时间"""
        cache_key = f"{content_hash}_{task}"
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT result, timestamp FROM processing_cache 
            WHERE cache_key = ? AND (julianday('now') - julianday(timestamp)) < 90
        """, (cache_key,))
        row = cursor.fetchone()

        if row:
            result, _ = row
            # 更新最后访问时间
            cursor.execute("""
                UPDATE processing_cache SET last_accessed = datetime('now') WHERE cache_key = ?
            """, (cache_key,))
            conn.commit()
            conn.close()
            log.debug(f"缓存命中: {cache_key}")
            return result
        else:
            conn.close()
            return None

# %% [markdown]
# ### set(self, content_hash: str, task: str, result: str)

    # %%
    def set(self, content_hash: str, task: str, result: str):
        """设置缓存结果"""
        cache_key = f"{content_hash}_{task}"
        now = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 使用 INSERT OR REPLACE 实现 upsert
        cursor.execute("""
            INSERT OR REPLACE INTO processing_cache 
            (cache_key, content_hash, task, result, timestamp, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (cache_key, content_hash, task, result, now, now))

        # 定期清理：如果总记录数超过20000，删除最旧且最少访问的1000条
        cursor.execute("SELECT COUNT(*) FROM processing_cache")
        count = cursor.fetchone()
        if not (cache_limit := getinivaluefromcloud("joplinai", "cache_limit")):
            cache_limit = 50000
        if count > cache_limit:
            cursor.execute("""
                DELETE FROM processing_cache 
                WHERE cache_key IN (
                    SELECT cache_key FROM processing_cache 
                    ORDER BY last_accessed ASC, timestamp ASC 
                    LIMIT 1000
                )
            """)
            log.info(f"执行缓存清理，删除了1000条旧记录。")

        conn.commit()
        conn.close()
        log.debug(f"缓存已保存: {cache_key}")

# %% [markdown]
# ### cleanup_old_entries(self, max_age_days: int = 90)

    # %%
    def cleanup_old_entries(self, max_age_days: int = 90):
        """清理超过指定天数的旧缓存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM processing_cache 
            WHERE (julianday('now') - julianday(timestamp)) > ?
        """, (max_age_days,))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        if deleted:
            log.info(f"清理了 {deleted} 条超过 {max_age_days} 天的缓存记录。")
