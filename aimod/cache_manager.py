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
# # 缓存管理器

# %%
# cache_manager.py

# %% [markdown]
# ## 导入库

# %% [markdown]
# ### 系统库

# %%
import argparse
import hashlib
import json
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# %% [markdown]
# ### func库

# %%
import pathmagic

with pathmagic.context():
    try:
        from func.first import getdirmain
        from func.jpfuncs import (
            createnote,
            getinivaluefromcloud,
            searchnotes,
            updatenote_body,
        )
        from func.logme import log
        from func.sysfunc import execcmd, not_IPython
        from func.wrapfuncs import timethis
    except ImportError as e:
        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger(__name__)
        log.error(f"导入项目模块失败: {e}")


# %% [markdown]
# # CacheResult类


# %%
# 新增：定义一个数据类来封装更丰富的返回信息
@dataclass
class CacheResult:
    """缓存查询结果封装"""

    content: Optional[str]  # 缓存的内容本身，如果为None表示未命中
    requires_validation: bool  # 本次命中后，是否达到了验证阈值，建议调用者进行验证
    cache_key: str
    current_hit_count: int  # 验证周期内的命中次数（触发验证后会被重置）
    total_hits: int


# %% [markdown]
# # SQLiteCacheManager类


# %%
class SQLiteCacheManager:
    """基于SQLite的高性能缓存管理器，用于DeepSeek增强结果。"""

# %% [markdown]
# ## 验证阈值

    # %%
    if not (
        VALIDATION_THRESHOLD := getinivaluefromcloud("joplinai", "validation_threshold")
    ):
        VALIDATION_THRESHOLD = 5000  # 验证阈值，可从配置读取

# %% [markdown]
# ## __init__(self, db_path: str = "data/.deepseek_cache/deepseek_cache.db")

    # %%
    def __init__(self, db_path: str = "data/.deepseek_cache/deepseek_cache.db"):
        self.db_path = db_path
        self._init_db()

# %% [markdown]
# ## _init_db(self)

    # %%
    def _init_db(self):
        """初始化数据库和表结构（增强版）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # 尝试创建新表，如果表已存在，后续可能需要用 ALTER TABLE 添加缺失的列（生产环境建议用迁移脚本）
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_cache (
                cache_key TEXT PRIMARY KEY,
                content_hash TEXT NOT NULL,
                task TEXT NOT NULL,
                result TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                last_accessed DATETIME NOT NULL,
                last_validated_at DATETIME,
                hit_count INTEGER DEFAULT 0,
                total_hits INTEGER DEFAULT 0,
                validation_result TEXT
            )
        """)
        # 创建索引
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_hash_task ON processing_cache (content_hash, task)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_last_accessed ON processing_cache (last_accessed)"
        )
        conn.commit()
        conn.close()

# %% [markdown]
# ## get(self, content_hash: str, task: str) -> Optional[str]

    # %%
    def get(self, content_hash: str, task: str) -> CacheResult:
        """
        获取缓存结果。
        核心职责：1. 返回缓存内容 2. 更新访问计数和时间 3. 判断并标记是否需要验证。
        绝不包含任何API调用逻辑。
        """
        cache_key = f"{content_hash}_{task}"
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 查询缓存内容和当前计数
        cursor.execute(
            """
            SELECT result, hit_count, total_hits FROM processing_cache
            WHERE cache_key = ? AND (julianday('now') - julianday(created_at)) < 90
        """,
            (cache_key,),
        )
        row = cursor.fetchone()

        if not row:
            conn.close()
            # 未命中
            return CacheResult(
                content=None,
                requires_validation=False,
                cache_key=cache_key,
                current_hit_count=0,
                total_hits=0,
            )

        cached_result, current_hit_count, total_hits = row
        new_hit_count = current_hit_count + 1
        new_total_hits = total_hits + 1
        now_iso = datetime.now().isoformat()

        # 判断本次命中后是否达到验证阈值
        should_validate = new_hit_count >= self.VALIDATION_THRESHOLD

        # 更新数据库：计数、最后访问时间
        update_sql = """
            UPDATE processing_cache
            SET hit_count = ?, total_hits = ?, last_accessed = ?
        """
        update_params = [new_hit_count, new_total_hits, now_iso]

        if should_validate:
            # 达到阈值，标记为需要验证，并重置周期计数
            update_sql += (
                ", hit_count = 0, last_validated_at = ?, validation_result = 'pending'"
            )
            update_params.extend([now_iso])
            # 注意：这里将 hit_count 重置为0，开始新的计数周期

        update_sql += " WHERE cache_key = ?"
        update_params.append(cache_key)

        cursor.execute(update_sql, update_params)
        conn.commit()
        conn.close()

        # log.debug(
        #     f"缓存查询: {cache_key[:12]}... (周期命中={new_hit_count}, 总计={new_total_hits}, 需验证={should_validate})"
        # )

        # 返回封装结果，告诉调用者缓存内容以及“是否需要验证”
        return CacheResult(
            content=cached_result,
            requires_validation=should_validate,
            cache_key=cache_key,
            current_hit_count=new_hit_count
            if not should_validate
            else 0,  # 返回重置前的计数或重置后的0
            total_hits=new_total_hits,
        )

# %% [markdown]
# ## update_on_validation(self, cache_key: str, new_result: Optional[str], validation_successful: bool)

    # %%
    def update_on_validation(
        self, cache_key: str, new_result: Optional[str], validation_successful: bool
    ):
        """
        由外部调用者在完成验证后调用，用于更新缓存。
        :param cache_key: 缓存键
        :param new_result: 验证得到的新结果。如果为 None 且 validation_successful=True，表示内容未变。
        :param validation_successful: 验证流程是否成功执行（API是否成功调用）
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now_iso = datetime.now().isoformat()

        if not validation_successful:
            # 验证失败（如网络错误），只更新验证时间和状态
            cursor.execute(
                """
                UPDATE processing_cache
                SET last_validated_at = ?, validation_result = 'failed'
                WHERE cache_key = ?
            """,
                (now_iso, cache_key),
            )
            log.warning(f"验证失败记录更新: {cache_key[:12]}...")
        elif new_result is None:
            # 验证成功，且内容未变化
            cursor.execute(
                """
                UPDATE processing_cache
                SET last_validated_at = ?, validation_result = 'valid'
                WHERE cache_key = ?
            """,
                (now_iso, cache_key),
            )
            log.info(f"验证完成，内容未变: {cache_key[:12]}...")
        else:
            # 验证成功，且内容已更新
            cursor.execute(
                """
                UPDATE processing_cache
                SET result = ?, created_at = ?, last_validated_at = ?, validation_result = 'updated'
                WHERE cache_key = ?
            """,
                (new_result, now_iso, now_iso, cache_key),
            )
            log.info(f"验证完成，缓存已更新: {cache_key[:12]}...")

        conn.commit()
        conn.close()

# %% [markdown]
# ## set(self, content_hash: str, task: str, result: str)

    # %%
    def set(self, content_hash: str, task: str, result: str):
        """设置新的缓存条目（首次保存或强制覆盖）"""
        cache_key = f"{content_hash}_{task}"
        now_iso = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO processing_cache
            (cache_key, content_hash, task, result, created_at, last_accessed,
             last_validated_at, hit_count, total_hits, validation_result)
            VALUES (?, ?, ?, ?, ?, ?, NULL, 0, 0, NULL)
        """,
            (cache_key, content_hash, task, result, now_iso, now_iso),
        )

        # 定期清理：如果总记录数超过20000，删除最旧且最少访问的1000条
        cursor.execute("SELECT COUNT(*) FROM processing_cache")
        count_result = cursor.fetchone()
        # 从元组中提取第一个元素（即计数值），如果结果为None则默认为0
        count = count_result[0] if count_result else 0
        # count = cursor.fetchone()
        if not (cache_limit := getinivaluefromcloud("joplinai", "cache_limit")):
            cache_limit = 50000
        if count > cache_limit:
            cursor.execute("""
                DELETE FROM processing_cache 
                WHERE cache_key IN (
                    SELECT cache_key FROM processing_cache 
                    ORDER BY last_accessed ASC
                    LIMIT 1000
                )
            """)
            log.info(f"执行缓存清理，删除了1000条旧记录。")

        conn.commit()
        conn.close()
        log.debug(f"缓存已设置: {cache_key}")


# %% [markdown]
# ## get_stats(self, cache_key: str = None) -> Dict[str, Any]

    # %%
    def get_stats(self, cache_key: str = None) -> Dict[str, Any]:
        """获取缓存统计信息"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        stats = {}
        if cache_key:
            cursor.execute(
                "SELECT * FROM processing_cache WHERE cache_key = ?", (cache_key,)
            )
            row = cursor.fetchone()
            if row:
                stats = dict(row)
        else:
            # 全局统计
            cursor.execute(
                "SELECT COUNT(*) as total, SUM(total_hits) as total_hits, SUM(hit_count) as current_hits FROM processing_cache"
            )
            row = cursor.fetchone()
            stats = dict(row) if row else {}
            cursor.execute(
                "SELECT validation_result, COUNT(*) as count FROM processing_cache WHERE validation_result IS NOT NULL GROUP BY validation_result"
            )
            # 将结果列表中的 Row 也转换为字典
            validation_items = cursor.fetchall()
            stats['validation_breakdown'] = {dict(r)['validation_result']: dict(r)['count'] for r in validation_items} if validation_items else {}
            # stats["validation_breakdown"] = {r: c for r, c in cursor.fetchall()}

        conn.close()
        return stats

