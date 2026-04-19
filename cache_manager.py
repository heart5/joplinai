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
# ## import_from_json(self, json_file_path: str, clear_existing: bool = False) -> Dict[str, int]

    # %%
    def import_from_json(
        self, json_file_path: str, clear_existing: bool = False
    ) -> Dict[str, int]:
        """
        从JSON文件导入缓存数据到SQLite数据库

        Args:
            json_file_path: JSON缓存文件路径
            clear_existing: 是否先清空现有缓存表（默认False，保留现有数据）

        Returns:
            导入统计信息字典：{"total": 总条目数, "success": 成功数, "failed": 失败数}
        """
        import_stats = {"total": 0, "success": 0, "failed": 0}

        # 检查JSON文件是否存在
        if not os.path.exists(json_file_path):
            log.error(f"JSON缓存文件不存在: {json_file_path}")
            return import_stats

        try:
            # 读取JSON文件
            with open(json_file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            import_stats["total"] = len(json_data)

            # 如果需要，先清空现有表
            if clear_existing:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM processing_cache")
                conn.commit()
                conn.close()
                log.info("已清空现有缓存表")

            # 批量导入数据
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for cache_key, cache_data in json_data.items():
                try:
                    # 解析缓存键，提取content_hash和task
                    # JSON格式的key可能是 "d1c2c1ef7537a312149f9bd262c871df_tags"
                    if "_" in cache_key:
                        # 尝试从key中提取哈希和任务类型
                        parts = cache_key.rsplit("_", 1)
                        if len(parts) == 2:
                            content_hash, task = parts
                        else:
                            # 如果解析失败，使用数据中的task字段
                            content_hash = cache_key
                            task = cache_data.get("task", "unknown")
                    else:
                        # 如果没有下划线分隔，使用默认解析
                        content_hash = cache_key
                        task = cache_data.get("task", "unknown")

                    # 获取缓存结果
                    result = cache_data.get("result", "")
                    timestamp_str = cache_data.get("timestamp", "")

                    # 转换时间戳格式
                    if timestamp_str:
                        try:
                            # 尝试解析ISO格式时间戳
                            timestamp_dt = datetime.fromisoformat(
                                timestamp_str.replace("Z", "+00:00")
                            )
                            created_at = timestamp_dt.isoformat()
                        except (ValueError, AttributeError):
                            # 如果解析失败，使用当前时间
                            created_at = datetime.now().isoformat()
                            log.warning(f"时间戳格式错误，使用当前时间: {cache_key}")
                    else:
                        created_at = datetime.now().isoformat()

                    # 构建完整的缓存记录
                    # 注意：JSON数据中没有hit_count等新字段，我们设置默认值
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO processing_cache
                        (cache_key, content_hash, task, result, created_at, last_accessed,
                         last_validated_at, hit_count, total_hits, validation_result)
                        VALUES (?, ?, ?, ?, ?, ?, NULL, 0, 0, NULL)
                    """,
                        (
                            cache_key,
                            content_hash,
                            task,
                            result,
                            created_at,
                            created_at,  # 初始时last_accessed与created_at相同
                        ),
                    )

                    import_stats["success"] += 1

                except Exception as e:
                    import_stats["failed"] += 1
                    log.error(f"导入缓存条目失败 {cache_key}: {e}")

            conn.commit()
            conn.close()

            log.info(
                f"JSON缓存导入完成: 总计{import_stats['total']}条, "
                f"成功{import_stats['success']}条, 失败{import_stats['failed']}条"
            )

            # 可选：导入后重命名或备份旧文件
            old_file = Path(json_file_path)
            backup_file = old_file.with_suffix(".json.backup")
            old_file.rename(backup_file)
            log.info(f"旧缓存文件{old_file}已备份为: {backup_file.name}")
        except json.JSONDecodeError as e:
            log.error(f"JSON文件格式错误: {e}")
            import_stats["failed"] = import_stats["total"]
        except Exception as e:
            log.error(f"导入过程发生错误: {e}")
            import_stats["failed"] = import_stats["total"]

        return import_stats

# %% [markdown]
# ## import_from_json_directory(self, json_dir_path: str, pattern: str = "*.json", clear_existing: bool = False) -> Dict[str, Any]

    # %%
    def import_from_json_directory(
        self, json_dir_path: str, pattern: str = "*.json", clear_existing: bool = False
    ) -> Dict[str, Any]:
        """
        从目录批量导入多个JSON缓存文件

        Args:
            json_dir_path: JSON文件目录路径
            pattern: 文件匹配模式（默认*.json）
            clear_existing: 是否先清空现有缓存表

        Returns:
            批量导入统计信息
        """
        import_stats = {
            "files_processed": 0,
            "total_entries": 0,
            "successful_entries": 0,
            "failed_entries": 0,
            "file_details": [],
        }

        json_dir = Path(json_dir_path)
        if not json_dir.exists() or not json_dir.is_dir():
            log.error(f"JSON目录不存在或不是目录: {json_dir_path}")
            return import_stats

        # 如果需要，先清空现有表（只在第一次导入前）
        if clear_existing:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM processing_cache")
            conn.commit()
            conn.close()
            log.info("已清空现有缓存表")

        # 遍历目录中的JSON文件
        for json_file in json_dir.glob(pattern):
            if json_file.is_file():
                file_stats = self.import_from_json(str(json_file), clear_existing=False)

                import_stats["files_processed"] += 1
                import_stats["total_entries"] += file_stats["total"]
                import_stats["successful_entries"] += file_stats["success"]
                import_stats["failed_entries"] += file_stats["failed"]

                import_stats["file_details"].append(
                    {
                        "file": str(json_file.name),
                        "total": file_stats["total"],
                        "success": file_stats["success"],
                        "failed": file_stats["failed"],
                    }
                )

                log.info(f"已处理文件: {json_file.name}")

        log.info(
            f"批量导入完成: 处理{import_stats['files_processed']}个文件, "
            f"总计{import_stats['total_entries']}条记录"
        )

        return import_stats

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
# ## cleanup_old_entries(self, max_age_days: int = 90)

    # %%
    def cleanup_old_entries(self, max_age_days: int = 90):
        """清理超过指定天数的旧缓存"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            DELETE FROM processing_cache 
            WHERE (julianday('now') - julianday(timestamp)) > ?
        """,
            (max_age_days,),
        )
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        if deleted:
            log.info(f"清理了 {deleted} 条超过 {max_age_days} 天的缓存记录。")

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

# %% [markdown]
# # 缓存统计报告生成器

# %%
# cache_report_generator.py
"""
缓存使用统计报告生成器
自动分析 SQLite 缓存数据库，生成使用报告并保存到 Joplin 笔记
"""

# %% [markdown]
# ## 配置

# %%
# 默认缓存数据库路径（与 cache_manager.py 保持一致）
DEFAULT_CACHE_DB = getdirmain() / "data" / ".deepseek_cache" / "deepseek_cache.db"

# 报告笔记配置
REPORT_NOTEBOOK = "ewmobile"  # 报告保存的笔记本
REPORT_NOTE_TITLE_PREFIX = "📊 缓存使用统计报告"  # 报告笔记标题前缀

# 统计时间范围配置
TIME_WINDOWS = {
    "24h": 1,
    "7d": 7,
    "30d": 30,
    "90d": 90,
    "all": None,  # 全部数据
}

# %% [markdown]
# # 核心统计类


# %%
class CacheStatsAnalyzer:
    """缓存统计数据分析器"""

# %% [markdown]
# ## 初始化

    # %%
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DEFAULT_CACHE_DB)
        self.conn = None
        self.cursor = None

    def connect(self):
        """连接数据库"""
        if not Path(self.db_path).exists():
            log.error(f"缓存数据库不存在: {self.db_path}")
            return False

        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # 返回字典格式
            self.cursor = self.conn.cursor()
            log.info(f"已连接缓存数据库: {self.db_path}")
            return True
        except Exception as e:
            log.error(f"连接数据库失败: {e}")
            return False

    def disconnect(self):
        """断开数据库连接"""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

# %% [markdown]
# ## _fetch_scalar(self, query: str, params: tuple = ()) -> Any:

    # %%
    def _fetch_scalar(self, query: str, params: tuple = ()) -> Any:
        """执行查询并返回单个标量值，简化fetchone()和空值处理"""
        self.cursor.execute(query, params)
        row = self.cursor.fetchone()
        if not row:
            return 0
        # 返回第一个字段的值（适用于SELECT COUNT(*), SUM(...)等查询）
        return row[0]

# %% [markdown]
# ## 基础统计

    # %%
    def get_basic_stats(self) -> Dict[str, Any]:
        stats = {}
        
        # 总记录数 - 使用fetchone()[0]或对Row使用键名获取值
        self.cursor.execute("SELECT COUNT(*) as total FROM processing_cache")
        row = self.cursor.fetchone()
        stats["total_entries"] = row["total"] if row else 0  # 关键修改：提取字段值
    
        # 总命中次数
        self.cursor.execute("SELECT SUM(total_hits) as total_hits FROM processing_cache")
        row = self.cursor.fetchone()
        # 注意：如果所有行的total_hits都为NULL，SUM()会返回NULL，需要处理
        stats["total_hits"] = row["total_hits"] if row and row["total_hits"] is not None else 0
    
        # 当前周期命中次数
        self.cursor.execute("SELECT SUM(hit_count) as current_hits FROM processing_cache")
        row = self.cursor.fetchone()
        stats["current_hits"] = row["current_hits"] if row and row["current_hits"] is not None else 0
    
        # 平均命中率（估算）
        if stats["total_entries"] > 0:
            stats["avg_hits_per_entry"] = stats["total_hits"] / stats["total_entries"]
        else:
            stats["avg_hits_per_entry"] = 0
    
        # 任务类型分布（这部分代码原本正确，因为fetchall()返回Row列表，后续dict(row)处理得当）
        self.cursor.execute("""
            SELECT task, COUNT(*) as count,
                    SUM(total_hits) as hits,
                   AVG(total_hits) as avg_hits
            FROM processing_cache 
            GROUP BY task 
            ORDER BY count DESC
        """)
        stats["task_distribution"] = [dict(row) for row in self.cursor.fetchall()]
    
        return stats

    # %%
    def get_time_analysis(self) -> Dict[str, Any]:
        """时间维度分析"""
        analysis = {}

        # 创建时间分布（按天）
        self.cursor.execute("""
            SELECT DATE(created_at) as date, COUNT(*) as count
            FROM processing_cache
            GROUP BY DATE(created_at)
            ORDER BY date DESC
            LIMIT 30
        """)
        analysis["creation_by_day"] = [dict(row) for row in self.cursor.fetchall()]

        # 最后访问时间分布
        self.cursor.execute("""
            SELECT DATE(last_accessed) as date, COUNT(*) as count
            FROM processing_cache
            GROUP BY DATE(last_accessed)
            ORDER BY date DESC
            LIMIT 30
        """)
        analysis["access_by_day"] = [dict(row) for row in self.cursor.fetchall()]

        # 缓存年龄分布
        self.cursor.execute("""
            SELECT 
                CASE 
                    WHEN julianday('now') - julianday(created_at) <= 1 THEN '1天内'
                    WHEN julianday('now') - julianday(created_at) <= 7 THEN '7天内'
                    WHEN julianday('now') - julianday(created_at) <= 30 THEN '30天内'
                    WHEN julianday('now') - julianday(created_at) <= 90 THEN '90天内'
                    ELSE '超过90天'
                END as age_group,
                COUNT(*) as count,
                AVG(total_hits) as avg_hits
            FROM processing_cache
            GROUP BY age_group
            ORDER BY 
                CASE age_group
                    WHEN '1天内' THEN 1
                    WHEN '7天内' THEN 2
                    WHEN '30天内' THEN 3
                    WHEN '90天内' THEN 4
                    ELSE 5
                END
        """)
        analysis["age_distribution"] = [dict(row) for row in self.cursor.fetchall()]

        # 最近活跃的缓存（最近7天有访问）
        # self.cursor.execute("""
        #     SELECT COUNT(*) as recent_active
        #     FROM processing_cache
        #     WHERE julianday('now') - julianday(last_accessed) <= 7
        # """)
        # analysis["recent_active"] = self.cursor.fetchone()
        sql_str = """
            SELECT COUNT(*) as recent_active
            FROM processing_cache
            WHERE julianday('now') - julianday(last_accessed) <= 7
        """
        analysis["recent_active"] = self._fetch_scalar(sql_str) or 0

        return analysis

# %% [markdown]
# ## get_validation_analysis(self) -> Dict[str, Any]

    # %%
    def get_validation_analysis(self) -> Dict[str, Any]:
        """验证状态分析"""
        analysis = {}

        # 验证结果分布
        self.cursor.execute("""
            SELECT 
                COALESCE(validation_result, 'not_validated') as validation_state,
                COUNT(*) as count,
                AVG(total_hits) as avg_hits,
                AVG(julianday('now') - julianday(created_at)) as avg_age_days
            FROM processing_cache
            GROUP BY validation_state
            ORDER BY count DESC
        """)
        analysis["validation_states"] = [dict(row) for row in self.cursor.fetchall()]

        # 需要验证的条目（hit_count接近阈值）
        validation_threshold = (
            getinivaluefromcloud("joplinai", "validation_threshold") or 5000
        )
        # self.cursor.execute(
        #     """
        #     SELECT COUNT(*) as nearing_validation
        #     FROM processing_cache
        #     WHERE hit_count >= ? * 0.8  -- 达到阈值的80%
        # """,
        #     (validation_threshold,),
        # )
        # analysis["nearing_validation"] = self.cursor.fetchone()
        sql_str = """
            SELECT COUNT(*) as nearing_validation
            FROM processing_cache
            WHERE hit_count >= ? * 0.8  -- 达到阈值的80%
        """
        sql_param = (validation_threshold,)
        analysis["nearing_validation"] = self._fetch_scalar(sql_str, sql_param) or 0

        # 最近验证时间
        # self.cursor.execute("""
        #     SELECT MAX(last_validated_at) as last_validation
        #     FROM processing_cache
        #     WHERE last_validated_at IS NOT NULL
        # """)
        # analysis["last_validation_time"] = self.cursor.fetchone()
        sql_str = """
            SELECT MAX(last_validated_at) as last_validation
            FROM processing_cache
            WHERE last_validated_at IS NOT NULL
        """
        analysis["last_validation_time"] = self._fetch_scalar(sql_str) or 0

        return analysis

# %% [markdown]
# ## get_performance_metrics(self) -> Dict[str, Any]

    # %%
    def get_performance_metrics(self) -> Dict[str, Any]:
        """性能指标分析"""
        metrics = {}

        # 高命中缓存（top 10）
        self.cursor.execute("""
            SELECT 
                cache_key,
                task,
                total_hits,
                hit_count,
                created_at,
                last_accessed
            FROM processing_cache
            ORDER BY total_hits DESC
            LIMIT 10
        """)
        metrics["top_hitters"] = [dict(row) for row in self.cursor.fetchall()]

        # 长期未访问的缓存（可能已过时）
        # self.cursor.execute("""
        #     SELECT COUNT(*) as stale_entries
        #     FROM processing_cache
        #     WHERE julianday('now') - julianday(last_accessed) > 30
        #       AND total_hits = 0  -- 从未被命中过
        # """)
        # metrics["stale_entries"] = self.cursor.fetchone()
        sql_str = """
            SELECT COUNT(*) as stale_entries
            FROM processing_cache
            WHERE julianday('now') - julianday(last_accessed) > 30
              AND total_hits = 0  -- 从未被命中过
        """
        metrics["stale_entries"] = self._fetch_scalar(sql_str) or 0

        # 缓存大小估算（基于记录数）
        # self.cursor.execute("SELECT COUNT(*) as count FROM processing_cache")
        # count = self.cursor.fetchone()
        count = self._fetch_scalar("SELECT COUNT(*) FROM processing_cache") or 0
        # 粗略估算：每条记录约 2KB
        metrics["estimated_size_mb"] = round(count * 2 / 1024, 2)

        # 缓存命中时间模式（按小时）
        self.cursor.execute("""
            SELECT 
                strftime('%H', last_accessed) as hour,
                COUNT(*) as access_count
            FROM processing_cache
            WHERE last_accessed IS NOT NULL
            GROUP BY hour
            ORDER BY hour
        """)
        metrics["access_by_hour"] = [dict(row) for row in self.cursor.fetchall()]

        return metrics

    # %%
    def get_growth_trends(self, days: int = 30) -> Dict[str, Any]:
        """增长趋势分析"""
        trends = {}

        # 每日新增缓存
        self.cursor.execute(f"""
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as new_entries
            FROM processing_cache
            WHERE DATE(created_at) >= DATE('now', '-{days} days')
            GROUP BY DATE(created_at)
            ORDER BY date
        """)
        trends["daily_growth"] = [dict(row) for row in self.cursor.fetchall()]

        # 累计增长
        self.cursor.execute(f"""
            SELECT 
                DATE(created_at) as date,
                SUM(COUNT(*)) OVER (ORDER BY DATE(created_at)) as cumulative
            FROM processing_cache
            WHERE DATE(created_at) >= DATE('now', '-{days} days')
            GROUP BY DATE(created_at)
            ORDER BY date
        """)
        trends["cumulative_growth"] = [dict(row) for row in self.cursor.fetchall()]

        # 预测未来增长（简单线性预测）
        if trends["daily_growth"]:
            avg_daily = sum([d["new_entries"] for d in trends["daily_growth"]]) / len(
                trends["daily_growth"]
            )
            trends["predicted_weekly_growth"] = round(avg_daily * 7, 1)
        else:
            trends["predicted_weekly_growth"] = 0

        return trends

    # %%
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """生成综合报告"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "database_path": self.db_path,
            "basic_stats": self.get_basic_stats(),
            "time_analysis": self.get_time_analysis(),
            "validation_analysis": self.get_validation_analysis(),
            "performance_metrics": self.get_performance_metrics(),
            "growth_trends": self.get_growth_trends(30),
            "summary": {},
        }

        # 生成执行摘要
        basic = report["basic_stats"]
        time_ana = report["time_analysis"]
        valid_ana = report["validation_analysis"]

        report["summary"] = {
            "total_cache_entries": basic["total_entries"],
            "total_cache_hits": basic["total_hits"],
            "cache_hit_rate": f"{basic['avg_hits_per_entry']:.2f} hits/entry",
            "recently_active": f"{time_ana['recent_active']} entries (last 7 days)",
            "validation_coverage": f"{sum([v['count'] for v in valid_ana['validation_states'] if v['validation_state'] != 'not_validated'])} entries validated",
            "estimated_size": f"{report['performance_metrics']['estimated_size_mb']} MB",
            "growth_rate": f"{report['growth_trends']['predicted_weekly_growth']} entries/week (predicted)",
        }

        return report


# %% [markdown]
# # 报告生成器


# %%
class CacheReportGenerator:
    """缓存报告生成器（负责格式化和保存）"""

    def __init__(self, analyzer: CacheStatsAnalyzer):
        self.analyzer = analyzer
        self.report_data = None

    def generate_markdown_report(self) -> str:
        """生成 Markdown 格式的报告"""
        if not self.report_data:
            self.report_data = self.analyzer.get_comprehensive_report()

        report = self.report_data
        md_lines = []

        # 标题和元信息
        md_lines.append(f"# 📊 缓存使用统计报告")
        md_lines.append(f"**生成时间**: {report['generated_at']}")
        md_lines.append(f"**数据库**: `{report['database_path']}`")
        md_lines.append("")

        # 执行摘要
        md_lines.append("## 🎯 执行摘要")
        summary = report["summary"]
        md_lines.append(f"- **总缓存条目**: {summary['total_cache_entries']}")
        md_lines.append(f"- **总命中次数**: {summary['total_cache_hits']}")
        md_lines.append(f"- **平均命中率**: {summary['cache_hit_rate']}")
        md_lines.append(f"- **近期活跃**: {summary['recently_active']}")
        md_lines.append(f"- **验证覆盖率**: {summary['validation_coverage']}")
        md_lines.append(f"- **估算大小**: {summary['estimated_size']}")
        md_lines.append(f"- **预测周增长**: {summary['growth_rate']}")
        md_lines.append("")

        # 基础统计
        md_lines.append("## 📈 基础统计")
        basic = report["basic_stats"]
        md_lines.append(f"- **总记录数**: {basic['total_entries']}")
        md_lines.append(f"- **总命中次数**: {basic['total_hits']}")
        md_lines.append(f"- **当前周期命中**: {basic['current_hits']}")
        md_lines.append(f"- **平均每条目命中**: {basic['avg_hits_per_entry']:.2f}")
        md_lines.append("")

        # 任务分布
        md_lines.append("### 任务类型分布")
        md_lines.append("| 任务类型 | 条目数 | 总命中 | 平均命中 |")
        md_lines.append("|----------|--------|--------|----------|")
        for task in basic["task_distribution"]:
            md_lines.append(
                f"| {task['task']} | {task['count']} | {task['hits'] or 0} | {task['avg_hits'] or 0:.1f} |"
            )
        md_lines.append("")

        # 时间分析
        md_lines.append("## ⏰ 时间分析")
        time_ana = report["time_analysis"]
        md_lines.append(f"- **最近活跃缓存（7天内）**: {time_ana['recent_active']}")
        md_lines.append("")

        md_lines.append("### 缓存年龄分布")
        md_lines.append("| 年龄分组 | 条目数 | 平均命中 |")
        md_lines.append("|----------|--------|----------|")
        for age in time_ana["age_distribution"]:
            md_lines.append(
                f"| {age['age_group']} | {age['count']} | {age['avg_hits'] or 0:.1f} |"
            )
        md_lines.append("")

        # 验证分析
        md_lines.append("## ✅ 验证状态分析")
        valid_ana = report["validation_analysis"]

        md_lines.append("### 验证结果分布")
        md_lines.append("| 验证状态 | 条目数 | 平均命中 | 平均年龄(天) |")
        md_lines.append("|----------|--------|----------|--------------|")
        for state in valid_ana["validation_states"]:
            md_lines.append(
                f"| {state['validation_state']} | {state['count']} | {state['avg_hits'] or 0:.1f} | {state['avg_age_days'] or 0:.1f} |"
            )
        md_lines.append("")

        md_lines.append(f"- **接近验证阈值**: {valid_ana['nearing_validation']} 条")
        md_lines.append(
            f"- **最后验证时间**: {valid_ana['last_validation_time'] or '从未验证'}"
        )
        md_lines.append("")

        # 性能指标
        md_lines.append("## 🚀 性能指标")
        perf = report["performance_metrics"]
        md_lines.append(f"- **估算缓存大小**: {perf['estimated_size_mb']} MB")
        md_lines.append(
            f"- **陈旧条目（30天未访问且零命中）**: {perf['stale_entries']}"
        )
        md_lines.append("")

        md_lines.append("### 高命中缓存（Top 10）")
        md_lines.append("| 缓存键（前20字符） | 任务 | 总命中 | 周期命中 | 创建时间 |")
        md_lines.append("|-------------------|------|--------|----------|----------|")
        for hit in perf["top_hitters"][:10]:
            short_key = (
                hit["cache_key"][:20] + "..."
                if len(hit["cache_key"]) > 20
                else hit["cache_key"]
            )
            md_lines.append(
                f"| `{short_key}` | {hit['task']} | {hit['total_hits']} | {hit['hit_count']} | {hit['created_at'][:10]} |"
            )
        md_lines.append("")

        # 增长趋势
        md_lines.append("## 📊 增长趋势（最近30天）")
        growth = report["growth_trends"]
        md_lines.append(f"- **预测周增长**: {growth['predicted_weekly_growth']} 条目")
        md_lines.append("")

        if growth["daily_growth"]:
            md_lines.append("### 每日新增缓存")
            md_lines.append("| 日期 | 新增条目 | 累计总数 |")
            md_lines.append("|------|----------|----------|")
            for i, daily in enumerate(growth["daily_growth"][-10:]):  # 最近10天
                cum = (
                    growth["cumulative_growth"][i]["cumulative"]
                    if i < len(growth["cumulative_growth"])
                    else "N/A"
                )
                md_lines.append(f"| {daily['date']} | {daily['new_entries']} | {cum} |")

        # 建议和洞察
        md_lines.append("")
        md_lines.append("## 💡 洞察与建议")

        # 基于数据分析生成建议
        insights = []

        if perf["stale_entries"] > 100:
            insights.append(
                "**清理建议**: 发现较多陈旧条目，考虑运行 `cleanup_old_entries()` 或调整清理策略"
            )

        if valid_ana["nearing_validation"] > 50:
            insights.append("**验证提醒**: 大量缓存接近验证阈值，建议安排批量验证")

        if time_ana["recent_active"] / basic["total_entries"] < 0.3:
            insights.append(
                "**活跃度低**: 近期活跃缓存比例较低，考虑优化缓存策略或检查数据新鲜度"
            )

        if not insights:
            insights.append("缓存系统运行良好，继续保持当前策略")

        for insight in insights:
            md_lines.append(f"- {insight}")

        # 数据快照
        md_lines.append("")
        md_lines.append("## 🔍 数据快照")
        md_lines.append(f"```json")
        md_lines.append(
            json.dumps(
                {
                    "timestamp": report["generated_at"],
                    "total_entries": basic["total_entries"],
                    "total_hits": basic["total_hits"],
                    "validation_states": [
                        s["validation_state"] for s in valid_ana["validation_states"]
                    ],
                },
                indent=2,
            )
        )
        md_lines.append(f"```")

        return "\n".join(md_lines)


# %% [markdown]
# ## save_to_joplin(self, notebook_title: str = None, note_title: str = None) -> bool

    # %%
    def save_to_joplin(
        self, notebook_title: str = None, note_title: str = None
    ) -> bool:
        """保存报告到 Joplin 笔记"""
        try:
            from func.jpfuncs import createnote, searchnotes, updatenote_body

            # 生成报告内容
            report_content = self.generate_markdown_report()

            # 确定笔记本和标题
            notebook = notebook_title or REPORT_NOTEBOOK
            title = (
                note_title
                or f"{REPORT_NOTE_TITLE_PREFIX} {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )

            # 搜索是否已存在今日报告
            search_query = f'title:"{REPORT_NOTE_TITLE_PREFIX}"'
            existing_notes = searchnotes(search_query)

            note_id = None
            if existing_notes:
                # 找到最近的相关笔记
                for note in existing_notes:
                    if notebook_title in getattr(note, "notebook_title", ""):
                        note_id = note.id
                        break

            if note_id:
                # 更新现有笔记
                success = updatenote_body(note_id, report_content)
                log.info(f"已更新缓存统计报告笔记: {title}")
            else:
                # 创建新笔记
                note_id = createnote(
                    title=title,
                    body=report_content,
                    parent_id=None,  # 将自动放入指定笔记本
                )
                if note_id:
                    log.info(f"已创建缓存统计报告笔记: {title}")
                    success = True
                else:
                    log.error("创建报告笔记失败")
                    success = False

            return success

        except Exception as e:
            log.error(f"保存报告到 Joplin 失败: {e}")
            return False


# %% [markdown]
# # 命令行接口


# %%
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="缓存使用统计报告生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                          # 生成报告并显示
  %(prog)s --save                   # 生成并保存到 Joplin
  %(prog)s --db path/to/cache.db    # 指定缓存数据库
  %(prog)s --export-json report.json # 导出 JSON 数据
  %(prog)s --auto --schedule daily  # 自动模式（用于定时任务）
        """,
    )

    parser.add_argument(
        "--db",
        type=str,
        default=str(DEFAULT_CACHE_DB),
        help=f"缓存数据库路径（默认: {DEFAULT_CACHE_DB}）",
    )

    parser.add_argument("--save", action="store_true", help="将报告保存到 Joplin 笔记")

    parser.add_argument(
        "--notebook",
        type=str,
        default=REPORT_NOTEBOOK,
        help=f"保存报告的笔记本（默认: {REPORT_NOTEBOOK}）",
    )

    parser.add_argument(
        "--export-json", type=str, help="将原始统计数据导出为 JSON 文件"
    )

    parser.add_argument(
        "--auto", action="store_true", help="自动模式（不交互，用于定时任务）"
    )

    parser.add_argument(
        "--schedule",
        choices=["daily", "weekly", "monthly"],
        default="daily",
        help="自动运行频率（与 --auto 一起使用）",
    )

    return parser.parse_args()


# %% [markdown]
# # 主函数


# %%
def main():
    """主函数"""
    args = parse_args()

    log.info("=" * 60)
    log.info("缓存使用统计报告生成器")
    log.info("=" * 60)

    # 检查数据库是否存在
    if not Path(args.db).exists():
        log.error(f"缓存数据库不存在: {args.db}")
        log.error("请先运行 deepseek_enhancer.py 或 joplinai.py 生成缓存数据")
        return 1

    try:
        # 初始化分析器
        with CacheStatsAnalyzer(args.db) as analyzer:
            # 生成报告
            generator = CacheReportGenerator(analyzer)
            report_content = generator.generate_markdown_report()

            # 显示报告
            if not args.auto:
                print("\n" + "=" * 60)
                print("缓存统计报告预览（前100行）:")
                print("=" * 60)
                report_lines = report_content.split("\n")
                for i, line in enumerate(report_lines[:100]):
                    print(line)
                if (len_report := len(report_lines)) > 100:
                    print(f"\n...（完整报告共 {len_report} 行）")

            # 导出 JSON 数据
            if args.export_json:
                json_data = generator.report_data
                with open(args.export_json, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                log.info(f"统计数据已导出到: {args.export_json}")

            # 保存到 Joplin
            if args.save or args.auto:
                success = generator.save_to_joplin(
                    notebook_title=args.notebook,
                    note_title=f"{REPORT_NOTE_TITLE_PREFIX} {datetime.now().strftime('%Y-%m-%d')} ({args.schedule})",
                )
                if success:
                    log.info("✅ 报告已成功保存到 Joplin")
                else:
                    log.warning("⚠️  报告保存到 Joplin 可能失败")

            # 显示关键指标
            summary = generator.report_data["summary"]
            log.info("\n" + "=" * 60)
            log.info("关键指标摘要:")
            log.info("=" * 60)
            for key, value in summary.items():
                log.info(f"  {key.replace('_', ' ').title()}: {value}")

            return 0

    except Exception as e:
        log.error(f"生成报告时出错: {e}", exc_info=True)
        return 1


# %% [markdown]
# ## 定时任务集成示例


# %%
def setup_scheduled_task():
    """设置定时任务（示例代码）"""
    schedule_code = """
# 将以下内容添加到系统的定时任务（crontab）中：
# 每天凌晨2点运行缓存统计报告
0 2 * * * cd /path/to/project && python cache_manager.py --auto --schedule daily --save

# 或者使用 Python 调度库：
import schedule
import time
from cache_manager import main as generate_report

def job():
    generate_report(['--auto', '--schedule', 'daily', '--save'])

# 每天运行
schedule.every().day.at("02:00").do(job)

while True:
    schedule.run_pending()
    time.sleep(60)
    """

    return schedule_code


# %% [markdown]
# ## 模块集成说明


# %%
def integration_guidance():
    """提供集成到现有系统的指导"""
    guidance = """
## 集成到现有系统的三种方式：

### 1. 独立运行（推荐）
直接作为独立工具使用：
bash
python cache_report_generator.py --save

"""


# %% [markdown]
# # 主函数

# %%
if __name__ == "__main__":
    sys.exit(main())
