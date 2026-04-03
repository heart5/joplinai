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
import hashlib
import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# %% [markdown]
# ### func库

# %%
try:
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
# ## CacheResult类

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
# ## SQLiteCacheManager类

# %%
class SQLiteCacheManager:
    """基于SQLite的高性能缓存管理器，用于DeepSeek增强结果。"""


# %% [markdown]
# ### 验证阈值

    # %%
    if not (VALIDATION_THRESHOLD := getinivaluefromcloud("joplinai", "validation_threshold")):
        VALIDATION_THRESHOLD = 5000  # 验证阈值，可从配置读取

# %% [markdown]
# ### __init__(self, db_path: str = "data/.deepseek_cache/deepseek_cache.db")

    # %%
    def __init__(self, db_path: str = "data/.deepseek_cache/deepseek_cache.db"):
        self.db_path = db_path
        self._init_db()

# %% [markdown]
# ### _init_db(self)

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
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hash_task ON processing_cache (content_hash, task)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON processing_cache (last_accessed)")
        conn.commit()
        conn.close()

# %% [markdown]
# ### import_from_json(self, json_file_path: str, clear_existing: bool = False) -> Dict[str, int]

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
            backup_file = old_file.with_suffix('.json.backup')
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
# ### import_from_json_directory(self, json_dir_path: str, pattern: str = "*.json", clear_existing: bool = False) -> Dict[str, Any]

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
# ### get(self, content_hash: str, task: str) -> Optional[str]

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
        cursor.execute("""
            SELECT result, hit_count, total_hits FROM processing_cache
            WHERE cache_key = ? AND (julianday('now') - julianday(created_at)) < 90
        """, (cache_key,))
        row = cursor.fetchone()

        if not row:
            conn.close()
            # 未命中
            return CacheResult(
                content=None,
                requires_validation=False,
                cache_key=cache_key,
                current_hit_count=0,
                total_hits=0
            )

        cached_result, current_hit_count, total_hits = row
        new_hit_count = current_hit_count + 1
        new_total_hits = total_hits + 1
        now_iso = datetime.now().isoformat()

        # 判断本次命中后是否达到验证阈值
        should_validate = (new_hit_count >= self.VALIDATION_THRESHOLD)

        # 更新数据库：计数、最后访问时间
        update_sql = """
            UPDATE processing_cache
            SET hit_count = ?, total_hits = ?, last_accessed = ?
        """
        update_params = [new_hit_count, new_total_hits, now_iso]

        if should_validate:
            # 达到阈值，标记为需要验证，并重置周期计数
            update_sql += ", hit_count = 0, last_validated_at = ?, validation_result = 'pending'"
            update_params.extend([now_iso])
            # 注意：这里将 hit_count 重置为0，开始新的计数周期

        update_sql += " WHERE cache_key = ?"
        update_params.append(cache_key)

        cursor.execute(update_sql, update_params)
        conn.commit()
        conn.close()

        log.debug(f"缓存查询: {cache_key[:12]}... (周期命中={new_hit_count}, 总计={new_total_hits}, 需验证={should_validate})")

        # 返回封装结果，告诉调用者缓存内容以及“是否需要验证”
        return CacheResult(
            content=cached_result,
            requires_validation=should_validate,
            cache_key=cache_key,
            current_hit_count=new_hit_count if not should_validate else 0, # 返回重置前的计数或重置后的0
            total_hits=new_total_hits
        )

# %% [markdown]
# ### update_on_validation(self, cache_key: str, new_result: Optional[str], validation_successful: bool)

    # %%
    def update_on_validation(self, cache_key: str, new_result: Optional[str], validation_successful: bool):
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
            cursor.execute("""
                UPDATE processing_cache
                SET last_validated_at = ?, validation_result = 'failed'
                WHERE cache_key = ?
            """, (now_iso, cache_key))
            log.warning(f"验证失败记录更新: {cache_key[:12]}...")
        elif new_result is None:
            # 验证成功，且内容未变化
            cursor.execute("""
                UPDATE processing_cache
                SET last_validated_at = ?, validation_result = 'valid'
                WHERE cache_key = ?
            """, (now_iso, cache_key))
            log.info(f"验证完成，内容未变: {cache_key[:12]}...")
        else:
            # 验证成功，且内容已更新
            cursor.execute("""
                UPDATE processing_cache
                SET result = ?, created_at = ?, last_validated_at = ?, validation_result = 'updated'
                WHERE cache_key = ?
            """, (new_result, now_iso, now_iso, cache_key))
            log.info(f"验证完成，缓存已更新: {cache_key[:12]}...")

        conn.commit()
        conn.close()

# %% [markdown]
# ### set(self, content_hash: str, task: str, result: str)

    # %%
    def set(self, content_hash: str, task: str, result: str):
        """设置新的缓存条目（首次保存或强制覆盖）"""
        cache_key = f"{content_hash}_{task}"
        now_iso = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO processing_cache
            (cache_key, content_hash, task, result, created_at, last_accessed,
             last_validated_at, hit_count, total_hits, validation_result)
            VALUES (?, ?, ?, ?, ?, ?, NULL, 0, 0, NULL)
        """, (cache_key, content_hash, task, result, now_iso, now_iso))

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
                    ORDER BY last_accessed ASC, timestamp ASC 
                    LIMIT 1000
                )
            """)
            log.info(f"执行缓存清理，删除了1000条旧记录。")

        conn.commit()
        conn.close()
        log.debug(f"缓存已设置: {cache_key}")

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

# %% [markdown]
# ### get_stats(self, cache_key: str = None) -> Dict[str, Any]

    # %%
    def get_stats(self, cache_key: str = None) -> Dict[str, Any]:
        """获取缓存统计信息"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        stats = {}
        if cache_key:
            cursor.execute("SELECT * FROM processing_cache WHERE cache_key = ?", (cache_key,))
            row = cursor.fetchone()
            if row:
                stats = dict(row)
        else:
            # 全局统计
            cursor.execute("SELECT COUNT(*) as total, SUM(total_hits) as total_hits, SUM(hit_count) as current_hits FROM processing_cache")
            row = cursor.fetchone()
            stats = dict(row) if row else {}
            cursor.execute("SELECT validation_result, COUNT(*) as count FROM processing_cache WHERE validation_result IS NOT NULL GROUP BY validation_result")
            stats['validation_breakdown'] = {r: c for r, c in cursor.fetchall()}

        conn.close()
        return stats
