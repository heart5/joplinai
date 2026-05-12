"""SQLiteCacheManager 单元测试。"""
from aimod.cache_manager import SQLiteCacheManager, CacheResult


class TestSQLiteCacheManager:
    """核心 CRUD + 验证流程。"""

    def test_init_creates_db(self, cache_db_path):
        mgr = SQLiteCacheManager(cache_db_path)
        stats = mgr.get_stats()
        assert stats["total"] == 0

    def test_set_and_get_hit(self, cache_db_path):
        mgr = SQLiteCacheManager(cache_db_path)
        mgr.set("hash123", "summary", "缓存结果内容")

        result = mgr.get("hash123", "summary")
        assert result.content == "缓存结果内容"
        assert result.requires_validation is False
        assert result.total_hits == 1

    def test_get_miss(self, cache_db_path):
        mgr = SQLiteCacheManager(cache_db_path)
        result = mgr.get("nonexistent", "summary")
        assert result.content is None
        assert result.requires_validation is False

    def test_validation_threshold(self, cache_db_path):
        """达到验证阈值后 requires_validation=True。"""
        mgr = SQLiteCacheManager(cache_db_path)
        mgr.VALIDATION_THRESHOLD = 3  # 压低阈值便于测试
        mgr.set("hash456", "tags", "标签A,标签B")

        # 前 2 次命中不应触发验证
        for _ in range(2):
            r = mgr.get("hash456", "tags")
            assert r.requires_validation is False

        # 第 3 次应触发
        r3 = mgr.get("hash456", "tags")
        assert r3.requires_validation is True
        assert r3.content == "标签A,标签B"  # 仍然返回缓存内容

    def test_update_on_validation_unchanged(self, cache_db_path):
        mgr = SQLiteCacheManager(cache_db_path)
        mgr.set("hash789", "summary", "原始内容")
        r = mgr.get("hash789", "summary")
        key = r.cache_key

        mgr.update_on_validation(key, None, validation_successful=True)
        r2 = mgr.get("hash789", "summary")
        assert r2.content == "原始内容"  # 未变

    def test_update_on_validation_updated(self, cache_db_path):
        mgr = SQLiteCacheManager(cache_db_path)
        mgr.set("hash789", "summary", "旧内容")
        r = mgr.get("hash789", "summary")
        key = r.cache_key

        mgr.update_on_validation(key, "新内容", validation_successful=True)
        r2 = mgr.get("hash789", "summary")
        assert r2.content == "新内容"

    def test_update_on_validation_failed(self, cache_db_path):
        mgr = SQLiteCacheManager(cache_db_path)
        mgr.set("hashfail", "summary", "内容")
        r = mgr.get("hashfail", "summary")
        key = r.cache_key

        # 不抛异常
        mgr.update_on_validation(key, None, validation_successful=False)
        r2 = mgr.get("hashfail", "summary")
        assert r2.content == "内容"  # 原内容不变

    def test_get_stats_per_key(self, cache_db_path):
        mgr = SQLiteCacheManager(cache_db_path)
        mgr.set("hash_stats", "summary", "内容")
        r = mgr.get("hash_stats", "summary")

        stats = mgr.get_stats(cache_key=r.cache_key)
        assert stats["content_hash"] == "hash_stats"
        assert stats["task"] == "summary"

    def test_different_tasks_same_hash(self, cache_db_path):
        mgr = SQLiteCacheManager(cache_db_path)
        mgr.set("hash_abc", "summary", "摘要内容")
        mgr.set("hash_abc", "tags", "标签内容")

        r1 = mgr.get("hash_abc", "summary")
        r2 = mgr.get("hash_abc", "tags")
        assert r1.content == "摘要内容"
        assert r2.content == "标签内容"
        assert r1.cache_key != r2.cache_key


class TestCacheResult:
    def test_dataclass_fields(self):
        cr = CacheResult(
            content="test",
            requires_validation=False,
            cache_key="key1",
            current_hit_count=5,
            total_hits=10,
        )
        assert cr.content == "test"
        assert cr.total_hits == 10
