"""deepseek_enhancer 缓存流程测试 — API 调用全部 mock。

注意：aimod.deepseek_enhancer 导入链会触发 func/datatools 模块级副作用，
故所有 SUT 导入均延迟到测试函数内部。
"""
from unittest.mock import MagicMock, patch

import pytest

# CacheResult 在 cache_manager 中，无副作用导入
from aimod.cache_manager import CacheResult


class TestCacheFlow:
    @pytest.fixture
    def de(self):
        import aimod.deepseek_enhancer as mod
        return mod

    def test_no_api_key_returns_none(self, de, monkeypatch):
        monkeypatch.setattr(de, "DEEPSEEK_API_KEY", "")
        result = de.deepseek_process_note("测试文本", task="summary")
        assert result is None

    def test_cache_hit_returns_cached(self, de, monkeypatch):
        monkeypatch.setattr(de, "DEEPSEEK_API_KEY", "fake-key")
        monkeypatch.setattr(de, "compute_content_hash", lambda x: "abc123")

        mock_cm = MagicMock()
        mock_cm.get.return_value = CacheResult(
            content="缓存摘要",
            requires_validation=False,
            cache_key="abc123_summary",
            current_hit_count=1,
            total_hits=10,
        )
        monkeypatch.setattr(de, "get_cache_manager", lambda: mock_cm)

        result = de.deepseek_process_note("测试文本", task="summary", use_cache=True)
        assert result == "缓存摘要"
        mock_cm.set.assert_not_called()

    def test_cache_hit_with_validation_triggers_async_check(self, de, monkeypatch):
        monkeypatch.setattr(de, "DEEPSEEK_API_KEY", "fake-key")
        monkeypatch.setattr(de, "compute_content_hash", lambda x: "xyz789")

        mock_cm = MagicMock()
        mock_cm.get.return_value = CacheResult(
            content="需验证的内容",
            requires_validation=True,
            cache_key="xyz789_tags",
            current_hit_count=1,
            total_hits=5001,
        )
        monkeypatch.setattr(de, "get_cache_manager", lambda: mock_cm)

        with patch.object(de, "_validate_cache_entry_async") as mock_val:
            result = de.deepseek_process_note("测试文本", task="tags", use_cache=True)
            assert result == "需验证的内容"
            mock_val.assert_called_once()

    def test_cache_miss_calls_api(self, de, monkeypatch):
        monkeypatch.setattr(de, "DEEPSEEK_API_KEY", "fake-key")
        monkeypatch.setattr(de, "compute_content_hash", lambda x: "miss123")

        mock_cm = MagicMock()
        mock_cm.get.return_value = CacheResult(
            content=None,
            requires_validation=False,
            cache_key="miss123_summary",
            current_hit_count=0,
            total_hits=0,
        )
        monkeypatch.setattr(de, "get_cache_manager", lambda: mock_cm)

        with patch.object(de, "_call_deepseek_api_directly", return_value="新结果") as mock_api:
            result = de.deepseek_process_note("测试", task="summary", use_cache=True)
            assert result == "新结果"
            mock_api.assert_called_once()
            mock_cm.set.assert_called_once_with("miss123", "summary", "新结果")

    def test_bypass_cache_direct_api(self, de, monkeypatch):
        monkeypatch.setattr(de, "DEEPSEEK_API_KEY", "fake-key")

        with patch.object(de, "_call_deepseek_api_directly", return_value="直调结果") as mock_api:
            result = de.deepseek_process_note("测试", task="tags", use_cache=False)
            assert result == "直调结果"
            mock_api.assert_called_once()


class TestValidationAsync:
    @pytest.fixture
    def de(self):
        import aimod.deepseek_enhancer as mod
        return mod

    def test_validation_updates_cache_on_change(self, de, monkeypatch):
        monkeypatch.setattr(de, "DEEPSEEK_API_KEY", "fake-key")

        mock_cm = MagicMock()
        monkeypatch.setattr(de, "get_cache_manager", lambda: mock_cm)

        with patch.object(de, "_call_deepseek_api_directly", return_value="新验证结果"):
            de._validate_cache_entry_async(
                original_text="原文",
                task="summary",
                cache_key="key123",
                cached_content="旧结果",
                model="deepseek-chat",
                max_retries=1,
            )

        mock_cm.update_on_validation.assert_called_once_with(
            "key123", "新验证结果", validation_successful=True
        )

    def test_validation_no_change(self, de, monkeypatch):
        monkeypatch.setattr(de, "DEEPSEEK_API_KEY", "fake-key")

        mock_cm = MagicMock()
        monkeypatch.setattr(de, "get_cache_manager", lambda: mock_cm)

        with patch.object(de, "_call_deepseek_api_directly", return_value="相同结果"):
            de._validate_cache_entry_async(
                original_text="原文",
                task="summary",
                cache_key="key456",
                cached_content="相同结果",
                model="deepseek-chat",
                max_retries=1,
            )

        mock_cm.update_on_validation.assert_called_once_with(
            "key456", None, validation_successful=True
        )

    def test_validation_api_failure(self, de, monkeypatch):
        monkeypatch.setattr(de, "DEEPSEEK_API_KEY", "fake-key")

        mock_cm = MagicMock()
        monkeypatch.setattr(de, "get_cache_manager", lambda: mock_cm)

        with patch.object(de, "_call_deepseek_api_directly", return_value=None):
            de._validate_cache_entry_async(
                original_text="原文",
                task="summary",
                cache_key="key789",
                cached_content="结果",
                model="deepseek-chat",
                max_retries=1,
            )

        mock_cm.update_on_validation.assert_called_once_with(
            "key789", None, validation_successful=False
        )

    def test_validation_exception_handled(self, de, monkeypatch):
        monkeypatch.setattr(de, "DEEPSEEK_API_KEY", "fake-key")

        mock_cm = MagicMock()
        monkeypatch.setattr(de, "get_cache_manager", lambda: mock_cm)

        with patch.object(de, "_call_deepseek_api_directly", side_effect=RuntimeError("boom")):
            de._validate_cache_entry_async(
                original_text="原文",
                task="summary",
                cache_key="key_err",
                cached_content="结果",
                model="deepseek-chat",
                max_retries=1,
            )

        mock_cm.update_on_validation.assert_called_once_with(
            "key_err", None, validation_successful=False
        )


class TestCallAPI:
    def test_unsupported_task_returns_none(self, monkeypatch):
        import aimod.deepseek_enhancer as de

        monkeypatch.setattr(de, "DEEPSEEK_API_KEY", "fake-key")
        result = de._call_deepseek_api_directly("text", "invalid_task", "model", 1)
        assert result is None
