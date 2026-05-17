"""RunTracker 单元测试 — 本地 SQLite 模式。"""
import pytest
from aimod.run_tracker import RunTracker


@pytest.fixture
def tracker(history_db_path):
    """返回无 history_client 的 RunTracker（纯本地 SQLite）。

    __init__ 中 history_db_path 使用 getdirmain()，需事后覆盖为 temp 路径。
    """
    config = {"ollama_embedding_model": "test/model:v1"}
    rt = RunTracker(config, history_client=None)
    rt.history_db_path = history_db_path
    rt._init_history_db()  # 在 temp DB 上重新建表
    return rt


class TestRunTracker:
    def test_init_no_client(self, tracker):
        assert tracker.history_client is None
        assert tracker.task_records == []
        assert tracker.summary_data == {}
        assert tracker.global_chunk_stats["total_chunks_processed"] == 0

    def test_add_notebook_record(self, tracker):
        stats = {
            "total_notes": 10,
            "updated_count": 8,
            "failed_count": 2,
            "notes_added": ["note1", "note2"],
            "notes_removed": ["note3"],
            "failed_notes": ["note4", "note5"],
            "chunk_stats": {
                "total_chunks": 50,
                "upserted": 45,
                "skipped": 3,
                "orphans_cleaned": 2,
            },
        }
        tracker.add_notebook_record("测试笔记本", stats)

        assert len(tracker.task_records) == 1
        assert "测试笔记本" in tracker.summary_data
        assert tracker.global_chunk_stats["total_chunks_processed"] == 50
        assert tracker.global_chunk_stats["chunks_upserted"] == 45

    def test_finalize_run(self, tracker):
        stats = {
            "total_notes": 5,
            "updated_count": 5,
            "failed_count": 0,
            "notes_added": ["n1"],
            "notes_removed": [],
            "failed_notes": [],
            "chunk_stats": {"total_chunks": 20, "upserted": 20, "skipped": 0, "orphans_cleaned": 0},
        }
        tracker.add_notebook_record("笔记本A", stats)
        tracker.finalize_run(success=True)

    def test_finalize_run_failure(self, tracker):
        stats = {
            "total_notes": 3,
            "updated_count": 0,
            "failed_count": 3,
            "notes_added": [],
            "notes_removed": [],
            "failed_notes": ["f1", "f2", "f3"],
            "chunk_stats": {"total_chunks": 0, "upserted": 0, "skipped": 0, "orphans_cleaned": 0},
        }
        tracker.add_notebook_record("失败笔记本", stats)
        tracker.finalize_run(success=False, error_msg="API timeout")

    def test_get_snapshot(self, tracker):
        stats = {
            "total_notes": 1,
            "updated_count": 1,
            "failed_count": 0,
            "notes_added": ["n"],
            "notes_removed": [],
            "failed_notes": [],
            "chunk_stats": {"total_chunks": 1, "upserted": 1, "skipped": 0, "orphans_cleaned": 0},
        }
        tracker.add_notebook_record("快照测试", stats)
        snapshot = tracker.get_snapshot()
        assert "summary_data" in snapshot
        assert snapshot["summary_data"]["快照测试"]["total_notes"] == 1

    def test_get_cumulative_stats_empty(self, tracker):
        """空数据库应返回合理默认值。"""
        result = tracker.get_cumulative_stats()
        assert isinstance(result, dict)
        assert "cumulative" in result
        # 空 DB 时 cumulative 字段为 0（SQLite COUNT 无匹配行返回 0）
        assert result["cumulative"]["total_runs"] == 0

    def test_get_cumulative_stats_after_run(self, tracker):
        stats = {
            "total_notes": 10,
            "updated_count": 10,
            "failed_count": 0,
            "notes_added": ["a", "b"],
            "notes_removed": [],
            "failed_notes": [],
            "chunk_stats": {"total_chunks": 40, "upserted": 40, "skipped": 0, "orphans_cleaned": 0},
        }
        tracker.add_notebook_record("统计测试", stats)
        tracker.finalize_run(success=True)

        result = tracker.get_cumulative_stats()
        assert result["cumulative"]["total_runs"] == 1

    def test_get_change_analysis(self, tracker):
        stats = {
            "total_notes": 5,
            "updated_count": 3,
            "failed_count": 2,
            "notes_added": ["x"],
            "notes_removed": ["y"],
            "failed_notes": ["z1", "z2"],
            "chunk_stats": {"total_chunks": 15, "upserted": 10, "skipped": 3, "orphans_cleaned": 2},
        }
        tracker.add_notebook_record("变化分析", stats)
        tracker.finalize_run(success=True)

        result = tracker.get_change_analysis()
        assert isinstance(result, dict)
        assert "added_count" in result
        assert "removed_count" in result

    def test_get_efficiency_metrics(self, tracker):
        result = tracker.get_efficiency_metrics()
        assert isinstance(result, dict)

    def test__get_current_run_id(self, tracker):
        rid = tracker._get_current_run_id()
        assert "_" in rid
        assert "model_v1" in rid
