"""
pytest 共享 fixtures。

pytest_configure 在测试收集前注入 func 层 mock，
确保所有从 func/ 导入的 SUT 模块在受控环境中初始化。
"""
import logging
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# pytest 钩子 — 早于任何测试模块导入
# ---------------------------------------------------------------------------

_PATCHERS = []


def pytest_configure(config):
    """在所有测试模块导入之前启动 patches。"""

    # getinivaluefromcloud: 返回 None，各测试可按需 override
    _start_patch("func.jpfuncs.getinivaluefromcloud", MagicMock(return_value=None))
    _start_patch("func.logme.log", MagicMock())

    # func/datatools 有模块级副作用（import 时调用 getkeysfromcloud()），
    # 无法安全 import。注入 mock 模块避免副作用。
    _mock_datatools = MagicMock()
    _mock_datatools.compute_content_hash = MagicMock(return_value="deadbeef1234")
    sys.modules["func.datatools"] = _mock_datatools


def pytest_unconfigure(config):
    for p in _PATCHERS:
        p.stop()
    _PATCHERS.clear()
    sys.modules.pop("func.datatools", None)


def _start_patch(target, mock):
    p = patch(target, mock)
    p.start()
    _PATCHERS.append(p)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def cache_db_path(temp_dir):
    return str(temp_dir / "test_cache.db")


@pytest.fixture
def user_db_path(temp_dir):
    return str(temp_dir / "test_users.db")


@pytest.fixture
def history_db_path(temp_dir):
    return str(temp_dir / "test_history.db")


@pytest.fixture
def clean_sqlite():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()
