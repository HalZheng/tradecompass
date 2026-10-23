"""共享测试夹具。

所有测试使用合成数据与临时目录，绝不读写真实的 TC_HOME。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tradecompass.bootstrap import initialize


@pytest.fixture()
def home(tmp_path: Path) -> Path:
    """一个刚 init 完的 TC_HOME（配置来自仓库模板）。"""
    target = tmp_path / "home"
    initialize(target)
    return target
