"""TC_HOME 初始化（SPEC §3.3）。

与 CLI 分离，方便测试；``init`` 命令只负责解析目标路径和打印结果。
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from tradecompass.config.loader import Paths, examples_dir
from tradecompass.storage.db import init_db

_REQUIRED_SUBDIRS: tuple[str, ...] = ("config", "config/strategies", "data", "reports", "logs")


@dataclass(frozen=True, slots=True)
class InitReport:
    home: Path
    copied: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    db_created: bool = False


def _copy_if_absent(src: Path, dst: Path, report: InitReport) -> None:
    if dst.exists():
        report.skipped.append(dst)
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    report.copied.append(dst)


def initialize(home: Path) -> InitReport:
    """建目录、复制模板、建库。任何已存在的文件都不会被覆盖。"""
    paths = Paths.from_home(home)
    for name in _REQUIRED_SUBDIRS:
        (paths.home / name).mkdir(parents=True, exist_ok=True)

    report = InitReport(home=home)
    source = examples_dir()

    for item in sorted(source.iterdir()):
        if item.is_dir():
            continue
        if item.name == ".env.example":
            _copy_if_absent(item, paths.home / ".env", report)
        elif item.name.endswith(".example.yaml"):
            target_name = item.name.replace(".example.yaml", ".yaml")
            _copy_if_absent(item, paths.config / target_name, report)

    strategies_src = source / "strategies"
    if strategies_src.is_dir():
        for item in sorted(strategies_src.glob("*.example.yaml")):
            target_name = item.name.replace(".example.yaml", ".yaml")
            _copy_if_absent(item, paths.strategies_dir / target_name, report)

    created = init_db(paths.db_path)
    return InitReport(
        home=home, copied=report.copied, skipped=report.skipped, db_created=created
    )
