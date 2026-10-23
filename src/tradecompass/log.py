"""日志配置（标准库 logging，SPEC §0.4）。

约定：不输出密钥；金额明细只进推送与本地报告，不进日志。
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_NOISY = ("urllib3", "httpx", "httpcore", "asyncio", "apscheduler.executors.default")


def configure(
    level: str = "INFO",
    log_dir: Path | None = None,
    filename: str = "tradecompass.log",
) -> None:
    """配置根 logger：控制台 + （可选）按大小轮转的文件。可重复调用。"""
    root = logging.getLogger()
    root.setLevel(level.upper())
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        rotating = RotatingFileHandler(
            log_dir / filename, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        rotating.setFormatter(formatter)
        root.addHandler(rotating)

    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
