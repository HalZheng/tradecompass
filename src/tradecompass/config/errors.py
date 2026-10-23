"""配置层错误类型。"""

from __future__ import annotations


class ConfigError(Exception):
    """配置文件缺失、语法错误或字段非法。

    消息必须以「文件名: 键路径 -> 原因」的形式给出，方便直接定位。
    """

    pass
