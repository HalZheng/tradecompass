"""飞书推送：webhook 最简单，失败时退化为本地预览。"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import httpx

from tradecompass.config.loader import AppConfig


def send_text(cfg: AppConfig, text: str, *, name: str = "daily") -> str:
    """推送一段文本；返回一句话状态，供 CLI 打印。"""
    if not cfg.system.notify.enabled:
        return _write_preview(cfg, text, name)

    url = os.environ.get("TC_FEISHU_WEBHOOK_URL", "").strip()
    if not url:
        return _write_preview(cfg, text, name) + "（未配置 TC_FEISHU_WEBHOOK_URL）"

    try:
        response = httpx.post(
            url,
            json={"msg_type": "text", "content": {"text": text}},
            timeout=10.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return _write_preview(cfg, text, name) + f"（飞书推送失败：{exc}）"

    return "已推送到飞书"


def _write_preview(cfg: AppConfig, text: str, name: str) -> str:
    directory = cfg.paths.logs / "push_preview"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{date.today()}-{name}.md"
    path.write_text(text, encoding="utf-8")
    return f"推送预览：{path}"


def preview_path(cfg: AppConfig, name: str = "daily") -> Path:
    return cfg.paths.logs / "push_preview" / f"{date.today()}-{name}.md"
