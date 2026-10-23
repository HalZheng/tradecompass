"""交割单导入（SPEC §11.1）。

支持 ``.csv`` / ``.txt``（制表符）/ ``.xlsx`` / ``.xls``；列名与操作文本的映射来自
``broker.yaml``。未识别的操作**不会静默丢弃**，会作为 issue 返回并写入 ``import_issues``。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from tradecompass.config.models import BrokerMapping

VALID_SIDES = ("buy", "sell", "dividend", "bonus_share", "tax")
ENCODINGS = ("utf-8-sig", "gbk", "utf-8")


class ImportError_(RuntimeError):
    """交割单无法读取或结构不符。"""


@dataclass(frozen=True, slots=True)
class ParsedTrade:
    trade_date: str
    trade_time: str | None
    symbol: str
    name: str | None
    side: str
    quantity: int
    price_raw: float | None
    amount: float | None
    fee_commission: float
    fee_stamp: float
    fee_transfer: float
    fee_other: float
    dedupe_key: str
    raw_json: str


@dataclass(frozen=True, slots=True)
class ParseIssue:
    row_no: int
    reason: str
    raw_json: str


@dataclass(frozen=True, slots=True)
class ParseResult:
    trades: tuple[ParsedTrade, ...]
    issues: tuple[ParseIssue, ...]
    ignored: int

    @property
    def ok(self) -> bool:
        return not self.issues


def read_statement(path: Path) -> pd.DataFrame:
    """读取交割单为 DataFrame（全部列名转成去空格的字符串）。"""
    if not path.is_file():
        raise ImportError_(f"文件不存在: {path}")
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        frame = pd.read_excel(path, dtype=object)
    elif suffix in (".csv", ".txt", ".tsv"):
        last_error: Exception | None = None
        for encoding in ENCODINGS:
            try:
                separator = "\t" if suffix in (".txt", ".tsv") else ","
                frame = pd.read_csv(
                    path, sep=separator, dtype=str, encoding=encoding, skip_blank_lines=True
                )
                break
            except UnicodeDecodeError as exc:  # 换下一个编码
                last_error = exc
        else:
            raise ImportError_(f"无法解码文件 {path.name}: {last_error}")
    else:
        raise ImportError_(f"不支持的文件类型: {suffix}")
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _pick(columns: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _to_float(value: object) -> float:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    text = str(value).strip().replace(",", "")
    if not text or text in ("-", "--"):
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _to_date(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime | date):
        return value.strftime("%Y-%m-%d")
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    try:
        return pd.Timestamp(text).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def normalize_symbol(value: object) -> str | None:
    """Excel 常把 000001 读成整数 1，统一补零到 6 位。"""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    return digits.zfill(6) if len(digits) <= 6 else digits[:6]


def parse_statement(frame: pd.DataFrame, mapping: BrokerMapping) -> ParseResult:
    columns = list(frame.columns)
    resolved: dict[str, str] = {}
    for field, candidates in mapping.columns.items():
        found = _pick(columns, candidates)
        if found is not None:
            resolved[field] = found
    missing = [
        field
        for field in ("trade_date", "symbol", "side_text", "quantity")
        if field not in resolved
    ]
    if missing:
        raise ImportError_(f"交割单缺少必要列（映射后）: {', '.join(missing)}；文件列有 {columns}")

    side_lookup: dict[str, str] = {}
    for side_name, texts in mapping.sides.items():
        for text in texts:
            side_lookup[text.strip()] = side_name

    trades: list[ParsedTrade] = []
    issues: list[ParseIssue] = []
    ignored = 0

    for position, (_, row) in enumerate(frame.iterrows(), start=2):
        raw = {str(k): str(v) for k, v in row.items() if not pd.isna(v)}
        side_text = str(row.get(resolved["side_text"], "")).strip()
        side = side_lookup.get(side_text)
        if side == "ignore":
            ignored += 1
            continue
        if side is None:
            issues.append(
                ParseIssue(
                    position, f"未识别的操作文本「{side_text}」",
                    json.dumps(raw, ensure_ascii=False),
                )
            )
            continue
        if side not in VALID_SIDES:
            issues.append(
                ParseIssue(
                    position, f"操作「{side_text}」映射到非法 side={side}",
                    json.dumps(raw, ensure_ascii=False),
                )
            )
            continue

        symbol = normalize_symbol(row.get(resolved["symbol"]))
        trade_date = _to_date(row.get(resolved["trade_date"]))
        if symbol is None or trade_date is None:
            issues.append(
                ParseIssue(
                    position, "证券代码或成交日期无法解析", json.dumps(raw, ensure_ascii=False)
                )
            )
            continue

        quantity = int(_to_float(row.get(resolved["quantity"])))
        if side in ("buy", "sell", "bonus_share") and quantity <= 0:
            issues.append(ParseIssue(position, "成交数量为 0", json.dumps(raw, ensure_ascii=False)))
            continue

        price = _to_float(row.get(resolved["price"])) if "price" in resolved else 0.0
        amount = _to_float(row.get(resolved["amount"])) if "amount" in resolved else 0.0
        trade_time: str | None = None
        if "trade_time" in resolved:
            raw_time = str(row.get(resolved["trade_time"], "")).strip()
            trade_time = raw_time or None

        trade_id = (
            str(row.get(resolved["trade_id"], "")).strip() if "trade_id" in resolved else ""
        )
        dedupe = (
            f"id:{trade_id}"
            if trade_id and trade_id.lower() != "nan"
            else "hash:"
            + hashlib.sha256(
                f"{trade_date}|{trade_time}|{symbol}|{side}|{quantity}|{price}".encode()
            ).hexdigest()[:32]
        )

        trades.append(
            ParsedTrade(
                trade_date=trade_date,
                trade_time=trade_time,
                symbol=symbol,
                name=str(row.get(resolved["name"], "")).strip() if "name" in resolved else None,
                side=side,
                quantity=quantity,
                price_raw=price or None,
                amount=amount or None,
                fee_commission=_to_float(row.get(resolved.get("fee_commission", ""), 0)),
                fee_stamp=_to_float(row.get(resolved.get("fee_stamp", ""), 0)),
                fee_transfer=_to_float(row.get(resolved.get("fee_transfer", ""), 0)),
                fee_other=_to_float(row.get(resolved.get("fee_other", ""), 0)),
                dedupe_key=dedupe,
                raw_json=json.dumps(raw, ensure_ascii=False),
            )
        )
    return ParseResult(tuple(trades), tuple(issues), ignored)


def import_into(
    conn: sqlite3.Connection, result: ParseResult, *, source: str = "broker"
) -> tuple[int, int, int]:
    """写入 trades 与 import_issues，返回 (新增, 重复, 问题)。"""
    inserted = 0
    duplicated = 0
    now = datetime.now().isoformat(timespec="seconds")
    for trade in result.trades:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO trades
            (dedupe_key, trade_date, trade_time, symbol, name, side, quantity, price_raw,
             amount, fee_commission, fee_stamp, fee_transfer, fee_other, source,
             raw_json, imported_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                trade.dedupe_key, trade.trade_date, trade.trade_time, trade.symbol, trade.name,
                trade.side, trade.quantity, trade.price_raw, trade.amount, trade.fee_commission,
                trade.fee_stamp, trade.fee_transfer, trade.fee_other, source, trade.raw_json, now,
            ),
        )
        if cursor.rowcount:
            inserted += 1
        else:
            duplicated += 1
    for issue in result.issues:
        conn.execute(
            "INSERT INTO import_issues (file, row_no, reason, raw_json, created_at)"
            " VALUES (?,?,?,?,?)",
            ("statement", issue.row_no, issue.reason, issue.raw_json, now),
        )
    conn.commit()
    return inserted, duplicated, len(result.issues)
