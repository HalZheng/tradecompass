"""东方财富“股票明细对账单”PDF 解析。

- 读取账户汇总、持仓快照、资金流水；
- 用“资金余额”逐行链式校验（上一行余额 + 本行发生金额 = 本行余额），保证不漏行、不错位；
- 天天宝申购/赎回与 OTC 现金宝调账是场外理财划转，校验后从结果中剔除；
- 不保留证件号、客户编号、股东账号等个人敏感信息。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

TRADE_TYPES = {"证券买入": "buy", "证券卖出": "sell"}
CATEGORY = {
    "证券买入": "trade",
    "证券卖出": "trade",
    "银行转证券": "transfer_in",
    "证券转银行": "transfer_out",
    "红利入账": "dividend",
    "股息红利差异扣税": "dividend_tax",
    "利息归本": "interest",
}
EXCLUDED_PREFIXES = ("天天宝", "OTC现金宝")

_NUM = r"-?\d+(?:\.\d+)?"
_FLOW = re.compile(
    rf"^(\d{{8}}) (\S+?)(?: (\d{{6}}) (.+?))? (-?\d+) ({_NUM}) ({_NUM}) ({_NUM}) ({_NUM}) "
    rf"({_NUM}) ({_NUM})$"
)
_HOLDING = re.compile(rf"^(\S+) (\d{{6}}) (.+?) (-?\d+) ({_NUM}) ({_NUM}) ({_NUM})$")
_FIELD = re.compile(r"(证券市值|总资产|资金余额|资金可用)\(RMB\)：\s*(-?[\d.]+)")
_PERIOD = re.compile(r"查询区间：(\d{4})/(\d{2})/(\d{2})-(\d{4})/(\d{2})/(\d{2})")
_PRINTED = re.compile(r"打印日期：(\d{4}-\d{2}-\d{2})")
_NOISE = re.compile(r"^(第\d+页|发生日期 |===|交易市场 )")


@dataclass(frozen=True, slots=True)
class FlowRow:
    day: date
    kind: str
    code: str
    name: str
    quantity: int
    price: float
    amount: float
    fee: float
    stamp: float
    transfer_fee: float
    balance: float

    @property
    def category(self) -> str:
        if self.kind.startswith(EXCLUDED_PREFIXES):
            return "excluded"
        return CATEGORY.get(self.kind, "other")


@dataclass(frozen=True, slots=True)
class HoldingRow:
    market: str
    code: str
    name: str
    quantity: int
    price: float
    cost: float
    value: float


@dataclass(slots=True)
class Statement:
    printed: date | None = None
    start: date | None = None
    end: date | None = None
    summary: dict[str, float] = field(default_factory=dict)
    holdings: list[HoldingRow] = field(default_factory=list)
    rows: list[FlowRow] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def kept_rows(self) -> list[FlowRow]:
        """剔除天天宝/现金宝后的流水，按时间正序。"""
        return [r for r in reversed(self.rows) if r.category != "excluded"]


def _day(text: str) -> date:
    return date(int(text[:4]), int(text[4:6]), int(text[6:8]))


def parse_text(text: str) -> Statement:
    st = Statement()
    section = ""
    last_flow: int | None = None
    last_holding: int | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if m := _PRINTED.search(line):
            st.printed = date.fromisoformat(m.group(1))
        if m := _PERIOD.search(line):
            g = [int(x) for x in m.groups()]
            st.start, st.end = date(g[0], g[1], g[2]), date(g[3], g[4], g[5])
        for key, value in _FIELD.findall(line):
            st.summary[key] = float(value)
        if line.startswith("汇总股票资料"):
            section = "holdings"
            continue
        if line.startswith("资金流水明细"):
            section = "flows"
            continue
        if _NOISE.match(line):
            continue
        if section == "holdings":
            if m := _HOLDING.match(line):
                st.holdings.append(HoldingRow(
                    m.group(1), m.group(2), m.group(3), int(m.group(4)),
                    float(m.group(5)), float(m.group(6)), float(m.group(7)),
                ))
                last_holding = len(st.holdings) - 1
            elif last_holding is not None:
                h = st.holdings[last_holding]
                st.holdings[last_holding] = HoldingRow(
                    h.market, h.code, h.name + line, h.quantity, h.price, h.cost, h.value
                )
        elif section == "flows":
            if m := _FLOW.match(line):
                st.rows.append(FlowRow(
                    _day(m.group(1)), m.group(2), m.group(3) or "", m.group(4) or "",
                    int(m.group(5)), float(m.group(6)), float(m.group(7)), float(m.group(8)),
                    float(m.group(9)), float(m.group(10)), float(m.group(11)),
                ))
                last_flow = len(st.rows) - 1
            elif last_flow is not None and st.rows[last_flow].code:
                # 证券名称过长时 PDF 会折到下一行
                r = st.rows[last_flow]
                st.rows[last_flow] = FlowRow(
                    r.day, r.kind, r.code, r.name + line, r.quantity, r.price, r.amount,
                    r.fee, r.stamp, r.transfer_fee, r.balance,
                )
            else:
                st.issues.append(f"无法识别的流水行：{line}")
    st.issues.extend(check_balance_chain(st.rows))
    st.issues.extend(check_holdings(st))
    return st


def check_balance_chain(rows: list[FlowRow]) -> list[str]:
    """流水按时间倒序；逐行验证余额连续。"""
    issues = []
    chronological = list(reversed(rows))
    for prev, cur in zip(chronological, chronological[1:], strict=False):
        if abs(prev.balance + cur.amount - cur.balance) > 0.011:
            issues.append(
                f"余额不连续：{cur.day} {cur.kind} {cur.code} 期望 "
                f"{prev.balance + cur.amount:.2f}，对账单 {cur.balance:.2f}"
            )
    return issues


def check_holdings(st: Statement) -> list[str]:
    issues = []
    total = 0.0
    for h in st.holdings:
        total += h.value
        if abs(h.quantity * h.price - h.value) > 0.05:
            issues.append(f"持仓市值不符：{h.code} {h.quantity}×{h.price}≠{h.value}")
    expected = st.summary.get("证券市值")
    if expected is not None and abs(total - expected) > 0.05:
        issues.append(f"持仓合计 {total:.2f} ≠ 证券市值 {expected:.2f}")
    return issues


def read_pdf(path: Path) -> Statement:
    import pdfplumber

    with pdfplumber.open(path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    return parse_text(text)


def implied_opening(st: Statement) -> dict[str, int]:
    """期末持仓 − 期内净买入；非 0 说明有期初持仓或流水之外的成交。"""
    net: dict[str, int] = {}
    for r in st.kept_rows():
        if r.kind in TRADE_TYPES:
            net[r.code] = net.get(r.code, 0) + (r.quantity if r.kind == "证券买入" else -r.quantity)
    held = {h.code: h.quantity for h in st.holdings}
    out = {}
    for code in sorted(set(net) | set(held)):
        diff = held.get(code, 0) - net.get(code, 0)
        if diff:
            out[code] = diff
    return out


def export(st: Statement, root: Path) -> dict[str, Path]:
    """写入 ``root``（私有仓库的 portfolio/eastmoney/）：流水、成交、持仓快照、账户汇总。"""
    import pandas as pd
    import yaml

    root.mkdir(parents=True, exist_ok=True)
    kept = st.kept_rows()
    flows = pd.DataFrame([
        {
            "date": r.day.isoformat(), "category": r.category, "kind": r.kind, "code": r.code,
            "name": r.name, "quantity": r.quantity, "price": r.price, "amount": r.amount,
            "fee": r.fee, "stamp": r.stamp, "transfer_fee": r.transfer_fee,
        }
        for r in kept
    ])
    trades = flows[flows["category"] == "trade"].assign(
        side=lambda f: f["kind"].map(TRADE_TYPES)
    )
    as_of = (st.end or date.today()).isoformat()
    holdings = pd.DataFrame([
        {
            "code": h.code, "name": h.name, "market": h.market, "quantity": h.quantity,
            "price": h.price, "cost": h.cost, "value": h.value,
            "pnl": round((h.price - h.cost) * h.quantity, 2),
            "pnl_pct": round(h.price / h.cost - 1.0, 4) if h.cost else None,
        }
        for h in st.holdings
    ])
    paths = {
        "flows": root / "flows.csv",
        "trades": root / "trades.csv",
        "holdings": root / f"holdings_{as_of}.csv",
        "account": root / f"account_{as_of}.yaml",
    }
    flows.to_csv(paths["flows"], index=False, encoding="utf-8-sig")
    trades.to_csv(paths["trades"], index=False, encoding="utf-8-sig")
    holdings.to_csv(paths["holdings"], index=False, encoding="utf-8-sig")
    excluded = [r for r in st.rows if r.category == "excluded"]
    account = {
        "source": "东方财富股票明细对账单 PDF",
        "period": [str(st.start), str(st.end)],
        "printed": str(st.printed),
        "summary": st.summary,
        "cash_note": "资金可用 = 资金余额 + 天天宝余额（券商口径），作为可用现金",
        "flow_rows": {
            "total": len(st.rows), "kept": len(kept), "excluded_tiantianbao": len(excluded)
        },
        "totals": {
            "bank_in": round(sum(r.amount for r in kept if r.category == "transfer_in"), 2),
            "bank_out": round(sum(r.amount for r in kept if r.category == "transfer_out"), 2),
            "dividend_net": round(
                sum(r.amount for r in kept if r.category in ("dividend", "dividend_tax")), 2
            ),
            "fees": round(sum(r.fee + r.stamp + r.transfer_fee for r in kept), 2),
        },
        "checks": {"balance_chain_and_holdings": st.issues or "OK"},
        "reconcile_opening_or_unlisted": implied_opening(st),
    }
    paths["account"].write_text(
        yaml.safe_dump(account, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return paths
