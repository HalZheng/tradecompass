"""对账单 PDF 文本解析：余额链校验、折行名称、天天宝剔除、持仓合计。"""

from __future__ import annotations

from datetime import date

from tradecompass.account.statement_pdf import implied_opening, parse_text

SAMPLE = """第1页，共1页
打印日期：2026-10-23 16:31:20
查询区间：2026/10/01-2026/10/23
证件号码：000000000000000000
证券市值(RMB)： 17890.00 证券市值(HKD)： 0.00
总资产(RMB)： 18000.00 总资产(HKD)： 0.00
资金余额(RMB)： 10.00 资金余额(HKD)： 0.00
资金可用(RMB)： 110.00 资金可用(HKD)： 0.00
汇总股票资料
交易市场 证券代码 证券名称 持仓数量 市价 成本价 证券市值
沪市A股 510300 沪深300ETF华泰柏瑞 1000 4.180 4.000 4180.00
沪市A股 601398 工商银行 1500 8.320 8.000 12480.00
深市A股 000099 中信海直 100 12.300 12.000 1230.00
资金流水明细(2026/10/01-2026/10/23)
发生日期 买卖类别 证券代码 证券名称 成交数量 成交价格 总发生金额 手续费 印花税 过户费 资金余额
20261022 红利入账 000099 中信海直 0 0.0000 10.00 0.00 0.00 0.00 10.00
20261022 证券买入 510300 沪深300ETF华泰柏 1000 4.0000 -4000.72 0.72 0.00 0.00 0.00
瑞
20261022 天天宝赎回 0 0.0000 4000.62 0.00 0.00 0.00 4000.72
20261020 证券买入 601398 工商银行 1500 8.0000 -12005.12 5.00 0.00 0.12 0.10
20261020 银行转证券 0 0.0000 12000.00 0.00 0.00 0.00 12005.22
20261019 OTC现金宝调账入 0 0.0000 5.22 0.00 0.00 0.00 5.22
"""


def test_parses_summary_holdings_and_flows() -> None:
    st = parse_text(SAMPLE)
    assert st.issues == []
    assert st.start == date(2026, 10, 1) and st.end == date(2026, 10, 23)
    assert st.summary["资金可用"] == 110.0
    assert [h.code for h in st.holdings] == ["510300", "601398", "000099"]
    assert len(st.rows) == 6
    etf = next(r for r in st.rows if r.code == "510300")
    assert etf.name == "沪深300ETF华泰柏瑞"
    assert etf.fee == 0.72


def test_excludes_tiantianbao_and_cash_sweeps() -> None:
    st = parse_text(SAMPLE)
    kept = st.kept_rows()
    assert all(not r.kind.startswith(("天天宝", "OTC现金宝")) for r in kept)
    assert [r.kind for r in kept] == ["银行转证券", "证券买入", "证券买入", "红利入账"]


def test_broken_balance_chain_is_reported() -> None:
    broken = SAMPLE.replace("-12005.12 5.00", "-12006.12 5.00")
    assert any("余额不连续" in issue for issue in parse_text(broken).issues)


def test_holding_total_mismatch_is_reported() -> None:
    broken = SAMPLE.replace("证券市值(RMB)： 17890.00", "证券市值(RMB)： 17000.00")
    assert any("持仓合计" in issue for issue in parse_text(broken).issues)


def test_implied_opening_flags_positions_without_trades() -> None:
    assert implied_opening(parse_text(SAMPLE)) == {"000099": 100}
