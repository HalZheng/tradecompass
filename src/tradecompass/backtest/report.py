"""回测报告（SPEC §12.7）：单文件 HTML，离线可开。

首屏是结论与核心指标，其后是权益/回撤曲线与交易明细。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import plotly.graph_objects as go
from jinja2 import Template
from plotly.subplots import make_subplots

from tradecompass.backtest.metrics import AcceptanceReport, Metrics
from tradecompass.backtest.types import BacktestResult, EquityPoint

_TEMPLATE = Template(
    """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>{{ title }}</title>
<style>
 body{font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;margin:0;background:#0f1116;color:#e6e6e6}
 .wrap{max-width:1080px;margin:0 auto;padding:28px 20px 60px}
 h1{font-size:20px;font-weight:500;margin:0 0 4px}
 .sub{color:#8b93a1;font-size:13px;margin-bottom:18px}
 .verdict{border-radius:10px;padding:14px 18px;margin:0 0 22px;font-size:14px;line-height:1.7}
 .pass{background:#10281c;border:1px solid #2f6b47;color:#8fe3b0}
 .fail{background:#2a1416;border:1px solid #7a3238;color:#f0a2a8}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px;margin-bottom:24px}
 .card{background:#171a21;border:1px solid #232833;border-radius:8px;padding:12px 14px}
 .k{color:#8b93a1;font-size:12px;margin-bottom:6px}
 .v{font-size:18px;font-weight:500}
 table{width:100%;border-collapse:collapse;font-size:13px;margin:10px 0 26px}
 th,td{text-align:right;padding:6px 8px;border-bottom:1px solid #232833}
 th:first-child,td:first-child{text-align:left}
 th{color:#8b93a1;font-weight:500}
 .neg{color:#63c98a}.pos{color:#e06c75}
 h2{font-size:15px;font-weight:500;margin:26px 0 8px;color:#c9d1d9}
</style></head><body><div class="wrap">
<h1>{{ title }}</h1>
<div class="sub">{{ subtitle }}</div>
<div class="verdict {{ 'pass' if acceptance.passed else 'fail' }}">
 <b>{{ '通过验收' if acceptance.passed else '未通过验收' }}</b><br>
 {% if acceptance.passed %}全部 {{ acceptance["items"]|length }} 项满足门槛。{% else %}
 未通过：{{ failed|join('、') }}{% endif %}
</div>
<div class="grid">
{% for label, value in cards %}<div class="card"><div class="k">{{ label }}</div><div class="v">{{ value }}</div></div>
{% endfor %}</div>
<h2>验收明细</h2>
<table><tr><th>验收项</th><th>结果</th><th>说明</th></tr>
{% for item in acceptance["items"] %}<tr><td>{{ item.name }}</td>
<td class="{{ 'neg' if item.ok else 'pos' }}">{{ 'PASS' if item.ok else 'FAIL' }}</td>
<td>{{ item.detail }}</td></tr>{% endfor %}</table>
<h2>权益与回撤</h2>{{ chart }}
<h2>交易明细{% if truncated %}（前 {{ rows|length }} 笔，共 {{ total }} 笔）{% endif %}</h2>
<table><tr><th>标的</th><th>买入</th><th>卖出</th><th>股数</th><th>买价</th><th>卖价</th>
<th>持有天</th><th>净盈亏</th><th>退出原因</th></tr>
{% for t in rows %}<tr><td>{{ t.symbol }}</td><td>{{ t.entry_date }}</td><td>{{ t.exit_date }}</td>
<td>{{ t.quantity }}</td><td>{{ '%.2f'|format(t.entry_price_raw) }}</td>
<td>{{ '%.2f'|format(t.exit_price_raw) }}</td><td>{{ t.holding_days }}</td>
<td class="{{ 'neg' if t.net_pnl >= 0 else 'pos' }}">{{ '%.0f'|format(t.net_pnl) }}</td>
<td>{{ t.exit_reason }}</td></tr>{% endfor %}</table>
<div class="sub">由 TradeCompass 生成 · 回测结果不代表未来收益 · 成本假设见 config/backtest.yaml</div>
</div></body></html>"""
)


def _chart(
    equity: tuple[EquityPoint, ...], benchmark: tuple[EquityPoint, ...] | None
) -> str:
    dates = [point.date for point in equity]
    values = [point.equity / equity[0].equity * 100 if equity[0].equity else 0 for point in equity]

    peaks: list[float] = []
    peak = 0.0
    for value in values:
        peak = max(peak, value)
        peaks.append((value - peak) / peak * 100 if peak else 0.0)

    figure = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.06,
        subplot_titles=("净值（期初 = 100）", "回撤 %"),
    )
    figure.add_trace(
        go.Scatter(x=dates, y=values, name="策略", line={"color": "#58a6ff", "width": 1.6}),
        row=1, col=1,
    )
    if benchmark:
        base = benchmark[0].equity
        figure.add_trace(
            go.Scatter(
                x=[p.date for p in benchmark],
                y=[p.equity / base * 100 if base else 0 for p in benchmark],
                name="等权买入持有",
                line={"color": "#8b93a1", "width": 1.2, "dash": "dot"},
            ),
            row=1, col=1,
        )
    figure.add_trace(
        go.Scatter(x=dates, y=peaks, name="回撤", fill="tozeroy",
                   line={"color": "#e06c75", "width": 1.0}),
        row=2, col=1,
    )
    figure.update_layout(
        template="plotly_dark", height=520, margin={"l": 50, "r": 20, "t": 40, "b": 30},
        paper_bgcolor="#0f1116", plot_bgcolor="#0f1116",
        legend={"orientation": "h", "y": 1.08, "x": 0}, font={"size": 11},
    )
    html: str = figure.to_html(full_html=False, include_plotlyjs="inline", div_id="tc-chart")
    return html


def write_report(
    path: Path,
    *,
    title: str,
    subtitle: str,
    result: BacktestResult,
    metrics: Metrics,
    acceptance: AcceptanceReport,
    benchmark: tuple[EquityPoint, ...] | None = None,
    max_rows: int = 200,
) -> Path:
    trades = list(result.trades)
    rows = sorted(trades, key=lambda t: t.net_pnl)[:max_rows]
    cards = [
        ("交易笔数", f"{metrics.trades}"),
        ("胜率", f"{metrics.win_rate:.1%}"),
        ("盈亏比", f"{metrics.avg_win / abs(metrics.avg_loss):.2f}" if metrics.avg_loss else "-"),
        ("利润因子", f"{metrics.profit_factor:.2f}"),
        ("单笔期望", f"{metrics.expectancy:.0f} 元"),
        ("总收益", f"{metrics.total_return:+.1%}"),
        ("年化", f"{metrics.annual_return:+.1%}"),
        ("最大回撤", f"{metrics.max_drawdown:.1%}"),
        ("资金占用", f"{metrics.capital_utilization:.0%}"),
        ("累计费用", f"{result.fees_total:.0f} 元"),
    ]
    html = _TEMPLATE.render(
        title=title,
        subtitle=subtitle,
        acceptance={"passed": acceptance.passed, "items": list(acceptance.items)},
        failed=[item.name for item in acceptance.failed()],
        cards=cards,
        chart=_chart(result.equity, benchmark),
        rows=rows,
        total=len(trades),
        truncated=len(trades) > len(rows),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def report_path(directory: Path, strategy_id: str, params_hash: str, today: date) -> Path:
    return directory / f"{strategy_id}_{params_hash}_{today:%Y%m%d}.html"
