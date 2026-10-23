# Handoff — TradeCompass 当前阶段

> 给新拉取仓库的 agent：**先读本文件，再读 [PLAN.md](PLAN.md)、[README.md](README.md)、
> [docs/STATE_MACHINE.md](docs/STATE_MACHINE.md)**。
> 私有配置、真实持仓、历史决策与对话不在公开仓库，见 §8。

## 1. 一句话状态

当前是「多因子信号系统」的 **I1 完成态**：数据 → 指标 → 因子 → B1/B2/S1/S2 →
仓位/历史过滤 → 每日报告 → 回测 → B2 归因，全链路已打通。

**结论先行：还不能实盘。**
5 年/2 年 B2 入场过滤后 PF 已转正（1.24 / 完整系统 1.14、1.31），
但最近 1 年仍亏损（PF 0.05），且远输等权买入持有；B1 仍是负期望。

## 2. 本轮已经完成

| 层 | 内容 | 关键文件 |
|---|---|---|
| 数据 | 不复权日线 + 复权因子 + 市场指数 + 行业指数；baostock 同步 | `src/tradecompass/data/` |
| 指标 | MA/KDJ/ATR/RSI6/12/24/MACD、价格位置 60/250、量比 5/20、量价方向、周线全套 | `src/tradecompass/indicators/` |
| 因子 | 每个标的每个交易日一个 `FactorSnapshot`（日线/周线/行业/大盘） | `src/tradecompass/factors/engine.py` |
| 信号 | B1/B2/S1/S2 加权打分 + regime/行业动态门槛 + 仓位/历史否决 | `src/tradecompass/signals/engine.py` |
| 每日 | `daily`：市场 → 行业 → 行动清单 → Markdown/JSON → 飞书｜预览 | `src/tradecompass/signals/daily.py` |
| 回测 | 独立信号回测：次日开盘、真实费率、T+1、涨跌停、滑点、随机对照 | `src/tradecompass/signals/backtest.py` |
| 归因 | B2 MFE/MAE/R 倍数、退出原因、入场因子分组 | `src/tradecompass/signals/b2_analysis.py` |
| 账户 | 交割单导入、期初持仓快照、交易历史/冷却期 | `src/tradecompass/account/` |

## 3. 当前信号逻辑（不要凭感觉改）

### B1 回调买点（动态）

- 绝对否决：`RISK_OFF`/`PANIC` 不新开仓；行业指数在 MA20 下方；250 日位置 ≥95%；
  初始止损距离 > `max_stop_pct`。
- 其余全部加权：周线趋势 25、周线超卖 25（`jw_pct` 自身两年分位 + 绝对 JW）、
  日线企稳 20、位置 10、量能 10、行业/大盘 10。
- 动态门槛：`RISK_ON` 55 / `NEUTRAL` 65；行业强 −3、弱 +5。

### B2 突破买点（严格入场 + 静态止损）

- 硬条件：收盘突破前 60 日最高、量比 20 ≥1.5、MACD 柱 >0、周线非空、
  大盘非 `RISK_OFF/PANIC`、行业在 MA20 上方、RSI6 ≤80、250 日位置 <95%。
- 打分里：量比 >3 只给 8 分（极端放量=衰竭）；250 日位置 50–70% 加分、
  70–85% 降权；行业 strong 加分从 6 降到 2。
- **不要重新放宽 B2 的硬条件**：放宽版 PF 从 0.80 掉到 0.40（D033）。

### S1 趋势卖点（动态）

- 纯打分：周线高位转弱 25 + 周线趋势破坏 25 + 日线走弱 20 + 位置/量能 15 +
  **早期亏损 15（−5% 即介入）** + 行业 5。
- 动态门槛：`RISK_ON` 70 / `NEUTRAL` 60 / `RISK_OFF` 50 / `PANIC` 45；
  行业强 +3、弱 −5。报告显示“置信度/门槛”（如 73/40）。

### S2 风控

- 硬风控：浮亏 ≤−10%、跌破前 10 日最低且量比 ≥1.5、单日 ≤−7% 且放量。
- 永远推送，优先于 S1；S1 可以在 −5% 提前介入。

### 移动止损

- **默认关闭**。实测 +0.5R 保本版 PF 0.63、松版 0.53，都会砍掉右尾。
- 参数保留：`breakeven_at_r`、`lock_half_r_at_r`、`trail_at_r`、`trail_atr_mult`；
  研究时用 `backtest b2 --breakeven-r ...` 打开。

## 4. 当前回测结论（必须知道）

口径：49 只自选、10 万初始资金、最多 5 仓、次日开盘、真实费率、T+1、
涨跌停限制、滑点 0.1%；随机对照使用同样的退出/仓位/行业与大盘准入。

| 样本 | 版本 | 笔数 | PF | 总收益 | 回撤 | 资金占用 |
|---|---|---|---|---|---|---|
| 5 年 | B2 过滤 + 静态止损（当前） | 66 | **1.14** | **+2.8%** | 13.1% | 21.5% |
| 2 年 | 同上 | 36 | **1.31** | **+3.2%** | 13.0% | 34.6% |
| 1 年 | 同上 | 17 | 0.05 | **−5.5%** | 6.7% | 10.8% |
| 5 年 | 等权买入持有 | — | — | +34.1% | 15.8% | 55.1% |
| 5 年 | 随机入场 3 组 | 36–49 | 0.00 | −17.4% / −3.4% / +13.0% | — | — |

B2 单独：5 年 51 笔、胜率 17.6%、PF **1.24**、平均 **+100.16 元/笔**。
B1 单独：5 年 15 笔、平均 −127/笔；2 年 13 笔、平均 −103/笔——**仍是负期望**。

**不要把 5 年/2 年转正当成稳定 alpha**：1 年仍亏，过滤条件有市场阶段依赖；
自选池是现在挑出来的，存在幸存者偏差；回测从空仓开始，不是当前持仓复现。

## 5. 下一步任务（按优先级）

1. **B1 归因**：用 `b2_analysis` 的思路做 B1 的 MFE/MAE 与入场分组；
   如果 5 年仍 <1，把 B1 从默认买点移除，避免拖累组合。
2. **部分止盈 / 时间止损**：+2R 卖 1/3~1/2，或 20 日未到 +0.5R 退出；
   目标是保护浮盈但不砍右尾。**不要再用紧移动止损。**
3. **滚动样本外验证**：现在的 B2 过滤条件来自 5 年全样本；
   用 2021–2023 训练、2024–2026 验证，或逐年滚动，确认不是过拟合。
4. **信号事后追踪**：每日信号落盘后回填 T+1/T+5/T+20 收益与命中率；
   按因子组合统计，而不是凭感觉调参。
5. **盘中条件单提醒**：只对当日行动清单 + 持仓做价位轮询，不漏报、不刷屏。
6. **调度与周报**：`daily --push` 可在收盘后定时跑；周报看遵从率、信号表现、持仓结构。

## 6. 运行与验证

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e ".[dev]"

# 个人 TC_HOME 在仓库外；公开仓库只有模板
tradecompass init --home C:\Users\<you>\Documents\tradecompass-private
tradecompass doctor --offline --home C:\Users\<you>\Documents\tradecompass-private
tradecompass data sync --home C:\Users\<you>\Documents\tradecompass-private

# 每日信号
tradecompass daily --home C:\Users\<you>\Documents\tradecompass-private

# 单票因子
tradecompass factors show 600000 --home C:\Users\<you>\Documents\tradecompass-private

# 回测
tradecompass backtest signals --home C:\Users\<you>\Documents\tradecompass-private `
  --start 2021-10-22 --end 2026-10-22 --random-seeds 3
tradecompass backtest b2 --home C:\Users\<you>\Documents\tradecompass-private `
  --start 2021-10-22 --end 2026-10-22
```

验证门槛：

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python -m ruff check src tests
.\.venv\Scripts\python -m mypy src
```

当前基线：`212 passed`，ruff/mypy strict 全绿。

## 7. 公开仓库与私有数据

公开仓库：<https://github.com/HalZheng/tradecompass>，只有一个干净的 root commit，
只包含代码、模板、测试、PLAN/README/HANDOFF。
**不包含**真实自选池、持仓、交割单、账户金额、真实回测产物、历史对话/决策原文。

> 安全：本机旧 git 历史（`master`）包含私有文件，**不要直接把 `master` push 到公开仓库**。
> 公开仓库由 `public-export/` 的独立 root commit 生成。

私有信息单独放在**私有仓库**：<https://github.com/HalZheng/tradecompass-private>。

私有仓库根目录就是 `TC_HOME`：

| 目录/文件 | 内容 | 进 git |
|---|---|---|
| `config/` | 自选池、期初持仓、风控、系统配置 | 是 |
| `reports/` | 每日信号与回测报告 | 是 |
| `private_docs/` | 背景、SPEC、决策原文、历史讨论 | 是 |
| `scripts/` | 研究脚本 | 是 |
| `.workbuddy/` | agent 记忆 | 是 |
| `HANDOFF_PRIVATE.md` | 私有交接 | 是 |
| `data/` | 行情缓存 | **否**（新机器 `data sync`） |
| `logs/` | 运行日志 | **否** |
| `.env` | 飞书 webhook / key | **否**（只保留 `.env.example`） |

新机器：

```powershell
git clone https://github.com/HalZheng/tradecompass.git
git clone https://github.com/HalZheng/tradecompass-private.git

cd tradecompass
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e ".[dev]"

tradecompass data sync --home ..\tradecompass-private
copy ..\tradecompass-private\.env.example ..\tradecompass-private\.env
tradecompass doctor --offline --home ..\tradecompass-private
tradecompass daily --home ..\tradecompass-private
```

旧的离线 zip 仍可作为一次性备份，但日常跨机同步用私有仓库。

## 8. 不可偏离的约束

1. **LLM 不参与买卖判断**；只做解释、归因、工程。
2. **回测是验证工具，不是产品**；不要回到“只测一个买点”的老路。
3. **不要重新放宽 B2 硬条件**；放宽版实测更差。
4. **不要再启用紧移动止损**；它会砍掉右尾。
5. **不要在 1 年样本仍亏的情况下放大仓位或宣称稳定 alpha**。
6. 每次改信号必须同步：定义（PLAN/DECISIONS）、单元测试、回测/随机对照。
7. 私有数据只放 `local/` / `.workbuddy/` / 私有包，绝不提交到公开仓库。
