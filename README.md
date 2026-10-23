# TradeCompass

A 股个人投资者的**多因子信号系统**：每个交易日收盘后，把位置/趋势/振荡/量能/周线/
行业/大盘与当前持仓放在一起，输出 B1/B2 买点、S1/S2 卖点及置信度；
只提醒高置信度、可执行的操作，不自动下单，LLM 不参与任何买卖判断。

- 方案：[`PLAN.md`](PLAN.md)
- 交接：[`HANDOFF.md`](HANDOFF.md)
- 私有规格、真实持仓与历史决策原文不随公开仓库分发；迁移说明见 `HANDOFF.md` §7。

> **2026-10-23 起的最新状态以 [`HANDOFF.md`](HANDOFF.md) 为准**：核心已改为
> “无幸存者偏差股票池上的分板块滚动多因子策略 + 每日收盘后 `tradecompass plan` 生成下一交易日操作提醒”，
> 下文描述的 B1/B2/S1/S2 `daily` 链路已降级为历史对照。

## 它是干什么的

收盘后取数 → 算日线/周线因子 → 结合行业与大盘 → 打分生成 B1/B2/S1/S2 →
用当前仓位与交易历史过滤 → 输出行动清单（参考价、止损价、建议数量、理由）→
落盘并推送到飞书。没有高置信度信号的日子，明确告诉你"今天不动"。

## 快速开始

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -e ".[dev]"

# 个人配置与数据放在仓库外，不入 git
tradecompass init --home C:\Users\<you>\Documents\tradecompass-private
tradecompass doctor --offline --home C:\Users\<you>\Documents\tradecompass-private

# 拉取不复权日线与复权因子（自选池 + market.yaml 里的指数）
tradecompass data sync --home C:\Users\<you>\Documents\tradecompass-private

# 打印某日的前复权价格与全部指标，拿去和券商软件比对
tradecompass data show 600000 --home C:\Users\<you>\Documents\tradecompass-private

# 每日信号：市场 → 行业 → B1/B2/S1/S2 → 行动清单
tradecompass daily --home C:\Users\<you>\Documents\tradecompass-private

# 查看单只标的的完整因子快照（日线/周线/行业/大盘）
tradecompass factors show 600570 --home C:\Users\<you>\Documents\tradecompass-private
```

> 没有交割单时，可在 `TC_HOME/config/opening_positions.yaml` 填期初持仓
> （股数 + 成本），`daily` 会用它与当前行情算浮盈亏；有 `import-trades` 流水后以流水为准。

> 依赖安装建议加国内镜像：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e ".[dev]"`。

## 信号回测

```powershell
# 当前 B1/B2/S1/S2 系统，5 年，含随机入场对照
tradecompass backtest signals --home local --start 2021-10-22 --end 2026-10-22
```

首轮结果（2026-10-22，见 [PLAN.md](PLAN.md) §10 与 [HANDOFF.md](HANDOFF.md)）：

| 版本 | 5 年 | 2 年 | 1 年 | 说明 |
|---|---|---|---|---|
| 静态硬条件 | PF 0.67 / −10.5% | — | PF 0.11 / −12.8% | B1 几乎不触发、S1 从未触发 |
| 动态阈值 | PF 0.80 / −8.6% | — | PF 0.11 / −13.7% | B1 3 笔、S1 6 次退出 |
| **B2 过滤 + 静态止损（当前）** | **PF 1.14 / +2.8%** | **PF 1.31 / +3.2%** | **PF 0.05 / −5.5%** | B2 PF 1.24；1 年仍亏 |
| 等权买入持有 | +34.1% | +12.x% | −6.5% | 基准 |
| 随机入场（5 年中位） | −4.2% ~ +3.7% | −8.9% ~ −3.6% | −7.5% ~ −3.4% | 同退出/仓位/准入 |

**B2 入场过滤让 5 年/2 年转正，但 1 年仍亏损，尚不是稳定 alpha，不能实盘。**
紧移动止损实测会砍掉右尾，已默认关闭。下一步：部分止盈/时间止损 + 滚动样本外验证 + B1 归因。

## 目录约定

| 类别 | 位置 | 进 git |
|---|---|---|
| 代码与通用默认值 | `src/` | 是 |
| 配置模板 | `config/examples/` | 是 |
| 个人配置、密钥、数据、报告 | `TC_HOME`（仓库外） | 否 |

`TC_HOME` 解析顺序：`--home` 参数 → 环境变量 `TC_HOME` → 仓库内 `local/`。

个人 `TC_HOME` 建议单独放在一个**私有仓库**（例如 `tradecompass-private`），
方便多台机器同步配置/报告/决策；`.env`、`data/`、`logs/` 仍排除 git 跟踪。

`init` 会把 `config/examples/` 下的模板复制到 `TC_HOME/config/`（去 `.example` 后缀），已存在的文件**不会被覆盖**。

## 当前进度

| 里程碑 | 状态 |
|---|---|
| M0 脚手架（配置、init/doctor、SQLite） | 完成 |
| M1 数据层（日线、复权因子、校验、日历、行业指数） | 完成 |
| M2 指标（MA/KDJ/RSI/MACD/ATR、完结周线、`data show`） | 完成，KDJ/MA 已用东财金标准校验 |
| M3 回测引擎（成交模拟、成本、绩效、防偏差测试） | 完成（降级为验证工具） |
| **I1 多因子信号链路（因子/信号/daily/推送）** | **完成** |
| I2 信号事后追踪（T+1/T+5/T+20 命中率） | 下一步 |
| I4 盘中条件单提醒 | 后续 |

## 使用流程

1. `init` → 填 `universe.yaml`、`risk.yaml`、`.env`
2. `data sync` → `data show` 与券商软件比对指标
3. `backtest run` → 读报告 → 通过的策略才设 `enabled: true`
4. `import-trades` → `checkup` → 按存量清理清单逐步调整
5. 每个交易日收盘后跑 `daily`（可加 `--push`），在券商 APP 按行动清单设条件单
6. 交易后导入交割单或更新持仓快照；每周复盘信号表现，更新 `market_view.yaml`

## 开发

```powershell
ruff check src tests
mypy src
pytest
```

联网测试全部标记 `@pytest.mark.network`，默认不运行（`pytest -m network` 显式启用）。
