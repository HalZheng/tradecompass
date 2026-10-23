# Handoff — TradeCompass（2026-10-23 交接版，权威文档）

> 接手的 agent：**完整读完本文件 §1–§10 再动代码**。与本文冲突的旧文档（PLAN.md、README.md、
> 附录 A、docs/STATE_MACHINE.md）一律以本文为准。私有信息（真实持仓、对账单、研究报告）在私有仓库
> `tradecompass-private`，见其 `HANDOFF_PRIVATE.md`。

## 1. 项目核心需求（不可偏离）

1. **产品是什么**：个人 A 股投资者的“纪律执行器”。**每个交易日收盘后**，基于最新行情 + **真实持仓** +
   **过往交易记录**，滚动生成**下一交易日**可直接执行的操作提醒：买什么、卖什么、什么价位（条件单）、多少股。
   周五生成的就是下周一的提醒——**不是周报**。
2. **本质是赚钱的量化策略**：用成熟的 A 股量化方法（横截面多因子、分板块建模、无幸存者偏差股票池、
   样本外检验、真实成本）。不是指标拼盘，也不是“只测一个买点”。
3. **滚动决策**：每天重算；当天决策只能用当天及以前的数据、交易记录（冷却期/持有期/清理起始日）和当前持仓。
   回测与实盘使用**同一个逐日 `step` 逻辑**（`research/rolling.py`）。
4. **不同行业秉性不同**：因子有效性与操作节奏按板块组区分（防御 / 消费医药 / 周期制造 / 科技成长），不能一刀切。
5. **ETF 的角色**（`research/etfs.py::etf_roles`）：**底仓**（510300/518880/511090）只做再平衡；
   **稳健**（宽基、红利/自由现金流、海外宽基）长期持有、不做波段、不新增；**波段**（行业/主题）属于波段仓，
   但**规则尚未验证**（已测的动量轮动无效），暂行规则见 §4。行业 ETF 仅在对应行业指数数据不全时作参考；
   目前板块指数由成分股合成、数据完整，不需要参考行业 ETF。
6. **每日结论必须由脚本可复现生成**：`tradecompass plan` 不调用任何 LLM，同样数据同样输出。
   LLM/agent 只做研究、工程、数据异常处理，**不参与每日买卖判断**，也不能往报告里写手工判断文字。
7. **用户约束**：上班族，只能收盘后/午休操作；容易情绪化，历史上“赚小钱就卖、亏的拿着、分散、补仓”。
   因此：换仓分批（每日最多新买 3 只）、深套不低点一次性割（反弹条件单）、不补仓摊低、提醒措辞直接可执行。
8. **隐私**：公开仓库只放代码、模板、测试和不含个人信息的文档；真实持仓/对账单/研究产物只在私有仓库；
   对账单 PDF 原件（含证件号）只放 TC_HOME `data/`（不进任何 git）。

## 2. 系统现状一览（数据流）

```
行情/估值（baostock / 新浪 / 东财 / 腾讯，见 §7）
  └─ research/data.py → TC_HOME/data/research/{constituents, bars, valuation, etf, index, industry}
       └─ research/panel.py  日期×代码宽表（无幸存者偏差：每月末沪深300+中证500 历史成分，含退市）
            └─ research/factors.py  候选因子（反转/低波/特异波动/换手/估值/ROE/价量相关…）
                 └─ research/rolling.py  分板块滚动引擎（打分、名额、行业层倾斜、逐日 step、回测）
券商对账单 PDF ─ account/statement_pdf.py → TC_HOME/portfolio/eastmoney/{flows,trades,holdings,account}
                 └─ account/behavior.py 交易行为体检
                      └─ research/weekly_plan.py + cli `plan` → TC_HOME/reports/plan/<下一交易日>.md
```

旧链路（`daily` 的 B1/B2/S1/S2、`backtest signals`）仍可运行，但**已降级为历史对照**，见附录 A。

## 3. 每日运行流程

```powershell
# 0) 有新成交时：从东方财富导出“股票明细对账单”PDF 后导入（会校验并覆盖期初持仓）
tradecompass import-statement <对账单.pdf> --home ..\tradecompass-private
# 1) 收盘后更新数据（个股日线+估值；ETF/指数见 §7）
tradecompass research sync --home ..\tradecompass-private --source fast --workers 4
# 2) 生成下一交易日提醒（只写文件，不推送）
tradecompass plan --home ..\tradecompass-private
```

研究命令：`research ic`（因子体检+事件研究）、`research portfolio`（月度组合）、`research rolling`（滚动引擎回测）。

## 4. 当前默认策略（改任何一项都必须先回测，见 §10）

- **资金结构**：底仓 50%（510300 50% / 518880 30% / 511090 20%，来自 `universe.yaml` 的 `target`），个股 50%。
- **股票池**：当时的沪深300+中证500 成分；剔除停牌、ST、上市不足 120 日；一手买不起的不选。
- **个股：4 个板块组（`research/sleeves.py` 证监会行业→8 板块→4 组）**

  | 组 | 因子（文献方向） | 调仓间隔 | 缓冲 | 最短持有 | 止损 |
  |---|---|---|---|---|---|
  | 防御（金融/公用交运/地产基建） | vol60, turn60, ep, bp, roe | 60 日 | ×3 | 40 日 | 无 |
  | 消费医药 | vol60, turn60, ep, roe, pv_corr20, rev20 | 20 日 | ×2 | 10 日 | 无 |
  | 周期制造 | vol60, ivol20, ep, bp, pv_corr20 | 20 日 | ×2 | 10 日 | 无 |
  | 科技成长 | rev20, pv_corr20, ivol20, turn20 | 20 日 | ×2 | 10 日 | 无 |

  因子权重 = 先验等权 50% + 滚动 ICIR 50%（只用**已实现**的月度 IC，近 36 个月）。12 个名额按各组可选股票数分配。
- **行业层**：每组成分股等权指数在 MA120 上 → 名额 ×1.3，下 → ×0.7（`trend_tilt`，默认开启）。
- **大盘层**：沪深300 状态机**只展示不执行**（实测控仓会追涨杀跌）。
- **每日提醒 `plan`**：
  - 持仓在所属组“名额×缓冲”内 → 保留；否则 → **存量清理**：反弹到 MA10 卖一半、MA20 卖完（条件单）；
    进入清理满 10 个交易日仍未到位 → “收盘前卖一半”（起始日在 TC_HOME `portfolio/plan_state.json`）。
  - 新买：按组排名，每日最多 3 只，限价 ≤ 收盘×1.02，近 20 个交易日卖出过的不回补，现金不足列候补。
  - 底仓：列出与目标的差额，清理回笼资金一半按差额补。
  - 稳健 ETF：持有，不清理、不新增；报告显示其相对 MA20 位置，不低位卖出。
  - 波段 ETF（**暂行规则，待研究替换**）：站上 MA20 → 持有观察、不新增；跌破 MA20 → 按存量清理规则反弹减仓。
  - 持仓差异人工确认：TC_HOME `portfolio/confirmations.yaml`（`代码: {quantity, note}`），数量与快照一致时
    报告显示“已确认”而非核对提示。
- **ETF 动量轮动**：代码保留在 `research/etf_rotation.py` 仅作研究对照，**不进入 plan**。

## 5. 研究证据（2019-01 → 2026-10，必须知道的数字）

- **旧 B1/B2 在大样本无超额**：B2 硬条件 5922 次 20 日超额 −0.47%（t −1.2）；B1 不显著。
- **分板块 IC**：低波/估值在防御类 t 3–4，在科技成长失效；科技成长只有反转/价量有效 → 支持分板块。
- **滚动引擎 + 趋势倾斜，12 个起点（2019 每月初）**：年化 8.6%（6.0%–11.5%），
  相对股票池等权超额 +2.6%（**12/12 为正**，最差 +0.7%），夏普 0.57 vs 0.31，最大回撤 −18% vs −32%。
- **优势是回撤小，不是高收益**：2020–2021 牛市明显跑输股票池；样本外（2023+）规则设计时看过全样本，偏乐观。
- **实测否定（不要再试同样的东西）**：
  1. 反转为主的板块加止损（科技 10 日+止损 12%：每笔 +0.2%）；
  2. 沪深300 状态机控总仓（年化 10.6% → 0.0%）；
  3. ETF 动量轮动（快参数/慢参数年化 ≈0%、回撤 −52%）；与个股引擎按 20–50% 组合只会稀释
     （只否定了这两版动量轮动，波段 ETF 的其他规则仍是开放研究，见 §9）；
  4. 板块组反转倾斜（样本内超额 −8.3%）。
- 研究脚本与结果文本在私有仓库 `scripts/`、`reports/research/`，可直接复跑。

## 6. 代码地图（新增部分）

| 模块 | 职责 | 测试 |
|---|---|---|
| `research/data.py` | 历史成分、日线、估值、行业、ETF/指数存取；baostock 超时重连；腾讯补数 | — |
| `research/fast_source.py` | 新浪日线 + 东财估值（多进程） | `test_research.py` |
| `research/panel.py` | 宽表面板、复权、涨停买不进、可选性 | `test_research.py` |
| `research/factors.py` / `evaluate.py` / `events.py` / `portfolio.py` | 因子、IC/分组、事件研究、月度组合 | `test_research.py` |
| `research/sleeves.py` | 行业 → 板块 → 组 | `test_rolling.py` |
| `research/rolling.py` | 滚动引擎、名额、趋势倾斜、回测报告 | `test_rolling.py`（含无前视测试） |
| `research/etfs.py` / `etf_rotation.py` | ETF 名单与角色（底仓/稳健/波段）、轮动（对照） | `test_etf_rotation.py`、`test_weekly_plan.py` |
| `research/weekly_plan.py` | 每日提醒的计算与渲染（文件名历史原因，实际是每日） | `test_weekly_plan.py` |
| `account/statement_pdf.py` | 东财对账单 PDF 解析、余额链校验、导出 | `test_statement_pdf.py` |
| `account/behavior.py` | 已实现盈亏、胜率、盈亏比、同日买卖 | `test_weekly_plan.py` |
| `cli.py` | `research sync/ic/portfolio/rolling`、`import-statement`、`plan` | — |

## 7. 数据源与已知坑（踩过的）

- **baostock**：唯一能取历史成分股；慢（4–40 秒/只）、**不能并发**（第二个会话会卡死）、会整体不可用；
  已加 60 秒 socket 超时 + 重新登录。
- **新浪 `stock_zh_a_daily`**：快，但依赖 py_mini_racer（V8）**不是线程安全**，只能多进程；无已退市标的。
- **东财 `stock_value_em`**：估值统一用它（与 baostock PB/市现率口径不同，不可混用）；本机 push2his 被代理拦截。
- **腾讯**：baostock 不可用时补 ETF/指数当日 K 线（`append_from_tencent`）；部分 ETF 无数据（如 518880、159501）。
- **ETF 份额拆分**：原始价会跳变 50% 以上，preclose 已调整；持仓必须按复权单位记账。
- **前视陷阱**：`Panel.eligible()` 含“次日是否涨停”，**决策时不能用**（滚动引擎自己构造可选性）。
- **日历**：`TradingCalendar.next_trading_day/previous_trading_day` 是**含当日**语义。
- **Windows**：非中文控制台已在 CLI 启动时改为 UTF-8；PowerShell 管道给 python 的中文会乱码，研究脚本写成文件。
- **行业分类**是当前快照（非历史时点），有轻微前视。

## 8. 已知局限 / 技术债（接手先读）

1. 旧 `daily`（B1/B2/S1/S2）与新 `plan` 并存，口径冲突；用户日常只看 `plan`。
2. `plan` 只从对账单 PDF 知道持仓；当天新成交需重新导出对账单。对账单最后一天的成交可能未进流水
   （报告会自动提示差异）。
3. 底仓再平衡只给差额表，没有生成具体买单。
4. 没有提醒落盘为结构化数据，也没有事后追踪（建议执行率、提醒后 T+5/T+20 表现）。
5. 数据更新要分别跑个股（sync fast）与 ETF/指数（baostock 或腾讯），没有一键命令和完整性检查。
6. `research/weekly_plan.py` 命名过时；`render_plan` 用 `dict[str, object]` 传参，类型较松。
7. 牛市跑输：缺少成长/盈利质量类因子（需要财报公告日对齐的 PIT 数据）。

## 9. 下一步路线（按优先级；每项都要有测试，跑通 §11 验证）

1. **提醒结构化落盘 + 事后追踪**：`plan` 同时写 `reports/plan/<日>.json`；新命令回填每条建议 T+1/T+5/T+20
   表现与是否执行（对比后续对账单）。验收：连续多日可统计“建议命中率/遵从率”。
2. **一键更新**：`tradecompass update`（个股 fast sync + ETF/指数 baostock→腾讯降级 + 完整性检查：
   持仓与当前成分股必须有最新交易日 K 线，否则 plan 拒绝生成并说明原因）。
3. **统一入口**：`daily` 改为调用 `plan` 的逻辑或明确标注 legacy；飞书推送保持关闭，除非用户明确同意。
4. **底仓再平衡买单**：按差额与可用现金生成 510300/518880/511090 的具体股数与限价（ETF 费率）。
5. **波段 ETF 规则研究**：替换 §4 的暂行 MA20 规则。候选：类似个股板块组的反转/价量规则、估值或价格
   分位、趋势过滤；必须 2019–2022 决策、2023+ 验证、12 起点，与“持有不动”和“暂行规则”对照。
   通过后才改 `weekly_plan.etf_action` 并允许波段 ETF 新增；不通过就保持暂行规则并在报告注明。
6. **持仓更新便捷化**：支持手工 `positions.yaml` 或券商 CSV 作为 PDF 之外的持仓来源，导入后自动对账。
7. **研究（必须样本外）**：PIT 财报因子（ROE 稳定性、盈利增速）补牛市短板；PIT 行业分类。
   规则：2019–2022 决策、2023+ 只验证；12 起点检验；不过关就不合入默认。
8. Windows 任务计划程序在收盘后自动运行 update + plan（只写文件）。

## 10. 给后续 agent 的硬约束

**必须**
- 改策略规则（因子、权重、节奏、倾斜、清理规则、名额）前：跑 `research rolling` 与私有仓库
  `scripts/start_offsets.py`（12 起点），样本内外分开报告，并更新本文 §4/§5。
- 新增任何决策逻辑都要有“改未来数据、过去决策不变”的无前视测试。
- 报告中的事实性文字必须由数据计算；不允许写死股票名、数量、日期或主观判断。
- 保持 `pytest`、`ruff check src tests`、`mypy src`（strict）全绿。
- 涉及个人数据的文件只进私有仓库；推公开仓库前检查 diff 中没有持仓、金额、证件号、账户号。

**禁止**
- 让 LLM 参与每日买卖结论；把 agent 的分析文字写进 `plan` 报告。
- 回到“49 只自选 + 单一买点择时”的路线，或重新放宽 B2 / 启用紧移动止损（旧结论，见附录 A）。
- 重新启用：反转板块止损、沪深300 状态机控仓、已测的两版 ETF 动量轮动——除非有新的、样本外的证据。
- 把稳健 ETF 当波段做，或在波段规则验证前让波段 ETF 进入新买清单。
- 在全样本上网格搜索挑参数，或只报最好的一个起点。
- 并发访问 baostock；多线程调用新浪接口。
- 把 TC_HOME（私有仓库）内容复制进公开仓库；提交 `.env` 或对账单原件。

## 11. 运行与验证

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e ".[dev]"
.\.venv\Scripts\python -m pytest -q          # 基线：241 passed
.\.venv\Scripts\python -m ruff check src tests
.\.venv\Scripts\python -m mypy src
```

新机器首次需要 `research sync`（默认 baostock 全字段，约 1–4 小时；或 `--source fast` 加速，
退市股仍需 baostock 补）与 ETF 数据同步，详见私有仓库 `HANDOFF_PRIVATE.md`。

---

## 附录 A：旧 B1/B2 信号系统（2026-10-22，已降级，仅供参考）

> 以下为上一阶段交接内容，**与上文冲突时以上文为准**。其中“下一步任务”已被 §9 取代。

### A1. 一句话状态（旧）

当前是「多因子信号系统」的 **I1 完成态**：数据 → 指标 → 因子 → B1/B2/S1/S2 →
仓位/历史过滤 → 每日报告 → 回测 → B2 归因，全链路已打通。

**结论先行：还不能实盘。**
5 年/2 年 B2 入场过滤后 PF 已转正（1.24 / 完整系统 1.14、1.31），
但最近 1 年仍亏损（PF 0.05），且远输等权买入持有；B1 仍是负期望。

### A2. 上一阶段完成的模块

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

### A3. 旧信号逻辑

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

### A4. 旧回测结论（49 只自选池）

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

### A5. 旧下一步任务（已被 §9 取代）

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

### A6. 旧运行方式

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

旧基线：`212 passed`（当前基线见 §11）。

### A7. 公开仓库与私有数据（仍然有效）

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

### A8. 旧约束（仍有效的已并入 §10）

1. **LLM 不参与买卖判断**；只做解释、归因、工程。
2. **回测是验证工具，不是产品**；不要回到“只测一个买点”的老路。
3. **不要重新放宽 B2 硬条件**；放宽版实测更差。
4. **不要再启用紧移动止损**；它会砍掉右尾。
5. **不要在 1 年样本仍亏的情况下放大仓位或宣称稳定 alpha**。
6. 每次改信号必须同步：定义（PLAN/DECISIONS）、单元测试、回测/随机对照。
7. 私有数据只放 `local/` / `.workbuddy/` / 私有包，绝不提交到公开仓库。
