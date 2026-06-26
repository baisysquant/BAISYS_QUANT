<p align="center">
  <img src="https://github.com/paiyuyen/Multi-factor-Quantitative-Stock-Selection-Analysis-System/raw/main/Images/logo.png" alt="LOGO" width="50%">
  <br/><br/>
  <b>百 思 量 化</b>
  <br/><br/>
  <b > 量 化 方 寸 间  ， 洞 悉 万 象 市 </b>
  <br/>  
</p>
<p align="center">
    <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/Data-AkShare-red?logo=databricks&logoColor=white" />
    <img src="https://img.shields.io/badge/Analysis-Pandas_TA-green?logo=pandas&logoColor=white" />
<img src="https://img.shields.io/badge/Performance-15_Thread_Parallel-brightgreen?logo=speedtest" />
    <br />
    <img src="https://img.shields.io/badge/MACD-Dual_Cycle_&_Momentum-ff4500?style=flat-square" />
    <img src="https://img.shields.io/badge/KDJ-Divergence_Detection-8a2be2?style=flat-square" />
    <img src="https://img.shields.io/badge/Output-Auto_Excel_Report-success?logo=microsoftexcel&style=social" />
    <img src="https://img.shields.io/badge/CCI-Professional_Tiering-7cfc00?style=flat-square" />
    <img src="https://img.shields.io/badge/Trend-MA_Bullish_Alignment-00ced1?style=flat-square" />
    
</p>
<br />

## 📖 项目简介

百思量化是一套面向 A 股的全链路量化系统，覆盖 **数据同步 → 信号预计算 → 策略回测 → 每日分析报告** 全流程。系统分为两大阶段：

### 阶段 A — 回测校准

通过 Walk-Forward 滚动窗口优化 + Grid Search 网格搜索，自动寻优 6 个核心策略参数（<font color="red">ATR 止损倍数</font>、<font color="red">Kelly 仓位比例</font>、<font color="red">基础仓位 A</font>、<font color="red">流动性否决比</font>、<font color="red">布林窄幅比</font>、<font color="red">金叉衰减天数</font>），将最优参数写入 `config.ini` 用于日常运行。

### 阶段 B — 每日分析管线

13 步流水线从数据库增量同步 K 线 → 计算技术指标 → 多门控评分 → 生成结构化 Excel 报告 → 同步结果到 PostgreSQL。

### 设计特点

- **单参数 MACD 管线** — 摒弃双周期冗余，聚焦 (12,26,9) 单参数 + ATR 波动率归一化，7 维评分维度权重可配置
- **6 道门控递进评分** — Gate 0（数据质量）→ 0.5（宏观）→ 1（信号共振）→ 2（波动率/背离）→ 3（资金流修饰）→ 4（仓位联动），Gate 5 组合级后处理
- **信号衰减模型** — 金叉 30 天半衰、背离 8 天半衰、K 线形态 10 天半衰
- **行业中性化** — 行业内百分位排名的信号校准
- **增量缓存续算** — 每日信号以 `signal_cache_{trade_date}/{symbol}.parquet` 按只写入，中断后可自动续算已完成的股票
- **全量配置化** — 所有参数收口在 `config.ini`，支持 `ENC:` 加密敏感字段，Pydantic 自动类型校验

### 数据源

| 数据 | 来源 | 方式 |
|------|------|------|
| 日 K 线（前复权） | AkShare `stock_zh_a_daily` | 增量同步到 PostgreSQL，除权自动检测全量重写 |
| 基础信息 / 行业分类 | AkShare 申万二级分类 | 并行抓取，按日缓存 |
| 资金流向 | AkShare / AShareHub API | 多周期（3/5/10/20 日）|
| 筹码分布 | AShareHub API | 获利比例 + 成本分位 + 集中度 |
| 交易日期历 | AkShare / chinesecalendar 兜底 | 24h 缓存 TTL |
| 强势股 / 连涨股 / 量价齐升 | AkShare 市场情绪接口 | 原始数据获取阶段一并拉取 |

<br>


## 🚀 核心功能与策略

### Walk-Forward 回测系统

- **Walk-Forward 滚动优化** — 以 in-sample（120 天）做网格搜索选出最优参数，在 out-of-sample（20 天）验证，滚动覆盖全历史
- **Grid Search 网格搜索** — 多参数组合并行评估：<font color="red">`atr_stop_mult`</font>(1.0~3.0)、<font color="red">`kelly_fraction`</font>(0.1~0.5)、<font color="red">`position_a`</font>(0.2~0.5)、<font color="red">`liq_veto_ratio`</font>(0.03~0.10)、<font color="red">`boll_narrow_ratio`</font>(0.6~1.2)、<font color="red">`cross_decay_days`</font>(15~60)
- **多进程并行评估** — 单个参数组合使用 `ProcessPoolExecutor` 并行回测，结果写入 parquet 共享
- **性能指标** — Sharpe、Sortino、Calmar、最大回撤、VaR(95%)、CVaR(95%)、年化收益率/波动率、胜率、盈亏比
- **仓位优化** — 支持风险平价、最小方差、均值-方差、评分加权四种组合权重分配
- **校准持久化** — 最优参数自动写入 `config.ini`，回测日志记录到 `backtest_calibration_log` 表
- **信号预计算缓存** — `prepare_backtest_data()` 按 `signal_cache_{trade_date}/{symbol}.parquet` 增量写入，中断后自动续算

### 数据同步（IncrementalSyncEngine）

- 增量同步 A 股日 K 线（Sina `stock_zh_a_daily`，HFQ 前复权），自动检测除权事件并全量重写
- 申万行业分类基础信息拉取（`ThreadPoolExecutor(10)`，~40s）
- 交易日期历本地缓存（24h TTL，chinesecalendar 兜底）
- 失败股票自动记录，下次运行重试
- 全局 HTTP 30s 超时（`AkshareConfig` 补丁）

### 技术指标信号

| 指标 | 周期 | 用途 |
|------|------|------|
| MACD | (12,26,9) | 7 维评分：趋势/金叉/动能/斜率/背离/量价/K 线形态 |
| ATR | 14 | 波动率归一化、止损/目标价计算、高波动过滤 |
| ADX | 14 | 趋势/反转情景切换（>25 高波动趋势，<20 低波反转） |
| BOLL | 20,2σ | 带宽/缩口/张口状态，与 MACD+CCI 共振评分 |
| CCI | 20 | 极度超买超卖，与 MACD+BOLL 共振 |
| RSI | 14 | 超卖及底背离，与 KDJ+量共振 |
| KDJ | 9,3,3 | 14 种信号模式 + 金叉死叉 + 三金叉共振 |
| K 线形态 | 25+ 种 | TA-Lib 吞没/十字星/锤子线等，评分 -10~+10 叠加衰减 |

### 评分管道（6 道门控）

```
Gate 0: 数据质量  →  K线<60日/ATR缺失/MA60缺失 → 否决
Gate 0.5: 宏观环境 →  涨跌比驱动等级门槛调节
Gate 1: 入场信号  →  无金叉/背离/反转 → C 级（拦截 ~50%）
Gate 2: 风险过滤  →  高波/顶背离/低成交额 → 否决（拦截 ~10~15%）
Gate 3: 资金修饰  →  资金流/量价修饰评分
Gate 4: 仓位联动  →  风险等级驱动 position_adjust 系数
Gate 5: 组合约束  →  行业集中度 <30%，总仓位 <100%
```

### 资金流 & 筹码

- 多周期资金净流入（3/5/10/20 日），主力/大户/散户细分
- 筹码分布：获利比例、成本分位（5%/50%/95%）、集中度、阻力位规则
- 市场状态分类：STRONG_TREND / WEAK_TREND / BOTTOM_REVERSAL / TOP_RISK / OSCILLATION

### 输出

- **Excel 报告** — 全市场 43+ 列结构化报表（评分、等级、止损、目标价），行业深度分析子表
- **数据库同步** — 结果写入 `ods_ak_ranking_stocks`、`ods_ak_industry_analysis`、`app_stock_strategy_report`

<br />

## 📊 打造个性化交易系统

通过修改 `config.ini` 适应不同交易风格：

**短线激进型**

```ini
[TECHNICAL_INDICATORS]
macd_params = 6,13,5              ; 超短敏感 MACD

[SYSTEM]
FUND_FLOW_PERIODS = [3, 5]        ; 短期资金流

[FILTER_RULES]
exempt_levels = 完全主升,趋势加速   ; 仅保留强势股

[BACKTEST]
atr_stop_mult = 2.0               ; 较宽止损
kelly_fraction = 0.5              ; 激进仓位
```

**中线稳健型**

```ini
[TECHNICAL_INDICATORS]
macd_params = 24,52,18            ; 中线趋势 MACD

[SYSTEM]
FUND_FLOW_PERIODS = [5, 10, 20]   ; 多周期验证

[FULL_BULL_SCORING]
conclusion_full_bull = 80         ; 提高完全主升门槛

[POSITION_SIZING]
kelly_fraction = 0.25             ; 保守仓位
max_single_position = 0.15        ; 单只上限 15%
```

**长线配置型（默认）**

```ini
[TECHNICAL_INDICATORS]
macd_params = 12,26,9             ; 经典均衡 MACD

[SYSTEM]
FUND_FLOW_PERIODS = [10, 20]      ; 中长期资金流

[POSITION_SIZING]
kelly_fraction = 0.2
position_a = 0.3                  ; A 级仓位 30%
max_single_position = 0.2
```
</br> </br> 
## 🛠️ 安装与配置

### 环境要求

- **Python 3.12+**（推荐 3.12~3.13）
- **PostgreSQL 14+** — 数据持久化存储
- **AkShare** — 免费使用，内置频率限制和 30s 全局超时

### 数据库准备

1. 创建数据库（名称任意，默认 `Corenews`）
2. 执行 `PostgreSQL建表语句.sql` 创建全部表结构
3. 配置 `config.ini` 中 `[DATABASE]` 节的连接参数

### AShareHub API（可选）

筹码分布数据需要 [AShareHub](https://www.asharehub.com) API 密钥。

```ini
[ASHAREHUB]
api_key = ENC:gAAAAAB...         ; 支持 ENC 加密
enable_chip_distribution = true
chip_limit = 1
```

密钥加密使用 `UtilsManager/ConfigCipher.py`，与数据库密码共用密钥。

<br></br> 

## ⚙️ 安装

**克隆项目仓库：**

git clone https://github.com/chowkuanyen/BAISYS_QUAN.git

cd BAISYS_QUAN

**安装依赖包:**

运行 `pip install -r requirements.txt` 安装全部依赖。

注：openpyxl 和 xlsxwriter 用于 Excel 文件的读写。psycopg2-binary 是 PostgreSQL 的 Python 驱动。

<br />

## 🛠️ 配置

所有配置统一存放于项目根目录的 `config.ini` 文件中，支持加密值（`ENC:` 前缀）。以下按 section 逐一说明。

---

### [DATABASE] — 数据库连接

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `user` | 字符串 | 是 | `postgres` | 数据库用户名 |
| `password` | 字符串 | 是 | - | 数据库密码（支持 ENC 加密） |
| `host` | 字符串 | 是 | - | 数据库主机地址 |
| `port` | 字符串 | 是 | - | 数据库端口号 |
| `db_name` | 字符串 | 是 | - | 数据库名称 |
| `main_board_only` | 布尔 | 否 | `true` | 是否仅获取主板股票（60/00开头） |
| `encryption_key_path` | 字符串 | 否 | `~/.baisys_quant_key` | 加密密钥文件路径（用于解密 `ENC:` 前缀的密码/密钥） |

---

### [SYSTEM] — 系统运行参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `HOME_DIRECTORY` | 字符串 | 否 | `~/Downloads/CoreNews_Reports` | 报告和缓存输出根目录 |
| `TEMP_DATA_DIR` | 字符串 | 否 | `.` | 临时数据子目录（相对 HOME_DIRECTORY） |
| `max_workers` | 整数 | 否 | `15` | 最大并发数据获取线程数 |
| `data_fetch_retries` | 整数 | 否 | `3` | 数据获取失败重试次数 |
| `data_fetch_delay` | 整数 | 否 | `5` | 重试间隔秒数 |
| `signal_processing_processes` | 整数 | 否 | CPU 核数 | 技术指标信号处理线程数 |

---

### [LOGGING] — 日志

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `log_level` | 字符串 | 否 | `INFO` | 日志级别 (DEBUG/INFO/WARNING/ERROR/CRITICAL) |
| `log_dir` | 字符串 | 否 | `Logs` | 日志子目录（相对 HOME_DIRECTORY） |

---

### [MULTI_HEAD_ARRANGEMENT] — 多头排列评分

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `full_bull_threshold` | 整数 | `85` | 完全主升浪阈值：最强多头排列，所有均线向上发散 |
| `trend_acceleration_threshold` | 整数 | `65` | 趋势加速阈值：较强多头排列，突破加速 |
| `trend_oscillation_threshold` | 整数 | `45` | 趋势震荡阈值：中等强度，均线收敛/震荡 |
| `trend_watch_threshold` | 整数 | `45` | 趋势观望阈值：弱/空头排列 |
| `moving_average_periods` | 逗号分隔整数 | `5,10,20,30,60` | 均线周期（用于本地评分计算，不影响 Akshare 均线突破） |

---

### [FILTER_RULES] — 弱势股过滤

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_weak_stock_filter` | 布尔 | `true` | 是否启用弱势股自动剔除 |
| `exempt_levels` | 逗号分隔字符串 | `完全主升,趋势加速` | 豁免条件：具备这些级别的股票不过滤 |
| `liq_veto_ratio` | 浮点 | <font color="red">`0.05`</font> | 流动性否决比（由回测优化） |

---

### [FUND_FLOW] — 资金流分析

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `fund_flow_periods` | 逗号分隔整数 | `5,10,20` | 资金流统计周期，必须 3 个。可选组合：`3,5,10`（短线）/ `3,5,20` / `5,10,20`（中线，默认）/ `3,10,20` |

---

### [TECHNICAL_INDICATORS] — 技术指标

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `macd_params` | 逗号分隔整数 | `12,26,9` | MACD (快线,慢线,信号线)，出厂默认 `12,26,9`，用户可自由修改 |

---

### [COLUMN_ALIASES] — 列名映射（一般不动）

| 参数 | 类型 | 说明 |
|------|------|------|
| `code_aliases` | 字符串 | 股票代码列名映射 |
| `name_aliases` | 字符串 | 股票名称列名映射 |
| `price_aliases` | 字符串 | 价格列名映射 |

---

### [RESEARCH_REPORT_FILTER] — 研报过滤

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_research_report_filter` | 布尔 | `true` | 是否启用研报数据过滤 |
| `research_report_min_count` | 整数 | `1` | 买入评级最低次数要求 |

---

### [KLINE_DATA] — K线数据获取

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `kline_history_days` | 整数 | `200` | 历史 K 线获取天数（建议 60～500） |

---

### [USER_FOCUS_STOCKS] — 用户关注股池

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `user_focus_stocks` | 竖线分隔 | 空 | 关注股票列表（例：`000001\|000002\|600000`），将在 Excel 中高亮置顶 |

---

### [FULL_BULL_SCORING] — MACD 管线评分权重

**权重维度（建议合计 90，不含量价配合）：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `weight_zero_axis` | `20` | MACD 趋势（零轴条件） |
| `weight_strategy_golden` | `15` | 金叉信号 |
| `weight_momentum` | `15` | 柱状动能 |
| `weight_dif_slope` | `10` | DIF 斜率 |
| `weight_divergence` | `10` | 背离信号 |
| `weight_volume_price` | `10` | 量价配合（独立奖励分） |
| `weight_kline_pattern` | `10` | K 线形态 |

**结论阈值：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `conclusion_full_bull` | `80` | 评分 ≥ 此值 → A 级（综合多头） |
| `conclusion_bullish` | `60` | 评分 ≥ 此值 → B 级（偏多） |
| `conclusion_oscillate` | `40` | 评分 ≥ 此值 → C 级（多空拉锯），否则 C 级（偏空） |

---

### [ASHAREHUB] — 筹码分布 API

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `api_key` | 字符串 | - | AShareHub API 密钥（支持 ENC 加密） |
| `enable_chip_distribution` | 布尔 | `true` | 是否获取筹码分布数据 |
| `chip_history_days` | 整数 | `90` | 筹码分布历史天数 |

---

### [MACRO_FILTER] — 宏观过滤器

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_macro_filter` | 布尔 | `true` | 是否启用宏观过滤 |
| `advance_ratio_ice` | 浮点 | `0.25` | 涨跌比"冰冻"阈值 |
| `advance_ratio_weak` | 浮点 | `0.35` | 涨跌比"弱势"阈值 |
| `advance_ratio_hot` | 浮点 | `0.70` | 涨跌比"过热"阈值 |

---

### [REGIME_DETECTION] — 市场状态分类参数

> ⚠️ **纯自定义经验值**，需根据回测结果调整。控制 `_detect_market_regime()` 中的市场状态判定阈值。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `boll_narrow_ratio` | 浮点 | <font color="red">`0.8`</font> | 窄布林判定：近期 BOLL 带宽 < 历史均值 × 此值 → 震荡（由回测优化） |
| `oscillation_hist_std_ratio` | 浮点 | `0.1` | 震荡模式：柱状图绝对值 < 此值 × close.std() → 震荡 |
| `top_risk_ma20_deviation` | 浮点 | `0.15` | 顶风险：收盘价偏离 MA20 超过此比例 → 顶部风险 |
| `oscillation_min_bars` | 整数 | `30` | 震荡判定所需最小 K 线数 |
| `reversal_lookback` | 整数 | `10` | 底/顶反转检测回溯长度（根） |

---

### [DIVERGENCE] — 背离检测参数

> ⚠️ **信号衰减模型为自研**，需根据回测调整。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `base_distance` | 整数 | `10` | 背离检测基础窗口 |
| `strength_threshold` | 浮点 | `0.15` | 背离有效强度门限，超过才生成信号 |
| `decay_half_life` | 整数 | `8` | 背离信号半衰期（天） |
| `slope_window` | 整数 | `5` | DIF 斜率线性回归窗口（根） |

---

### [POSITION_SIZING] — 仓位管理

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_single_position` | `0.25` | 单只股票最大仓位比例 |
| `kelly_fraction` | <font color="red">`0.25`</font> | Kelly 仓位比例系数（由回测优化） |
| `default_win_rate` | `0.55` | 默认胜率 |
| `position_a` | <font color="red">`0.35`</font> | A 级基础仓位（回测优化） |
| `position_b` | `0.25` | B 级基础仓位 |
| `position_c` | `0.15` | C 级基础仓位 |
| `position_d` | `0.05` | D 级基础仓位 |
| `max_industry_exposure` | `0.30` | 单行业最大暴露 |
| `risk_budget` | `0.02` | 风险预算（组合波动率上限） |

---

### [BACKTEST] — 回测系统

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `true` | 是否启用回测校准 |
| `optimize_frequency` | `monthly` | 校准频率 |
| `backtest_start_date` | `20200101` | 回测起始日期 |
| `out_of_sample_days` | `20` | Walk-Forward 样本外窗口天数 |
| `initial_cash` | `1000000` | 初始资金 |
| `full_a_share_mode` | `false` | 是否全 A 股回测 |
| `commission_rate` | `0.0003` | 佣金费率 |
| `stamp_tax_rate` | `0.001` | 印花税率 |
| `slippage` | `0.001` | 滑点 |
| `max_position_pct` | `0.1` | 单只上限 |
| `portfolio_method` | `score_weighted` | 组合权重方法 |

**网格搜索参数范围（逗号分隔 min,max,step）：**

| 参数 | 默认值 | 寻优对象 |
|------|--------|----------|
| `atr_stop_mult_range` | `1.0,3.0,0.5` | ATR 止损倍数 |
| `atr_t1_mult_range` | `2.0,6.0,1.0` | T1 目标倍数 |
| `kelly_fraction_range` | `0.1,0.5,0.1` | Kelly 比例 |
| `position_a_range` | `0.2,0.5,0.05` | A 级仓位 |
| `liq_veto_ratio_range` | `0.03,0.10,0.01` | 流动性否决比 |
| `boll_narrow_ratio_range` | `0.6,1.2,0.1` | 布林窄幅比 |
| `cross_decay_days_range` | `15,60,5` | 金叉衰减天数 |

---

### [SCORING_PARAMS] — 评分计算参数

> ⚠️ **纯自研参数**，控制衰减模型、波动率归一化、退出策略。

**衰减相关：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `cross_decay_days` | <font color="red">`30`</font> | 金叉信号衰减半衰期（天，由回测优化） |
| `cross_decay_min` | `0.3` | 金叉衰减下限（原始权重的 30%） |
| `kline_decay_days` | `10` | K 线形态衰减半衰期（天） |
| `kline_decay_min` | `0.2` | K 线衰减下限（原始权重的 20%） |

**波动率归一化：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vol_norm_denominator` | `0.15` | 金叉强度波动率归一化分母：(DIF-DEA)/ATR ÷ 此值 → vol_factor |

**退出策略（ATR 倍数）：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `atr_stop_mult` | <font color="red">`1.5`</font> | 止损价 = close - ATR × 此值（由回测优化） |
| `atr_t1_mult` | <font color="red">`3.0`</font> | T1 目标价 = close + ATR × 此值（由回测优化） |
| `atr_t2_mult` | `5.0` | T2 目标价 = close + ATR × 此值 |

**移动止损：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `trailing_stop_high_ratio` | `0.98` | close ≥ 近 N 日最高价 × 此值 → 激活移动止损 |
| `trailing_stop_high_lookback` | `20` | 参考高点回溯窗口（根） |
| `trailing_stop_lookback` | `10` | 移动止损价取近 N 日最低价（根） |

**预期盈亏比：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `expected_return_lookback` | `20` | 计算预期盈亏比时取近 N 日价格区间 |

---

### [TECHNICAL_CONSTANTS] — 标准技术指标参数

> ✅ **行业标准参数**，一般无需修改。放在此处仅为了统一入口和对比调参。

| 参数 | 默认值 | 标准来源 | 说明 |
|------|--------|----------|------|
| `atr_length` | `14` | Wilder | ATR 计算周期 |
| `adx_length` | `14` | Wilder | ADX 计算周期 |
| `rsi_length` | `14` | Wilder | RSI 计算周期 |
| `boll_length` | `20` | Bollinger | BOLL 计算周期 |
| `boll_std` | `2.0` | Bollinger | BOLL 标准差倍数 |
| `stoch_k` | `9` | Lane | Stoch %K 周期 |
| `stoch_d` | `3` | Lane | Stoch %D 平滑周期 |
| `kline_scan_window` | `60` | - | K 线形态扫描窗口（根数） |

## 🚀 使用方法

执行 `MainShareAnalysis.py` 启动全自动化流程（含回测校准 + 每日分析）。

### CLI 参数

| 参数 | 说明 |
|------|------|
| `--force` | 强制重新执行回测校准（忽略频率检查） |
| `--pipeline-only` | 仅运行每日分析管线，跳过回测阶段 |
| `--backtest-only` | 仅运行回测校准，跳过每日分析 |
| `--schedule` | 启动持久化调度守护进程（每日 02:00 检查是否需要运行） |

### 运行流程

```
MainShareAnalysis
  │
  ├── [阶段 A] 回测校准 (run_backtest_pipeline)
  │     ├── 解析股票列表 → 拉取 K 线
  │     ├── 信号预计算 (prepare_backtest_data)
  │     │   └── 并行 ProcessPoolExecutor + 增量 parquet 缓存
  │     ├── Walk-Forward 滚动优化
  │     │   ├── 滑动窗口: in-sample 120 天 grid search
  │     │   └── out-of-sample 20 天验证
  │     ├── 全量回测 (run_full_backtest) — 最优参数
  │     ├── 绩效指标计算 (Sharpe/Sortino/Calmar/VaR/胜率)
  │     └── 保存校准结果 → calibration_result.json + config.ini
  │
  └── [阶段 B] 每日分析管线 (StockAnalysisCoordinator 13步)
        ├─ Step 01: 同步历史K线 (IncrementalSyncEngine)
        ├─ Step 02: 格式化股票代码 (CodeNormalizer)
        ├─ Step 03: 获取原始数据 (资金流/强势股/行业板块)
        ├─ Step 04: 获取K线数据及最新价
        ├─ Step 05: 处理技术指标信号 (MACD 7维/KDJ/CCI/RSI/BOLL/K线形态)
        ├─ Step 06: 行业分析 (IndustryFlowAnalyzer)
        ├─ Step 07: 均线突破数据
        ├─ Step 08: 合并数据字典
        ├─ Step 09: 合并分析数据 (DataProcessingService)
        ├─ Step 10: 行业信号映射 + 行业中性化
        ├─ Step 11: 剔除弱势股
        ├─ Step 12: 生成Excel报告
        └─ Step 13: 同步结果到数据库
```

### 调度模式

```bash
# 启动后台守护进程（每日 02:00 自动检查运行）
python MainShareAnalysis.py --schedule

# 手动指定参数
python MainShareAnalysis.py --force      # 强制重跑回测
python MainShareAnalysis.py --pipeline-only  # 仅分析
python MainShareAnalysis.py --backtest-only  # 仅回测
```

<br /></br> 

## 📊 输出结果

所有报告和缓存文件生成在 `config.ini` 中 `HOME_DIRECTORY` 指定的目录下（默认 `~/Downloads/CoreNews_Reports`）。

### Excel 报告

**`审计报告_YYYYMMDD.xlsx`** — 每日全市场分析结果，包含 43+ 列及多个子表：

| 区块 | 列数 | 包含列 |
|------|------|--------|
| 基础信息 | 7 | 股票代码, 股票简称, 行业, 所属行业信号, 最新价, 主力成本, 成本位置 |
| 资金流信号 | 5 | 强势股, 量价齐升, 量价配合, 连涨天数, 放量天数 |
| MACD 评分 | 4 | MACD趋势, 金叉信号, 柱状动能, DIF斜率 |
| 技术指标 | 5 | KDJ/CCI/RSI/BOLL/K线形态信号 |
| 均线参考 | 3 | 10/30/60 日均线价 |
| 背离 | 3 | 背离信号, 距今, 位置 |
| 风控 | 1 | 风险等级 |
| 综合报告 | 9 | 多头排列趋势, 综合分析结论/评分/级别, 止损价, T1/T2目标价, 移动止损, 盈亏比 |
| 资金 | 5 | 研报买入次数, 资金动能, 5/10/20 日资金流入 |
| 链接 | 1 | 股票链接 |

子表：行业深度分析、主力研报筛选、均线多头排列、资金流向、强势股池、技术指标信号等。

### 回测校准结果

每次回测运行后，结果保存在 `calibration_result.json` 中，最优参数自动写入 `config.ini`。运行日志记录到 `backtest_calibration_log` 数据库表。

### 缓存文件

| 文件/目录 | 说明 |
|-----------|------|
| `backtest_signal_cache/{trade_date}/{symbol}.parquet` | 信号预计算缓存（按日 + 按只，支持中断续算）|
| `calendar/official_trading_dates.json` | 交易日历缓存（24h TTL）|
| `StockIndes_YYYYMMDD.txt` | 股票基础信息缓存 |
| `ShareData/` | 原始数据及清洗后数据缓存 |
| `failed_symbols_{YYYYMMDD}.txt` | 当日同步失败的股票，下次运行自动重试 |

## ⚠️ 注意事项

- 请确保 PostgreSQL 服务已启动且 `config.ini` 中数据库连接信息正确
- 首次运行前：`pip install -r requirements.txt`
- 数据同步依赖 AkShare，建议在交易日 15:30 后运行
- 信号预计算阶段使用 `ProcessPoolExecutor`，需确保 Python 环境支持 multiprocessing spawn
- 若 `config.ini` 缺少某些节，系统会自动补全默认值（`ConfigValidator`）
- 敏感信息（数据库密码、API Key）支持 `ENC:` 加密前缀，使用 `ConfigCipher` 工具生成
- 信号缓存按交易日后缀存储，旧日期的缓存目录在下次运行时自动清理

<br />

## ⚠️ 免责声明

本项目提供的所有数据、分析报告和投资建议仅供学习、研究和参考，不构成任何投资建议。投资者应自行承担投资风险，并根据自身情况做出独立的投资决策。本项目的开发者不对任何使用本系统数据或分析结果而导致的投资损失承担责任。

请务必理解并同意以上声明后，再使用本项目。

<br>
