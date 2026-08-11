<p align="center">
  <img src="https://github.com/paiyuyen/Multi-factor-Quantitative-Stock-Selection-Analysis-System/raw/main/Images/logo.png" alt="LOGO" width="50%">
  <br/><br/>
  <b>百 思 量 化</b>
  <br/><br/>
  <b > 量 化 方 寸 间  ， 洞 悉 万 象 市 </b>
  <br/>  
</p>
<p align="center">
     <img src="https://img.shields.io/badge/Lib-PostgreSQL-ff4500?style=flat-square" />
     <img src="https://img.shields.io/badge/Lib-Pydantic-ff4500?style=flat-square" />
     <img src="https://img.shields.io/badge/Lib-Loguru-ff4500?style=flat-square" />
     <img src="https://img.shields.io/badge/Lib-TA_LIB-ff4500?style=flat-square" />
    </br> 
       <img src="https://img.shields.io/badge/Data-AkShare-red?logo=databricks&logoColor=white" />
    <img src="https://img.shields.io/badge/Data-Pandera-red?style=flat-square" />
     <img src="https://img.shields.io/badge/Data-Numpy-red?style=flat-square" />
     <img src="https://img.shields.io/badge/Data-Akquant-red?style=flat-square" />
         </br> 
   <img src="https://img.shields.io/badge/Analysis-WalkForward-green?logo=pandas&logoColor=white" />
     <img src="https://img.shields.io/badge/Analysis-KDJ-green?logo=pandas&logoColor=white" />
     <img src="https://img.shields.io/badge/Analysis-MACD-green?logo=pandas&logoColor=white" />
     <img src="https://img.shields.io/badge/Analysis-BOLL-green?logo=pandas&logoColor=white" />
     <img src="https://img.shields.io/badge/Analysis-CCI-green?style=flat-square" />
     <img src="https://img.shields.io/badge/Analysis-ADX-green?style=flat-square" />
     <img src="https://img.shields.io/badge/Analysis-XGBoost-brightgreen?style=flat-square" />
     <img src="https://img.shields.io/badge/Analysis-Vectorized-9cf?style=flat-square" />
</p>
<br />

## 📖 项目简介

百思量化是一套面向 A 股的全链路量化系统，覆盖 **数据同步 → 信号预计算 → 策略回测 → 每日分析报告** 全流程。系统分为两大阶段：

### 阶段 A — 回测校准

通过 Walk-Forward 滚动窗口优化 + 贝叶斯优化（Bayesian Optimization），自动寻优 12 个核心策略参数（4 个信号参数 + 8 个组合参数）用于日常运行。信号计算已全向量化（`numpy` + `np.select` / `np.correlate`，无 per-bar Python 循环），结合 Phase 0 指标预计算缓存，单次信号计算 3000 只股票约 10 分钟。采用两级成本分层（信号计算分钟级 + 纯回测秒级）和 GP 代理模型（Gaussian Process + qEI 采集函数），在相同计算预算下覆盖完整参数空间。

### 阶段 B — 每日分析管线

19 步 DAG 流水线从数据库增量同步 K 线 → 多门控评分 → 多因子 Alpha → 行业百分位过滤 → 组合构建 → 基准对比 → 因子衰减监控 → 跟仓回测 → 生成 Excel 报告 → 同步结果到 PostgreSQL。断点续跑 + run_id 版本管理。

### 设计特点

- **MACD 管线** — 默认12,26,9周期 + ATR 波动率归一化，7 维评分维度权重可配置
- **5 道门控递进评分 + 组合约束** — Gate 0（数据质量）→ 1（信号共振，否决）→ 2（波动率/背离，否决）→ 3（资金流修饰）→ 4（仓位联动），Gate 5 组合级后处理（行业集中度 <30%）
- **信号衰减模型** — 金叉 30 天半衰、背离 8 天半衰、K 线形态 10 天半衰
- **多因子 Alpha 评分** — 5 因子加权：MACD(25%) + 动量(25%) + 资金流(20%) + 质量(15%) + 估值(15%)，因子权重可配置
- **因子衰减监控** — 因子 IC 值、ICIR、Decile Spread 持续跟踪，超阈值自动告警
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
| 估值因子（PE/PB/市值）| AShareHub API | 日频增量同步 |
| 质量因子（ROE/毛利率）| AkShare 财务摘要 | 季度数据，增量缓算 |

<br>


## 🚀 核心功能与策略

### Walk-Forward 回测系统

- **全向量化信号引擎** — 信号计算已从 per-bar 逐行 Python 循环重构为纯 numpy 向量化运算。`compute_signals()` 使用 `np.where`、`np.correlate`、`np.select`、前缀和、rolling window broadcast 等技术，消除所有逐行 Python 回调。次核心指标（MACD/BOLL/KDJ/RSI/CCI/ADX）及其峰值/谷值检测在 Phase 0 预计算阶段一次性完成，后续所有参数评估直接复用。
- **机器学习信号增强** — 引入 XGBoost 模型（`BackTrading/model.py` `BacktestXGBScorer`）作为信号修偏层。`BacktestXGBScorer` 以历史金叉/背离信号及量价特征为输入，输出概率校准后的信号置信度，在 `compute_signals` 中叠加到综合评分，提升回测信号质量。
- **Phase 0 指标预计算缓存** — `indicator_cache.py` 在每个 WFO 窗口首次运行时预计算全部技术指标 + 峰值/谷值检测，结果写磁盘（`.indicators.parquet` + `.peaks.npy` + `.troughs.npy`）。窗口内后续贝叶斯评估（信号参数变化）跳过指标复算，仅重跑评分逻辑，单次评估从 ~1 小时降至 ~5 分钟。
- **Walk-Forward 滚动优化** — 以 in-sample 训练窗口做贝叶斯优化选出最优参数，在 out-of-sample 验证，滚动覆盖全历史。默认 IS=120 天、OOS=20 天，多路径偏移取中位数聚合，相邻窗口 GP 状态热启动加速收敛
- **贝叶斯优化引擎** — 4 阶段优化器：Sobol 准随机初始化（~15 组）→ GP+qEI Level 1 搜索（~35 次）→ GP+qEI Level 2 搜索（~150 次，信号固定）→ GP 代理 L-BFGS-B 精炼 Top-3。联合寻优 12 个参数：boll_narrow_ratio, cross_decay_days, golden_cross_bonus, divergence_penalty（信号参数），atr_stop_mult, atr_t1_mult, atr_t2_mult, kelly_fraction, position_a, liq_veto_ratio, conclusion_full_bull, risk_none_multiplier（组合参数）
- **两级成本分层** — 信号参数（4 个）触发完整管线（`prepare_backtest_data`，分钟级）；纯组合参数（8 个）直接从缓存加载信号（秒级）。`FidelityController` 自动检测输入数据是否有预计算信号列，避免不必要的重算
- **高斯过程代理模型** — `ConstantKernel × Matern(ARD) + WhiteKernel` 组合核，自动相关性长度尺度学习。`save_gp_state` / `restore_gp_state` 序列化核参数实现跨窗口迁移学习
- **混合采集函数** — `Expected Improvement - λ·σ/|μ|`（DSR 惩罚项，λ=0.05），抑制高不确定性低预期区域的采样，L-BFGS-B 多起点优化
- **性能指标** — Sharpe、Sortino、Calmar、最大回撤、VaR(95%)、CVaR(95%)、年化收益率/波动率、胜率、盈亏比、PBO（概率过拟合）、DSR（缩水 Sharpe 比）
- **仓位优化** — 支持风险平价、最小方差、均值-方差（含换手率惩罚 + 行业约束）、评分加权四种组合权重分配
- **校准持久化** — 最优参数自动写入 `config.ini [BACKTEST_CALIBRATED]` 分区，回测日志记录到 `backtest_calibration_log` 表
- **信号预计算缓存** — `prepare_backtest_data()` 按 `signal_cache_{trade_date}_{config_hash}_{param_hash}/{bucket}/{symbol}.parquet` 增量写入，分桶减小单目录文件数，zstd 压缩，断点自动续算。Phase 0 缓存位于 `CACHE_DIR/indicator_cache_v1/<bucket>/<symbol>.indicators.parquet`，子进程通过磁盘共享。

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

### 评分管道（5 道门控 + 组合约束）

```
Gate 0: 数据质量   →  K线<60日/ATR缺失/MA60缺失 → 否决
Gate 1: 入场信号   →  无金叉/背离/反转 → C 级（拦截 ~50%）
Gate 2: 风险过滤   →  高波/顶背离/低成交额 → 否决（拦截 ~10~15%）
Gate 3: 资金修饰   →  资金流/量价修饰评分
Gate 4: 仓位联动   →  风险等级驱动 position_adjust 系数 (ATR/止损比)
Gate 5: 组合约束   →  行业集中度 <30%，总仓位 <100%（由 PortfolioBuilder 执行）
```

### 资金流 & 筹码

- 多周期资金净流入（3/5/10/20 日），主力/大户/散户细分
- 筹码分布：获利比例、成本分位（5%/50%/95%）、集中度、阻力位规则
- 市场状态分类：STRONG_TREND / WEAK_TREND / BOTTOM_REVERSAL / TOP_RISK / OSCILLATION

### 输出

- **Excel 报告** — 输出个股分析报告
- **数据库同步** — 结果写入 `ods_ak_ranking_stocks`、`ods_ak_industry_analysis`、`app_stock_strategy_report`、`ods_factor_ic_history`、`dash_pipeline_checkpoint`

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

[BACKTEST_CALIBRATED]
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

[BACKTEST_CALIBRATED]
kelly_fraction = 0.25             ; 保守仓位
max_single_position = 0.15        ; 单只上限 15%
```

**长线配置型（默认）**

```ini
[TECHNICAL_INDICATORS]
macd_params = 12,26,9             ; 经典均衡 MACD

[SYSTEM]
FUND_FLOW_PERIODS = [10, 20]      ; 中长期资金流

[BACKTEST_CALIBRATED]
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

### AShareHub API

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

所有配置统一存放于项目根目录的 `config.ini` 文件中，支持加密值（`ENC:` 前缀）。
文件按两大分区组织：**⚙️ 系统配置**（基础设施、性能、外部 API）和 **📊 业务配置**（策略参数、评分、风控、回测）。

### ⚙️ 系统配置

---

#### [DATABASE] — 数据库连接

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `user` | 字符串 | 是 | `postgres` | 数据库用户名 |
| `password` | 字符串 | 是 | - | 数据库密码（支持 ENC 加密） |
| `host` | 字符串 | 是 | - | 数据库主机地址 |
| `port` | 字符串 | 是 | - | 数据库端口号 |
| `db_name` | 字符串 | 是 | - | 数据库名称 |
| `main_board_only` | 布尔 | 否 | `true` | 是否仅获取主板股票（60/00开头） |
| `encryption_key_path` | 字符串 | 否 | `~/.baisys_quant_key` | 加密密钥文件路径 |

---

#### [SYSTEM] — 系统运行参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `HOME_DIRECTORY` | 字符串 | 否 | `~/Downloads/CoreNews_Reports` | 报告和缓存输出根目录 |
| `TEMP_DATA_DIR` | 字符串 | 否 | `cache` | 临时数据子目录（相对 HOME_DIRECTORY） |
| `max_workers` | 整数 | 否 | `15` | 最大并发数据获取线程数 |
| `data_fetch_retries` | 整数 | 否 | `3` | 数据获取失败重试次数 |
| `data_fetch_delay` | 整数 | 否 | `5` | 重试间隔秒数 |
| `stock_basic_info_expire_days` | 整数 | 否 | `30` | 基础信息缓存过期天数 |
| `signal_processing_processes` | 整数 | 否 | CPU 核数 | 技术指标信号处理线程数 |

---

#### [LOGGING] — 日志

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `log_level` | 字符串 | 否 | `INFO` | 日志级别 |
| `log_dir` | 字符串 | 否 | `Logs` | 日志子目录（相对 HOME_DIRECTORY） |

---

#### [COLUMN_ALIASES] — 列名映射

| 参数 | 类型 | 说明 |
|------|------|------|
| `code_aliases` | 字符串 | 股票代码列名映射 |
| `name_aliases` | 字符串 | 股票名称列名映射 |
| `price_aliases` | 字符串 | 价格列名映射 |

---

#### [ASHAREHUB] — 外部 API 配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `api_key` | 字符串 | - | AShareHub API 密钥（支持 ENC 加密） |
| `enable_chip_distribution` | 布尔 | `true` | 是否获取筹码分布数据 |
| `moneyflow_retry` | 整数 | `3` | 资金流向 API 重试次数 |
| `moneyflow_page_delay` | 浮点 | `1.0` | 资金流向分页间隔（秒） |

---

### 📊 业务配置

---

#### [USER_FOCUS_STOCKS] — 用户关注股池

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `user_focus_stocks` | 竖线分隔 | 空 | 关注股票列表（`000001\|000002\|600000`），Excel 中高亮置顶 |

---

#### [TECHNICAL_INDICATORS] — 技术指标

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `macd_params` | 逗号分隔整数 | `12,26,9` | MACD (快线,慢线,信号线) |

---

#### [TECHNICAL_CONSTANTS] — 标准技术指标参数

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

---

#### [MULTI_HEAD_ARRANGEMENT] — 多头排列评分

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `full_bull_threshold` | 整数 | `85` | ≥85 → 完全主升浪 |
| `trend_acceleration_threshold` | 整数 | `65` | 65~84 → 趋势加速 |
| `trend_oscillation_threshold` | 整数 | `45` | 45~64 → 趋势震荡 |
| `moving_average_periods` | 逗号分隔整数 | `5,10,20,30,60` | 均线周期 |

---

#### [FULL_BULL_SCORING] — MACD 评分权重

**7 维权重（建议合计 90，不含量价配合）：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `weight_zero_axis` | `20` | MACD 趋势（零轴条件） |
| `weight_strategy_golden` | `15` | 金叉信号 |
| `weight_momentum` | `15` | 柱状动能 |
| `weight_dif_slope` | `10` | DIF 斜率 |
| `weight_divergence` | `10` | 背离信号 |
| `weight_volume_price` | `10` | 量价配合（奖励分） |
| `weight_kline_pattern` | `10` | K 线形态 |

**结论阈值：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `conclusion_full_bull` | `80` | ≥80 → A 级 |
| `conclusion_bullish` | `60` | ≥60 → B 级 |
| `conclusion_oscillate` | `40` | ≥40 → C 级，否则 C 级（偏空） |

---

#### [REGIME_DETECTION] — 市场状态分类

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `oscillation_hist_std_ratio` | 浮点 | `0.1` | 柱状图标准差比 |
| `top_risk_ma20_deviation` | 浮点 | `0.15` | 顶风险 MA20 偏离阈值 |
| `oscillation_min_bars` | 整数 | `30` | 震荡判定最小 K 线数 |
| `reversal_lookback` | 整数 | `10` | 反转检测回溯长度 |

---

#### [DIVERGENCE] — 背离检测

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `base_distance` | 整数 | `10` | 背离检测基础窗口 |
| `strength_threshold` | 浮点 | `0.15` | 背离有效强度门限 |
| `decay_half_life` | 整数 | `8` | 背离信号半衰期（天） |
| `slope_window` | 整数 | `5` | DIF 斜率回归窗口 |

---

#### [SCORING_PARAMS] — 评分计算参数

**信号衰减：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `cross_decay_min` | `0.3` | 金叉衰减下限（30%） |
| `kline_decay_days` | `10` | K 线形态衰减半衰期（天） |
| `kline_decay_min` | `0.2` | K 线衰减下限（20%） |

**波动率归一化：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vol_norm_denominator` | `0.15` | (DIF-DEA)/ATR ÷ 此值 → vol_factor |

**Rule 评分偏移量（Walk-Forward 可校准）：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `golden_cross_bonus` | `10` | R04: 金叉量价确认加分 |
| `divergence_penalty` | `20` | R41: 顶背离量缩扣分 |

**退出策略（ATR 倍数）：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `atr_t2_mult` | `5.0` | T2 目标价 = close + ATR × 此值 |

**移动止损：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `trailing_stop_high_ratio` | `0.98` | 近 N 日最高价 × 此值 → 激活移动止损 |
| `trailing_stop_lookback` | `10` | 移动止损价取近 N 日最低价 |
| `trailing_stop_high_lookback` | `20` | 参考高点回溯窗口 |

**预期盈亏比：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `expected_return_lookback` | `20` | 计算回溯窗口（天） |

---

#### [FILTER_RULES] — 弱势股过滤

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_weak_stock_filter` | 布尔 | `true` | 是否启用弱势股三级过滤 |
| `exempt_levels` | 逗号分隔字符串 | `完全主升,趋势加速` | 豁免级别列表 |
| `industry_pct_hard` | 整数 | `10` | Stage 2 硬剔除：行业内评分后 N% |
| `industry_pct_d` | 整数 | `30` | Stage 3 辅助剔除：行业内评分后 N% |
| `industry_pct_exempt` | 整数 | `80` | Stage 1 单因子前 N% 豁免 |
| `liq_w_section` | 浮点 | `0.4` | 流动性评分截面权重 |
| `liq_w_timeseries` | 浮点 | `0.4` | 流动性评分时序权重 |
| `liq_w_marketcap` | 浮点 | `0.2` | 流动性评分规模权重 |
| `liq_min_discount` | 浮点 | `0.3` | 流动性最差时仓位最低比例 |

---

#### [FUND_FLOW] — 资金流分析

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `fund_flow_periods` | 逗号分隔整数 | `5,10,20` | 统计周期，可选：`3,5,10` / `3,5,20` / `5,10,20` / `3,10,20` |

---

#### [RESEARCH_REPORT_FILTER] — 研报过滤

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_research_report_filter` | 布尔 | `false` | 是否启用研报过滤 |
| `research_report_min_count` | 整数 | `1` | 买入评级最低次数 |

---

#### [MULTI_FACTOR_ALPHA] — 多因子 Alpha 评分

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | 布尔 | `true` | 启用多因子 Alpha 评分 |
| `financial_quality_cache_days` | 整数 | `90` | 质量因子缓存天数 |
| `financial_quality_batch_size` | 整数 | `500` | 质量因子每批采集股票数 |
| `financial_quality_batch_sleep` | 整数 | `20` | 质量因子批间休眠秒数 |
| `financial_quality_file_cache_days` | 整数 | `30` | 质量因子离线文件缓存天数 |
| 因子权重 | — | — | 权重定义已迁移至 `config/factor_registry.yaml` |


---

#### [POSITION_SIZING] — 仓位管理

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_single_position` | `0.33` | 单只股票最大仓位比例 |
| `default_win_rate` | `0.50` | 默认胜率 |
| `position_b` | `0.15` | B 级基础仓位 |
| `position_c` | `0.05` | C 级基础仓位 |
| `max_industry_exposure` | `0.30` | 单行业最大暴露 |
| `max_day_turnover` | `0.20` | 单日最大双边换手率 |
| `risk_aversion` | `1.0` | 风险厌恶系数 |
| `risk_budget` | `0.02` | 风险预算（组合波动率上限） |
| `risk_none_multiplier` | `1.0` | NONE 风险等级仓位系数 |

---

#### [BACKTEST_CALIBRATED] — 回测自动校准参数

这些参数由 Walk-Forward 寻优引擎在回测期间自动搜索最优值并写回本分区，日常运行无需手动修改。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `boll_narrow_ratio` | <font color="red">`0.9`</font> | 窄布林判定：带宽/历史均值 < 此值 → 震荡（由回测优化） |
| `cross_decay_days` | <font color="red">`37`</font> | 金叉信号衰减半衰期，天（由回测优化） |
| `atr_stop_mult` | <font color="red">`2`</font> | ATR 止损倍数：止损价 = close - ATR × 此值（由回测优化） |
| `atr_t1_mult` | <font color="red">`4`</font> | T1 目标价 ATR 倍数（由回测优化） |
| `liq_veto_ratio` | <font color="red">`0.065`</font> | 流动性否决比（由回测优化） |
| `kelly_fraction` | <font color="red">`0.3`</font> | Kelly 仓位比例系数（由回测优化） |
| `position_a` | <font color="red">`0.35`</font> | A 级基础仓位（由回测优化） |
| `atr_t2_mult` | <font color="red">`5.0`</font> | T2 目标价 ATR 倍数（由回测优化） |
| `conclusion_full_bull` | <font color="red">`80`</font> | MACD 综合评分 ≥ 此值 → A 级（由回测优化） |
| `golden_cross_bonus` | <font color="red">`10`</font> | R04: 金叉量价确认加分（由回测优化） |
| `divergence_penalty` | <font color="red">`20`</font> | R41: 顶背离量缩扣分（由回测优化） |
| `risk_none_multiplier` | <font color="red">`1.0`</font> | NONE 风险等级仓位系数（由回测优化） |

---

#### [BACKTEST] — 回测系统

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `true` | 是否启用回测校准 |
| `optimize_frequency` | `monthly` | 校准频率 |
| `backtest_start_date` | `20230101` | 回测起始日期 |
| `out_of_sample_days` | `60` | Walk-Forward 样本外窗口天数 |
| `initial_cash` | `1000000` | 初始资金 |
| `full_a_share_mode` | `false` | 是否全 A 股回测 |
| `signal_pipelines` | `3` | 信号预计算并行管道数 |
| `execution_model` | `next_open` | 成交时点模型：`close` 信号日收盘成交（老行为）/ `next_open` 信号次日开盘成交（默认，符合A股T+1）/ `vwap` 信号次日VWAP成交 |

**网格搜索参数范围（逗号分隔 min,max,step）：**

**趋势 + 止损：**

| 参数 | 默认值 | 寻优对象 |
|------|--------|----------|
| `atr_stop_mult_range` | `1.0,3.0,0.5` | ATR 止损倍数 |
| `atr_t1_mult_range` | `2.0,6.0,1.0` | T1 目标倍数 |
| `atr_t2_mult_range` | `3.0,10.0,1.0` | T2 目标倍数 |
| `boll_narrow_ratio_range` | `0.6,1.2,0.1` | 布林窄幅比 |
| `cross_decay_days_range` | `15,60,5` | 金叉衰减天数 |

**仓位 + 风控：**

| 参数 | 默认值 | 寻优对象 |
|------|--------|----------|
| `kelly_fraction_range` | `0.1,0.5,0.1` | Kelly 比例 |
| `position_a_range` | `0.2,0.5,0.05` | A 级仓位 |
| `liq_veto_ratio_range` | `0.03,0.10,0.01` | 流动性否决比 |
| `conclusion_full_bull_range` | `60,95,5` | A 级评分阈值 |
| `risk_none_multiplier_range` | `0.5,2.0,0.25` | NONE 风险系数 |

**Rule 评分偏移：**

| 参数 | 默认值 | 寻优对象 |
|------|--------|----------|
| `golden_cross_bonus_range` | `5,20,5` | R04 金叉加分 |
| `divergence_penalty_range` | `10,40,5` | R41 顶背离扣分 |

---

#### [TRADING_COST] — A股交易成本（跟仓回测）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `commission_rate` | `0.0003` | 佣金费率（万三） |
| `stamp_tax_rate` | `0.001` | 印花税费率（卖出单向，千一） |
| `transfer_fee_rate` | `0.00001` | 过户费率（双向，万0.1） |

---

#### [POSITION_BACKTEST] — 跟仓回测

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pool_file_path` | `证券交割单.xlsx` | 历史交易记录池文件路径（FIFO 匹配） |

---

## 🚀 使用方法

执行 `MainShareAnalysis.py` 启动全自动化流程（含回测校准 + 每日分析）。

### CLI 参数

| 参数 | 说明 |
|------|------|
| `--force` | 强制重跑回测校准 + 管线所有步骤（忽略 checkpoint，数小时） |
| `--pipeline-only` | 仅运行每日分析管线，跳过回测阶段（DAG 支持断点续跑）|
| `--backtest-only` | 仅运行回测校准，跳过每日分析 |
| `--schedule` | 启动持久化调度守护进程（每日 02:00 检查是否需要运行） |

### 运行流程

```
MainShareAnalysis
  │
  ├── [阶段 A] 回测校准 (run_backtest_pipeline)
  │     ├── 解析股票列表 → 拉取 K 线
  │     ├── 信号预计算 (prepare_backtest_data)
  │     │   ├── Phase 0: 技术指标预计算 (indicator_cache.py)
  │     │   │   └── MACD/A股/BOLL/KDJ/CCI/RSI/ADX → 磁盘缓存 + 峰值谷值检测
  │     │   ├── 全向量化信号评分 (vectorized_signal.py)
  │     │   │   └── numpy 向量化: 背离广播/斜率卷积/形态前缀和/金叉衰减曲线
  │     │   ├── XGBoost 信号修偏 (BacktestXGBScorer)
  │     │   │   └── 概率校准 → 叠加到综合评分
  │     │   └── 并行 ProcessPoolExecutor + 增量 parquet 缓存
  │     ├── Walk-Forward 滚动优化
  │     │   ├── 滑动窗口: in-sample 120 天 grid search
  │     │   └── out-of-sample 20 天验证
  │     ├── 全量回测 (run_full_backtest) — 最优参数
  │     ├── 绩效指标计算 (Sharpe/Sortino/Calmar/VaR/胜率)
  │     └── 保存校准结果 → calibration_result.json + config.ini
  │
  └── [阶段 B] 每日分析管线 DAG 19 步 (StockAnalysisCoordinator)
        ├─ 01 同步历史K线 (IncrementalSyncEngine)
        ├─ 02 格式化股票代码 (CodeNormalizer)
        ├─ 03 获取原始数据 (资金流/强势股/行业板块)              ─┐
        ├─ 04 获取K线数据及最新价                                ─┤ 并行可独立执行
        ├─ 05 处理技术指标信号 (MACD 7维/KDJ/CCI/RSI/BOLL)      ─┤
        ├─ 06 行业分析 (IndustryFlowAnalyzer)                   ─┘
        ├─ 07 处理均线突破数据 (依赖 03+04)
        ├─ 08 准备处理数据字典 (合并 05+06+07)
        ├─ 09 合并分析数据 (DataProcessingService)
        ├─ 10 数据质量检查 (DataQualityChecker)
        ├─ 11 行业信号映射 + 行业中性化
        ├─ 12 多因子 Alpha 评分 (5 因子加权 → fuse_scores)
        │     └── 行业百分位 Stage 1：in-cache 缓存
        ├─ 13 剔除弱势股 (三级过滤：趋势豁免→硬百分位→D/C级)
        ├─ 14 组合构建 (PortfolioBuilder: Kelly+风险平价+流动性)
        ├─ 15 基准对比 (BenchmarkEvaluator: 沪深300/全A等权)
        ├─ 16 因子衰减监控 (FactorDecayMonitor: IC/ICIR 跟踪)
        ├─ 17 跟仓回测分析 (PositionTrackingService: FIFO 匹配)
        ├─ 18 生成Excel报告 (ReportService: 58列+行业/因子子表)
        └─ 19 同步结果到数据库 (ods_* / app_stock_strategy_report)
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

**`审计报告_YYYYMMDD.xlsx`** — 每日全市场分析结果，包含 58+ 列及多个子表：

| 区块 | 列数 | 包含列 |
|------|------|--------|
| 基础信息 | 8 | 股票代码, 股票简称, 行业, 所属行业信号, 最新价, 95%筹码价, 主力成本, 成本位置 |
| 资金流信号 | 5 | 强势股, 量价齐升, 量价配合, 连涨天数, 放量天数 |
| MACD 评分 | 4 | MACD趋势, 金叉信号, 柱状动能, DIF斜率 |
| 技术指标 | 5 | KDJ/CCI/RSI/BOLL/K线形态信号 |
| 均线参考 | 3 | 10/30/60 日均线价 |
| 背离 | 4 | 背离信号, 背离距今, 背离位置, MACD上穿零轴时间 |
| 风控 | 2 | 风险等级, 宏观风险 |
| 仓位 | 3 | 建议仓位比例, 目标权重, 仓位依据 |
| 退出策略 | 5 | 止损价, T1/T2目标价, 移动止损, 盈亏比 |
| 综合报告 | 3 | 多头排列趋势, 综合分析结论/评分/级别 |
| 多因子评分 | 5 | 基本面评分, 估值评分, 动量评分, 资金流评分, MACD评分 |
| 行业百分位 | 4 | 行业内评分百分位, 动量百分位, 基本面百分位, 估值百分位 |
| 资金 | 5 | 研报买入次数, 资金动能, 5/10/20 日资金流入 |
| 链接 | 1 | 股票链接 |

子表：行业深度分析、主力研报筛选、均线多头排列、资金流向、强势股池、技术指标信号、因子衰减监控、跟仓回测详情等。

### 回测校准结果

每次回测运行后，结果保存在 `calibration_result.json` 中，最优参数自动写入 `config.ini`。运行日志记录到 `backtest_calibration_log` 数据库表。

### 缓存文件

| 文件/目录 | 说明 |
|-----------|------|
| `backtest_signal_cache/{trade_date}/{symbol}.parquet` | 信号预计算缓存（按日 + 按只，支持中断续算）|
| `CACHE_DIR/indicator_cache_v1/<bucket>/<symbol>.indicators.parquet` | Phase 0 技术指标预计算缓存（含峰值/谷值）|
| `calendar/official_trading_dates.json` | 交易日历缓存（24h TTL）|
| `StockIndes_YYYYMMDD.txt` | 股票基础信息缓存 |
| `ShareData/` | 原始数据及清洗后数据缓存 |
| `failed_symbols_{YYYYMMDD}.txt` | 当日同步失败的股票，下次运行自动重试 |

## ⚠️ 特别提醒

- 回测测试默认使用后复权
- 复盘中除价格字段之外各因子的计算默认使用后复权
- 复盘计算后输出的报告中价格字段均采用不复权（以方便用户直接查看）

## ⚠️ 注意事项

- 请确保 PostgreSQL 服务已启动且 `config.ini` 中数据库连接信息正确
- 首次运行前：`pip install -r requirements.txt`
- 数据同步依赖 AkShare，建议在交易日 15:30 后运行
- 信号预计算阶段使用 `ProcessPoolExecutor`，需确保 Python 环境支持 multiprocessing spawn
- 若 `config.ini` 缺少某些节，系统会自动补全默认值（`ConfigValidator`）
- 敏感信息（数据库密码、API Key）支持 `ENC:` 加密前缀，使用 `ConfigCipher` 工具生成
- 信号缓存按交易日后缀存储，历史缓存文件永不删除，仅在缺少当前交易日缓存时重新计算
- DAG 流水线 checkpoint 存于 `dash_pipeline_checkpoint` 表，run_id 隔离，支持断点续跑和跨日隔离

<br />

## ⚠️ 免责声明

本项目提供的所有数据、分析报告和投资建议仅供学习、研究和参考，不构成任何投资建议。投资者应自行承担投资风险，并根据自身情况做出独立的投资决策。本项目的开发者不对任何使用本系统数据或分析结果而导致的投资损失承担责任。

请务必理解并同意以上声明后，再使用本项目。

<br>

## 📜 开源协议

本项目基于 **MIT License** 授权发布。

```
MIT License

Copyright (c) 2026 BAISYS_QUANT

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
