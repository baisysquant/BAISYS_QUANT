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
     <img src="https://img.shields.io/badge/Data-AShareHub-red?style=flat-square" />
         </br> 
   <img src="https://img.shields.io/badge/Analysis-WalkForward-green?logo=pandas&logoColor=white" />
     <img src="https://img.shields.io/badge/Analysis-KDJ-green?logo=pandas&logoColor=white" />
     <img src="https://img.shields.io/badge/Analysis-MACD-green?logo=pandas&logoColor=white" />
     <img src="https://img.shields.io/badge/Analysis-BOLL-green?logo=pandas&logoColor=white" />
     <img src="https://img.shields.io/badge/Analysis-CCI-green?style=flat-square" />
     <img src="https://img.shields.io/badge/Analysis-ADX-green?style=flat-square" />
<img src="https://img.shields.io/badge/Analysis-XGBoost-brightgreen?style=flat-square" />
     <img src="https://img.shields.io/badge/Analysis-Vectorized-9cf?style=flat-square" />
     <img src="https://img.shields.io/badge/Analysis-FastAPI-blue?style=flat-square" />
     <img src="https://img.shields.io/badge/Analysis-Scikit--learn-blue?style=flat-square" />
</p>
<br />

## 📖 项目简介

百思量化是一套面向 A 股的全链路量化系统，覆盖 **数据同步 → 信号预计算 → 策略回测 → 每日分析报告** 全流程。系统分为两大阶段：

### 阶段 A — 回测校准

通过 Walk-Forward 滚动窗口优化 + 贝叶斯优化（Bayesian Optimization），自动寻优 8 个核心策略参数（5 个信号参数 + 3 个组合参数）用于日常运行。信号计算已全向量化（`numpy` + `np.select` / `np.correlate`，无 per-bar Python 循环），结合 Phase 0 指标预计算缓存与 ML 预测冻结缓存，单次信号计算 3000 只股票约 10 分钟。采用两级成本分层（信号计算分钟级 + 纯回测秒级）和 GP 代理模型（Gaussian Process + qEI 采集函数），在相同计算预算下覆盖完整参数空间。

### 阶段 B — 每日分析管线

20 步 DAG 流水线从数据库增量同步 K 线 → 多门控评分 → 13 因子 Alpha → 行业百分位过滤 → 组合构建 → 基准对比 → 风险分析（VaR/Brinson 归因）→ 因子衰减监控 → 跟仓回测 → 生成 Excel 报告 → 同步结果到 PostgreSQL。断点续跑 + run_id 版本管理。

### 设计特点

- **MACD 管线** — 默认12,26,9周期 + ATR 波动率归一化，7 维评分维度权重可配置
- **多门控递进评分 + 组合约束** — 数据质量（DAG 步骤 9a）→ 信号评分（金叉/背离/反转加分，无信号→评分参考）→ 风险否决（波动率/背离）→ 资金流修饰 → 仓位联动，组合级后处理（行业集中度 <30%、流动性否决、冲击成本控制）
- **信号衰减模型** — 金叉线性衰减（默认 37 天，回测校准）、背离 8 天半衰、K 线形态 10 天半衰
- **多因子 Alpha 评分** — 13 因子注册中心（`config/factor_registry.yaml`）：MACD(0.14) + 动量(0.16) + 资金流(0.17) + 质量(0.11) + 估值(0.11) 为核心权重，叠加龙虎榜/流动性/波动率/宏观/财务前瞻/事件驱动/舆情等卫星因子，权重经 IR-Weighted 正交化自动配权
- **因子衰减监控** — 因子 IC 值、ICIR、Decile Spread 持续跟踪，超阈值自动告警
- **行业中性化** — 行业内百分位排名的信号校准 + 申万一级行业映射
- **增量缓存续算** — 每日信号以 `signal_cache_{trade_date}_{config_hash}_{param_hash}_{data_fp}/{bucket}/{symbol}.parquet` 按只写入，中断后可自动续算已完成的股票
- **全量配置化** — 所有参数收口在 `config.ini`，支持 `ENC:` 加密敏感字段，Pydantic 自动类型校验

### 数据源

| 数据 | 来源 | 方式 |
|------|------|------|
| 日 K 线（后复权 + 不复权收盘 + 复权因子） | 腾讯行情 + AShareHub 复权因子 | 增量同步到 PostgreSQL，除权自动检测全量重写 |
| 基础信息 / 行业分类 | AkShare 申万二级分类 | 并行抓取，按日缓存 |
| 资金流向 | AkShare / AShareHub API | 多周期（3/5/10/20 日）|
| 筹码分布 | AShareHub API | 获利比例 + 成本分位 + 集中度 |
| 交易日期历 | AkShare / chinesecalendar 兜底 | 24h 缓存 TTL |
| 强势股 / 连涨股 / 量价齐升 | AkShare 市场情绪接口 | 原始数据获取阶段一并拉取 |
| 估值因子（PE/PB/市值）| AShareHub API | 日频增量同步 |
| 质量因子（ROE/毛利率）| AkShare 财务摘要 | 季度数据，增量缓算 |
| 龙虎榜 | AkShare 沪深交易所 | 上榜净买入 + 20 日聚合 |
| 宏观因子（PMI/M2/CPI）| AkShare | 宏观周期 → 申万一级行业 tilt |
| 舆情因子（新闻情感）| AkShare 百度新闻 | NLP 情感打分 [-1,1] |
| 业绩预告 / 分析师评级 | AShareHub / AkShare | 预告超预期分 + 共识分 |
| 事件驱动（回购/增减持/分红）| AkShare + AShareHub | 事件驱动总分 |
| 基准指数日线 | AShareHub | 基准对比 / Brinson 归因 |

<br>


## 🚀 核心功能与策略

### Walk-Forward 回测系统

- **全向量化信号引擎** — 信号计算已从 per-bar 逐行 Python 循环重构为纯 numpy 向量化运算。`compute_signals()` 使用 `np.where`、`np.correlate`、`np.select`、前缀和、rolling window broadcast 等技术，消除所有逐行 Python 回调。次核心指标（MACD/BOLL/KDJ/RSI/CCI/ADX）在 Phase 0 预计算阶段一次性完成，后续所有参数评估直接复用。
- **机器学习信号增强** — 引入 XGBoost/Ridge 双模型（`LogicAnalyzer/ml/signal_model.py`）作为信号修偏层。以历史金叉/背离信号及量价特征为输入，输出概率校准后的信号置信度（`apply_ml_signal`），叠加到综合评分。预测结果按 (config_hash, data_fp) 冻结缓存，窗口内仅重训一次，避免重复计算。
- **Phase 0 指标预计算缓存** — `indicator_cache.py` 在每个 WFO 窗口首次运行时预计算全部技术指标，结果写磁盘（`.indicators.parquet`）；背离峰值/谷值在 `_divergence_scores` 中滚动计算（避免未来函数 lookahead），缓存于 `.divergence.npz`。窗口内后续贝叶斯评估（信号参数变化）跳过指标复算，仅重跑评分逻辑，单次评估从 ~1 小时降至 ~5 分钟。
- **Walk-Forward 滚动优化** — 以 in-sample 训练窗口做贝叶斯优化选出最优参数，在 out-of-sample 验证，滚动覆盖全历史。OOS 默认 120 天（`out_of_sample_days` 可配置），IS 按数据长度自适应（`train_period = max(120, 总长度 - OOS - 路径数×OOS)`），多路径偏移取中位数聚合，相邻窗口 GP 状态热启动加速收敛
- **贝叶斯优化引擎** — 4 阶段优化器：Sobol 准随机初始化 → GP+qEI 信号层搜索 → GP+qEI 组合层搜索（信号固定）→ GP 代理 L-BFGS-B 精炼 Top-3。联合寻优 8 个参数：boll_narrow_ratio, cross_decay_days, golden_cross_bonus, divergence_penalty, conclusion_full_bull（信号参数），atr_stop_mult, buy_threshold, max_holdings（组合参数）。预算可在 `config.ini [BACKTEST]` 配置（默认 12 组初始化 + 24 次信号层迭代 + 20 组初始化 + 60 次组合层迭代）
- **两级成本分层** — 信号参数（5 个）触发完整管线（`prepare_backtest_data`，分钟级）；纯组合参数（3 个）直接从缓存加载信号（秒级）。`FidelityController` 自动检测输入数据是否有预计算信号列，避免不必要的重算
- **高斯过程代理模型** — `ConstantKernel × Matern(ARD) + WhiteKernel` 组合核，自动相关性长度尺度学习。`save_gp_state` / `restore_gp_state` 序列化核参数实现跨窗口迁移学习
- **混合采集函数** — `Expected Improvement - λ·σ/|μ|`（DSR 惩罚项，λ=0.05），抑制高不确定性低预期区域的采样，L-BFGS-B 多起点优化
- **性能指标** — Sharpe、Sortino、Calmar、最大回撤、VaR(95%)、CVaR(95%)、年化收益率/波动率、胜率、盈亏比、PBO（概率过拟合）、DSR（缩水 Sharpe 比）
- **仓位分配** — 引擎执行 Top-K 评分等权分配（`core.py`），候选按综合评分排序后等权买入，遵守单票上限/行业暴露约束
- **校准持久化** — 最优参数自动写入 `config.ini [BACKTEST_CALIBRATED]` 分区，回测日志记录到 `backtest_calibration_log` 表
- **信号预计算缓存** — `prepare_backtest_data()` 按 `signal_cache_{trade_date}_{config_hash}_{param_hash}_{data_fp}/{bucket}/{symbol}.parquet` 增量写入（`data_fp` 为数据指纹，用于 WFO 窗口切片隔离），分桶减小单目录文件数，断点自动续算。Phase 0 缓存位于 `CACHE_DIR/indicator_cache_v1/<bucket>/<symbol>.indicators.parquet`，线程间通过磁盘共享。
- **独立验证集模拟验证** — WFO 选参后优先使用与选参区间**无交集**的 holdout 独立验证集（默认 `holdout_ratio=0.20`，全程禁触）做 Sharpe/Sortino 衰减校验（衰减 >30% 拒绝上线），替代旧的自引用"最近 N 日"验证；独立验证集未提供时回退并告警
- **过拟合防护体系** — CPCV 净化 + 禁运（purge 5 天 / embargo 3 天剔除训练窗口重叠样本）、Diebold-Mariano (DM) 检验（Newey-West HAC，仅采纳 DM 显著窗口参数）、多重测试惩罚（同区间调参超 10 次 Sharpe/Sortino 扣 20%，超 30 次判失效）、参数稳健性自检（Sharpe>2 时最优参数 ±10% 扰动验证）、统一采纳门控（模拟验证 / OOS 衰减 / 多重测试 / 统计显著性 / 参数稳健性 / PBO≤5% / DSR≥50% 七项硬门槛，任一不过则不写入校准结果）
- **WFO 可靠性保障** — 时间预算熔断（默认 8 小时）+ 连续无改进早停（3 个窗口）、系统性失败拦截（连续 3 窗口 OOS 失效即中断流水线）、四方绑定调度（数据版本 + 配置哈希 + 频率 + 时间）、失败快照持久化（保留 14 天）、窗口预检（RELAX/OK/SKIP/LOW_CONFIDENCE/NEED_FILL 多档）+ 指标降级缩窗重算

### 真实价格体系（P0-11 审计重构）

回测引擎的成交、现金、净值、费用一律使用**不复权真实价**，复权价仅用于信号/止损/市场状态判定：

- **成交价格** — 开盘/收盘/VWAP 均按不复权真实价（VWAP = 成交额/成交量，不复权），买入成交价 = 当日真实开盘价，与 A 股实际成交口径一致
- **净值与费用** — 市值按真实价估值，佣金/印花税/过户费按真实成交额计费（旧体系按复权额计费，被复权因子放大）
- **除权处理** — 除权日按复权因子跳变比率（af_now/af_prev）自动调整持仓股数，净值跨除权日零跳变（如 10 送 10 → 股数翻倍）
- **涨跌停取整** — 涨跌停价按交易所口径 ROUND_HALF_UP 四舍五入（前收 9.55 涨停价 10.51，而非银行家舍入的 10.50）
- **可成交量** — 涨跌停开盘撮合的可成交量取**前日成交量**（PIT 合规，避免当日量前视）；数据首日（无前收）标的禁买
- **前视合规** — 当日指标一律 shift(1) 后使用（如 AMOUNT_MA20），信号日收盘决策 → 次日开盘执行（A 股 T+1）
- **效果** — 修复三类失真：①净值收益被持仓加权复权因子放大；②真实仓位低于目标仓位（目标 10% 实际仅 3.3%）；③费用按复权成交额高估

### 数据同步（IncrementalSyncEngine）

- 增量同步 A 股日 K 线（腾讯行情接口，HFQ 后复权 + 不复权收盘 + 复权因子），自动检测除权事件并全量重写
- 申万行业分类基础信息拉取（`ThreadPoolExecutor(10)`，~40s），含申万一级行业映射（`SwIndustrySync`）
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

### 评分管道（多门控评分 + 组合约束）

```
Gate 0: 数据质量   →  由 DAG 步骤 9a DataQualityChecker 执行（写 dash_quality_log 表）
Gate 1: 入场信号   →  金叉/背离/反转等加分规则评分；无信号时追加"评分参考"，级别由综合评分阈值决定
Gate 2: 风险过滤   →  高波/顶背离/低成交额 → 否决（拦截 ~10~15%）
Gate 3: 资金修饰   →  资金流/量价修饰评分
Gate 4: 仓位联动   →  风险等级驱动 position_adjust 系数 (ATR/止损比)
Gate 5: 组合约束   →  行业集中度 <30%，总仓位 <100%（由 PortfolioBuilder 执行）+ 流动性否决（liq_veto_ratio）
```

### 资金流 & 筹码

- 多周期资金净流入（3/5/10/20 日），主力/大户/散户细分
- 筹码分布：获利比例、成本分位（5%/50%/95%）、集中度、阻力位规则
- 市场状态分类：STRONG_TREND / WEAK_TREND / BOTTOM_REVERSAL / TOP_RISK / OSCILLATION
- 宏观前置过滤：L1 上证指数 MA60/120 空头排列 → L2 量价验证 → L3 全 A 上涨比例；HIGH_RISK → 跳过全部标的，MEDIUM → 评分阈值上浮 15%
- 资金动能双因子：趋势（权重 0.4）+ 速度（权重 0.6）

### 输出

- **Excel 报告** — 输出个股分析报告
- **数据库同步** — 结果写入 `ods_ak_ranking_stocks`、`ods_ak_industry_analysis`、`app_stock_strategy_report`、`ods_factor_ic_history`、`dash_pipeline_checkpoint`、`dash_run_log`、`dash_quality_log` 等表

<br />

## 📊 打造个性化交易系统

通过修改 `config.ini` 适应不同交易风格：

**短线激进型**

```ini
[TECHNICAL_INDICATORS]
macd_params = 6,13,5              ; 超短敏感 MACD

[FUND_FLOW]
fund_flow_periods = 3,5,10        ; 短期资金流（仅允许 3,5,10 / 3,5,20 / 5,10,20 / 3,10,20 四种三周期组合）

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

[FUND_FLOW]
fund_flow_periods = 5,10,20       ; 多周期验证

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

[FUND_FLOW]
fund_flow_periods = 5,10,20       ; 中长期资金流

[POSITION_SIZING]
kelly_fraction = 0.3
position_a = 0.35                 ; A 级仓位 35%
max_single_position = 0.33
```
</br> </br> 
## 🛠️ 安装与配置

### 环境要求

- **Python 3.10+**（推荐 3.12~3.13）
- **PostgreSQL 14+** — 数据持久化存储
- **AkShare** — 免费使用，项目通过 `AkshareConfig` 补丁提供全局 30s 超时

### 数据库准备

1. 创建数据库（名称任意，默认 `Corenews`）
2. 执行 `PostgreSQL建表语句.sql` 创建全部表结构
3. 配置 `config.ini` 中 `[DATABASE]` 节的连接参数

### AShareHub API

筹码分布、资金流向、估值、财务预告等数据需要 [AShareHub](https://www.asharehub.com) API 密钥。

```ini
[ASHAREHUB]
api_key = ENC:gAAAAAB...         ; 支持 ENC 加密
enable_chip_distribution = true
enable_fundamentals = true
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
| `enable_fundamentals` | 布尔 | `true` | 是否获取估值/财务等基本面数据 |
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
| `atr_t1_mult` | `4.0` | T1 目标价 = close + ATR × 此值 |
| `atr_t2_mult` | `5.0` | T2 目标价 = close + ATR × 此值 |

**行业估值聚合：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `industry_valuation_agg_method` | `aggregate_profitable` | 行业估值聚合方式 |

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
| `fundamentals_retry` | 整数 | `3` | 基本面数据拉取重试次数 |
| 因子权重 | — | — | 权重定义已迁移至 `config/factor_registry.yaml`（13 因子注册表） |


---

#### [POSITION_SIZING] — 仓位管理

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_single_position` | `0.33` | 单只股票最大仓位比例 |
| `default_win_rate` | `0.55` | 默认胜率 |
| `position_b` | `0.15` | B 级基础仓位 |
| `position_c` | `0.05` | C 级基础仓位 |
| `max_industry_exposure` | `0.30` | 单行业最大暴露 |
| `max_day_turnover` | `0.20` | 单日最大双边换手率 |
| `risk_aversion` | `1.0` | 风险厌恶系数 |
| `risk_budget` | `0.02` | 风险预算（组合波动率上限） |
| `risk_none_multiplier` | `1.0` | NONE 风险等级仓位系数 |
| `max_single_impact` | `0.02` | 单票最大冲击成本 |
| `impact_threshold` | `0.01` | 冲击成本阈值 |
| `impact_base` | `0.002` | 冲击成本基数 |

---

#### [BACKTEST_CALIBRATED] — 回测自动校准参数

其中 8 个参数（boll_narrow_ratio / cross_decay_days / conclusion_full_bull / golden_cross_bonus / divergence_penalty / atr_stop_mult / buy_threshold / max_holdings）由 Walk-Forward 寻优引擎在回测期间自动搜索最优值并写回本分区；其余为静态默认值，日常运行无需手动修改。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `boll_narrow_ratio` | <font color="red">`0.9`</font> | 窄布林判定：带宽/历史均值 < 此值 → 震荡（由回测优化） |
| `cross_decay_days` | <font color="red">`37`</font> | 金叉信号衰减天数，天（由回测优化） |
| `atr_stop_mult` | <font color="red">`2`</font> | ATR 止损倍数：止损价 = close - ATR × 此值（由回测优化） |
| `conclusion_full_bull` | <font color="red">`80`</font> | MACD 综合评分 ≥ 此值 → A 级（由回测优化） |
| `golden_cross_bonus` | <font color="red">`10`</font> | R04: 金叉量价确认加分（由回测优化） |
| `divergence_penalty` | <font color="red">`20`</font> | R41: 顶背离量缩扣分（由回测优化） |
| `buy_threshold` | <font color="red">`17`</font> | 买入评分阈值（由回测优化） |
| `max_holdings` | <font color="red">`11`</font> | 最大持仓数（由回测优化） |
| `atr_t1_mult` | `4` | T1 目标价 ATR 倍数（静态） |
| `liq_veto_ratio` | `0.065` | 流动性否决比（静态） |
| `kelly_fraction` | `0.3` | Kelly 仓位比例系数（静态） |
| `position_a` | `0.35` | A 级基础仓位（静态） |
| `atr_t2_mult` | `5.0` | T2 目标价 ATR 倍数（静态） |
| `risk_none_multiplier` | `1.0` | NONE 风险等级仓位系数（静态） |

---

#### [BACKTEST] — 回测系统

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | `true` | 是否启用回测校准 |
| `optimize_frequency` | `monthly` | 校准频率 |
| `backtest_start_date` | `20240101` | 回测起始日期 |
| `out_of_sample_days` | `120` | Walk-Forward 样本外窗口天数 |
| `holdout_ratio` | `0.20` | holdout 独立验证集比例（全程禁触） |
| `wfo_num_paths` | `3` | Walk-Forward 多路径数 |
| `exclude_st` | `true` | 是否剔除 ST/退市整理股 |
| `initial_cash` | `1000000` | 初始资金 |
| `signal_pipelines` | `6` | 信号预计算并行管道数 |
| `execution_model` | `next_open` | 成交时点模型：`next_open` 信号次日开盘成交（默认，符合A股T+1）/ `vwap` 信号次日VWAP成交 |
| `bayesian_n_init_signal` | `12` | 信号层 Sobol 初始化组数 |
| `bayesian_n_iter_signal` | `24` | 信号层 GP+qEI 迭代次数 |
| `bayesian_n_init_portfolio` | `20` | 组合层 Sobol 初始化组数 |
| `bayesian_n_iter_portfolio` | `60` | 组合层 GP+qEI 迭代次数 |
| `bayesian_cpcv_purge_days` | `5` | CPCV 净化天数 |
| `bayesian_cpcv_embargo_days` | `3` | CPCV 禁运天数 |
| `bayesian_time_budget_seconds` | `28800` | 贝叶斯优化时间预算（秒，8h） |
| `bayesian_max_no_improve_windows` | `3` | 连续无改进早停窗口数 |
| `snapshot_enabled` | `true` | 失败快照持久化 |
| `precheck_mode` | `RELAX` | 窗口预检模式（RELAX/OK/SKIP/LOW_CONFIDENCE/NEED_FILL） |
| `calendar_align_mode` | `on` | 交易日历对齐（官方日轴） |
| `indicator_degradation` | `RELAX` | 指标降级模式（窗口不足时缩窗重算） |
| `simulate_limit_up_down` | `true` | 涨跌停分档撮合模拟 |
| `limit_seal_ratio` / `limit_tradable_ratio` / `limit_intraday_ratio` / `limit_seal_decay` | — | 一字/盘中/冲板三类成交比例 + 连板衰减 |
| `max_order_pct` | `0.30` | 单笔委托占成交量上限 |
| `resume_gap_up` / `resume_gap_down` | `0.05` | 复牌跳空处理阈值 |
| `handling_fee_rate` | `0.0000341` | 经手费率 |
| `csrc_fee_rate` | `0.00002` | 证管费率 |
| `stamp_tax_segments` | `2023-08-28:0.0005;2000-01-01:0.001` | 印花税历史分段费率 |

**贝叶斯寻优参数范围（逗号分隔 min,max,step）：**

**信号参数：**

| 参数 | 默认值 | 寻优对象 |
|------|--------|----------|
| `boll_narrow_ratio_range` | `0.6,1.2,0.1` | 布林窄幅比 |
| `cross_decay_days_range` | `3,60,3` | 金叉衰减天数 |
| `conclusion_full_bull_range` | `60,95,5` | A 级评分阈值 |
| `golden_cross_bonus_range` | `5,20,5` | R04 金叉加分 |
| `divergence_penalty_range` | `10,40,5` | R41 顶背离扣分 |

**组合参数：**

| 参数 | 默认值 | 寻优对象 |
|------|--------|----------|
| `atr_stop_mult_range` | `1.0,3.0,0.5` | ATR 止损倍数 |
| `buy_threshold_range` | `5,30,5` | 买入评分阈值 |
| `max_holdings_range` | `3,20,1` | 最大持仓数 |

---

#### [TRADING_COST] — A股交易成本（回测 + 跟仓）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `commission_rate` | `0.0003` | 佣金费率（万三） |
| `stamp_tax_rate` | `0.0005` | 印花税费率（卖出单向，2023-08-28 起万五） |
| `stamp_tax_segments` | `2023-08-28:0.0005;2000-01-01:0.001` | 印花税历史分段费率 |
| `transfer_fee_rate` | `0.00001` | 过户费率（双向，万0.1） |
| `handling_fee_rate` | `0.0000341` | 经手费率 |
| `csrc_fee_rate` | `0.00002` | 证管费率 |
| `min_commission_per_trade` | `5.0` | 单笔最低佣金（元） |

---

#### [POSITION_BACKTEST] — 跟仓回测

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pool_file_path` | `证券交割单.xlsx` | 历史交易记录池文件路径（FIFO 匹配） |

---

#### [API] — API 服务（ApiServer）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | 布尔 | `0` | 是否启用 FastAPI 服务 |
| `host` | 字符串 | `127.0.0.1` | 监听地址 |
| `port` | 整数 | `8000` | 监听端口 |
| `alert_webhook_url` | 字符串 | 空 | 告警 Webhook 地址 |
| `alert_channel` | 字符串 | `generic` | 告警渠道：generic/wecom/feishu/dingtalk |
| `alert_on_failure` | 布尔 | `1` | 管线失败时告警 |
| `alert_on_success` | 布尔 | `0` | 管线成功时告警 |

---

#### [DISTRIBUTION] — 东方财富主力成本 API

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `api_token` | 字符串 | - | 主力成本数据 API Token（当前为明文，建议改用 ENC 加密） |

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
  │     ├── 解析股票列表 → 拉取 K 线（数据库 advisory lock 防并发）
  │     ├── 信号预计算 (prepare_backtest_data)
  │     │   ├── Phase 0: 技术指标预计算 (indicator_cache.py)
  │     │   │   └── MACD/ATR/BOLL/KDJ/CCI/RSI/ADX → 磁盘缓存 + 背离峰值谷值 (divergence.npz)
  │     │   ├── 全向量化信号评分 (vectorized_signal.py)
  │     │   │   └── numpy 向量化: 背离广播/斜率卷积/形态前缀和/金叉衰减曲线
  │     │   ├── XGBoost 信号修偏 (LogicAnalyzer/ml/signal_model.py)
  │     │   │   └── 概率校准 → 叠加到综合评分（预测冻结缓存，仅重训一次）
  │     │   └── 并行 ThreadPoolExecutor + 增量 parquet 缓存
  │     ├── Walk-Forward 滚动优化（多路径 + 贝叶斯优化）
  │     │   ├── 滑动窗口: in-sample 数据自适应（≥120 天）贝叶斯寻优
  │     │   └── out-of-sample 120 天验证 + holdout 独立终验集
  │     ├── 全量回测 (run_full_backtest) — 最优参数
  │     ├── 绩效指标计算 (Sharpe/Sortino/Calmar/VaR/胜率/PBO/DSR)
  │     ├── 统一采纳门控（7 项硬门槛校验）
  │     └── 保存校准结果 → calibration_result.json + config.ini
  │
  └── [阶段 B] 每日分析管线 DAG 20 步 (StockAnalysisCoordinator)
        ├─ 01 同步历史K线 (IncrementalSyncEngine)
        ├─ 02 格式化股票代码 (CodeNormalizer)
        ├─ 03 获取原始数据 (资金流/强势股/行业板块/龙虎榜/宏观/舆情/事件)  ─┐
        ├─ 04 获取K线数据及最新价                                        ─┤ 并行可独立执行
        ├─ 05 处理技术指标信号 (MACD 7维/KDJ/CCI/RSI/BOLL)              ─┤
        ├─ 06 行业分析 (IndustryFlowAnalyzer)                           ─┘
        ├─ 07 均线突破及过滤 (依赖 03+04)
        ├─ 08 准备处理数据字典 (合并 05+06+07)
        ├─ 09 合并分析数据 (DataProcessingService)
        ├─ 9a 数据质量检查 (DataQualityChecker, 写 dash_quality_log)
        ├─ 10 映射行业信号 + 行业中性化（含申万一级 tilt）
        ├─ 11 多因子 Alpha 评分 (13 因子注册表加权 → fuse_scores + 正交化)
        │     └── 行业百分位 Stage 1：in-cache 缓存
        ├─ 12 剔除弱势股 (三级过滤：趋势豁免→硬百分位→D/C级)
        ├─ 13 组合构建 (PortfolioBuilder: Kelly+流动性+冲击成本)
        ├─ 14 基准对比 (BenchmarkEvaluator: 沪深300/全A等权)
        ├─ 15 因子衰减监控 (FactorDecayMonitor: IC/ICIR 跟踪)
        ├─ 16 跟仓回测分析 (PositionTrackingService: FIFO 匹配)
        ├─ 风险分析 (VaR/ES、Brinson 归因、因子风险归因)
        ├─ 17 生成Excel报告 (ReportService: 52列+行业/因子/风险子表)
        └─ 18 同步结果到数据库 (ods_* / app_stock_strategy_report / dash_*)
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

**`审计报告_YYYYMMDD.xlsx`** — 每日全市场分析结果，默认 52 列及多个子表：

| 区块 | 列数 | 包含列 |
|------|------|--------|
| 基础信息 | 8 | 股票代码, 股票简称, 行业, 所属行业信号, 最新价, 95%筹码价, 主力成本, 成本位置 |
| 资金流信号 | 5 | 强势股, 量价齐升, 量价配合, 连涨天数, 放量天数 |
| MACD 评分 | 4 | MACD趋势, 金叉信号, 柱状动能, DIF斜率 |
| 技术指标 | 5 | KDJ/CCI/RSI/BOLL/K线形态信号 |
| 均线参考 | 3 | 10/30/60 日均线价 |
| 背离 | 4 | 背离信号, 背离距今, 背离位置, MACD上穿零轴时间 |
| 风控 | 2 | 风险等级, 宏观风险 |
| 仓位 | 2 | 建议仓位比例, 目标权重 |
| 退出策略 | 4 | 止损价, T1/T2目标价, 移动止损 |
| 综合报告 | 3 | 多头排列趋势, 综合分析结论/评分/级别 |
| 多因子评分 | 11+ | 基本面/估值/动量/资金流/MACD 等 13 因子评分 |
| 行业百分位 | 10 | 行业内评分/动量/基本面/估值等百分位 |
| 资金 | 5 | 研报买入次数, 资金动能, 5/10/20 日资金流入 |
| 链接 | 1 | 股票链接 |

子表：数据汇总、行业深度分析、主力研报筛选、主力成本分析、跟仓回测、基准对比，以及风险分析子表（风险 VaR 分析、Brinson 归因、因子风险归因）。

### 回测校准结果

每次回测运行后，结果保存在 `calibration_result.json` 中，通过 7 项硬门槛（模拟验证 / OOS 衰减 / 多重测试惩罚 / 统计显著性 / 参数稳健性 / PBO≤5% / DSR≥50%）后方写入 `config.ini`。运行日志记录到 `backtest_calibration_log` 数据库表。

### 缓存文件

| 文件/目录 | 说明 |
|-----------|------|
| `signal_cache_{trade_date}_{config_hash}_{param_hash}_{data_fp}/<bucket>/<symbol>.parquet` | 信号预计算缓存（按日 + 按只 + 分桶，支持中断续算）|
| `CACHE_DIR/indicator_cache_v1/<bucket>/<symbol>.indicators.parquet` | Phase 0 技术指标预计算缓存（含 `.divergence.npz` 背离峰谷）|
| `failure_snapshots/<yyyy-mm-dd>/<bucket>/<snapshot_id>.parquet` | 回测失败快照（保留 14 天）|
| `calendar/official_trading_dates.json` | 交易日历缓存（24h TTL）|
| `StockIndes_YYYYMMDD.txt` | 股票基础信息缓存 |
| `ShareData/` | 原始数据及清洗后数据缓存 |
| `failed_symbols_{YYYYMMDD}.txt` | 当日同步失败的股票，下次运行自动重试 |

## 🖥️ API 服务（ApiServer）

内置 FastAPI 服务（`python -m ApiServer.app`），默认关闭（`[API] enabled=0`）。提供：

- `GET /api/v1/health` — 数据库探活
- `POST /api/v1/pipeline/run` + `GET /pipeline/status` / `runs` / `runs/{run_id}` / `steps/{run_id}` — 管线触发与状态查询（基于 `dash_run_log` / `dash_pipeline_checkpoint`）
- `GET /api/v1/factors/registry` / `ic-history` — 因子注册表与 IC 历史
- `GET /api/v1/pipeline/quality-log` — 数据质量日志
- `GET/POST /api/v1/alert/config` + `POST /alert/test` — 告警配置
- 告警渠道：企业微信 / 飞书 / 钉钉 / 通用 Webhook（管线完成/失败通知）

## 🧰 工具集（TreasureBox）

独立运行的辅助分析工具：

| 工具 | 功能 |
|------|------|
| `SingleStockAnalyzer.py` | 交互式单只股票技术分析（MACD 7 维/K线形态/KDJ/CCI/RSI/BOLL 全信号 + 综合结论）|
| `DoubleBottomAnalyzer.py` | 全市场 MACD+KDJ 双谷（双重底）形态扫描，输出"双谷探底机会点"Excel |
| `KDJ_Signal.py` | 从 `app_stock_strategy_report` 筛选 KDJ 信号并验证信号日后股价表现 |
| `MacdZeroAxisPattern.py` | 全市场 MACD"零轴完整反弹形态"扫描 |
| `个股新闻查询.py` | 东方财富个股新闻批量查询导出 |

<br />

## ⚠️ 特别提醒

- 复盘计算中除价格字段之外各因子的计算默认使用后复权
- 复盘计算后输出的报告中价格字段均采用不复权（以方便用户直接查看）

## ⚠️ 注意事项

- 请确保 PostgreSQL 服务已启动且 `config.ini` 中数据库连接信息正确
- 首次运行前：`pip install -r requirements.txt`
- 数据同步依赖 AkShare，建议在交易日 15:30 后运行
- 信号预计算阶段使用 `ThreadPoolExecutor`（Windows spawn 下进程池会死锁）
- 若 `config.ini` 缺少必需节（DATABASE/SYSTEM/LOGGING/MULTI_HEAD_ARRANGEMENT/FUND_FLOW/TECHNICAL_INDICATORS），`ConfigValidator` 自动补全默认值并备份 `config.ini.bak`
- 敏感信息（数据库密码、API Key）支持 `ENC:` 加密前缀，使用 `ConfigCipher` 工具生成；注意 `[DISTRIBUTION]` 的 `api_token` 目前为明文，建议改用加密值
- 信号缓存按交易日后缀存储，历史缓存文件永不删除，仅在缺少当前交易日缓存时重新计算
- DAG 流水线 checkpoint 存于 `dash_pipeline_checkpoint` 表，run_id 隔离，支持断点续跑和跨日隔离
- 回测与流水线通过数据库 advisory lock 互斥，避免并发冲突

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
