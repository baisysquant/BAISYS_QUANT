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

百思量化是一套面向 A 股的全链路量化分析系统，覆盖从数据获取、技术指标计算、信号评分到 Excel 报告生成的全流程。

### 核心管线

系统以 `StockAnalysisCoordinator` 13 步流水线驱动，每日自动完成：

1. **数据层** — 从 PostgreSQL 同步历史 K 线，从 AkShare 获取实时行情、资金流向、强势股池、行业板块等原始数据
2. **指标层** — `TASignalProcessor` 并行计算 MACD、KDJ、CCI、RSI、BOLL 五大指标，结合 MACD 7 维管线评分体系（趋势/金叉/动能/斜率/背离/量价/K 线形态）
3. **评分层** — 6 道门控规则递进式评分管道：Gate 0（数据质量筛查）→ Gate 0.5（宏观环境注入）→ Gate 1（共振信号评分）→ Gate 2（波动率/背离/风险过滤）→ Gate 3（资金流/量价修饰）→ Gate 4（仓位联动调整），叠加多时间帧对齐、波动率状态切换、信号衰减模型
4. **合并层** — 基础信息、资金流信号、技术指标、行业信号、筹码分布等多源数据统一合并
5. **输出层** — 生成 44 列结构化 Excel 报告（含建议仓位比例），同时同步到 PostgreSQL 数据库

### 设计特点

- **单参数 MACD 管线** — 摒弃双周期冗余，聚焦 (12,26,9) 单参数 + ATR 波动率归一化，7 维评分维度权重可配置
- **行业中性化** — 行业内百分位排名的信号校准，避免行业偏倚
- **信号衰减模型** — 金叉 30 天半衰、背离 8 天半衰、K 线形态 10 天半衰，保证信号时效性
- **退出策略层** — ATR 倍数驱动的止损/目标价/移动止损输出，作为独立信息层不参与评分
- **配置化** — 所有阈值、权重、参数统一收口在 `config.ini`，单文件管理

### 4 道门控规则的递进式评分管道

评分管道是系统的决策核心，采用递进式门控设计，逐层过滤高风险标的：

```
┌─ Gate 1: 入场信号检查 ──────────────────────────┐
│  条件：signal_list 为空 → 直接返回 C 级          │
│  判定：无 MACD 金叉、无底背离、无 K 线反转信号    │
│  → 视为"无明确入场信号"，不进入后续评分             │
└───────────────────── 拦截约 50% 标的 ─────────────┘
                        ↓ 通过
┌─ Gate 2: 风险等级上浮 ───────────────────────────┐
│  条件：risk_level == HIGH → 返回当前状态          │
│  触发：筹码密集区高位 + 弱势/顶部风险市场           │
│  → 视为"高风险"，不计算最终评分                     │
└───────────────────── 拦截约 5~10% 标的 ───────────┘
                        ↓ 通过
┌─ Gate 3: 筹码风控 ───────────────────────────────┐
│  执行：_apply_chip_risk(state, df)                │
│  检查：获利比例 > 80%、筹码集中度、成本阻力位       │
│  → 上浮 risk_level（LOW→MEDIUM→HIGH）             │
└──────────── 影响 risk_level，不直接拦截 ──────────┘
                        ↓ 通过
┌─ 最终评分层 ─────────────────────────────────────┐
│  7 维加权求和 → 多时间帧乘数 → 资金流奖励          │
│  → 输出: level(A/B/C/D) + score(0~100)            │
└────────────────── 全量输出 ───────────────────────┘
```

**Gate 1** 负责拦截无信号标的 — 这是最主要的空值来源（约 50% 股票无金叉/背离/反转信号，直接判 C 级）。

**Gate 2** 负责识别筹码结构风险 — 主要在弱势趋势或顶部风险市场中生效。

**Gate 3** 负责动态上浮风险等级 — 结合筹码分布数据做细粒度风控，不影响评分计算但影响最终风险等级输出。

> 核心目标：将复杂的 A 股市场波动转化为可量化的、可复现的每日分析报告，辅助投资决策。

<br>


## 🚀 核心功能与策略

**多源数据整合**

AkShare：获取实时行情、主力研报、财务摘要、市场资金流向、强势股池、连涨股、量价齐升、持续放量、均线突破等数据。

AShareHub：通过 API 获取筹码分布数据（成本分布、获利比例），用于筹码风控规则和评分修饰。
</br> </br> 

**历史K线数据同步**

PostgreSQL 数据库：自动检测并同步 A 股主要上市公司的日 K 线数据，支持增量更新和除权信息自动校验及全量重写，确保历史数据的准确性和完整性。
</br> </br> 

**全面股票分析**

基础行情：最新价、股票简称。

研报洞察：分析师买入评级次数。

趋势捕捉：均线多头排列判断（10/30/60日均线）。强势股、量价齐升、连涨天数、持续放量。所属行业是否为当日涨幅 Top10 行业。
</br> </br> 

**资金流三周期分析**
- **灵活配置**：用户可自定义资金流观察周期（3/5/10/20日）
- **预设优化组合**：提供四种经实战验证的参数组合：
  - `3510`：3日、5日、10日资金流向
  - `3520`：3日、5日、20日资金流向  
  - `51020`：5日、10日、20日资金流向
  - `31020`：3日、10日、20日资金流向

**技术指标信号**

**向量化MACD策略**
- **默认周期**：(12, 26, 9) 经典 MACD 配置，可通过 `macd_params` 自由修改
- **信号分析**：7 维评分体系 — MACD 趋势（20 分）、金叉信号（15 分）、柱状动能（15 分）、DIF 斜率（10 分）、背离信号（10 分）、量价配合（10 分）、K 线形态（10 分）
- **MACD 趋势分类**：SUPER_STRONG / STRONG / WEAK / SUPER_WEAK，决定评分乘数与仓位乘数
- **DIF 斜率**：线性回归拟合斜率 + R² 拟合优度，识别趋势方向与确定性
- **背离检测**：顶背离（一票否决）、底背离，含强度阈值（0.15）和半衰期衰减（默认 8 天），输出背离距今天数及价格
- **多重时间帧**：日线 + 周线 MACD 对齐评分乘数（共振多头 ×1.1，周线空头 ×0.5）

**ATR（平均真实波幅，14 日）：** 波动率中枢识别，金叉强度归一化分母，仓位波动率上限（风险预算 2%），止损/止盈价计算（T1=ATR×3.0, T2=ATR×5.0），高波动过滤规则（ATR/价格 >5%~8%）。

**ADX（趋势强度，14 日）：** 区分高波动趋势（ADX>25）与低波动反转（ADX<20），切换评分策略。

**CCI（14 日）：** 极度超买/超卖，强势/弱势超买/超卖，常态波动。与 MACD、BOLL 共振评分。

**RSI（14 日）：** 超卖低位及底背离判断。与 KDJ、成交量共振评分。

**BOLL（20, 2σ）：** 带宽（上轨-下轨）/价格，低波/缩口/张口状态。与 MACD、CCI 共振评分。

**KDJ（9, 3, 3）：** 14 种信号模式 + 金叉/死叉检测：
- 极值 J 线反转、底背离金叉、趋势确认金叉、低位超卖金叉
- 深度超卖反弹、J 线高位拐头、K 线快速拉升、三线聚合突破
- 死叉回踩支撑、J 线极限值回归、背离信号、振荡区间突破
- KDJ 三线同步、超卖修复启动
- 与 MACD、RSI、成交量三金叉共振（+5 分）

**K 线形态（TA-Lib，25+ 种）：** 吞没、十字星、锤子线、启明星、黄昏星等。评分 -10~+10，按强反转/中反转/弱信号/持续分级，叠加时间衰减。

**量价分析：**
- 量价趋势评分：5 日价格变化 + 成交量趋势
- 分类：量价齐升、价涨量缩、放量下跌、缩量下跌
- 三连阳量递增（+5 分）
- 成交量健康检查：close>MA60 + vol>MA5 均量

**资金流分析：**
- 多周期资金净流入（3/5/10/20 日，万元）
- 资金动能信号（主力/大户/中户/散户买卖细分）
- 规则：资金净流入 + MACD 看涨（+3），资金净流出 + MACD 看跌（-3）

**筹码分布（AShareHub）：**
- 获利比例（winner_rate）：底仓确认、高位获利盘风险
- 成本分位：cost_5pct（下沿），cost_50pct（中位），cost_95pct（上沿）
- 筹码集中度：(cost_95 - cost_5) / cost_5
- 筹码阻力位规则

**市场状态（Regime Detection）：**
- STRONG_TREND：多头排列 + 正斜率 + 正动量
- WEAK_TREND：空头排列 + DIF<0
- BOTTOM_REVERSAL：DIF 从负值上升
- TOP_RISK：价格偏离 MA20 + DIF 下降
- OSCILLATION：窄布林带 + 低柱状图
- 波动率情景：HIGH_VOL_TREND（ATR↑>30% + ADX>25），LOW_VOL_REVERSAL（ATR↓<30% + ADX<20）
</br> </br> 

**主力成本分析：**

成本差价计算：自动计算当前价格与主力成本的差价及百分比

成本位置判断：智能识别股价相对主力成本的位置（大幅/略高于/低于成本）

机构参与度分级：将机构参与度划分为低/中低/中高/高等四个等级

主力控盘强度评估：综合分析主力控盘情况（高度/中度/轻度/低度控盘）

行业趋势分析：基于即时、3日、5日、10日、20日行业资金流向，计算行业资金分、价格分、换手分、趋势得分，并识别“资金主攻”、“退潮预警”、“黄金坑潜入”、“低位强异动”等行业信号。
</br> </br> 

**智能筛选与报告生成**

根据多重信号组合，智能剔除“弱势且加速下跌”的股票，聚焦潜力标的。

生成结构清晰、多工作表的 Excel 报告，方便查阅所有分析结果。
</br></br> 

**本地缓存与并发**

利用本地文件缓存机制，减少重复 API 调用，加快程序运行速度。

多线程并发处理股票数据获取和技术分析，提高整体效率。

个股新闻查询工具：提供独立的脚本，支持查询指定股票在过去 30 或 60 天内的新闻资讯，并保存为 Excel 文件。

<br />

## 📊打造个性化交易系统


**场景示例：短线交易者**

[TECHNICAL_INDICATORS]

macd_params = 6,13,5  超短线敏感周期

[SYSTEM]

FUND_FLOW_PERIODS = [3, 5]  关注短期资金流

EXEMPT_LEVELS = ["完全主升", "趋势加速"] 仅保留强势股

</br> 

**场景示例：中线投资者**

[TECHNICAL_INDICATORS]

macd_params = 24,52,18  中线趋势周期

[SYSTEM]

FUND_FLOW_PERIODS = [5, 10, 20]    多周期资金验证

FULL_BULL_THRESHOLD = 80           提高完全主升标准

</br> 

**场景示例：稳定投资者（默认）**

[TECHNICAL_INDICATORS]

macd_params = 12,26,9  经典均衡参数

[SYSTEM]

FUND_FLOW_PERIODS = [10, 20]      关注中长期资金

ENABLE_COST_ANALYSIS = True       用主力成本分析
</br> </br> 
## 🛠️ 安装与配置

**Python 环境**

确保您已安装 Python 3.13+ 版本。
</br> </br> 

**数据库**

PostgreSQL：请确保您的系统已安装并运行 PostgreSQL 数据库。

数据库创建：在 PostgreSQL 中创建一个名为 Corenews 的数据库，并记住您的数据库用户名和密码。

创建表结构：执行PostgreSQL建表语句.sql中的 SQL 语句，创建必要的数据表。
</br> </br> 

**AShareHub API Key**

筹码分布数据需要 AShareHub API 密钥。免费版每日 100 次调用。

1. 前往 [AShareHub 官网](https://www.asharehub.com) 注册并获取 API Key
2. 编辑 `config.ini` 中 `[ASHAREHUB]` 节：

```ini
[ASHAREHUB]
api_key = ENC:gAAAAAB...         # 你的 API Key（支持 ENC 加密）
enable_chip_distribution = true  # 是否获取筹码分布数据
chip_limit = 1                   # 拉取快照数：1=仅最新快照，>1 拉取多日历史
```

密钥加密方式与数据库密码相同，使用 `ConfigCipher` 工具类。
若不需要筹码分布功能，可将 `enable_chip_distribution` 设为 `false` 跳过。

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `api_key` | 字符串 | - | API 密钥（支持 `ENC:` 加密） |
| `enable_chip_distribution` | 布尔 | `false` | 是否启用筹码分布获取 |
| `chip_limit` | 整数 | `1` | 拉取历史快照数：`1`=最新快照，`>1`=多日历史（最大 200） |

**AkShare**：大部分接口无需 Token，但有访问频率限制。本项目已内置缓存文件和延迟机制以应对。

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
| `boll_narrow_ratio` | 浮点 | `0.8` | 窄布林判定：近期 BOLL 带宽 < 历史均值 × 此值 → 震荡 |
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

### [SCORING_PARAMS] — 评分计算参数

> ⚠️ **纯自研参数**，控制衰减模型、波动率归一化、退出策略。

**衰减相关：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `cross_decay_days` | `30` | 金叉信号衰减半衰期（天） |
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
| `atr_stop_mult` | `1.5` | 止损价 = close - ATR × 此值（行业范围 1.5～3.0） |
| `atr_t1_mult` | `3.0` | T1 目标价 = close + ATR × 此值 |
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

执行 `MainShareAnalysis.py` 启动全自动化分析流程。

```
MainShareAnalysis
 └─ StockAnalysisCoordinator (13步流水线)
      ├─ Step 01: 同步历史数据 (sync_data)
      ├─ Step 02: 格式化股票代码 (format_codes)
      ├─ Step 03: 获取原始数据 (get_raw_data)
      ├─ Step 04: 获取K线数据及最新价
      │   └─ stock_daily_kline (DB) → hist_df
      ├─ Step 05: 处理技术指标信号
      │   └─ TASignalProcessor (并行) → 7个DataFrame
      │        ├─ MACD_FULL_BULL  ← pipeline_analysis (7维评分)
      │        ├─ KDJ / CCI / RSI / BOLL / KLINE_PATTERN
      │        └─ 均线突破 (xstp)
      ├─ Step 06: 运行行业分析
      │   └─ IndustryFlowAnalyzer → sw_data_cache
      ├─ Step 07: 处理均线突破数据
      ├─ Step 08: 准备处理数据字典
      ├─ Step 09: 合并处理数据 (consolidate_data)
      │   ├─ merge_basic_info (名称+行业+最新价)
      │   ├─ calculate_bull_scores (多头排列评分)
      │   ├─ merge_fund_flow_data (资金流+信号)
      │   ├─ merge_technical_indicators (MACD/KDJ/CCI/RSI/BOLL)
      │   ├─ merge_special_data (主力成本+均线突破)
      │   ├─ filter_signal_stocks
      │   └─ sort_and_format_report → reorder_columns
      ├─ Step 10: 映射行业信号
      │   └─ merge_industry_signal + industry_neutralization
      ├─ Step 11: 剔除弱势股
      ├─ Step 12: 生成Excel报告
      └─ Step 13: 同步结果到数据库
```

您可以在控制台看到详细的日志输出，追踪程序的运行状态。

<br /></br> 

## 📊 输出结果

所有报告和缓存文件生成在 `config.ini` 中 `HOME_DIRECTORY` 指定的目录下（默认 `~/Downloads/CoreNews_Reports`）。

### Excel 报告

**`审计报告_YYYYMMDD.xlsx`** — 每日全市场分析结果，第一工作表（数据汇总）包含 43 列，按功能区块排列：

| 区块 | 列数 | 包含列 |
|------|------|--------|
| 基础信息 | 7 | 股票代码, 股票简称, 行业, 所属行业信号, 最新价, 主力成本, 成本位置 |
| 资金流信号 | 5 | 强势股, 量价齐升, 量价配合, 连涨天数, 放量天数 |
| MACD 7 维评分 | 4 | MACD趋势, 金叉信号, 柱状动能, DIF斜率 |
| 独立技术指标 | 5 | KDJ_Signal, CCI_Signal, RSI_Signal, BOLL_Signal, K线形态信号 |
| 均线参考 | 3 | 10日均线价, 30日均线价, 60日均线价 |
| 背离 | 3 | 背离信号, 背离距今, 背离位置 |
| 风控 | 1 | 风险等级 |
| 综合报告 | 9 | 多头排列趋势, 综合分析结论, 综合分析评分, 综合级别, 止损价, T1目标价, T2目标价, 移动止损, 盈亏比 |
| 资金 | 5 | 研报买入次数, 资金动能, 5日资金流入万元, 10日资金流入万元, 20日资金流入万元 |
| 链接 | 1 | 股票链接 |

其他工作表包括：行业深度分析、主力研报筛选、均线多头排列、资金流向、强势股池、技术指标信号等。

### 缓存文件

| 文件 | 说明 |
|------|------|
| `tradeCalendar_YYYYMMDD.txt` | 交易日历缓存 |
| `StockIndes_YYYYMMDD.txt` | 股票基础信息缓存（股票池） |
| `ShareData/` | 各类原始数据及清洗后的数据缓存 |

## ⚠️ 注意事项：

请确保 PostgreSQL 服务已启动

首次运行需安装依赖：`pip install -r requirements.txt`（请确保 `psycopg2-binary`、`akshare`、`pandas`、`pandas_ta` 等已安装）

数据同步依赖 Akshare，建议在交易日 15:30 后运行，数据最全

<br />

## ⚠️ 免责声明

本项目提供的所有数据、分析报告和投资建议仅供学习、研究和参考，不构成任何投资建议。投资者应自行承担投资风险，并根据自身情况做出独立的投资决策。本项目的开发者不对任何使用本系统数据或分析结果而导致的投资损失承担责任。

请务必理解并同意以上声明后，再使用本项目。

<br>
