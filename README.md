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

本产品深度融合股票行情、财务数据、资金流向、行业趋势等基本面信息，结合双周期MACD、KDJ、CCI、RSI、BOLL、成本分布、趋势买点、双底买点等多维度技术指标，构建全链路量化分析体系。

通过数据建模与算法迭代，帮助投资者复盘当下走势、捕捉潜在机会，将复杂的市场波动转化为可量化的决策依据，让投资从“经验驱动”升级为“数据驱动”，在万象变幻的A股市场中，锚定确定性，做好有章可循的投资艺术。产品持续迭代更新，详细记录详见WIKI。

核心目标：提供一个自动化、多维度、数据驱动的工具，帮助投资者每日快速洞察 A 股市场，辅助投资决策。

<br>


## 🚀 核心功能与策略

**多源数据整合**

Akshare：获取实时行情、主力研报、财务摘要、市场资金流向、强势股池、连涨股、量价齐升、持续放量、均线突破等数据。

Tushare (Pro)：获取 A 股主板股票基础信息和交易日历。
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

**向量化MACD双周期策略**
- **标准周期**：(12, 26, 9) 保持经典配置
- **自定义周期**：第二周期完全用户自定义（支持如 9,18,6 或 24,52,18 等任意组合）
- **智能共振识别**：
  - 优化多头评分逻辑，精准识别"战略层"（标准周期）与"战术层"（自定义周期）信号共振
  - 支持背离检测、动能分析、量价验证三维验证
  - 动态衰减机制，确保信号时效性

**CCI：** 极度超买/超卖，强势/弱势超买/超卖，常态波动。

**RSI：** 超卖低位及底背离判断。

**BOLL：** 低波/缩口状态。

**KDJ:**

- 极值J线反转 当J线从负值区域快速反弹，且K线与D线形成金叉时的信号

- 底背离金叉 价格创新低但KDJ指标未创新低，随后出现金叉的强买入信号

- 趋势确认金叉 在超卖区域金叉，且价格站稳短期均线上方的趋势确认信号

- 低位超卖金叉 在K线和D线均处于超卖区域时出现的金叉信号
 
- 深度超卖反弹 K线和D线均处于极低位（<10）时的金叉反弹信号
 
- J线高位拐头 J线处于超买状态（>100）时出现拐头向下的信号
 
- K线快速拉升 K线在短时间内（如单日）出现快速上涨的信号
 
- 三线聚合突破 KDJ三线紧密聚合后向上突破的强势信号
 
- 死叉回踩支撑 出现死叉后价格回踩前期支撑位的信号
 
- J线极限值回归 J线处于极端值区域（>120或<-20）后的回归信号
 
- 背离信号 价格走势与KDJ指标走势相反的背离现象
 
- 振荡区间突破 在震荡行情中K线突破近期震荡区间的信号
 
- KDJ三线同步 KDJ三线同时向上或向下运动的一致性信号
 
- 超卖修复启动 在长时间超卖状态后的修复性上涨信号
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

[SYSTEM]

MACD_SECOND_PARAMS = (6, 13, 5)  超短线敏感周期

FUND_FLOW_PERIODS = [3, 5]  关注短期资金流

EXEMPT_LEVELS = ["完全主升", "趋势加速"] 仅保留强势股

</br> 

**场景示例：中线投资者**

[SYSTEM]

MACD_SECOND_PARAMS = (24, 52, 18)  中线趋势周期  

FUND_FLOW_PERIODS = [5, 10, 20]    多周期资金验证

FULL_BULL_THRESHOLD = 80           提高完全主升标准

</br> 

**场景示例：稳定投资者**

[SYSTEM]
MACD_SECOND_PARAMS = (12, 26, 9)  仅用标准周期

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

**Tushare Pro Token**

前往 Tushare Pro官网 注册账号并获取您的 Token。您需要在 HistDataEngine.py 和 Ts_GetStockBasicinfo.py 文件中更新 self.token 变量为您的实际 Token。

HistDataEngine.py 和 Ts_GetStockBasicinfo.py 中
self.token = "YOUR_TUSHARE_TOKEN_HERE" # 请替换为你的 Tushare Token

Akshare：Akshare 大部分接口无需 Token，但有访问频率限制。本项目已内置缓存文件和延迟机制以应对。

<br></br> 

## ⚙️ 安装

**克隆项目仓库：**

git clone https://github.com/chowkuanyen/BAISYS_QUAN.git

cd BAISYS_QUAN

**安装依赖包:**

pip install -r requirements.txt

注：openpyxl 和 xlsxwriter 用于 Excel 文件的读写。psycopg2-binary 是 PostgreSQL 的 Python 驱动。

<br />

## 🛠️ 配置

在运行之前，您可能需要根据您的环境调整以下配置：

[DATABASE] 节 - 数据库配置

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `user` | 字符串 | 是 | - | 数据库用户名 |
| `password` | 字符串 | 是 | - | 数据库密码 |
| `host` | 字符串 | 是 | - | 数据库主机地址 |
| `port` | 整数 | 是 | 5432 | 数据库端口号 |
| `db_name` | 字符串 | 是 | stock_data | 数据库名称 |
| `tushare_token` | 字符串 | 是 | - | TuShare API 访问令牌 |

[SYSTEM] 节 - 系统配置

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `HOME_DIRECTORY` | 字符串 | 否 | - | 系统主目录路径 |
| `TEMP_DATA_DIR` | 字符串 | 否 | `ShareData` | 临时数据存储子目录名 |
| `MAX_WORKERS` | 整数 | 否 | `15` | 最大并发工作线程数 |
| `DATA_FETCH_RETRIES` | 整数 | 否 | `3` | 数据获取失败重试次数 |
| `DATA_FETCH_DELAY` | 整数 | 否 | `5` | 数据获取失败后重试延迟秒数 |

[LOGGING] 节 - 日志配置

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `LOG_LEVEL` | 字符串 | 否 | `INFO` | 日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL） |
| `LOG_DIR` | 字符串 | 否 | `Logs` | 日志文件存储子目录名（相对于 HOME_DIRECTORY） |

业务向量


| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| MACD_SECOND_PARAMS | 元组 | 是 | (6,13,5) | 新增：第二周期MACD参数 (快线,慢线,信号线) |
| FUND_FLOW_PERIODS | 列表 | 是 | [5,10,20] | 新增：资金流观察周期（3/5/10/20日） |

</br> 

**配置详解**

**数据库配置说明**

数据库连接：系统使用 PostgreSQL 数据库，需要正确配置连接参数

TuShare Token：这是必须配置的参数，用于访问 TuShare 的金融数据接口

安全考虑：建议使用专用的数据库账户，权限最小化

**系统配置说明**

HOME_DIRECTORY：建议使用绝对路径或 ~ 表示用户主目录，系统会自动展开

TEMP_DATA_DIR：临时数据目录，用于存储中间处理结果

MAX_WORKERS：根据系统 CPU 核心数和内存大小调整，过高可能导致系统负载过大

重试机制：DATA_FETCH_RETRIES 和 DATA_FETCH_DELAY 用于处理网络不稳定导致的数据获取失败

**日志配置说明**

日志级别：生产环境建议使用 INFO 或 WARNING，调试时可使用 DEBUG

日志目录：系统会自动创建日志目录，无需手动创建
</br> </br> 

## 🚀 使用方法

执行 Corenews_Main.py 文件将启动整个自动化分析流程.

程序将执行以下步骤：

数据同步：通过 HistDataEngine 检查并同步 A 股主要上市公司的日 K 线数据到 PostgreSQL。

数据获取：从 Akshare 获取最新的市场行情、研报、资金流向、行业板块等数据。

深度分析：计算各项技术指标、筹码分布，进行行业趋势分析，并根据预设信号进行智能筛选。

报告生成：将所有分析结果汇总生成一份多工作表的 Excel 报告。

数据库存储：将最终的分析报告数据也同步到 PostgreSQL 数据库的 daily_stock_report 和 daily_industry_analysis 表中。

您可以在控制台看到详细的日志输出，追踪程序的运行状态。

<br /></br> 

## 📊 输出结果

所有报告和缓存文件将生成在 Corenews_Main.py 中 Config 类定义的 SAVE_DIRECTORY 下 (默认为 C:\Users\YourUsername\Downloads\CoreNews_Reports)。

**分析报告**

审计报告_YYYYMMDD.xlsx：包含 数据汇总、行业深度分析、主力研报筛选、均线多头排列、资金流向、强势股池、技术指标信号 等多个工作表。

**缓存文件**

tradeCalendar_YYYYMMDD.txt：交易日历缓存。

StockIndes_YYYYMMDD.txt：股票基础信息缓存（股票池）。

ShareData/ 目录：包含各类原始数据和清洗后的数据缓存文件 (例如 A股实时行情_经清洗_YYYYMMDD.txt, MACD_hist_data_cache_经清洗_YYYYMMDD.txt 等)。

<br />

<br />

## ⚠️ 注意事项：

请确保 PostgreSQL 服务已启动

首次运行需安装依赖：pip install psycopg2-binary akshare pandas ta tushare

数据同步依赖 Akshare，建议在交易日 15:30 后运行，数据最全

<br />

## ⚠️ 免责声明

本项目提供的所有数据、分析报告和投资建议仅供学习、研究和参考，不构成任何投资建议。投资者应自行承担投资风险，并根据自身情况做出独立的投资决策。本项目的开发者不对任何使用本系统数据或分析结果而导致的投资损失承担责任。

请务必理解并同意以上声明后，再使用本项目。

<br>
