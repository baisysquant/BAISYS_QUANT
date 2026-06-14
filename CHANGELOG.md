# Changelog

## v1.0.0 (2026-06-09)

百思量化 (BAISYS_QUANT) 首个正式版本发布。面向 A 股的全链路量化分析系统，覆盖数据获取、技术指标计算、信号评分到 Excel 报告生成的全流程。

### 新特性

- **全市场 MACD 7 维管线评分体系**：趋势、金叉、柱状动能、DIF 斜率、背离、量价配合、K 线形态等 7 维度评分 + 6 道门控规则递进式过滤管道
- **单参数 MACD 聚焦策略**：专注 (12,26,9) 参数组合，ATR 波动率归一化，行业中性化百分位校准
- **信号衰减模型**：金叉 30 天半衰、背离 8 天半衰、K 线形态 10 天半衰
- **ATR 倍数退出策略层**：多时间帧对齐、波动率状态切换
- **多源数据整合**：PostgreSQL 历史 K 线 + AkShare 实时行情 + AShareHub 筹码分布 + 资金流向 + 强势股池 + 行业板块
- **五大技术指标并行计算**：MACD、KDJ、CCI、RSI、BOLL，各自独立输出至 Excel
- **资金流三周期分析**：3/5/10/20 日可配置
- **智能筛选与 44 列结构化 Excel 审计报告**
- **15 线程并发处理 + 本地缓存机制**

### 更新内容

#### 架构优化

- **缓存系统统一**：废弃旧 `CacheManager`，迁移至 `UnifiedCacheManager`，提供 `load_cache`/`save_cache`/`cache_exists`/`get_cache_path` 向后兼容 API，零迁移成本
- **指标模块重写**：`indicators/boll.py`、`indicators/rsi.py`、`indicators/cci.py` 重写为纯函数风格（`pd.Series` → `dataclass`），支持显式信号生成和背离检测

#### Bug 修复

- **MACDAnalyzer 导入修复**：`LogicAnalyzer/pipeline.py` 模块名大小写修复（`Pipeline` → `pipeline`）
- **合并数据 KeyError 修复**：`MACD_FULL_BULL` 空数据时保留列结构，`filter_signal_stocks` 防御性过滤不存在的列
- **资金流分页死循环修复**：API 返回不足页数据时外层 `while True` 未退出，反复请求同一页触发 429 限流

#### 依赖

- Python 3.8+
- AkShare、Pandas TA、PostgreSQL
- 完整依赖见 `requirements.txt`

### 注意事项

- 需 PostgreSQL 数据库（数据库名 `Corenews`）
- AShareHub API Key 可选（仅筹码分布数据需要）
- 所有配置在 `config.ini` 中统一管理，支持 ENC 加密值
- 本项目仅供学习研究参考，不构成投资建议
