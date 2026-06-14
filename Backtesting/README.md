# Backtesting — BAISYS_QUANT 回测模块

基于 AKQuant (akshare 官方回测引擎) 构建的量化回测子系统。

## 目录结构

```
Backtesting/
├── __init__.py              # 模块入口
├── data_provider.py         # BacktestDataProvider 扩展 (backtest_kline 表读写)
├── strategy.py              # PipelineAdapter (AKQuant on_bar 实现)
├── optimizer.py             # Walk-Forward / Grid Search 调度
├── calibration.py           # calibration_result.json 读写 + Config 覆盖
├── runner.py                # 对外入口 (供 scheduler 每月调用)
├── data/                    # 校准结果缓存目录
│   └── calibration_result.json
└── examples/                # 示例脚本
    ├── 01_quick_backtest.py       # 快速单次回测
    ├── 02_grid_search.py          # 网格参数寻优
    └── 03_walk_forward.py         # Walk-Forward 滚动验证
```

## 安装依赖

```bash
# AKQuant (Rust 核心，需安装 Rust 编译工具链)
# 参见: https://akquant.akfamily.xyz/start/installation/
pip install akquant
```

---

## 工程计划

### Phase 1 — 基础设施 (Day 1-2)

| # | 任务 | 文件 | 交付物 |
|---|------|------|--------|
| 1.1 | `config.ini` 新增 `[BACKTEST]` 节 | `config.ini` | 寻优范围定义 |
| 1.2 | 新增 `BacktestConfig` Pydantic 模型 | `ConfigParser.py` | `app_config.backtest` 属性 |
| 1.3 | 创建 `backtest_kline` 表 + 迁移脚本 | `data_provider.py` | SQL DDL + 首次初始化 |
| 1.4 | 实现 `BacktestDataProvider` 扩展：从 `backtest_kline` 读取 + akshare 兜底补全 | `data_provider.py` | `get_backtest_data(start, end) → pd.DataFrame` |
| 1.5 | 实现 `KlineAdjuster.hfq()` 适配 | `DataManager/KlineAdjuster.py` | 后复权保证无未来数据 |

**依赖：** AKQuant 安装成功 + DB 可写

### Phase 2 — 策略适配 (Day 3-5)

| # | 任务 | 文件 | 交付物 |
|---|------|------|--------|
| 2.1 | 提取 `PipelineState._should_exit()` 独立按日接口 | `LogicAnalyzer/PipelineState.py` | `should_exit(state, bar) → bool` |
| 2.2 | 提取 `PipelineScoring._calc_entry_signal()` 独立按日接口 | `LogicAnalyzer/PipelineScoring.py` | `calc_entry_signal(bar) → Signal | None` |
| 2.3 | 实现 `PipelineAdapter` → AKQuant `on_bar` 策略 | `strategy.py` | 完整的入场/出场/仓位逻辑 |
| 2.4 | 冒烟测试：单只股票单次回测跑通 | `examples/01_quick_backtest.py` | 回测结果 + HTML 报告 |

**依赖：** Phase 1 完成

### Phase 3 — 参数寻优 (Day 6-8)

| # | 任务 | 文件 | 交付物 |
|---|------|------|--------|
| 3.1 | 接入 `akquant.run_grid_search` 实现网格寻优 | `optimizer.py` | 125 组合并行全市场回测 |
| 3.2 | 接入 `akquant.run_walk_forward` 实现滚动验证 | `optimizer.py` | 训练/测试集分离验证 |
| 3.3 | 实现 `calibration_result.json` 读写 | `calibration.py` | 最优参数持久化 |
| 3.4 | 实现 Config 参数覆盖机制 | `ConfigParser.py` + `calibration.py` | `Config.xxx` 自动返回校准值 |
| 3.5 | 验证端到端闭环：全量数据→寻优→覆写→管线使用 | `examples/03_walk_forward.py` | 完整月周期测试 |

**依赖：** Phase 2 完成

### Phase 4 — 集成上线 (Day 9-10)

| # | 任务 | 文件 | 交付物 |
|---|------|------|--------|
| 4.1 | `runner.py` 入口：参数配置、执行、结果写入 | `runner.py` | `run_backtest_session()` |
| 4.2 | 接入 scheduler 每月触发 | `runner.py` / scheduler | 自动执行 |
| 4.3 | 回测日志 + 异常告警 | `runner.py` | loguru 集成 |
| 4.4 | `docs/回测系统设计文档.md` 最终定稿 | `docs/` | 竣工文档 |

**依赖：** Phase 3 完成

---

## 关键决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 回测引擎 | AKQuant | akshare 生态原生，内置 Walk-Forward / Grid Search，Rust 性能 |
| K 线数据 | `backtest_kline` 独立表 | 不与每日管线共用 `stock_daily_kline`，解耦 |
| 复权方式 | 后复权 (hfq) | 避免前复权的未来数据偏差 |
| 寻优方法 | Walk-Forward | 训练/测试集严格分离，比 Grid Search 抗过拟合 |
| 参数覆盖 | `_calibration` 属性覆盖 | 下游代码零改动 |
| 调度频率 | 月度 (config 可配) | 非高频，平衡效果与耗时 |

## 约束

- `Backtesting/` 可引用 `LogicAnalyzer/` 和 `DataManager/`，反之不行
- 不改动现有因子计算代码（`SignalManager`, `ProfessionalIndicators` 等）
- 下游代码通过 `Config.xxx` 取值，不感知校准来源
