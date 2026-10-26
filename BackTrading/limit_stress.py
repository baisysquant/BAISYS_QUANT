"""涨跌停专项压力测试 — 一字涨停/开盘竞价触板/炸板高发窗口 worst-case 成本报告。

技术债修复（涨跌停/一字板可成交量模型）：固定比例模型（limit_*_ratio）无法
暴露流动性极端场景的实际成交成本。本模块在既有危机段压力测试（stress_test.py）
之外，新增**涨跌停场景专项压力测试**：

    场景（自动检测历史高发窗口，无需人工标注时段）：
        seal_up_dense       一字涨停密集期（买入踏空敞口）
        seal_down_dense     一字跌停密集期（卖出套牢敞口）
        break_board_dense   炸板高发期（开盘触板后炸板，开盘价附近成交拥挤）
        auction_touch_dense 开盘集合竞价触板高发期（竞价可成交量枯竭）

    报告口径（引擎 stats_sink 逐单统计，见 engine/core.py）：
        - 竞价触板买入/卖出 成交率（fill_value / (fill+unfilled)）
        - 部分成交/未成交单数；未成交金额（worst-case 敞口）
        - 一字板撤销（踏空/套牢）金额与单数
        - 单日最大未成交金额（worst-case 单日成本）
        - 校准指引：经验填充模型分位数（与 fixed 固定比例对照，见
          limit_calibration.calibrate_limit_ratios）

用法：
    from BackTrading.limit_stress import run_limit_stress
    report = run_limit_stress(kline_df, engine_cfg, best_params)
"""

from __future__ import annotations

import bisect
from typing import Any

import pandas as pd
from loguru import logger

from BackTrading.engine import EngineConfig, _run_single_backtest
from BackTrading.limit_calibration import (
    DAY_OPEN_UP,
    DAY_SEAL_DOWN,
    DAY_SEAL_UP,
    _classify_days,
    calibrate_limit_ratios,
)
from BackTrading.prepare import _build_params, prepare_backtest_data

# 窗口 warmup 天数（高发窗口前预留的交易日 buffer，让引擎建立 ADV/仓位状态）
_LIMIT_STRESS_WARMUP_DAYS = 30

# 场景定义：{场景名: (分类口径, 描述)}
LIMIT_SCENARIOS: dict[str, tuple[str, str]] = {
    "seal_up_dense": (DAY_SEAL_UP, "一字涨停密集期（买入踏空敞口）"),
    "seal_down_dense": (DAY_SEAL_DOWN, "一字跌停密集期（卖出套牢敞口）"),
    "break_board_dense": (DAY_OPEN_UP, "炸板高发期（开盘触板后炸板）"),
    "auction_touch_dense": ("auction_touch", "开盘集合竞价触板高发期（竞价可成交量枯竭）"),
}


def _daily_type_counts(data: pd.DataFrame) -> pd.DataFrame:
    """按交易日统计市场级触板计数（复用 limit_calibration 分类口径）。"""
    _cls = _classify_days(data)
    _cls["auction_touch"] = _cls["auction_type"].ne("")
    _cols = [DAY_SEAL_UP, DAY_SEAL_DOWN, DAY_OPEN_UP, "auction_touch"]
    _rows = []
    for _d, _g in _cls.groupby("trade_date", sort=True):
        _row = {"trade_date": str(_d)}
        for _c in _cols:
            _row[_c] = int((_g["day_type"] == _c).sum()) if _c != "auction_touch" else int(_g["auction_touch"].sum())
        _rows.append(_row)
    return pd.DataFrame(_rows)


def detect_limit_windows(
    data: pd.DataFrame,
    kind: str,
    min_win_days: int = 5,
    top_n: int = 3,
) -> list[dict[str, Any]]:
    """检测指定场景的高发窗口（滚动 min_win_days 日计数最高的 top_n 个不重叠窗口）。

    Returns:
        [{start, end, days, count, mean_per_day, kind}] 按计数降序。
    """
    if kind not in LIMIT_SCENARIOS:
        raise ValueError(f"未知涨跌停压力场景 {kind!r}，可选 {list(LIMIT_SCENARIOS)}")
    _col = LIMIT_SCENARIOS[kind][0]
    counts = _daily_type_counts(data)
    if counts.empty:
        return []
    _dates = counts["trade_date"].tolist()
    _vals = counts[_col].to_numpy(dtype=float)
    if _vals.sum() <= 0:
        return []
    _win = int(min_win_days)
    _n = len(_vals)
    # 滚动窗口计数（前 min_win_days-1 日不满窗不参与）
    _roll = pd.Series(_vals).rolling(_win, min_periods=_win).sum()
    windows: list[dict[str, Any]] = []
    _used: list[tuple[int, int]] = []

    def _overlaps(a: int, b: int) -> bool:
        return any(not (a > _e or b < _s) for (_s, _e) in _used)

    for _end in range(_win - 1, _n):
        _start = _end - _win + 1
        if _overlaps(_start, _end):
            continue
        _used.append((_start, _end))
        windows.append({
            "start": _dates[_start],
            "end": _dates[_end],
            "days": _win,
            "count": float(_roll.iloc[_end]),
            "mean_per_day": float(_roll.iloc[_end]) / _win,
        })
    windows.sort(key=lambda w: w["count"], reverse=True)
    return windows[:top_n]


def run_limit_stress(
    kline_df: pd.DataFrame,
    engine_cfg: EngineConfig,
    params: dict[str, float],
    prepared: pd.DataFrame | None = None,
    min_win_days: int = 5,
    top_n: int = 3,
) -> dict[str, Any]:
    """涨跌停场景专项压力测试（一字涨停/竞价触板/炸板高发窗口 worst-case 成本）。

    Args:
        kline_df: 全量 K 线数据（prepare 管线输入；prepared 已注入时仅用于校准指引）
        engine_cfg: 引擎配置
        params: WFO 最佳参数。P1-18：调用方需确保 params 包含以下夹具
            键，否则 ST 股 5% 涨跌幅与次新股豁免将被停用：
            - params["_st_history"]: ST/退市逐日状态 {symbol: {date: (is_st, is_delisted)}}
            - params["_listing_days"]: 上市日期映射 {symbol: "YYYY-MM-DD"}
            - params["_exclude_st"]: 是否启用 ST 剔除（bool）
        prepared: 可选已 prepare 的数据（测试注入；None 时内部走 prepare 管线）
        min_win_days: 高发窗口最小交易日数
        top_n: 每场景窗口数

    Returns:
        {
          "calibration": 经验填充模型分位数指引（calibrate_limit_ratios 输出）,
          "windows": {场景名: [窗口报告, ...]},
          "worst_case": 全场景最差统计汇总,
        }
    """
    if prepared is None:
        structured = _build_params(type("cfg", (), {})(), params)
        prepared = prepare_backtest_data(
            kline_df, params=structured, compute_exit_strategy=True, vectorized=True
        )

    date_range = prepared["trade_date"].astype(str)
    unique_dates = sorted(date_range.unique())

    calib_guidance = calibrate_limit_ratios(kline_df)
    logger.info(
        "[涨跌停压力] 经验填充模型校准指引生成完毕（全天口径 V_t/V_prev 分位数，"
        "与 fixed 固定比例对照评估）"
    )

    windows_report: dict[str, list[dict[str, Any]]] = {}
    worst = {
        "min_buy_fill_rate": 1.0,
        "min_sell_fill_rate": 1.0,
        "max_unfilled_buy_value": 0.0,
        "max_unfilled_sell_value": 0.0,
        "max_seal_buy_rejected_value": 0.0,
        "max_seal_sell_rejected_value": 0.0,
    }

    for kind, (_col, desc) in LIMIT_SCENARIOS.items():
        _win_list = detect_limit_windows(prepared, kind, min_win_days, top_n)
        kind_reports: list[dict[str, Any]] = []
        for _w in _win_list:
            _start_pos = bisect.bisect_left(unique_dates, _w["start"])
            _warmup_pos = max(0, _start_pos - _LIMIT_STRESS_WARMUP_DAYS)
            _warmup_start = unique_dates[_warmup_pos]
            _mask = (date_range >= _warmup_start) & (date_range <= _w["end"])
            _scene_data = prepared[_mask].copy()
            if _scene_data.empty:
                continue
            _stats: dict[str, Any] = {}
            _tl: list[dict[str, Any]] = []
            _ec: list[dict[str, Any]] = []
            _run_single_backtest(
                _scene_data, params, engine_cfg, _tl, _ec,
                stats_sink=_stats,
            )

            _cs_ts = pd.Timestamp(_w["start"])
            _ce_ts = pd.Timestamp(_w["end"])
            _ec_win = [
                r for r in _ec
                if _cs_ts <= pd.Timestamp(r.get("time", _w["start"])) <= _ce_ts
            ]
            _risk: dict[str, Any] = {}
            if _ec_win:
                from LogicAnalyzer.backtest_metrics import compute_risk_metrics
                _risk = compute_risk_metrics(_ec_win) or {}

            _buy_fill = float(_stats.get("buy_limit_fill_value", 0.0))
            _buy_unf = float(_stats.get("buy_limit_unfilled_value", 0.0))
            _sell_fill = float(_stats.get("sell_limit_fill_value", 0.0))
            _sell_unf = float(_stats.get("sell_limit_unfilled_value", 0.0))
            _buy_fr = _buy_fill / (_buy_fill + _buy_unf) if (_buy_fill + _buy_unf) > 0 else 1.0
            _sell_fr = _sell_fill / (_sell_fill + _sell_unf) if (_sell_fill + _sell_unf) > 0 else 1.0
            _seal_buy_v = float(_stats.get("seal_buy_rejected_value", 0.0))
            _seal_sell_v = float(_stats.get("seal_sell_rejected_value", 0.0))

            worst["min_buy_fill_rate"] = min(worst["min_buy_fill_rate"], _buy_fr)
            worst["min_sell_fill_rate"] = min(worst["min_sell_fill_rate"], _sell_fr)
            worst["max_unfilled_buy_value"] = max(
                worst["max_unfilled_buy_value"], _buy_unf
            )
            worst["max_unfilled_sell_value"] = max(
                worst["max_unfilled_sell_value"], _sell_unf
            )
            worst["max_seal_buy_rejected_value"] = max(
                worst["max_seal_buy_rejected_value"], _seal_buy_v
            )
            worst["max_seal_sell_rejected_value"] = max(
                worst["max_seal_sell_rejected_value"], _seal_sell_v
            )

            _report = {
                "kind": kind,
                "description": desc,
                "start": _w["start"],
                "end": _w["end"],
                "days": _w["days"],
                "touch_count": _w["count"],
                "touch_per_day": round(_w["mean_per_day"], 2),
                "risk": {
                    "total_return": _risk.get("total_return", 0),
                    "max_drawdown": _risk.get("max_drawdown", 0),
                    "annual_vol": _risk.get("annual_vol", 0),
                    "trading_days": len(_ec_win),
                },
                "limit_stats": {
                    "buy_orders": int(_stats.get("buy_limit_orders", 0)),
                    "buy_fill_rate": round(_buy_fr, 4),
                    "buy_unfilled_value": round(_buy_unf, 2),
                    "buy_rejected": int(_stats.get("buy_limit_rejected", 0)),
                    "buy_partial": int(_stats.get("buy_limit_partial", 0)),
                    "sell_orders": int(_stats.get("sell_limit_orders", 0)),
                    "sell_fill_rate": round(_sell_fr, 4),
                    "sell_unfilled_value": round(_sell_unf, 2),
                    "sell_rejected": int(_stats.get("sell_limit_rejected", 0)),
                    "sell_partial": int(_stats.get("sell_limit_partial", 0)),
                    "seal_buy_rejected": int(_stats.get("seal_buy_rejected", 0)),
                    "seal_buy_rejected_value": round(_seal_buy_v, 2),
                    "seal_sell_rejected": int(_stats.get("seal_sell_rejected", 0)),
                    "seal_sell_rejected_value": round(_seal_sell_v, 2),
                    "buy_worst_day": _stats.get("buy_limit_worst_day"),
                    "sell_worst_day": _stats.get("sell_limit_worst_day"),
                },
            }
            kind_reports.append(_report)
            logger.info(
                f"  涨跌停压力 [{kind} {_w['start']}~{_w['end']}]: "
                f"触板={_w['count']:.0f}日次 买入成交率={_buy_fr:.2%} "
                f"卖出成交率={_sell_fr:.2%} 买入未成交={_buy_unf:.0f}元 "
                f"卖出未成交={_sell_unf:.0f}元"
            )
        if kind_reports:
            windows_report[kind] = kind_reports

    return {
        "calibration": calib_guidance,
        "windows": windows_report,
        "worst_case": worst,
    }