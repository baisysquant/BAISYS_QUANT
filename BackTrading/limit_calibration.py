"""涨跌停经验填充模型校准 — 用历史日线统计替代固定比例常量（Limit Calibration）。

技术债（涨跌停/一字板可成交量模型过度简化）：
    EngineConfig 中 limit_seal_ratio / limit_tradable_ratio / limit_intraday_ratio /
    auction_fill_ratio 以固定比例建模"一字板/冲板"可成交量，比例取值缺乏经验依据。
    分钟/tick 与盘口（成交薄/分价位）数据当前不在本系统数据源内，无法直接构建
    逐笔撮合模型；本模块用**历史日线可观测量**构建经验填充模型，作为可行替代：

    经验可成交量比例 r = 触板日实际成交量 / 前日成交量（V_t / V_prev）
        - 对"次日开盘集合竞价"路径：竞价时点（9:25）仅知 open 与前日量，
          可成交量 = 前日量 × r。r 的分布（分位数）由全样本历史统计，
          属静态参数选择（等价于选择常量），应用时不带前视。
        - 语义：一字板日全部成交均在限价（流动性最差但仍可成交），炸板日
          限价附近成交占比更高——实际可观测的 V_t/V_prev 分布是这些固定比例
          的经验锚点（median=中性口径，p10=worst-case 保守口径）。

使用：
    EngineConfig.limit_ratio_mode:
        fixed            = 旧行为（固定比例常量，默认）
        empirical_median = 经验中位数（中性口径，V_t/V_prev 的 50% 分位）
        empirical_p10    = 经验 10% 分位（保守口径，暴露 worst-case 可成交量）
    单元格样本数 < limit_calib_min_samples → 回退 fixed 档（防稀疏样本噪声）。

与引擎规则的一致性说明：
    涨跌停价按主板 ±10%（limit_pricing.calc_limit_prices_batch）计算——校准
    统计不感知 ST 5%/退市整理期/上市豁免（K 线数据无这些标注），个别误分类
    对分位数估计影响可忽略（统计口径，文档化假设）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from BackTrading.limit_pricing import calc_limit_prices_batch

# 校准近似用涨跌幅（主板 ±10%；ST/退市/上市豁免不可感知，统计近似，见模块 docstring）
_CALIB_RATIO = 0.10
# 数值比较容差（与 limit_pricing._LIMIT_EPS 一致）
_LIMIT_EPS = 1e-9
# 连板数分桶上限（>3 板并入 3+ 桶，保证样本量）
_MAX_STREAK_BUCKET = 3
# 异常量比上限（数据污染保护：V_t/V_prev 超过该值视为异常剔除）
_MAX_RATIO_CLIP = 10.0

# 开盘触板（竞价路径）类型：open 触及限价，无论收盘（= 一字 + 炸板的并集）
AUCTION_TOUCH_UP = "auction_up"
AUCTION_TOUCH_DOWN = "auction_down"
# 全天口径类型（校准指引/报告用，引擎盘中档未消费，仅作固定比例校准建议）
DAY_SEAL_UP = "seal_up"
DAY_SEAL_DOWN = "seal_down"
DAY_OPEN_UP = "open_up"  # 开盘触板后炸板（open≥限价, close<限价）
DAY_OPEN_DOWN = "open_down"
DAY_INTRADAY_UP = "intraday_up"  # 盘中冲板（open<限价, high≥限价）
DAY_INTRADAY_DOWN = "intraday_down"

_ALL_DAY_TYPES = (
    DAY_SEAL_UP, DAY_SEAL_DOWN, DAY_OPEN_UP, DAY_OPEN_DOWN,
    DAY_INTRADAY_UP, DAY_INTRADAY_DOWN,
)


@dataclass(frozen=True)
class EmpiricalCell:
    """单一（触板类型 × 连板桶）单元格的经验统计。"""

    count: int
    p10: float
    p50: float
    p90: float

    def ratio_at(self, percentile: float) -> float:
        """按校准分位返回经验可成交量比例（越保守取越低分位，返回 ≤1 截断）。"""
        if percentile <= 0.11:
            v = self.p10
        elif percentile >= 0.49:
            v = self.p50
        else:
            v = self.p90
        return float(np.clip(v, 0.0, 1.0))


class EmpiricalCalibration:
    """经验填充模型校准表 — 引擎撮合层消费。

    auction_table: { (auction_type, streak_bucket): EmpiricalCell }
        auction_type ∈ {auction_up, auction_down}，streak_bucket ∈ {1, 2, 3}。
        竞价路径查找：开盘触板日（open≥涨停价 或 ≤跌停价，仅 9:25 已知信息）
        可成交量比例 = 分位数（V_t/V_prev），PIT 合规。
    day_type_table: { (day_type, streak_bucket): EmpiricalCell } — 全天口径，
        供校准指引报告（固定比例建议值）与后续盘中档模型扩展。
    """

    def __init__(
        self,
        auction_table: dict[tuple[str, int], EmpiricalCell],
        day_type_table: dict[tuple[str, int], EmpiricalCell],
        percentile: float,
        min_samples: int,
    ) -> None:
        self.auction_table = auction_table
        self.day_type_table = day_type_table
        self.percentile = percentile
        self.min_samples = min_samples

    def auction_fill_ratio(
        self,
        *,
        open_at_limit_up: bool,
        open_at_limit_down: bool,
        streak: int,
        fallback: float,
    ) -> float:
        """开盘触板日集合竞价可成交量比例（经验分位；样本不足/缺失回退 fallback）。

        Args:
            open_at_limit_up: 开盘价 ≥ 涨停价（9:25 已知）
            open_at_limit_down: 开盘价 ≤ 跌停价（9:25 已知）
            streak: 信号日连板数（含信号日，引擎口径 abs(streak_prev)+1）
            fallback: fixed 档比例（auction_fill_ratio_for × auction_fill_ratio 封顶）
        """
        if open_at_limit_up == open_at_limit_down:
            return fallback
        _typ = AUCTION_TOUCH_UP if open_at_limit_up else AUCTION_TOUCH_DOWN
        _cell = self.auction_table.get((_typ, _streak_bucket(streak)))
        if _cell is None or _cell.count < self.min_samples:
            return fallback
        return _cell.ratio_at(self.percentile)

    def summary(self) -> dict[str, Any]:
        """校准表摘要（日志/报告用）。"""
        return {
            "percentile": self.percentile,
            "min_samples": self.min_samples,
            "cells": {
                f"{_t}/{_s}板": {
                    "n": _c.count,
                    "p10": round(_c.p10, 4),
                    "p50": round(_c.p50, 4),
                    "p90": round(_c.p90, 4),
                }
                for (_t, _s), _c in sorted(self.auction_table.items())
            },
        }


def _streak_bucket(streak) -> int:
    """连板数分桶（1/2/3+；接受标量或数组）。"""
    _s = np.asarray(streak, dtype=np.int64)
    _b = np.clip(_s, 1, _MAX_STREAK_BUCKET)
    if _b.ndim == 0:
        return int(_b)
    return _b.astype(np.int64)


def _classify_days(df: pd.DataFrame) -> pd.DataFrame:
    """对日线 df 逐行分类触板类型（全天口径 + 竞价口径 + 连板数）。

    Args:
        df: 含 symbol / trade_date / open / high / low / close / close_raw /
            volume 列的日线数据（缺失 open/high/low 时按 close 填充近似）。

    Returns:
        副本 + 列：limit_up / limit_down / day_type / auction_type / streak
        （streak = 当日收盘连板数含当日，符号方向与引擎一致：涨停正、跌停负）
    """
    out = df.copy()
    _close = out["close"].astype(float)
    _close_raw = (
        out["close_raw"].astype(float) if "close_raw" in out.columns else _close
    )
    _open = out["open"].astype(float) if "open" in out.columns else _close
    _high = out["high"].astype(float) if "high" in out.columns else _close
    _low = out["low"].astype(float) if "low" in out.columns else _close

    g = out.groupby("symbol", sort=False)
    prev_close = g["close_raw"].shift(1) if "close_raw" in out.columns else g["close"].shift(1)
    prev_close = prev_close.astype(float)

    limit_up = np.full(len(out), np.inf)
    limit_down = np.full(len(out), -np.inf)
    _valid = prev_close.notna() & (prev_close > 0)
    if _valid.any():
        _pc = prev_close[_valid].to_numpy()
        _ru = np.full(_pc.shape, _CALIB_RATIO)
        _rd = np.full(_pc.shape, _CALIB_RATIO)
        _lu, _ld = calc_limit_prices_batch(_pc, _ru, _rd)
        limit_up[_valid.to_numpy()] = _lu
        limit_down[_valid.to_numpy()] = _ld
    out["limit_up"] = limit_up
    out["limit_down"] = limit_down

    _open_v = _open.to_numpy()
    _high_v = _high.to_numpy()
    _low_v = _low.to_numpy()
    _close_v = _close_raw.to_numpy()
    _lu_v = out["limit_up"].to_numpy()
    _ld_v = out["limit_down"].to_numpy()
    _has_prev = _valid.to_numpy()

    _open_at_up = _has_prev & (_open_v >= _lu_v - _LIMIT_EPS)
    _open_at_down = _has_prev & (_open_v <= _ld_v + _LIMIT_EPS)
    _close_at_up = _has_prev & (_close_v >= _lu_v - _LIMIT_EPS)
    _close_at_down = _has_prev & (_close_v <= _ld_v + _LIMIT_EPS)
    _touch_up = _close_at_up | (_high_v >= _lu_v - _LIMIT_EPS)
    _touch_down = _close_at_down | (_low_v <= _ld_v + _LIMIT_EPS)

    day_type = np.full(len(out), "", dtype=object)
    day_type[_close_at_up & _open_at_up] = DAY_SEAL_UP
    day_type[_close_at_down & _open_at_down] = DAY_SEAL_DOWN
    day_type[(_open_at_up & ~_close_at_up) & (day_type == "")] = DAY_OPEN_UP
    day_type[(_open_at_down & ~_close_at_down) & (day_type == "")] = DAY_OPEN_DOWN
    day_type[(_touch_up & ~_open_at_up) & (day_type == "")] = DAY_INTRADAY_UP
    day_type[(_touch_down & ~_open_at_down) & (day_type == "")] = DAY_INTRADAY_DOWN
    out["day_type"] = day_type

    auction_type = np.full(len(out), "", dtype=object)
    auction_type[_open_at_up] = AUCTION_TOUCH_UP
    auction_type[_open_at_down] = AUCTION_TOUCH_DOWN
    out["auction_type"] = auction_type

    # 连板数（符号编码与引擎 _build_day_limit_model 一致：涨停 +1 / 跌停 -1 / 断板归零）
    # 向量化 run-length：键值变化或换股 → 新 run，run 内 cumcount+1
    _key_s = pd.Series(
        _close_at_up.astype(np.int8) - _close_at_down.astype(np.int8),
        index=out.index,
    )
    out["_key"] = _key_s
    _change = (_key_s != _key_s.shift(1).fillna(0).astype(np.int8))
    _change = _change | (out["symbol"] != out["symbol"].shift(1).fillna(""))
    _run_id = _change.cumsum()
    _run_len = out.groupby(_run_id, sort=False).cumcount() + 1
    streak = np.where(_key_s.to_numpy() > 0, _run_len,
                      np.where(_key_s.to_numpy() < 0, -_run_len, 0))
    out["streak"] = streak.astype(np.int32)
    out.drop(columns=["_key"], inplace=True)
    return out


def _build_table(
    df: pd.DataFrame,
    type_col: str,
    percentile: float,
    min_samples: int,
) -> dict[tuple[str, int], EmpiricalCell]:
    """按（类型 × 连板桶）聚合 V_t/V_prev 经验分位数。"""
    _prev_vol = df.groupby("symbol", sort=False)["volume"].shift(1).astype(float)
    _r = df["volume"].astype(float) / _prev_vol.replace(0.0, np.nan)
    _r = _r.where(_r.notna() & (_r > 0) & (_r <= _MAX_RATIO_CLIP))

    table: dict[tuple[str, int], EmpiricalCell] = {}
    _types = df[type_col].to_numpy()
    _streaks = df["streak"].to_numpy()
    for _typ in sorted(set(_types)):
        if not _typ:
            continue
        _mask = _types == _typ
        _buckets = np.where(_streaks > 0, _streak_bucket(_streaks),
                            _streak_bucket(np.abs(_streaks)))
        for _bk in range(1, _MAX_STREAK_BUCKET + 1):
            _vals = _r.to_numpy()[_mask & (_buckets == _bk)]
            _vals = _vals[np.isfinite(_vals)]
            if len(_vals) == 0:
                continue
            _q = np.quantile(_vals, [0.1, 0.5, 0.9])
            table[(_typ, _bk)] = EmpiricalCell(
                count=int(len(_vals)),
                p10=float(_q[0]),
                p50=float(_q[1]),
                p90=float(_q[2]),
            )
    return table


def build_empirical_calibration(
    data: pd.DataFrame,
    percentile: float = 0.5,
    min_samples: int = 20,
) -> EmpiricalCalibration:
    """从历史日线构建经验填充模型校准表（引擎撮合层消费）。

    Args:
        data: 全量日线（symbol / trade_date / open / high / low / close /
            close_raw / volume）。
        percentile: 校准分位。0.5=中性（中位数），0.1=保守 worst-case。
        min_samples: 单元格最少样本数，不足的单元格在查找时回退 fixed 档。

    Returns:
        EmpiricalCalibration（auction_table 供竞价路径查找，day_type_table
        供校准指引报告）。
    """
    _d = data.copy()
    _d["trade_date"] = _d["trade_date"].astype(str)
    _d["symbol"] = _d["symbol"].astype(str)
    _d = _d.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    _cls = _classify_days(_d)
    auction_table = _build_table(_cls, "auction_type", percentile, min_samples)
    day_type_table = _build_table(_cls, "day_type", percentile, min_samples)
    return EmpiricalCalibration(
        auction_table=auction_table,
        day_type_table=day_type_table,
        percentile=float(percentile),
        min_samples=int(min_samples),
    )


def calibrate_limit_ratios(data: pd.DataFrame) -> dict[str, Any]:
    """校准指引：全天口径各触板类型的 V_t/V_prev 分位数（固定比例建议值）。

    输出 {day_type: {streak_bucket: {n, p10, p50, p90}}}——供研究报告/日志
    对照现有固定比例（seal 0.05 / tradable 0.30 / intraday 0.10 / auction 0.12）
    评估其经验合理性，并作为后续盘中档模型扩展的经验锚点。
    """
    _d = data.copy()
    _d["trade_date"] = _d["trade_date"].astype(str)
    _d["symbol"] = _d["symbol"].astype(str)
    _d = _d.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    _cls = _classify_days(_d)
    table = _build_table(_cls, "day_type", 0.5, 1)
    return {
        _t: {
            _s: {"n": _c.count, "p10": round(_c.p10, 4),
                 "p50": round(_c.p50, 4), "p90": round(_c.p90, 4)}
            for (_t2, _s), _c in sorted(table.items()) if _t2 == _t
        }
        for _t in _ALL_DAY_TYPES
    }