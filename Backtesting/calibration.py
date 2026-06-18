from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import pandas as pd

from Backtesting.akquant_strategy import QuantPipelineStrategy

# config.ini 中参数名 → (section, key) 映射
CALIB_PARAM_MAP: dict[str, tuple[str, str]] = {
    "atr_stop_mult": ("SCORING_PARAMS", "atr_stop_mult"),
    "atr_t1_mult": ("SCORING_PARAMS", "atr_t1_mult"),
    "kelly_fraction": ("POSITION_SIZING", "kelly_fraction"),
    "position_a": ("POSITION_SIZING", "position_a"),
    "liq_veto_ratio": ("FILTER_RULES", "liq_veto_ratio"),
    "boll_narrow_ratio": ("REGIME_DETECTION", "boll_narrow_ratio"),
    "cross_decay_days": ("SCORING_PARAMS", "cross_decay_days"),
}

CONFIG_INI = Path("config.ini")


def write_calibration_to_ini(params: dict[str, float]) -> None:
    """将寻优后的参数写回 config.ini，保留注释和格式。"""
    if not CONFIG_INI.exists():
        return

    lines = CONFIG_INI.read_text(encoding="utf-8").splitlines(keepends=True)
    current_section: str | None = None
    updated_keys: set[str] = set()

    def _format_val(key: str, val: float) -> str:
        if key.endswith("_days"):
            return str(int(val))
        s = f"{val:.6f}".rstrip("0").rstrip(".")
        return s if s else "0"

    for i, line in enumerate(lines):
        # 检测 section 头
        sec_match = re.match(r"^\s*\[(\w+)\]", line)
        if sec_match:
            current_section = sec_match.group(1)
            continue

        if current_section is None:
            continue

        # 检测 key = value
        kv_match = re.match(r"^\s*(\w+)\s*=", line)
        if not kv_match:
            continue

        raw_key = kv_match.group(1).lower()
        # 反向查找 param_map 中有无匹配
        for param_key, (sec, ini_key) in CALIB_PARAM_MAP.items():
            if sec == current_section and ini_key.lower() == raw_key and param_key in params:
                new_val = _format_val(param_key, params[param_key])
                old_val = line.split("=", 1)[1].strip()
                if old_val != new_val:
                    lines[i] = line.replace(old_val, new_val, 1)
                    updated_keys.add(param_key)
                break

    if updated_keys:
        CONFIG_INI.write_text("".join(lines), encoding="utf-8")


@dataclass
class CalibrationResult:
    """寻优结果数据类。"""

    params: dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    total_return: float = 0.0
    timestamp: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CalibrationResult:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


CALIBRATION_FILE = Path("calibration_result.json")


def run_grid_search(
    kline_df: pd.DataFrame,
    param_grid: dict[str, list[float]] | None = None,
    **backtest_kwargs: Any,
) -> pd.DataFrame:
    from akquant import run_grid_search as _ak_grid

    if param_grid is None:
        param_grid = {
            "atr_stop_mult": [1.0, 1.5, 2.0, 2.5, 3.0],
            "kelly_fraction": [0.1, 0.25, 0.5],
            "position_a": [0.2, 0.3, 0.4],
        }
    result_df = _ak_grid(
        strategy=QuantPipelineStrategy,
        param_grid=param_grid,
        data=kline_df,
        sort_by="sharpe_ratio",
        ascending=False,
        return_df=True,
        **backtest_kwargs,
    )
    return result_df


def run_walk_forward(
    kline_df: pd.DataFrame,
    param_grid: dict[str, list[float]] | None = None,
    train_period: int = 120,
    test_period: int = 20,
    initial_cash: float = 1_000_000.0,
    **backtest_kwargs: Any,
) -> pd.DataFrame:
    from akquant import run_walk_forward as _ak_wf

    if param_grid is None:
        param_grid = {
            "atr_stop_mult": [1.0, 1.5, 2.0, 2.5, 3.0],
            "kelly_fraction": [0.1, 0.25, 0.5],
        }

    result_df = _ak_wf(
        strategy=QuantPipelineStrategy,
        param_grid=param_grid,
        data=kline_df,
        train_period=train_period,
        test_period=test_period,
        initial_cash=initial_cash,
        metric="sharpe_ratio",
        ascending=False,
        **backtest_kwargs,
    )
    return result_df


def save_calibration(result: CalibrationResult) -> None:
    CALIBRATION_FILE.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_calibration() -> CalibrationResult | None:
    if not CALIBRATION_FILE.exists():
        return None
    try:
        data = json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))
        return CalibrationResult.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def apply_calibration_to_config(config: object) -> None:
    from ConfigParser import Config

    assert isinstance(config, Config), f"需要 Config 实例，收到 {type(config).__name__}"
    cfg = config
    result = load_calibration()
    if result is None:
        return
    overrides = result.params.copy()

    ps = cfg.app_config.position_sizing
    for key, attr in (
        ("atr_stop_mult", "ATR_STOP_MULT"),
        ("atr_t1_mult", "ATR_T1_MULT"),
        ("kelly_fraction", "KELLY_FRACTION"),
        ("position_a", "POSITION_A"),
        ("liq_veto_ratio", "LIQ_VETO_RATIO"),
    ):
        if key in overrides:
            setattr(ps, attr, overrides[key])

    rd = cfg.app_config.regime_detection
    if "boll_narrow_ratio" in overrides:
        rd.BOLL_NARROW_RATIO = overrides["boll_narrow_ratio"]

    sp = cfg.app_config.scoring_params
    if "cross_decay_days" in overrides:
        sp.CROSS_DECAY_DAYS = int(overrides["cross_decay_days"])
