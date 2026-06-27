from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    p = Path(__file__).resolve().parent  # Backtesting/
    for _ in range(10):
        if (p / "config.ini").exists():
            return p
        parent = p.parent
        if parent == p:
            break
        p = parent
    return Path.cwd()


PROJECT_ROOT = _project_root()

import pandas as pd

# config.ini 中参数名 → (section, key) 映射
CALIB_PARAM_MAP: dict[str, tuple[str, str]] = {
    "atr_stop_mult": ("BACKTEST_CALIBRATED", "atr_stop_mult"),
    "atr_t1_mult": ("BACKTEST_CALIBRATED", "atr_t1_mult"),
    "kelly_fraction": ("BACKTEST_CALIBRATED", "kelly_fraction"),
    "position_a": ("BACKTEST_CALIBRATED", "position_a"),
    "liq_veto_ratio": ("BACKTEST_CALIBRATED", "liq_veto_ratio"),
    "boll_narrow_ratio": ("BACKTEST_CALIBRATED", "boll_narrow_ratio"),
    "cross_decay_days": ("BACKTEST_CALIBRATED", "cross_decay_days"),
}

CONFIG_INI = PROJECT_ROOT / "config.ini"


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
    sortino: float = 0.0
    calmar: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    total_return: float = 0.0
    annual_return: float = 0.0
    annual_vol: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    timestamp: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CalibrationResult:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


CALIBRATION_FILE = PROJECT_ROOT / "calibration_result.json"


def run_grid_search(
    kline_df: pd.DataFrame,
    param_grid: dict[str, list[float]] | None = None,
    **backtest_kwargs: Any,
) -> pd.DataFrame:
    from Backtesting.engine import EngineConfig, grid_search as _gs

    cfg = _build_engine_config(backtest_kwargs)
    if param_grid is None:
        param_grid = {
            "atr_stop_mult": [1.0, 1.5, 2.0, 2.5, 3.0],
            "kelly_fraction": [0.1, 0.25, 0.5],
            "position_a": [0.2, 0.3, 0.4],
        }
    results = _gs(
        data=kline_df,
        param_grid=param_grid,
        engine_cfg=cfg,
        show_progress=backtest_kwargs.get("show_progress", False),
    )
    return pd.DataFrame(results)


def run_walk_forward(
    kline_df: pd.DataFrame,
    param_grid: dict[str, list[float]] | None = None,
    train_period: int = 120,
    test_period: int = 20,
    initial_cash: float = 1_000_000.0,
    **backtest_kwargs: Any,
) -> pd.DataFrame:
    from Backtesting.engine import EngineConfig, walk_forward as _wf

    cfg = _build_engine_config(initial_cash, backtest_kwargs)
    if param_grid is None:
        param_grid = {
            "atr_stop_mult": [1.0, 1.5, 2.0, 2.5, 3.0],
            "kelly_fraction": [0.1, 0.25, 0.5],
        }

    results = _wf(
        data=kline_df,
        engine_cfg=cfg,
        train_period=train_period,
        test_period=test_period,
        param_grid=param_grid,
        show_progress=backtest_kwargs.get("show_progress", False),
    )
    return pd.DataFrame(results)


def _build_engine_config(initial_cash_or_kwargs: float | dict[str, Any], kwargs: dict[str, Any] | None = None) -> Any:
    from Backtesting.engine import EngineConfig

    if isinstance(initial_cash_or_kwargs, dict):
        kwargs = initial_cash_or_kwargs
        initial_cash = kwargs.get("initial_cash", 1_000_000)
    else:
        initial_cash = initial_cash_or_kwargs
        kwargs = kwargs or {}

    return EngineConfig(
        initial_cash=initial_cash,
        commission_rate=kwargs.get("commission", 0.0003),
        stamp_tax_rate=kwargs.get("stamp_tax", 0.001),
        slippage=kwargs.get("slippage", 0.001),
        max_position_pct=kwargs.get("max_position_pct", 0.1),
        portfolio_method=kwargs.get("portfolio_method", "score_weighted"),
        point_in_time=kwargs.get("point_in_time", True),
    )


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
    if not overrides:
        return

    rd = cfg.app_config.regime_detection
    sc = cfg.app_config.scoring_params
    fr = cfg.app_config.filter_rules
    ps = cfg.app_config.position_sizing

    for key, val in overrides.items():
        attr = key.upper()
        if key == "boll_narrow_ratio":
            rd.BOLL_NARROW_RATIO = val
        elif key == "cross_decay_days":
            sc.CROSS_DECAY_DAYS = int(val)
        elif key in ("atr_stop_mult", "atr_t1_mult"):
            setattr(sc, attr, val)
        elif key == "liq_veto_ratio":
            fr.LIQ_VETO_RATIO = val
        elif key in ("kelly_fraction", "position_a"):
            setattr(ps, attr, val)
