from __future__ import annotations

import json
import re
import subprocess
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
from loguru import logger

from BackTrading.bayesian.space import build_spaces, split_by_cost, describe

# config.ini 中参数名 → (section, key) 映射
CALIB_PARAM_MAP: dict[str, tuple[str, str]] = {
    "atr_stop_mult": ("BACKTEST_CALIBRATED", "atr_stop_mult"),
    "atr_t1_mult": ("BACKTEST_CALIBRATED", "atr_t1_mult"),
    "atr_t2_mult": ("BACKTEST_CALIBRATED", "atr_t2_mult"),
    "kelly_fraction": ("BACKTEST_CALIBRATED", "kelly_fraction"),
    "position_a": ("BACKTEST_CALIBRATED", "position_a"),
    "liq_veto_ratio": ("BACKTEST_CALIBRATED", "liq_veto_ratio"),
    "boll_narrow_ratio": ("BACKTEST_CALIBRATED", "boll_narrow_ratio"),
    "cross_decay_days": ("BACKTEST_CALIBRATED", "cross_decay_days"),
    "conclusion_full_bull": ("BACKTEST_CALIBRATED", "conclusion_full_bull"),
    "golden_cross_bonus": ("BACKTEST_CALIBRATED", "golden_cross_bonus"),
    "divergence_penalty": ("BACKTEST_CALIBRATED", "divergence_penalty"),
    "risk_none_multiplier": ("BACKTEST_CALIBRATED", "risk_none_multiplier"),
}


@dataclass
class CalibrationResult:
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
    git_commit: str = ""
    config_hash: str = ""
    pbo: float = 0.0
    dsr: float = 0.0
    num_trials: int = 0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CalibrationResult:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


CALIBRATION_FILE = PROJECT_ROOT / "calibration_result.json"


def run_bayesian_walk_forward(
    kline_df: pd.DataFrame,
    train_period: int = 120,
    test_period: int = 20,
    num_paths: int = 3,
    initial_cash: float = 1_000_000.0,
    spaces: dict | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """贝叶斯 Walk-Forward 优化入口。

    Args:
        kline_df: K 线数据
        train_period: IS 训练窗口（交易日）
        test_period: OOS 验证窗口
        num_paths: 多路径数
        initial_cash: 初始资金
        spaces: 预构建的 ParamSpace dict（None 时从 config 自动构建）
        **kwargs: 透传给引擎的额外参数

    Returns:
        DataFrame, 每行一个 WFO 窗口，与旧 walk_forward 返回格式兼容。
    """
    from UtilsManager.ConfigParser import Config as _Config

    if spaces is None:
        cfg = _Config().app_config.backtest
        spaces = build_spaces(cfg)

    signal_sp, portfolio_sp = split_by_cost(spaces)
    logger.info(f"贝叶斯 WFO 参数空间:\n{describe(spaces)}")
    logger.info(f"  信号参数: {len(signal_sp)} | 组合参数: {len(portfolio_sp)}")

    n_dates = len(kline_df["trade_date"].unique())
    logger.info(f"  交易日数: {n_dates} | IS={train_period} | OOS={test_period}")
    if test_period < 60:
        logger.warning(f"OOS 窗口仅 {test_period} 天，Sharpe 估计标准误约 {1.96/ (test_period-1)**0.5:.2f}，建议 ≥60 天")

    # ── 正式调用贝叶斯 WFO 引擎 ─────────────────────────
    from BackTrading.bayesian.meta_optimizer import bayesian_walk_forward_multi

    result = bayesian_walk_forward_multi(
        kline_df=kline_df,
        train_period=train_period,
        test_period=test_period,
        num_paths=num_paths,
        initial_cash=initial_cash,
        spaces=spaces,
        **kwargs,
    )
    return result



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
    from UtilsManager.ConfigParser import Config

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
        elif key in ("atr_stop_mult", "atr_t1_mult", "atr_t2_mult"):
            setattr(sc, attr, val)
        elif key == "liq_veto_ratio":
            fr.LIQ_VETO_RATIO = val
        elif key in ("kelly_fraction", "position_a"):
            setattr(ps, attr, val)
        elif key == "conclusion_full_bull":
            cfg.app_config.full_bull_scoring.CONCLUSION_FULL_BULL = int(val)
        elif key == "golden_cross_bonus":
            sc.GOLDEN_CROSS_BONUS = int(val)
        elif key == "divergence_penalty":
            sc.DIVERGENCE_PENALTY = int(val)
        elif key == "risk_none_multiplier":
            ps.RISK_NONE_MULTIPLIER = float(val)


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return ""


def write_calibration_to_ini(config: object) -> None:
    from UtilsManager.ConfigParser import Config

    assert isinstance(config, Config), f"需要 Config 实例，收到 {type(config).__name__}"
    result = load_calibration()
    if result is None:
        logger.warning("未找到校准结果，跳过写入 config.ini")
        return
    overrides = result.params
    if not overrides:
        logger.info("无校准参数，跳过写入")
        return

    ini_path = Path(config.config_path) if hasattr(config, "config_path") and config.config_path else Path("config.ini")
    if not ini_path.exists():
        logger.warning(f"config.ini 不存在: {ini_path}")
        return

    raw = ini_path.read_text(encoding="utf-8")
    section_header = "[BACKTEST_CALIBRATED]"

    if section_header not in raw:
        raw += f"\n\n{section_header}\n"

    def _update_section(text: str, section: str, key: str, value: Any) -> str:
        pat = re.compile(rf"^({key}\s*=\s*).*", re.MULTILINE)
        in_section = False
        new_lines: list[str] = []
        for line in text.splitlines(keepends=True):
            if line.strip().startswith("["):
                in_section = line.strip().startswith(section)
            if in_section and pat.match(line):
                line = pat.sub(rf"\g<1>{value}", line)
            new_lines.append(line)
        return "".join(new_lines)

    for key, val in overrides.items():
        raw = _update_section(raw, section_header, key, val)

    ini_path.write_text(raw, encoding="utf-8")
    logger.info(f"校准参数已写入 {ini_path} [{section_header}]")
