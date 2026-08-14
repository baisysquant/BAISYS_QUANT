from __future__ import annotations

import json
import os
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
# 注：kelly_fraction / position_a / liq_veto_ratio / risk_none_multiplier
# 为引擎死参数（审计确认引擎仓位恒等权），不再寻优亦不再回写。
CALIB_PARAM_MAP: dict[str, tuple[str, str]] = {
    "atr_stop_mult": ("BACKTEST_CALIBRATED", "atr_stop_mult"),
    "boll_narrow_ratio": ("BACKTEST_CALIBRATED", "boll_narrow_ratio"),
    "cross_decay_days": ("BACKTEST_CALIBRATED", "cross_decay_days"),
    "conclusion_full_bull": ("BACKTEST_CALIBRATED", "conclusion_full_bull"),
    "golden_cross_bonus": ("BACKTEST_CALIBRATED", "golden_cross_bonus"),
    "divergence_penalty": ("BACKTEST_CALIBRATED", "divergence_penalty"),
    "buy_threshold": ("BACKTEST_CALIBRATED", "buy_threshold"),
    "max_holdings": ("BACKTEST_CALIBRATED", "max_holdings"),
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
CONFIG_INI = PROJECT_ROOT / "config.ini"

# 写入 config.ini 时需取整的整数参数
# P0-7 ②：补齐 buy_threshold/max_holdings —— 此前缺失导致以 "17.0" 浮点落盘，
# 一旦加载端 int() 解析即崩溃；现强制整值落盘（写入前另有类型断言）。
_INT_KEYS = frozenset({
    "cross_decay_days", "conclusion_full_bull",
    "golden_cross_bonus", "divergence_penalty",
    "buy_threshold", "max_holdings",
})


def run_bayesian_walk_forward(
    kline_df: pd.DataFrame,
    train_period: int = 120,
    test_period: int = 60,
    num_paths: int = 3,
    initial_cash: float = 1_000_000.0,
    spaces: dict | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """贝叶斯 Walk-Forward 优化入口。

    Args:
        kline_df: K 线数据
        train_period: IS 训练窗口（交易日）
        test_period: OOS 验证窗口（审计强制 ≥ 60 天，低于此值拒绝执行）
        num_paths: 多路径数（≥ 5，路径间偏移 ≥ 40 天以降低相关性）
        initial_cash: 初始资金
        spaces: 预构建的 ParamSpace dict（None 时从 config 自动构建）
        **kwargs: 透传给引擎的额外参数

    Returns:
        DataFrame, 每行一个 WFO 窗口，与旧 walk_forward 返回格式兼容。
    """
    # P0 审计修复：OOS 窗口硬约束 ≥ 60 天
    if test_period < 60:
        raise ValueError(
            f"OOS 验证窗口 {test_period} 天 < 60 天最小要求，统计效力不足（Sharpe 标准误 ≈ 1.96/√{test_period-1}）"
            " — 请缩短 IS 窗口或增加数据跨度以提供至少 60 天 OOS。"
        )
    # P1 审计修复：路径数 ≥ 5 以降低路径间相关性
    if num_paths < 5:
        logger.warning(
            f"路径数 {num_paths} < 5，WFO 中位数聚合统计效力不足，建议 ≥ 5 且路径起始偏移 ≥ 40 天"
        )

    from UtilsManager.ConfigParser import Config as _Config

    if spaces is None:
        cfg = _Config().app_config.backtest
        spaces = build_spaces(cfg)

    signal_sp, portfolio_sp = split_by_cost(spaces)
    logger.info(f"贝叶斯 WFO 参数空间:\n{describe(spaces)}")
    logger.info(f"  信号参数: {len(signal_sp)} | 组合参数: {len(portfolio_sp)}")

    n_dates = len(kline_df["trade_date"].unique())
    logger.info(f"  交易日数: {n_dates} | IS={train_period} | OOS={test_period}")

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
    tmp_path = CALIBRATION_FILE.with_name(CALIBRATION_FILE.name + ".tmp")
    tmp_path.write_text(
        json.dumps(asdict(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_path, CALIBRATION_FILE)


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
        elif key == "atr_stop_mult":
            setattr(sc, attr, val)
        elif key == "conclusion_full_bull":
            cfg.app_config.full_bull_scoring.CONCLUSION_FULL_BULL = int(val)
        elif key == "golden_cross_bonus":
            sc.GOLDEN_CROSS_BONUS = int(val)
        elif key == "divergence_penalty":
            sc.DIVERGENCE_PENALTY = int(val)
        # P0-7 ②：校准闭环补齐 —— buy_threshold/max_holdings 曾只写不读，
        # 现覆写到 backtest 配置（与 ConfigParser [BACKTEST_CALIBRATED] 覆写同目标）
        elif key == "buy_threshold":
            cfg.app_config.backtest.BUY_THRESHOLD = int(val)
        elif key == "max_holdings":
            cfg.app_config.backtest.MAX_HOLDINGS = int(val)


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return ""


def _format_val(key: str, val: Any) -> str:
    """按字段语义格式化配置值：整数参数取整，浮点去尾零，避免 int('37.0') 崩溃。"""
    if key in _INT_KEYS:
        return str(int(round(float(val))))
    try:
        v = float(val)
    except (TypeError, ValueError):
        return str(val)
    s = f"{v:.10f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def write_calibration_to_ini(params: dict) -> None:
    """将校准参数写入 config.ini 的 [BACKTEST_CALIBRATED]。

    已有键原位替换（保留行尾注释），新键追加到 section 末尾；
    整型参数取整写入，避免下次加载时 int() 解析崩溃；原子写防半截文件。
    """
    if not params:
        logger.info("无校准参数，跳过写入")
        return
    # P0-7 ②：落盘前类型断言 —— 整数参数必须为整数值，否则以 "17.0" 落盘
    # 会在加载端 int() 解析崩溃；此处 fail-fast，拒绝污染 config.ini。
    for k in sorted(_INT_KEYS & set(params)):
        v = params[k]
        try:
            is_integral = isinstance(v, (int, float, str)) and float(v).is_integer()
        except (TypeError, ValueError):
            is_integral = False
        if not is_integral:
            raise ValueError(
                f"[校准写回] 整数参数 {k} = {v!r} 必须为整数值，拒绝写入 config.ini（防 int() 解析崩溃）"
            )
    ini_path = CONFIG_INI
    if not ini_path.exists():
        logger.warning(f"config.ini 不存在: {ini_path}")
        return

    text = ini_path.read_text(encoding="utf-8")
    section_header = "[BACKTEST_CALIBRATED]"
    lines = text.splitlines(keepends=True)

    # 定位 section（或在其后插入）
    sec_idx = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("["):
            if s == section_header:
                sec_idx = i
                break
            if sec_idx is not None:
                break
    if sec_idx is None:
        if text and not text.endswith("\n"):
            text += "\n"
        text += f"\n{section_header}\n"
        lines = text.splitlines(keepends=True)
        sec_idx = len(lines) - 1

    written: set[str] = set()
    out: list[str] = []
    in_section = False
    for ln in lines:
        s = ln.strip()
        if s.startswith("["):
            in_section = s == section_header
            out.append(ln)
            continue
        if in_section and "=" in s and not s.startswith(("#", ";")):
            key = s.split("=", 1)[0].strip().lower()
            if key in params:
                comment = ""
                cm = re.search(r"#.*$", ln)
                if cm:
                    comment = cm.group(0)
                body = f"{key} = {_format_val(key, params[key])}"
                ln = body + ("  " + comment if comment else "") + ("\n" if ln.endswith("\n") else "")
                written.add(key)
        out.append(ln)

    # 追加未写出的键（插到 section 内容末尾）
    missing = {k for k in params if k not in written}
    if missing:
        insert_at = len(out)
        for i in range(len(out) - 1, -1, -1):
            if out[i].strip().startswith("["):
                insert_at = i + 1
                break
        extra = "".join(f"{k} = {_format_val(k, params[k])}\n" for k in sorted(missing))
        out.insert(insert_at, extra)

    tmp_path = ini_path.with_name(ini_path.name + ".tmp")
    tmp_path.write_text("".join(out), encoding="utf-8")
    os.replace(tmp_path, ini_path)
    logger.info(f"校准参数已写入 {ini_path} [{section_header}]")
