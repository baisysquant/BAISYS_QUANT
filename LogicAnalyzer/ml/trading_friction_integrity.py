"""1.8 交易摩擦合规自检（TradingFrictionCompliance）。

业务定义：策略 Alpha 必须能够覆盖真实的微观市场成本。
审计口径：
  1. 双边显性成本：单回合完整交易必须强制扣除 A 股真实佣金（单边 0.03%）、
     卖出端国家印花税（2023-08-28 减半后 0.05%，此前 0.1%）以及各交易所过户费（0.001% 双边）。
  2. 隐性滑点硬编码：固定滑点不得低于单边 0.05%（0.0005）；
     大资金调仓占个股日成交量（ADV）比例超过阈值时，必须引入平方根模型
     等非线性动态冲击成本调增滑点。
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from BackTrading._engine_legacy import (
    _MIN_SLIPPAGE_FLOOR,
    _STAMP_TAX_RECENT,
)
from LogicAnalyzer.ml.split_integrity import SplitReport

# 政策常量（与 _engine_legacy 同源，避免两处漂移）
_STAMP_TAX_CUTOFF = "2023-08-28"
_COMMISSION_FLOOR = 0.0003
_MIN_COMMISSION_CNY = 5.0
_TRANSFER_FEE_FLOOR = 0.00001

_ATTRS = (
    "commission_rate",
    "stamp_tax_rate",
    "transfer_fee_rate",
    "min_commission_per_trade",
    "market_slippage",
    "limit_slippage",
    "impact_threshold",
    "impact_base",
    "impact_cap",
)


def _read_cfg(obj: Any) -> dict[str, float]:
    return {k: float(getattr(obj, k)) for k in _ATTRS if getattr(obj, k, None) is not None}


def _slippage_rates(obj: Any) -> tuple[float, float]:
    """统一取 (市价滑点, 限价滑点)：EngineConfig 只暴露 slippage（限价=0.5×），CostModel 有独立字段。"""
    cfg = _read_cfg(obj)
    slip = cfg.get("slippage", getattr(obj, "slippage", None))
    market = cfg.get("market_slippage", slip or _MIN_SLIPPAGE_FLOOR)
    limit = cfg.get("limit_slippage", (slip or market) * 0.5)
    return market, limit


# ── 1. 双边显性成本 ─────────────────────────────────────────

def check_double_sided_explicit_costs(
    cm_or_cfg: Any,
    _log: bool = True,
) -> SplitReport:
    """双边显性成本校验：佣金≥0.03%（且最低 5 元）、印花税≥0.05%、过户费≥0.001%。

    Args:
        cm_or_cfg: CostModel 或 EngineConfig。
        _log: 是否输出日志。

    Returns:
        SplitReport。
    """
    check = SplitReport(check_name="双边显性成本（佣金/印花税/过户费）", passed=True)
    cfg = _read_cfg(cm_or_cfg)
    if "commission_rate" in cfg and cfg["commission_rate"] < _COMMISSION_FLOOR:
        check.passed = False
        check.details.append(
            f"佣金 {cfg['commission_rate']:.6f} < 单边 0.03%（{_COMMISSION_FLOOR}），"
            "单回合显性成本被低估"
        )
    if "stamp_tax_rate" in cfg and cfg["stamp_tax_rate"] < _STAMP_TAX_RECENT:
        check.passed = False
        check.details.append(
            f"印花税 {cfg['stamp_tax_rate']:.6f} < 现行政策 0.05%（{_STAMP_TAX_RECENT}）"
        )
    if "transfer_fee_rate" in cfg and cfg["transfer_fee_rate"] < _TRANSFER_FEE_FLOOR:
        check.passed = False
        check.details.append(
            f"过户费 {cfg['transfer_fee_rate']:.6f} < 0.001%（{_TRANSFER_FEE_FLOOR}）"
        )
    if "min_commission_per_trade" in cfg and cfg["min_commission_per_trade"] < _MIN_COMMISSION_CNY:
        check.passed = False
        check.details.append(f"最低佣金 {cfg['min_commission_per_trade']:.2f} 元 < 5 元")
    if check.passed:
        check.details.append("佣金/印花税/过户费/最低佣金全部满足 A 股政策下限")
    if _log:
        check.log()
    return check


# ── 2. 固定滑点下限 ─────────────────────────────────────────

def check_slippage_floor(
    cm_or_cfg: Any,
    _log: bool = True,
) -> SplitReport:
    """固定滑点下限校验：基础滑点必须 ≥ 单边 0.05%（0.0005）。

    配置值低于下限即违规（引擎已强制抬升，此处审计配置源头）。
    """
    check = SplitReport(check_name="隐性滑点硬编码（≥ 0.05%）", passed=True)
    market, limit = _slippage_rates(cm_or_cfg)
    for rate, label in ((market, "市价单"), (limit, "限价单")):
        if rate < _MIN_SLIPPAGE_FLOOR:
            check.passed = False
            check.details.append(
                f"{label}基础滑点 {rate:.6f} < 强制下限 {_MIN_SLIPPAGE_FLOOR}（0.05%）"
            )
    if check.passed:
        check.details.append("市价/限价基础滑点均不低于 0.05%")
    if _log:
        check.log()
    return check


# ── 3. 动态冲击成本（平方根模型） ───────────────────────────

def check_dynamic_impact(
    cm: Any,
    _log: bool = True,
) -> SplitReport:
    """动态冲击成本校验：参与率超过阈值后，冲击成本按非线性幂函数（指数 1.5）上升。

    校验点：
      - 低参与率（< 所有档阈值）不得触发冲击，滑点=基础滑点；
      - 高参与率必须显著放大滑点（> 基础滑点）；
      - 极端参与率下冲击受 cap 约束，不得无限放大；
      - 参与率单调递增时滑点单调不减。
    """
    check = SplitReport(check_name="大单动态冲击成本（ADV 占比平方根模型）", passed=True)
    slip_low = cm.calc_slippage(1_000, 100_000_000, amount_ma20=1e9)  # 参与率 0.001%
    slip_high = cm.calc_slippage(10_000_000, 100_000_000, amount_ma20=1e9)  # 10%
    slip_mid = cm.calc_slippage(5_000_000, 100_000_000, amount_ma20=1e9)  # 5%
    base = cm.calc_slippage(1_000, 100_000_000)  # 无档位时基础滑点
    if slip_low < base - 1e-12:
        check.passed = False
        check.details.append(f"低参与率滑点 {slip_low:.6f} < 基础滑点 {base:.6f}")
    if slip_high <= slip_mid:
        check.passed = False
        check.details.append("参与率 10% 滑点未高于 5% 滑点，冲击项未随参与率上升")
    if slip_high <= base:
        check.passed = False
        check.details.append("参与率 10% 未触发非线性冲击，大单成本被低估")
    if slip_mid <= base:
        check.passed = False
        check.details.append("参与率 5% 未触发非线性冲击（若阈值高于 5% 需复核档位参数）")
    cap = min(cm.impact_cap, *cm.liquidity_tier_cap)
    if slip_high > base + cap + 1e-9:
        check.passed = False
        check.details.append("极端参与率冲击成本越过档位 cap，模型失控")
    if check.passed:
        check.details.append("冲击成本随参与率非线性上升且受 cap 约束（平方根模型）")
    if _log:
        check.log()
    return check


# ── 4. 卖出端完整性（引擎成交记录热路径） ───────────────────

def check_sell_side_completeness(
    trade_log: Sequence[dict[str, Any]],
    cm_or_cfg: Any,
    _log: bool = True,
) -> SplitReport:
    """引擎成交记录完整性：每笔卖出成本必须覆盖 佣金+印花税+过户费（及滑点）。

    引擎成交记录 value=卖出净得、cost=总成本，故成交额 = value + cost。
    校验：cost >= 成交额×(印花税+过户费) + min(最低佣金, 成交额×佣金率)。
    任何低于此下限的卖出记录 = 显性成本漏扣回归。

    Args:
        trade_log: 引擎成交记录（须含 action/value/cost）。
        cm_or_cfg: CostModel 或 EngineConfig。
        _log: 是否输出日志。

    Returns:
        SplitReport。
    """
    check = SplitReport(check_name="卖出端显性成本完整性（热路径）", passed=True)
    cfg = _read_cfg(cm_or_cfg)
    commission = cfg.get("commission_rate", _COMMISSION_FLOOR)
    stamp = cfg.get("stamp_tax_rate", _STAMP_TAX_RECENT)
    transfer = cfg.get("transfer_fee_rate", _TRANSFER_FEE_FLOOR)
    min_comm = cfg.get("min_commission_per_trade", _MIN_COMMISSION_CNY)
    checked = 0
    violations: list[str] = []
    for t in trade_log:
        if str(t.get("action", "")).startswith("sell"):
            amount = float(t.get("value", 0.0)) + float(t.get("cost", 0.0))
            if amount <= 0:
                continue
            checked += 1
            comm = min(min_comm, amount * commission)
            floor_cost = amount * (stamp + transfer + _MIN_SLIPPAGE_FLOOR) + comm
            if float(t.get("cost", 0.0)) < floor_cost - 1e-6:
                violations.append(
                    f"sell {t.get('symbol')} {t.get('time')}: cost={t.get('cost')} < "
                    f"理论下限 {floor_cost:.2f}（佣金+印花税+过户费）"
                )
    check.n_checked = checked
    check.n_violations = len(violations)
    check.passed = check.n_violations == 0
    if violations:
        check.details.append(
            f"{check.n_violations}/{check.n_checked} 笔卖出成本低于显性成本下限"
            "（佣金/印花税/过户费漏扣）"
        )
        check.details.append("示例: " + "; ".join(violations[:5]))
    if check.passed and checked:
        check.details.append(f"{checked} 笔卖出记录全部覆盖佣金+印花税+过户费")
    if _log:
        check.log()
    return check


# ── 一站式入口 ──────────────────────────────────────────────

def run_trading_friction_check(
    cm_or_cfg: Any,
    trade_log: Sequence[dict[str, Any]] | None = None,
    _log: bool = True,
) -> dict[str, Any]:
    """一站式交易摩擦合规检查。

    Args:
        cm_or_cfg: CostModel 或 EngineConfig。
        trade_log: 可选引擎成交记录（卖出端完整性）。
        _log: 是否输出日志。

    Returns:
        {"passed": bool, "reports": list[SplitReport], "summary": str}。
    """
    reports: list[SplitReport] = [
        check_double_sided_explicit_costs(cm_or_cfg, _log=_log),
        check_slippage_floor(cm_or_cfg, _log=_log),
    ]
    if hasattr(cm_or_cfg, "calc_slippage"):
        reports.append(check_dynamic_impact(cm_or_cfg, _log=_log))
    else:
        rep = SplitReport(check_name="大单动态冲击成本（ADV 占比平方根模型）", passed=False)
        rep.details.append("未配置 CostModel：缺少 ADV 动态冲击成本，大单摩擦被低估")
        reports.append(rep)
    if trade_log:
        reports.append(check_sell_side_completeness(trade_log, cm_or_cfg, _log=_log))
    passed = all(r.passed for r in reports)
    summary = "PASS" if passed else "FAIL"
    return {"passed": passed, "reports": reports, "summary": summary}


def check_trading_friction_config(
    cm_or_cfg: Any,
    _log: bool = True,
) -> SplitReport:
    """引擎运行期 O(1) 热路径检查（费率/滑点/冲击参数合规）。

    未挂载 CostModel 的配置视为违规：fallback 路径无 ADV 动态冲击成本，
    大资金调仓摩擦被系统性低估。
    """
    check = SplitReport(check_name="交易摩擦配置合规（热路径）", passed=True)
    try:
        r1 = check_double_sided_explicit_costs(cm_or_cfg, _log=False)
        r2 = check_slippage_floor(cm_or_cfg, _log=False)
        for r in (r1, r2):
            if not r.passed:
                check.passed = False
                check.details.extend(r.details)
        if not hasattr(cm_or_cfg, "calc_slippage"):
            check.passed = False
            check.details.append(
                "未配置 CostModel：缺少 ADV 动态冲击成本（平方根模型），大单摩擦被低估"
            )
        else:
            r3 = check_dynamic_impact(cm_or_cfg, _log=False)
            if not r3.passed:
                check.passed = False
                check.details.extend(r3.details)
        if check.passed:
            check.details.append("显性成本/滑点下限/动态冲击全部合规")
    except Exception as e:  # noqa: BLE001
        check.passed = False
        check.details.append(f"检查执行异常: {e}")
    if _log:
        check.log()
    return check
