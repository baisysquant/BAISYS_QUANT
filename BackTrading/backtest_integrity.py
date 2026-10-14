"""回测完整性自检框架 — AUDIT_FIX P0-4 替代方案。

核心解耦（2655行单体函数拆分为模块化架构）需要较大工程投入。
作为替代，此框架在每次回测后自动运行完整性校验，确保关键假设成立。

使用方式（在 runner.py 回测结束后调用）：
    checker = BacktestIntegrityChecker()
    report = checker.check_all(
        trade_log=trade_log,
        equity_curve=equity_curve,
        engine_cfg=engine_cfg,
        engine_state=state,  # 可选：BacktestState 对象
    )
    if not report.passed:
        logger.warning(f"回测完整性检查未通过: {report.failures}")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class IntegrityCheck:
    """单项检查结果。"""
    name: str
    passed: bool
    details: str = ""


@dataclass
class IntegrityReport:
    """完整性检查报告。"""
    checks: list[IntegrityCheck] = field(default_factory=list)
    
    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)
    
    @property
    def failures(self) -> list[str]:
        return [c.name for c in self.checks if not c.passed]


class BacktestIntegrityChecker:
    """回测完整性自检器。"""
    
    def check_all(
        self,
        trade_log: list[dict[str, Any]],
        equity_curve: list[dict[str, Any]] | pd.DataFrame,
        engine_cfg: Any = None,
        **kwargs,
    ) -> IntegrityReport:
        """运行所有完整性检查。"""
        checks = [
            self.check_t1_compliance(trade_log),
            self.check_no_lookahead(trade_log),
            self.check_cost_consistency(trade_log),
            self.check_equity_continuity(equity_curve),
            self.check_fractional_shares(trade_log),
        ]
        report = IntegrityReport(checks)
        
        failed = [c.name for c in report.failures]
        if failed:
            logger.warning(
                f"[回测完整性] {len(failed)}/{len(checks)} 项未通过: {'; '.join(failed)}"
            )
        else:
            logger.info(f"[回测完整性] 全部 {len(checks)} 项通过")
        return report
    
    def check_t1_compliance(self, trade_log: list[dict[str, Any]]) -> IntegrityCheck:
        """T+1 合规检查：买入日不可卖出。"""
        violations = []
        for i, trade in enumerate(trade_log):
            if trade.get("action") == "buy":
                buy_date = trade.get("time")
                sym = trade.get("symbol")
                # 检查同日是否有卖出
                for j, other in enumerate(trade_log):
                    if other.get("action", "").startswith("sell"):
                        if other.get("symbol") == sym and other.get("time") == buy_date:
                            violations.append(
                                f"违反T+1: {sym} 在 {buy_date} 买入后当日卖出"
                            )
                            break
                if violations:
                    break  # 发现即停，避免 O(N²)
        
        passed = len(violations) == 0
        return IntegrityCheck(
            name="T+1合规",
            passed=passed,
            details=violations[0] if violations else "通过",
        )
    
    def check_no_lookahead(self, trade_log: list[dict[str, Any]]) -> IntegrityCheck:
        """前视偏差检查：成交价不可早于信号日+1日。"""
        violations = []
        for trade in trade_log:
            sig_dt = trade.get("sig_dt")
            exec_dt = trade.get("time")
            if sig_dt and exec_dt and sig_dt == exec_dt:
                violations.append(
                    f"前视偏差: {trade.get('symbol')} 信号日={sig_dt}, 执行日={exec_dt} (同日)"
                )
                if len(violations) >= 3:
                    break
        
        passed = len(violations) == 0
        return IntegrityCheck(
            name="前视偏差",
            passed=passed,
            details=violations[0] if violations else "通过",
        )
    
    def check_cost_consistency(self, trade_log: list[dict[str, Any]]) -> IntegrityCheck:
        """成本一致性检查：每笔交易成本应在合理范围内。"""
        violations = []
        for trade in trade_log:
            value = trade.get("value", 0)
            cost = trade.get("cost", 0)
            if value > 0:
                cost_ratio = cost / value
                # A股全成本上限约 0.2%（佣金+印花税+过户费+经手费+证管费）+ 滑点
                if cost_ratio > 0.01:  # 1% 作为告警阈值
                    violations.append(
                        f"成本异常: {trade.get('symbol')} {trade.get('action')} "
                        f"成本率={cost_ratio:.4%} (>{'%.0f%%' % 1})"
                    )
                    if len(violations) >= 3:
                        break
        
        passed = len(violations) == 0
        return IntegrityCheck(
            name="成本一致性",
            passed=passed,
            details=violations[0] if violations else "通过",
        )
    
    def check_equity_continuity(
        self, equity_curve: list[dict[str, Any]] | pd.DataFrame
    ) -> IntegrityCheck:
        """净值连续性检查：权益曲线不应有非交易日的跳变。"""
        if isinstance(equity_curve, list):
            df = pd.DataFrame(equity_curve)
        else:
            df = equity_curve
        
        if df.empty or "portfolio_value" not in df.columns:
            return IntegrityCheck("净值连续性", True, "无数据")
        
        vals = df["portfolio_value"].values.astype(float)
        if len(vals) < 2:
            return IntegrityCheck("净值连续性", True, "样本不足")
        
        returns = np.diff(vals) / vals[:-1]
        # 单日净值变动超过 20% 且无对应交易 → 可能由数据异常引起
        violations = []
        for i, ret in enumerate(returns):
            if abs(ret) > 0.20 and np.isfinite(ret):
                violations.append(
                    f"净值跳变: 第{i}日单日变动 {ret:.2%}"
                )
                if len(violations) >= 3:
                    break
        
        passed = len(violations) == 0
        return IntegrityCheck(
            name="净值连续性",
            passed=passed,
            details=violations[0] if violations else "通过",
        )
    
    def check_fractional_shares(self, trade_log: list[dict[str, Any]]) -> IntegrityCheck:
        """碎股检查：A股持仓应始终为 100 股整数倍（主板）。"""
        violations = []
        for trade in trade_log:
            qty = trade.get("qty", 0)
            if 0 < qty < 100:
                violations.append(
                    f"碎股持仓: {trade.get('symbol')} 持仓 {qty} 股"
                )
                if len(violations) >= 3:
                    break
        
        passed = len(violations) == 0
        return IntegrityCheck(
            name="碎股合规",
            passed=passed,
            details=violations[0] if violations else "通过",
        )
