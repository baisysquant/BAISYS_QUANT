"""
P0-5 统一参数采纳门控 测试

runner.py 曾门控不一致：write_calibration_to_ini 未应用 PBO/DSR 门控
（save_calibration 有）→ PBO 过拟合参数集仍可落盘进生产 config.ini。
_acceptance_gate 为两处共用的统一门控，本文件覆盖全部门控项与回归场景。
"""

from __future__ import annotations

from BackTrading.runner import _acceptance_gate


def _all_pass(**overrides: bool) -> dict[str, bool]:
    kw = dict(
        promote=True, oos_decay_pass=True, overfitting_critical=False,
        sig_pass=True, robust_pass=True, pbo_gate=True, dsr_gate=True,
    )
    kw.update(overrides)
    return kw


def test_gate_all_pass() -> None:
    passed, reasons = _acceptance_gate(**_all_pass())
    assert passed is True
    assert reasons == []


def test_gate_pbo_rejection_blocks_ini_write_path() -> None:
    """P0-5 回归：PBO 未过但其余全过 → 必须拒绝采纳。

    旧 write_calibration_to_ini 条件（_promote and _oos_decay_pass and
    not _overfitting_critical and _sig_pass and _robust_pass）不含 PBO/DSR，
    该场景下过拟合参数集仍会落盘进生产——统一门控后必须拒绝。
    """
    passed, reasons = _acceptance_gate(**_all_pass(pbo_gate=False))
    assert passed is False
    assert any("PBO" in r for r in reasons)


def test_gate_dsr_rejection() -> None:
    passed, reasons = _acceptance_gate(**_all_pass(dsr_gate=False))
    assert passed is False
    assert any("DSR" in r for r in reasons)


def test_gate_each_component_flips_verdict() -> None:
    cases = [
        ("promote", False, "模拟验证"),
        ("oos_decay_pass", False, "OOS"),
        ("overfitting_critical", True, "CRITICAL"),
        ("sig_pass", False, "统计显著性"),
        ("robust_pass", False, "稳健性"),
    ]
    for key, val, marker in cases:
        passed, reasons = _acceptance_gate(**_all_pass(**{key: val}))
        assert passed is False, f"{key} 应使门控失败"
        assert any(marker in r for r in reasons), f"{key} 原因缺失: {reasons}"


def test_gate_multiple_failures_collected() -> None:
    passed, reasons = _acceptance_gate(
        **_all_pass(pbo_gate=False, dsr_gate=False, sig_pass=False)
    )
    assert passed is False
    assert len(reasons) == 3