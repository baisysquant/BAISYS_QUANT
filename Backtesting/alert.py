from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from Backtesting.calibration import PROJECT_ROOT, CalibrationResult, load_calibration


class BacktestAlert:
    """回测告警 — 失败通知 + 参数漂移检测。"""

    DRIFT_THRESHOLD = 0.15  # 单参数相对变化 > 15% 触发漂移告警
    DRIFT_LOG = PROJECT_ROOT / "backtest_drift.json"

    def __init__(self, config: Any) -> None:
        self.config = config

    def on_success(self, result: CalibrationResult) -> None:
        logger.info(f"回测成功: Sharpe={result.sharpe:.2f}, Sortino={result.sortino:.2f}, "
                    f"Return={result.total_return:.2%}, DD={result.max_drawdown:.2%}, "
                    f"VaR={result.var_95:.2%}, 交易={result.total_trades}笔")
        self._check_drift(result.params)

    def on_failure(self, exc: Exception) -> None:
        logger.error(f"回测失败: {exc}")
        self._write_alert("failure", {"error": str(exc), "time": datetime.now().isoformat()})

    def _check_drift(self, new_params: dict[str, float]) -> None:
        old = load_calibration()
        if old is None:
            return

        drifts: list[dict[str, Any]] = []
        for key, new_val in new_params.items():
            old_val = old.params.get(key)
            if old_val is None or old_val == 0:
                continue
            ratio = abs(new_val - old_val) / abs(old_val)
            if ratio > self.DRIFT_THRESHOLD:
                drifts.append({
                    "param": key,
                    "old": old_val,
                    "new": new_val,
                    "drift_pct": round(ratio * 100, 1),
                })

        if drifts:
            logger.warning(f"参数漂移告警: {len(drifts)} 个参数变化超过 {self.DRIFT_THRESHOLD:.0%}")
            for d in drifts:
                logger.warning(f"  {d['param']}: {d['old']} -> {d['new']} ({d['drift_pct']:+.1f}%)")
            self._write_alert("drift", {
                "time": datetime.now().isoformat(),
                "drifts": drifts,
            })
        else:
            logger.info("参数漂移检测通过（无显著变化）")

    def _write_alert(self, alert_type: str, data: dict[str, Any]) -> None:
        records: list[dict[str, Any]] = []
        if self.DRIFT_LOG.exists():
            try:
                records = json.loads(self.DRIFT_LOG.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, TypeError):
                records = []
        records.append({"type": alert_type, **data})
        self.DRIFT_LOG.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
