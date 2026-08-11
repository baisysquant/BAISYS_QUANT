from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from BackTrading.calibration import PROJECT_ROOT, CalibrationResult, load_calibration
from BackTrading.calibration_log import MAX_TUNING_ATTEMPTS

# 参数过度优化告警：同区间调参超过此次数触发（与多重测试惩罚阈值统一，避免双源漂移）
OVEROPTIMIZE_THRESHOLD = MAX_TUNING_ATTEMPTS


class BacktestAlert:
    """回测告警 — 失败通知 + 参数漂移检测 + 参数过度优化告警。"""

    DRIFT_THRESHOLD = 0.15  # 单参数相对变化 > 15% 触发漂移告警
    DRIFT_LOG = PROJECT_ROOT / "backtest_drift.json"

    def __init__(self, config: Any) -> None:
        self.config = config

    def on_success(self, result: CalibrationResult) -> None:
        logger.info(f"回测成功: Sharpe={result.sharpe:.2f}, Sortino={result.sortino:.2f}, "
                    f"Return={result.total_return:.2%}, DD={result.max_drawdown:.2%}, "
                    f"VaR={result.var_95:.2%}, 交易={result.total_trades}笔")
        self._check_drift(result.params)
        self._check_overoptimize(result)

    def on_failure(self, exc: Exception, snapshot_id: str | None = None) -> None:
        logger.error(f"回测失败: {exc}")
        data: dict[str, Any] = {"error": str(exc), "time": datetime.now().isoformat()}
        if snapshot_id:
            data["snapshot_id"] = snapshot_id
            data["error_code"] = "PIPELINE_FAILED"
            logger.error(
                f"失败快照已落盘，可用 load_snapshot('{snapshot_id}') 本地复现"
                f"（目录: <CACHE_DIRECTORY>/failure_snapshots）"
            )
        self._write_alert("failure", data)

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

    def _check_overoptimize(self, result: CalibrationResult) -> None:
        """参数过度优化告警。

        通过读取 calibration_log 中同区间的记录数来判断调参次数。
        如果次数超过阈值，写入告警文件并输出高危风险警告。
        """
        try:
            from DataManager.DbEngine import get_engine
            from BackTrading.calibration_log import count_tuning_attempts

            engine = get_engine(self.config)
            # 从 result 或 config 获取回测区间信息
            bt_cfg = self.config.app_config.backtest
            attempt_count = count_tuning_attempts(
                engine, bt_cfg.BACKTEST_START_DATE, bt_cfg.OUT_OF_SAMPLE_DAYS,
            )

            if attempt_count > OVEROPTIMIZE_THRESHOLD:
                level = "CRITICAL" if attempt_count > 30 else "WARNING"
                logger.warning(
                    f"[参数过度优化] {level}: 同区间调参 {attempt_count} 次（阈值 {OVEROPTIMIZE_THRESHOLD}），"
                    f"Sharpe={result.sharpe:.2f}，建议人工复核策略稳健性"
                )
                self._write_alert("overoptimize", {
                    "time": datetime.now().isoformat(),
                    "attempt_count": attempt_count,
                    "threshold": OVEROPTIMIZE_THRESHOLD,
                    "sharpe": round(result.sharpe, 4),
                    "sortino": round(result.sortino, 4),
                    "level": level,
                    "params": result.params,
                })
        except Exception as e:
            logger.warning(f"[参数过度优化] 检查异常: {e}")
