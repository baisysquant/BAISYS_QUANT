from __future__ import annotations

from typing import Any

import pandas as pd

from ConfigParser import Config
from LogicAnalyzer.PipelineScoring import calc_entry_signal
from LogicAnalyzer.PipelineState import should_exit

_RISK_MAP: dict[str, float] = {
    "NONE": 1.0, "LOW": 1.5, "MEDIUM": 3.0, "HIGH": 5.0, "D": 8.0, "E": 10.0,
}


def _risk_to_numeric(risk_str: str) -> float:
    return _RISK_MAP.get(risk_str.upper().strip(), 3.0)


class PipelineAdapter:
    """将每日管线适配为 AKQuant 按日回调策略。

    使用 PipelineState.should_exit 和 PipelineScoring.calc_entry_signal
    做入场/出场判断。回测开始前需通过 ``prepare_backtest_data``
    预计算评分列。
    """

    def __init__(self, config: Config, initial_cash: float = 1_000_000.0) -> None:
        self.config = config
        self.initial_cash = initial_cash
        self._state: dict[str, Any] = {
            "date": None,
            "portfolio_value": initial_cash,
            "positions": {},
        }

    def on_start(self) -> None:
        self._state = {
            "date": None,
            "portfolio_value": self.initial_cash,
            "positions": {},
        }

    def on_bar(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        self._state["date"] = df.get("trade_date", pd.Timestamp.today())

        exit_mask = should_exit(df)
        entry_mask = calc_entry_signal(df)

        orders: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            sym = str(row.get("symbol", ""))
            if not sym:
                continue
            if exit_mask.get(row.name, False):
                orders.append({"symbol": sym, "action": "sell", "weight": 1.0})
            elif entry_mask.get(row.name, False):
                risk = _risk_to_numeric(str(row.get("风险等级", "MEDIUM")))
                weight = 1.0 / risk
                orders.append({"symbol": sym, "action": "buy", "weight": weight})
        return orders

    def on_end(self) -> dict[str, Any]:
        return {
            "final_value": self._state["portfolio_value"],
            "total_return": (self._state["portfolio_value"] / self.initial_cash) - 1,
        }

    def set_params(self, params: dict[str, float]) -> None:
        for key, value in params.items():
            upper = key.upper()
            for section_attr in ("position_sizing", "scoring_params", "regime_detection", "divergence"):
                section = getattr(self.config.app_config, section_attr, None)
                if section and hasattr(section, upper):
                    setattr(section, upper, value)
                    break
