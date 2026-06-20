from __future__ import annotations

from typing import Any

from akquant import Bar, Strategy
from akquant.params import FloatParam, IntParam, ParamModel


_trade_log: list[dict[str, Any]] = []
_equity_curve: list[dict[str, Any]] = []


def get_trade_log() -> list[dict[str, Any]]:
    return list(_trade_log)


def clear_trade_log() -> None:
    _trade_log.clear()


def get_equity_curve() -> list[dict[str, Any]]:
    return list(_equity_curve)


def clear_equity_curve() -> None:
    _equity_curve.clear()



class QuantPipelineParams(ParamModel):
    """回测策略可寻优参数。"""

    atr_stop_mult: float = FloatParam(1.5, ge=0.5, le=5.0, description="ATR止损倍数")
    kelly_fraction: float = FloatParam(0.25, ge=0.0, le=1.0, description="半凯利系数")
    position_a: float = FloatParam(0.3, ge=0.0, le=1.0, description="A级基础仓位")
    liq_veto_ratio: float = FloatParam(0.05, ge=0.01, le=0.2, description="流动性否决阈值")
    boll_narrow_ratio: float = FloatParam(0.8, ge=0.3, le=2.0, description="窄布林判定阈值")
    cross_decay_days: int = IntParam(30, ge=5, le=120, description="金叉衰减半衰期")


class QuantPipelineStrategy(Strategy):
    """AKQuant 策略 — 复用管线进场/出场信号。

    策略假设 K 线 DataFrame 中已预计算以下列：
    ``进场评分``、``退出评分``、``风险等级``、``止损价``。

    可寻优参数通过 ``PARAM_MODEL`` 声明，AKQuant 自动注入到策略实例。
    """

    PARAM_MODEL = QuantPipelineParams

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        model_fields = set(QuantPipelineParams.model_fields)
        param_kwargs = {k: v for k, v in kwargs.items() if k in model_fields}
        self.params = QuantPipelineParams(**param_kwargs)
        self._entry_cache: dict[str, float] = {}
        self._cost = {
            "commission": float(kwargs.get("commission", 0.0003)),
            "stamp_tax": float(kwargs.get("stamp_tax", 0.001)),
            "slippage": float(kwargs.get("slippage", 0.001)),
            "max_position_pct": float(kwargs.get("max_position_pct", 0.1)),
        }

    def on_start(self) -> None:
        self._entry_cache.clear()
        _trade_log.clear()
        _equity_curve.clear()

    def on_bar(self, bar: Bar) -> None:
        symbol = bar.symbol

        exit_signal = self._check_exit(bar)
        entry_signal = self._check_entry(bar) if not exit_signal else False

        if exit_signal:
            self.sell(symbol)
            self._entry_cache.pop(symbol, None)
            _trade_log.append({
                "time": str(getattr(bar, "trade_date", "")),
                "symbol": symbol,
                "action": "sell",
                "price": float(bar.close),
            })
        elif entry_signal:
            if symbol not in self._entry_cache:
                weight = self._calc_weight(bar)
                self.order_target_percent(symbol, weight)
                self._entry_cache[symbol] = weight
                _trade_log.append({
                    "time": str(getattr(bar, "trade_date", "")),
                    "symbol": symbol,
                    "action": "buy",
                    "price": float(bar.close) * (1 + self._cost["slippage"]),
                    "weight": weight,
                })

        pv = getattr(self, "portfolio_value", None)
        if pv is None:
            cash = getattr(self, "cash", 0)
            pos_val = 0
            for s, w in self._entry_cache.items():
                pos_val += w * getattr(self, "portfolio_value", 0) if hasattr(self, "portfolio_value") else 0
            pv = cash + pos_val if cash else 0
        _equity_curve.append({
            "time": str(getattr(bar, "trade_date", "")),
            "portfolio_value": float(pv or 0),
        })

    def _check_exit(self, bar: Bar) -> bool:
        risk = str(getattr(bar, "风险等级", "LOW")).upper()
        if risk in ("HIGH", "D"):
            return True
        exit_score = float(getattr(bar, "退出评分", 0))
        entry_score = float(getattr(bar, "进场评分", 0))
        if exit_score > entry_score and exit_score > 0:
            return True
        stop_loss = float(getattr(bar, "止损价", 0) or 0)
        if stop_loss > 0 and bar.close < stop_loss:
            return True
        return False

    def _check_entry(self, bar: Bar) -> bool:
        entry_score = float(getattr(bar, "进场评分", 0))
        if entry_score < 60:
            return False
        risk = str(getattr(bar, "风险等级", "LOW")).upper()
        if risk in ("HIGH", "D", "E"):
            return False
        return True

    def _calc_weight(self, bar: Bar) -> float:
        risk_str = str(getattr(bar, "风险等级", "MEDIUM")).upper()
        risk_map = {"NONE": 1.0, "LOW": 1.5, "MEDIUM": 3.0, "HIGH": 5.0, "D": 8.0}
        base = 1.0 / risk_map.get(risk_str, 3.0)
        capped = min(base, self._cost["max_position_pct"])
        return capped
