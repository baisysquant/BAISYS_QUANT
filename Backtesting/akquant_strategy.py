from __future__ import annotations

from typing import Any

from akquant import Bar, Strategy
from akquant.params import FloatParam, IntParam, ParamModel



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

    def on_start(self) -> None:
        self._entry_cache.clear()

    def on_bar(self, bar: Bar) -> None:
        symbol = bar.symbol

        exit_signal = self._check_exit(bar)
        entry_signal = self._check_entry(bar) if not exit_signal else False

        if exit_signal:
            self.sell(symbol)
            self._entry_cache.pop(symbol, None)
        elif entry_signal:
            if symbol not in self._entry_cache:
                weight = self._calc_weight(bar)
                self.order_target_percent(symbol, weight)
                self._entry_cache[symbol] = weight

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
        return 1.0 / risk_map.get(risk_str, 3.0)
