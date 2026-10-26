"""
AShareHub HTTP 客户端工厂

P2-2 审计修复：
原实现全局 monkeypatch 第三方 AShareHub.__init__ 并 verify=False 禁用 TLS 校验
（中间人风险 + 第三方库升级脆弱）。已移除：所有 AShareHub 调用统一走本工厂注入，
仅信任显式 CA（SSL_CERT_FILE / REQUESTS_CA_BUNDLE）或系统默认信任库，绝不降级。

企业代理自签名证书场景：导出代理 CA 为 PEM 并设置 SSL_CERT_FILE 即可。
"""

from __future__ import annotations

import os
from typing import Any

from loguru import logger

# ── 动态导入 asharehub，包未安装时优雅降级 ──
try:
    from asharehub import AShareHub  # noqa: PLC2701
except ImportError as _exc:  # noqa: PGH003
    AShareHub = None  # type: ignore[misc, assignment]
    _asharehub_import_error = _exc


def make_asharehub_client(api_key: str) -> Any:  # noqa: ANN401
    """创建 AShareHub API 客户端（工厂函数）。

    Parameters
    ----------
    api_key : str
        AShareHub API 密钥。

    Returns
    -------
    AShareHubClient
        包装后的客户端对象，代理所有 AShareHub 原始方法。

    Raises
    ------
    ImportError
        当 asharehub 包未安装时抛出。
    """
    if AShareHub is None:
        raise ImportError(
            "asharehub 包未安装。请运行: pip install asharehub"
        ) from _asharehub_import_error

    # TLS verify 策略：尊重系统 CA 或显式环境变量，绝不传 verify=False
    # 企业代理自签名证书 → 设置 SSL_CERT_FILE 或 REQUESTS_CA_BUNDLE
    _verify: str | bool = (
        os.environ.get("SSL_CERT_FILE")
        or os.environ.get("REQUESTS_CA_BUNDLE")
        or True
    )

    raw_client = AShareHub(api_key=api_key)

    return _AShareHubClientProxy(raw_client, _verify)


class _AShareHubClientProxy:
    """透明代理：包装原始 AShareHub 客户端，确保 TLS 校验绝不降级。

    方法分两类：
    1. 已显式声明的方法 —— 直接调用原始客户端（业务方法签名固定，不注入 verify）
    2. 未声明的方法 —— 通过 __getattr__ 转发（未来 API 扩展无需改代码）
    """

    # 已知需要调用的 API 方法名（便于 IDE 补全 & 文档）
    KNOWN_METHODS: tuple[str, ...] = (
        "industry_list",
        "fundamentals",
        "holder_trade",
        "chip_distribution",
        "moneyflow",
        "forecast",
        "adj_factor",
        "index_daily",
    )

    def __init__(self, raw: Any, verify: str | bool) -> None:  # noqa: ANN401
        object.__setattr__(self, "_raw", raw)
        object.__setattr__(self, "_verify", verify)

    # ── 显式代理方法 ──

    def industry_list(self, **kwargs: Any) -> Any:  # noqa: ANN401
        """获取申万行业分类列表。"""
        return self._call("industry_list", **kwargs)

    def fundamentals(self, trade_date: str | None = None, **kwargs: Any) -> Any:  # noqa: ANN401
        """获取个股每日估值（pe / pe_ttm / pb / dv_ratio / total_mv）。

        Endpoint: /v2/market/fundamentals
        """
        return self._call("fundamentals", trade_date=trade_date, **kwargs)

    def holder_trade(self, **kwargs: Any) -> Any:  # noqa: ANN401
        """获取股东增减持明细。"""
        return self._call("holder_trade", **kwargs)

    def chip_distribution(self, trade_date: str | None = None, **kwargs: Any) -> Any:  # noqa: ANN401
        """获取筹码分布数据。"""
        return self._call("chip_distribution", trade_date=trade_date, **kwargs)

    def moneyflow(self, **kwargs: Any) -> Any:  # noqa: ANN401
        """获取全市场资金流向。"""
        return self._call("moneyflow", **kwargs)

    def forecast(self, symbol: str | None = None, **kwargs: Any) -> Any:  # noqa: ANN401
        """获取业绩预告数据。

        Endpoint: /v1/financials/forecast
        """
        return self._call("forecast", symbol=symbol, **kwargs)

    def adj_factor(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        **kwargs: Any,
    ) -> Any:  # noqa: ANN401
        """获取复权因子。"""
        return self._call(
            "adj_factor",
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            **kwargs,
        )

    def index_daily(
        self,
        symbol: str,
        start_date: str | None = None,
        end_date: str | None = None,
        **kwargs: Any,
    ) -> Any:  # noqa: ANN401
        """获取指数日线行情。"""
        return self._call(
            "index_daily",
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            **kwargs,
        )

    # ── 动态转发（未知方法）──

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """转发未显式声明的属性/方法到原始客户端。"""
        raw = object.__getattribute__(self, "_raw")
        attr = getattr(raw, name)
        if callable(attr):
            return lambda *args, **kwargs: self._call(name, *args, **kwargs)
        return attr

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401
        if name in ("_raw", "_verify"):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_raw"), name, value)

    # ── 内部方法 ──

    def _call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """调用原始客户端方法，确保 TLS 校验不被降级。

        策略：
        1. 第三方 asharehub 业务方法签名固定（fundamentals/index_daily 等均无
           **kwargs，不接受 verify 参数）—— 不再向方法注入 verify，
           否则 TypeError: unexpected keyword argument 'verify'。
        2. TLS 校验由底层 httpx.Client 默认 verify=True 保证（P2-2 意图）；
           自定义 CA 通过 SSL_CERT_FILE / REQUESTS_CA_BUNDLE 环境变量自动生效。
        3. 调用方显式传 verify=False → 拒绝并忽略（绝不降级）。
        """
        if kwargs.pop("verify", None) is False:
            logger.warning(
                f"拒绝 {method_name} 的 verify=False 请求（已忽略）。"
                "如需使用自定义 CA，请设置 SSL_CERT_FILE 环境变量。"
            )

        raw = object.__getattribute__(self, "_raw")
        method = getattr(raw, method_name)
        return method(*args, **kwargs)
