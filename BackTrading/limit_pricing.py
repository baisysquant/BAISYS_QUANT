"""涨跌停价计算 + 可成交量规则 — 撮合层约束建模（Limit Matching）。

目标：在撮合层引入涨跌停约束与可成交量规则，提升成交模拟真实度。

业务范围：仅沪深主板（60x/00x），创业板/科创板/北交所已从系统中剔除。

涨跌停规则（基于前收盘与当期涨跌幅规则）：
    - 主板（60x/00x）：±10%；ST/*ST 一律 ±5%
    - 退市整理期（P0-6）：整理期首日无涨跌幅限制，次日起 ±10%
      （进入退市整理期后涨跌幅独立于 ST 5% 规则，按 10% 计算）
    - 上市初期豁免：
        主板注册制（2023-04-10 起）：上市前 5 个交易日无涨跌幅限制
        主板核准制（2023-04-10 前）：上市首日 ±44%/-36%，次日起 ±10%
    - 限价取整：前收 × (1±比例) 四舍五入到分

可成交量规则（simulate_limit_up_down=true 时撮合层消费）：
    - 非涨跌停日：可成交量比例 1.0（不限）
    - 涨停/跌停日：可成交量比例 = 一字板(开盘=收盘=限价) 用 seal_ratio，
      开盘触板后炸板(open≥限价, close<限价) 用 tradable_ratio，
      盘中冲板(open<限价, high≥限价) 用 intraday_ratio；
      连续板数 ≥ 2 时按 seal_decay 逐板衰减
    - 可成交量 = 当日成交量 × 比例，向下取整到最小交易单位；
      小于一手 → 未成交（日志 [撮合约束] 可追溯）

配置（[BACKTEST]，simulate_limit_up_down=true 开启本模型，false 回退简化撮合）：
    simulate_limit_up_down = true
    limit_seal_ratio = 0.05       # 一字板可成交量比例
    limit_tradable_ratio = 0.30   # 开盘触板后炸板可成交量比例
    limit_intraday_ratio = 0.10   # 盘中冲板可成交量比例
    limit_seal_decay = 0.5        # 连续板每板衰减系数
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

import numpy as np
from loguru import logger

# 涨跌停价取整精度（分）
LIMIT_PX_PRECISION = 100.0
# 主板注册制首批上市日（此后主板新股前 5 日无涨跌幅限制）
MAIN_BOARD_REFORM_DATE = "2023-04-10"
# ST/*ST 涨跌幅限制
ST_LIMIT_RATIO = 0.05
# 主板常规涨跌幅限制
MAIN_BOARD_LIMIT_RATIO = 0.10
# 退市整理期涨跌幅（P0-6）：首日无涨跌幅限制，次日起 ±10%
# （进入退市整理期后独立于 ST 5% 规则；引擎按 is_delisting 状态机判定首日/期间）
DELISTING_PERIOD_LIMIT_RATIO = 0.10
# 核准制主板上市首日限制（44% / -36%）
MAIN_BOARD_FIRST_DAY_UP = 0.44
MAIN_BOARD_FIRST_DAY_DOWN = 0.36
# 主板注册制后上市前 5 个交易日无涨跌幅限制
MAIN_BOARD_REGISTERED_EXEMPT_DAYS = 5
# 数值比较容差（浮点取整后相等判断）
_LIMIT_EPS = 1e-9

_CODE_RE = re.compile(r"(?:[a-zA-Z]*)(\d+)")


def listing_exempt_days(trade_date: Optional[str] = None) -> int:
    """主板上市初期无涨跌幅限制的交易日数。

    Args:
        trade_date: "YYYY-MM-DD"；主板注册制改革(2023-04-10)前后规则不同。

    Returns:
        豁免交易日数（listing_days <= 该值即豁免）。
        注册制后前 5 日豁免；核准制无豁免（首日用 44%/36%）。
    """
    if trade_date is not None and str(trade_date)[:10] >= MAIN_BOARD_REFORM_DATE:
        return MAIN_BOARD_REGISTERED_EXEMPT_DAYS
    return 0


def _round_half_up_scalar(x: float) -> float:
    """交易所"四舍五入到分"（ROUND_HALF_UP）。

    Python round / np.round 采用银行家舍入（round-half-even）：第三位小数恰为 5
    时舍向偶数。交易所（沪深交易规则）为 ROUND_HALF_UP——第三位恰为 5 一律进位。
    例：前收 9.55 → 涨停 9.55×1.1 = 10.505 → 交易所 10.51，round(10.505,2)=10.5。
    用 Decimal(str(x)) 规避二进制浮点表示误差（10.505 实际 ≈10.5049999...）。
    """
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _round_half_up_vec(x: np.ndarray) -> np.ndarray:
    """向量化 half-up（等效 _round_half_up_scalar）。

    前收×比例是 0.01 精度的有限十进制，第三位小数为精确 0-9；仅二进制表示有
    尾差。np.nextafter 将二进制略低于整数的边界值推过 0.5 进位阈值：第三位=5
    （浮点 10.505→1050.4999...）→ 进位；第三位=0（恰为整数）→ 不进位。
    数值结果为 {0, 0.005, 0.01, ...} 的半格边界判定，无 ROUND_HALF_UP 误判。
    """
    y = np.nextafter(x * 100.0, np.inf)
    return np.floor(y + 0.5) / 100.0


def calc_limit_prices(
    prev_close: float,
    ratio_up: float,
    ratio_down: float,
    precision: float = LIMIT_PX_PRECISION,
) -> tuple[float, float]:
    """涨跌停价 = 前收 × (1±比例)，四舍五入到分。

    P0-11 审计修复：以交易所 ROUND_HALF_UP（四舍五入）替代 Python round 的
    银行家舍入（round-half-even）——第三位小数恰为 5 时 round 会舍向偶数，
    与交易所规则系统性偏差 0.01 元，影响封板/一字板/可成交量判定。
    """
    up = _round_half_up_scalar(prev_close * (1 + ratio_up))
    down = _round_half_up_scalar(prev_close * (1 - ratio_down))
    return up, down


def calc_limit_prices_batch(
    prev_close: np.ndarray,
    ratio_up: np.ndarray,
    ratio_down: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """P2 审计修复：向量化涨跌停价批量计算，替代逐股循环调用 calc_limit_prices。

    P0-11：取整采用 ROUND_HALF_UP（_round_half_up_vec），与标量版一致，
    不再使用 np.round（银行家舍入）。

    Args:
        prev_close: 前收盘价数组 (n,)
        ratio_up: 涨停比例数组 (n,)
        ratio_down: 跌停比例数组 (n,)

    Returns:
        (limit_up, limit_down) 各 (n,)
    """
    up = _round_half_up_vec(prev_close * (1.0 + ratio_up))
    down = _round_half_up_vec(prev_close * (1.0 - ratio_down))
    return up, down


def lot_size_for(symbol: str) -> int:
    """A 股沪深主板每手申报单位 = 100 股（一处定义，引擎买入/卖出/可成交量撮合全链路复用）。

    注：系统仅覆盖沪深主板（60x/00x），100 股/手。科创板 200 股/手已随板块剔除。
    """
    return 100


@dataclass(frozen=True)
class LimitPriceInfo:
    """单只股票当日涨跌停信息（供撮合层消费）。"""

    ratio_up: float
    ratio_down: float
    limit_up: float
    limit_down: float
    exempt: bool = False
    is_st: bool = False

    def at_limit_up(self, close: float) -> bool:
        return close >= self.limit_up - _LIMIT_EPS

    def at_limit_down(self, close: float) -> bool:
        return close <= self.limit_down + _LIMIT_EPS


def limit_prices_for(
    prev_close: float,
    symbol: str,
    *,
    is_st: bool = False,
    listing_days: Optional[int] = None,
    trade_date: Optional[str] = None,
) -> LimitPriceInfo:
    """计算单只主板股票当日涨跌停价。

    Args:
        prev_close: 前收盘（不复权）。
        symbol: 股票代码（sh600000 / 600000.SH / 600000）— 保留参数兼容签名，
                 系统仅覆盖沪深主板，不再按代码分派板块规则。
        is_st: 当日是否为 ST/*ST（涨跌幅 5%）。
        listing_days: 上市第 N 个交易日（1=首日）；None 视为已上市多日（无豁免）。
        trade_date: "YYYY-MM-DD"，主板注册制前后规则切换用。

    Returns:
        LimitPriceInfo：exempt=True 时 limit_up/down 用 ±100% 近似无限制。
    """
    # P1 审计修复：非主板代码进入时发出 warning，防止 silently 使用错误涨跌幅
    _digits = _CODE_RE.search(symbol)
    if _digits is not None:
        _code = _digits.group(1)
        if _code.startswith("300"):
            logger.warning(
                f"[涨跌停] {symbol} 为创业板（300），应使用 ±20% 规则而非主板 ±10%。"
                f"请确认 stock_basic_info 过滤正确或关闭 main_board_only。"
            )
        elif _code.startswith("688"):
            logger.warning(
                f"[涨跌停] {symbol} 为科创板（688），应使用 ±20% 规则而非主板 ±10%。"
                f"请确认 stock_basic_info 过滤正确或关闭 main_board_only。"
            )
        elif _code.startswith("8") or _code.startswith("4"):
            logger.warning(
                f"[涨跌停] {symbol} 为北交所（{ _code[0] }xxx），应使用 ±30% 规则而非主板 ±10%。"
                f"请确认 stock_basic_info 过滤正确或关闭 main_board_only。"
            )

    exempt = False
    if listing_days is not None:
        exempt = listing_days <= listing_exempt_days(trade_date)

    if exempt:
        ratio_up = ratio_down = 1.0
    elif (
        listing_days == 1
        and (trade_date is None or str(trade_date)[:10] < MAIN_BOARD_REFORM_DATE)
    ):
        # 核准制主板首日：44% / -36%
        ratio_up, ratio_down = MAIN_BOARD_FIRST_DAY_UP, MAIN_BOARD_FIRST_DAY_DOWN
    else:
        ratio = ST_LIMIT_RATIO if is_st else MAIN_BOARD_LIMIT_RATIO
        ratio_up = ratio_down = ratio

    up, down = calc_limit_prices(prev_close, ratio_up, ratio_down)
    return LimitPriceInfo(
        ratio_up=ratio_up,
        ratio_down=ratio_down,
        limit_up=up,
        limit_down=down,
        exempt=exempt,
        is_st=is_st,
    )


def fill_ratio_for(
    close: float,
    open_price: Optional[float],
    high: Optional[float],
    low: Optional[float],
    limit_up: float,
    limit_down: float,
    board_streak: int = 1,
    *,
    seal_ratio: float = 0.05,
    tradable_ratio: float = 0.30,
    intraday_ratio: float = 0.10,
    seal_decay: float = 0.5,
) -> float:
    """可成交量比例（撮合层消费）——三档触板口径。

    触板判定（日线近似，无需分钟数据）：
        - 涨停方向：high >= limit_up 或 open >= limit_up（一字/高开触及涨停价）
        - 跌停方向：low <= limit_down 或 open <= limit_down
    三档口径（按触板方式区分，全天量 × 比例 = 可成交量）：
        1. 一字板（开=收=限价）：seal_ratio（默认 5%）—— 全天封死，流动性最差
        2. 开盘触板后炸板（open≥限价 但 close<限价）：tradable_ratio（默认 30%）
           —— 开盘即封板、随后打开，早盘限价附近有大量挂单，30% 合理
        3. 盘中冲板（open<限价, high≥限价）：intraday_ratio（默认 10%）
           —— 盘中才触板，全天大部分成交量不在限价附近，30%×全天量严重高估，
              用更低比例折算
    - 非触板日：1.0（不限制）
    - 连续板数 n 板 → 比例 × seal_decay^(n-1)
    """
    touched_up = limit_up is not None and (
        (high is not None and high >= limit_up - _LIMIT_EPS)
        or (open_price is not None and open_price >= limit_up - _LIMIT_EPS)
    )
    if touched_up:
        sealed = (
            open_price is not None and open_price >= limit_up - _LIMIT_EPS
            and close is not None and close >= limit_up - _LIMIT_EPS
        )
        if sealed:
            r = seal_ratio
        elif open_price is not None and open_price >= limit_up - _LIMIT_EPS:
            # 开盘涨停后炸板：open≥limit 但 close<limit
            r = tradable_ratio
        else:
            # 盘中冲板：open<limit, high≥limit
            r = intraday_ratio
        return r * (seal_decay ** max(0, board_streak - 1))

    touched_down = limit_down is not None and (
        (low is not None and low <= limit_down + _LIMIT_EPS)
        or (open_price is not None and open_price <= limit_down + _LIMIT_EPS)
    )
    if touched_down:
        sealed = (
            open_price is not None and open_price <= limit_down + _LIMIT_EPS
            and close is not None and close <= limit_down + _LIMIT_EPS
        )
        if sealed:
            r = seal_ratio
        elif open_price is not None and open_price <= limit_down + _LIMIT_EPS:
            # 开盘跌停后炸板：open≤limit 但 close>limit
            r = tradable_ratio
        else:
            # 盘中触跌停：open>limit, low≤limit
            r = intraday_ratio
        return r * (seal_decay ** max(0, board_streak - 1))

    return 1.0
