"""涨跌停价计算 + 可成交量规则 — 撮合层约束建模（Limit Matching）。

目标：在撮合层引入涨跌停约束与可成交量规则，提升成交模拟真实度。

涨跌停规则（基于前收盘与当期涨跌幅规则）：
    - 主板（60x/00x）：±10%；ST/*ST 一律 ±5%
    - 创业板（30x）/ 科创板（688/689）：±20%；ST ±5%（创业板 2020-08 起）
    - 北交所（8x/43x/92x）：±30%；首日无限制
    - 上市初期豁免：
        创业板/科创板：上市前 5 个交易日无涨跌幅限制
        主板注册制（2023-04-10 起）：上市前 5 个交易日无涨跌幅限制
        主板核准制（2023-04-10 前）：上市首日 ±44%/-36%，次日起 ±10%
        北交所：上市首日无涨跌幅限制
    - 限价取整：前收 × (1±比例) 四舍五入到分

可成交量规则（simulate_limit_up_down=true 时撮合层消费）：
    - 非涨跌停日：可成交量比例 1.0（不限）
    - 涨停/跌停日：可成交量比例 = 一字板(开盘=收盘=限价) 用 seal_ratio，
      盘中触板用 tradable_ratio；连续板数 ≥ 2 时按 seal_decay 逐板衰减
    - 可成交量 = 当日成交量 × 比例，向下取整到最小交易单位；
      小于一手 → 未成交（日志 [撮合约束] 可追溯）

配置（[BACKTEST]，simulate_limit_up_down=true 开启本模型，false 回退简化撮合）：
    simulate_limit_up_down = true
    limit_seal_ratio = 0.05       # 一字板可成交量比例
    limit_tradable_ratio = 0.30   # 盘中触板可成交量比例
    limit_seal_decay = 0.5        # 连续板每板衰减系数
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

# 涨跌停价取整精度（分）
LIMIT_PX_PRECISION = 100.0
# 主板注册制首批上市日（此后主板新股前 5 日无涨跌幅限制）
MAIN_BOARD_REFORM_DATE = "2023-04-10"
# ST/*ST 涨跌幅限制
ST_LIMIT_RATIO = 0.05
# 核准制主板上市首日限制（44% / -36%）
MAIN_BOARD_FIRST_DAY_UP = 0.44
MAIN_BOARD_FIRST_DAY_DOWN = 0.36
# 数值比较容差（浮点取整后相等判断）
_LIMIT_EPS = 1e-9

_CODE_RE = re.compile(r"(?:[a-zA-Z]*)(\d+)")


class Board(str, Enum):
    """A股板块分类（决定常规涨跌幅限制）。"""

    MAIN = "MAIN"  # 主板 10%
    GEM = "GEM"    # 创业板 20%
    STAR = "STAR"  # 科创板 20%
    BSE = "BSE"    # 北交所 30%


_BOARD_LIMIT_RATIO = {
    Board.MAIN: 0.10,
    Board.GEM: 0.20,
    Board.STAR: 0.20,
    Board.BSE: 0.30,
}


def classify_board(symbol: str) -> Board:
    """按代码前缀分类板块（兼容 sh600000 / 600000.SH / 600000 格式）。"""
    m = _CODE_RE.search(str(symbol))
    code = m.group(1) if m else str(symbol)
    if code.startswith(("688", "689")):
        return Board.STAR
    if code.startswith("30"):
        return Board.GEM
    if code.startswith(("8", "92", "43")):
        return Board.BSE
    return Board.MAIN


def listing_exempt_days(board: Board, trade_date: Optional[str] = None) -> int:
    """上市初期无涨跌幅限制的交易日数。

    Args:
        board: 板块分类。
        trade_date: "YYYY-MM-DD"；主板注册制改革(2023-04-10)前后规则不同。

    Returns:
        豁免交易日数（listing_days <= 该值即豁免）。
    """
    if board in (Board.GEM, Board.STAR):
        return 5
    if board == Board.BSE:
        return 1
    # 主板：注册制后前 5 日豁免；核准制无豁免（首日用 44%/36%）
    if trade_date is not None and str(trade_date)[:10] >= MAIN_BOARD_REFORM_DATE:
        return 5
    return 0


def board_limit_ratio(board: Board, is_st: bool = False) -> float:
    """常规交易日涨跌幅限制比例（ST 一律 5%）。"""
    if is_st:
        return ST_LIMIT_RATIO
    return _BOARD_LIMIT_RATIO[board]


def calc_limit_prices(
    prev_close: float,
    ratio_up: float,
    ratio_down: float,
    precision: float = LIMIT_PX_PRECISION,
) -> tuple[float, float]:
    """涨跌停价 = 前收 × (1±比例)，四舍五入到分。"""
    import math

    up = math.floor(prev_close * (1 + ratio_up) * precision + 0.5) / precision
    down = math.floor(prev_close * (1 - ratio_down) * precision + 0.5) / precision
    return up, down


def lot_size_for(symbol: str) -> int:
    """A 股每手申报单位（一处定义，引擎买入/卖出/可成交量撮合全链路复用）。

    - 科创板（688/689）：200 股/手（申报起点，卖出可零股）
    - 其余板块（主板 60x/00x、创业板 30x、北交所 8x/43x/92x）：100 股/手

    注：创业板（30x）与主板一致为 100 股/手 —— 科创板才是 200 股申报起点。
    """
    return 200 if classify_board(symbol) == Board.STAR else 100


@dataclass(frozen=True)
class LimitPriceInfo:
    """单只股票当日涨跌停信息（供撮合层消费）。"""

    board: Board
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
    """计算单只股票当日涨跌停价。

    Args:
        prev_close: 前收盘（不复权）。
        symbol: 股票代码（sh600000 / 600000.SH / 600000）。
        is_st: 当日是否为 ST/*ST（涨跌幅 5%）。
        listing_days: 上市第 N 个交易日（1=首日）；None 视为已上市多日（无豁免）。
        trade_date: "YYYY-MM-DD"，主板注册制前后规则切换用。

    Returns:
        LimitPriceInfo：exempt=True 时 limit_up/down 用 ±100% 近似无限制。
    """
    board = classify_board(symbol)
    exempt = False
    if listing_days is not None:
        exempt = listing_days <= listing_exempt_days(board, trade_date)

    if exempt:
        ratio_up = ratio_down = 1.0
    elif (
        board == Board.MAIN
        and listing_days == 1
        and (trade_date is None or str(trade_date)[:10] < MAIN_BOARD_REFORM_DATE)
    ):
        # 核准制主板首日：44% / -36%
        ratio_up, ratio_down = MAIN_BOARD_FIRST_DAY_UP, MAIN_BOARD_FIRST_DAY_DOWN
    else:
        ratio = board_limit_ratio(board, is_st)
        ratio_up = ratio_down = ratio

    up, down = calc_limit_prices(prev_close, ratio_up, ratio_down)
    return LimitPriceInfo(
        board=board,
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
    seal_decay: float = 0.5,
) -> float:
    """可成交量比例（撮合层消费）——盘中触板口径（0.3）。

    触板判定（日线近似，无需分钟数据）：
        - 涨停方向：high >= limit_up 或 open >= limit_up（一字/高开触及涨停价）
        - 跌停方向：low <= limit_down 或 open <= limit_down
    两档口径：
        - 一字板（开=收=限价）：seal_ratio —— 全天封死，流动性最差
        - 盘中板（触板但开/收未同时封住）：tradable_ratio
          —— 含"high>=limit_up 且 close<limit_up"的炸板回落，当日买入按此折算，
             而非完全无约束
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
        r = seal_ratio if sealed else tradable_ratio
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
        r = seal_ratio if sealed else tradable_ratio
        return r * (seal_decay ** max(0, board_streak - 1))
    return 1.0
