"""
上市日期表同步（IPO 日期显式注入）

P0-6 ④ 审计修复：引擎曾从行情数据推断上市日期（"数据期间新出现"股票的首个
交易日 = 上市日），数据缺口/中途加入的股票会被误判为新股，错误激活"注册制前
5 日无涨跌幅"豁免（放大收益）。本模块建立 stock_listing_days 表（主键 symbol），
从 AkShare stock_info_a_code_name（上市日期列）回填，回测引擎仅消费显式注入的
上市日期（params._listing_days），缺失时豁免逻辑整体停用并告警。

数据源（AkShare，防御性调用，网络失败优雅降级并告警）：
  - stock_info_a_code_name  沪深 A 股基本信息（代码 / 名称 / 上市日期）

用法：
    from DataManager.ListingDaysSync import (
        ensure_listing_days_table, sync_listing_days, load_listing_days,
    )
    ensure_listing_days_table(engine)
    sync_listing_days(engine, symbols)
    listing_days = load_listing_days(engine, symbols, start_date)
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger
from sqlalchemy import text

from UtilsManager.CodeNormalizer import CodeNormalizer

_LISTING_TABLE = "stock_listing_days"


def _norm_pool(symbols: list[str]) -> set[str]:
    """池内符号归一化为带市场前缀（sh/sz/bj）形式。"""
    out: set[str] = set()
    for s in symbols:
        s = str(s).strip()
        if s.startswith(("sh", "sz", "bj")):
            out.add(s)
        else:
            digits = "".join(ch for ch in s if ch.isdigit())
            if len(digits) == 6:
                out.add(CodeNormalizer.add_market_prefix(digits))
    return out


def _fetch_a_code_name() -> pd.DataFrame | None:
    """沪深 A 股基本信息（含上市日期）。失败返回 None。"""
    try:
        import akshare as ak

        df = ak.stock_info_a_code_name()
        if df is None or df.empty:
            return None
        return df
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[上市日] stock_info_a_code_name 拉取失败（IPO 日期表未更新）: {e}")
        return None


def ensure_listing_days_table(engine: Any) -> None:
    """确保 stock_listing_days 表存在（幂等）。"""
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {_LISTING_TABLE} (
                symbol VARCHAR(16) NOT NULL PRIMARY KEY,
                ipo_date DATE NOT NULL,
                updated_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """))


def sync_listing_days(engine: Any, symbols: list[str]) -> dict[str, int]:
    """同步池内股票的上市日期（AkShare stock_info_a_code_name → stock_listing_days）。

    Returns:
        {"upserted": int} 本次写入行数；网络失败返回 {"upserted": 0}（不阻断回测）。
    """
    df = _fetch_a_code_name()
    if df is None:
        return {"upserted": 0}
    code_col = next((c for c in ("代码", "code") if c in df.columns), None)
    date_col = next((c for c in ("上市日期", "list_date", "ipo_date") if c in df.columns), None)
    if code_col is None or date_col is None:
        logger.warning(f"[上市日] 数据缺列（code={code_col!r}, date={date_col!r}），跳过同步")
        return {"upserted": 0}
    pool = _norm_pool(symbols)
    if not pool:
        return {"upserted": 0}
    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        syms = set()
        digits = "".join(ch for ch in str(r[code_col]) if ch.isdigit())
        if len(digits) == 6:
            syms.add(CodeNormalizer.add_market_prefix(digits))
        if not syms:
            continue
        sym = next(iter(syms))
        if sym not in pool:
            continue
        if pd.notna(r[date_col]):
            d = pd.to_datetime(r[date_col], errors="coerce").date()
        else:
            d = None
        if d is None or pd.isna(d):
            continue
        rows.append({"symbol": sym, "ipo_date": d})
    if not rows:
        return {"upserted": 0}
    sql = text(f"""
        INSERT INTO {_LISTING_TABLE} (symbol, ipo_date, updated_at)
        VALUES (:symbol, :ipo_date, NOW())
        ON CONFLICT (symbol) DO UPDATE SET
            ipo_date = EXCLUDED.ipo_date,
            updated_at = NOW()
    """)
    with engine.begin() as conn:
        conn.execute(sql, rows)
    logger.info(f"[上市日] IPO 日期同步完成: {len(rows)} 只（池内 {len(pool)} 只）")
    return {"upserted": len(rows)}


def load_listing_days(
    engine: Any, symbols: list[str], start_date: str
) -> dict[str, str]:
    """加载池内股票上市日期（显式注入用，参数化 ANY(:syms) 杜绝注入）。

    Returns:
        {symbol: "YYYY-MM-DD"}；查询失败/无数据返回 {}（引擎将停用新股豁免并告警）。
    """
    if not symbols:
        return {}
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT symbol, ipo_date
                FROM {_LISTING_TABLE}
                WHERE symbol = ANY(:syms)
            """), {"syms": symbols}).fetchall()
        out: dict[str, str] = {}
        for symbol, ipo_date in rows:
            d = str(ipo_date)[:10]
            if d >= start_date:
                out[symbol] = d
        logger.info(f"加载上市日期: {len(out)} 只（查询起点 {start_date}）")
        return out
    except Exception as e:  # noqa: BLE001
        logger.warning(f"加载上市日期失败（引擎将停用新股豁免逻辑）: {e}")
        return {}