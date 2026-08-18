"""申万一级行业映射表同步（P0-7 ①：行业一级中性化静默失效修复）。

stock_basic_info_sw 表为申万二级语义（外部管线维护），本模块独立维护
stock_basic_info_sw_l1：每只股票 → 申万一级行业（l1_name 命名与
DataCollection/MacroFactorFetcher.py 的 _SW1_MACRO_CLASS 键一致，
供行业一级中性化与宏观 tilt 映射使用）。

数据源：AkShare 申万一级行业指数成分（sw_index_first_info 获取一级列表，
sw_index_third_cons 逐一级取成分股）。失败时记录 error 日志（不吞异常），
由调用方决定是否降级。
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd
from loguru import logger
from sqlalchemy import text

L1_TABLE = "stock_basic_info_sw_l1"

_PREFIXES = ("sh", "sz", "bj")


def _match_columns(df: pd.DataFrame, *candidates: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"缺少候选列 {candidates}，实际列: {list(df.columns)}")


def _normalize_code(raw: Any) -> str | None:
    s = str(raw).strip()
    for pfx in _PREFIXES:
        if s.lower().startswith(pfx):
            s = s[len(pfx):]
            break
    # akShare 成分股代码带交易所后缀（"000019.SZ"）→ 去后缀
    if "." in s:
        s = s.split(".")[0]
    s = s.zfill(6)
    return s if s.isdigit() else None


def _fetch_with_retry(fn, desc: str, retries: int = 3) -> Any:
    """带指数退避的 AkShare 拉取包装（legulegu.com 偶发 DNS/连接抖动，重试可显著提高成功率）。"""
    import random

    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            if attempt < retries:
                _backoff = 2 ** attempt + random.uniform(0, 1)
                logger.warning(
                    f"[申万一级] {desc} 拉取失败({attempt}/{retries}): {type(e).__name__}: {e}，"
                    f"等待 {_backoff:.1f}s 后重试"
                )
                time.sleep(_backoff)
    assert last is not None
    raise last


def fetch_sw_l1_memberships() -> pd.DataFrame:
    """AkShare 拉取 申万一级行业 → 成分股 映射。

    Returns:
        DataFrame: l1_code / l1_name / stock_code / stock_name
    """
    import akshare as ak

    l1_df = _fetch_with_retry(ak.sw_index_first_info, "申万一级列表")
    if l1_df is None or l1_df.empty:
        raise RuntimeError("ak.sw_index_first_info() 返回空数据")
    code_col = _match_columns(l1_df, "行业代码", "指数代码", "代码")
    name_col = _match_columns(l1_df, "行业名称", "指数名称", "名称")

    frames: list[pd.DataFrame] = []
    for _, row in l1_df.iterrows():
        l1_code = str(row[code_col]).strip()
        l1_name = str(row[name_col]).strip()
        if not l1_code:
            continue
        try:
            cons = _fetch_with_retry(
                lambda: ak.sw_index_third_cons(symbol=l1_code), f"{l1_name}({l1_code}) 成分"
            )
        except Exception as e:
            logger.error(
                f"[申万一级] 拉取 {l1_name}({l1_code}) 成分失败: {type(e).__name__}: {e}"
            )
            continue
        if cons is None or cons.empty:
            logger.error(f"[申万一级] {l1_name}({l1_code}) 成分股为空")
            continue
        c_code = _match_columns(cons, "股票代码", "代码")
        c_name = _match_columns(cons, "股票名称", "名称", "股票简称")
        frames.append(pd.DataFrame({
            "l1_code": l1_code,
            "l1_name": l1_name,
            "stock_code": cons[c_code].astype(str),
            "stock_name": cons[c_name].astype(str),
        }))

    if not frames:
        raise RuntimeError("申万一级成分股拉取全部失败（检查网络或 AkShare 接口变更）")

    merged = pd.concat(frames, ignore_index=True)
    merged["stock_code"] = merged["stock_code"].map(_normalize_code)
    merged = merged.dropna(subset=["stock_code"])
    merged = merged.drop_duplicates(subset=["l1_code", "stock_code"], keep="first")
    return merged[["l1_code", "l1_name", "stock_code", "stock_name"]]


def sync_sw_l1_industries(engine: Any, trade_date: date | None = None) -> int:
    """同步 stock_basic_info_sw_l1（当日快照语义：先删当日再全量插入）。

    Args:
        engine: SQLAlchemy engine（PostgreSQL 生产；sqlite 亦可，便于测试）。
        trade_date: 记录日期，默认今天。

    Returns:
        写入行数。

    Raises:
        RuntimeError: 拉取或入库失败（不吞异常，由调用方记录监控）。
    """
    day = trade_date or datetime.now().date()
    members = fetch_sw_l1_memberships()
    with engine.connect() as conn:
        conn.execute(text(f"DELETE FROM {L1_TABLE} WHERE record_date = :d"), {"d": day})
        rows = [
            {
                "l1_code": r.l1_code,
                "l1_name": r.l1_name,
                "stock_code": r.stock_code,
                "stock_name": r.stock_name,
                "record_date": day,
            }
            for r in members.itertuples(index=False)
        ]
        if rows:
            conn.execute(
                text(
                    f"INSERT INTO {L1_TABLE} "
                    "(l1_code, l1_name, stock_code, stock_name, record_date) "
                    "VALUES (:l1_code, :l1_name, :stock_code, :stock_name, :record_date)"
                ),
                rows,
            )
        conn.commit()
    logger.info(f"[申万一级] 同步完成: {len(rows)} 条 (record_date={day})")
    return len(rows)