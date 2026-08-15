"""
ST/退市状态 PIT（Point-In-Time）逐日表同步

P0-5 审计修复：stock_st_history 曾为「最近快照」（旧代码注释自认），历史 ST
期缺失 → 历史 5% 涨跌幅被错按 10%、ST 禁买/强平逻辑失效，价格与成交双失真；
且 _st_syms_by_day 只对表内出现过的日期生效。本模块建立逐日 ST/退市状态表
（主键 symbol+trade_date）并回填全历史 PIT 序列，回测引擎按日 O(1) 查询。

数据源（AkShare，逐个防御性调用，网络失败优雅降级并告警）：
  - stock_zh_a_st_em           当前 ST/*ST 列表 → 每日增量归档（快照日 = 今天）
  - stock_zh_a_spot_em         当前全市场列表 → 快照日显式 is_st=False
  - stock_info_sz_change_name  深交所简称变更历史（变更日期+简称）
       → 按简称含 ST/退 重建 SZ 股历史 ST/退市整理期（PIT 核心）
  - stock_info_sz_delist / stock_info_sh_delist  终止上市日期
       → P1-4 修复：退市整理期 = 摘牌日前 N 个交易日
       （退市新规 2020-12-31 后摘牌 15 日 / 此前 30 日），方向在摘牌日之前

P1-1 修复说明（沪市 ST 历史缺失）：
  深市历史 ST 期由简称变更历史精确回填；沪市存续股的"曾 ST 期"无免费
  带日期数据源（SSE 官网简称变更页无公开接口；新浪/东财曾用名列表无日期），
  仅能保证：① 每日快照（今日起逐日累积）② 退市股退市整理期（摘牌日前
  N 交易日，精确）③ 退市时简称含 ST 的股票在整理期前的 ST 历史无法回填
  （见同步日志统计）。覆盖检查按"PIT 行 / K 线交易日"真实覆盖率判定，
  不再因"窗口内出现过"误判已覆盖（旧漏洞：每日快照只写今天一行即跳过）。

行粒度：仅对标的实际交易日（stock_daily_kline.trade_date）展开，与回测 K 线
日键完全对齐，不写入非交易日。

用法：
    from DataManager.StPitSync import ensure_st_history_table, sync_st_pit, load_st_pit
    ensure_st_history_table(engine)
    sync_st_pit(engine, symbols, start_date="2024-01-01", end_date="2026-08-14")
    st_history = load_st_pit(engine, symbols, start_date, end_date)
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd
from loguru import logger
from sqlalchemy import text

from UtilsManager.CodeNormalizer import CodeNormalizer

_ST_TABLE = "stock_st_history"

# 简称判定：含 "ST"（覆盖 ST/*ST/S*ST/SST 等变体）→ is_st；含 "退"（退市整理期，
# 如 "退市金亚" / "*ST金泰退"）→ is_delisting。简称变化以变更日为生效日。
_ST_NAME_KEY = "ST"

# P1-4 修复：退市整理期规则分段（退市新规）——
# 2020-12-31（含）后摘牌的整理期为 15 个交易日；此前为 30 个交易日。
DELISTING_PERIOD_DAYS_NEW = 15
DELISTING_PERIOD_DAYS_OLD = 30
DELISTING_REFORM_DATE = date(2020, 12, 31)


def _name_flags(name: str | None) -> tuple[bool, bool]:
    """由证券简称判定 (is_st, is_delisting)。"""
    n = (name or "").upper().replace(" ", "")
    return ("ST" in n), ("退" in n)


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


def _norm_codes(codes: list[Any]) -> set[str]:
    """AkShare 6 位代码 → 带市场前缀符号集。"""
    out: set[str] = set()
    for c in codes:
        digits = "".join(ch for ch in str(c) if ch.isdigit())
        if len(digits) == 6:
            out.add(CodeNormalizer.add_market_prefix(digits))
    return out


# ── AkShare 取数封装（独立函数便于测试打桩与降级） ──


def _fetch_st_list() -> pd.DataFrame | None:
    """当前 ST/*ST 列表（每日快照源）。失败返回 None。"""
    try:
        import akshare as ak

        df = ak.stock_zh_a_st_em()
        if df is None or df.empty:
            return None
        return df
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[ST PIT] stock_zh_a_st_em 拉取失败（跳过今日 ST 归档）: {e}")
        return None


def _fetch_spot_list() -> pd.DataFrame | None:
    """当前全市场 A 股列表（快照日显式 is_st=False 用）。失败返回 None。"""
    try:
        import akshare as ak

        df = ak.stock_zh_a_spot_em()
        if df is None or df.empty:
            return None
        return df
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[ST PIT] stock_zh_a_spot_em 拉取失败（跳过快照日显式非 ST 标记）: {e}")
        return None


def _fetch_sz_change_names() -> pd.DataFrame | None:
    """深交所简称变更历史。失败返回 None。"""
    try:
        import akshare as ak

        df = ak.stock_info_sz_change_name("简称变更")
        if df is None or df.empty:
            return None
        return df
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[ST PIT] stock_info_sz_change_name 拉取失败（SZ 历史 ST 期无法回填）: {e}")
        return None


def _fetch_sz_delist() -> pd.DataFrame | None:
    """深交所终止上市公司（含终止上市日期）。失败返回 None。"""
    try:
        import akshare as ak

        df = ak.stock_info_sz_delist("终止上市公司")
        if df is None or df.empty:
            return None
        return df
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[ST PIT] stock_info_sz_delist 拉取失败: {e}")
        return None


def _fetch_sh_delist() -> pd.DataFrame | None:
    """上交所终止上市公司（含终止/暂停上市日期）。失败返回 None。"""
    try:
        import akshare as ak

        df = ak.stock_info_sh_delist("全部")
        if df is None or df.empty:
            return None
        return df
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[ST PIT] stock_info_sh_delist 拉取失败: {e}")
        return None


def fetch_delisted_symbols() -> set[str] | None:
    """独立生存偏差数据源：AkShare 深/沪终止上市公司列表（带市场前缀符号集）。

    P3-5（审计）：生存偏差评估与 stock_st_history PIT 表解耦——PIT 同步失败时
    退市标记缺失不应导致"生存偏差受控"误报。本函数直接消费交易所退市列表
    （stock_info_sz_delist / stock_info_sh_delist），任一源成功即返回（可部分），
    全部失败返回 None（调用方降级到 PIT 标记口径并注明）。
    """
    out: set[str] = set()
    ok = False
    for dl_df in (_fetch_sz_delist(), _fetch_sh_delist()):
        if dl_df is None:
            continue
        code_col = "证券代码" if "证券代码" in dl_df.columns else "公司代码"
        if code_col not in dl_df.columns:
            continue
        ok = True
        out |= _norm_codes([str(v) for v in dl_df[code_col].tolist()])
    return out if ok else None


# ── 表结构与写入 ──


def ensure_st_history_table(engine: Any) -> None:
    """确保 stock_st_history 逐日状态表存在（幂等）。"""
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {_ST_TABLE} (
                symbol VARCHAR(16) NOT NULL,
                trade_date DATE NOT NULL,
                is_st BOOLEAN NOT NULL DEFAULT FALSE,
                is_delisting BOOLEAN NOT NULL DEFAULT FALSE,
                PRIMARY KEY (symbol, trade_date)
            )
        """))
        conn.execute(text(
            f"CREATE INDEX IF NOT EXISTS idx_{_ST_TABLE}_symbol ON {_ST_TABLE} (symbol)"
        ))


def _upsert_rows(engine: Any, rows: list[dict[str, Any]]) -> None:
    """批量幂等写入（ON CONFLICT UPDATE）。"""
    if not rows:
        return
    sql = text(f"""
        INSERT INTO {_ST_TABLE} (symbol, trade_date, is_st, is_delisting)
        VALUES (:symbol, :trade_date, :is_st, :is_delisting)
        ON CONFLICT (symbol, trade_date) DO UPDATE SET
            is_st = EXCLUDED.is_st,
            is_delisting = EXCLUDED.is_delisting
    """)
    with engine.begin() as conn:
        conn.execute(sql, rows)


def _kline_days(engine: Any, symbol: str, start: date, end: date) -> list[date]:
    """标的在 [start, end] 内的实际交易日（与回测 K 线日键对齐）。"""
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT trade_date FROM stock_daily_kline "
                "WHERE symbol = :s AND trade_date >= :a AND trade_date <= :b "
                "ORDER BY trade_date"
            ), {"s": symbol, "a": start, "b": end}).fetchall()
        return [r[0] for r in rows]
    except Exception as e:  # noqa: BLE001
        logger.warning(f"[ST PIT] 读取 {symbol} 交易日失败: {e}")
        return []


def _has_row_for_date(engine: Any, d: date) -> bool:
    try:
        with engine.connect() as conn:
            return bool(conn.execute(text(
                f"SELECT 1 FROM {_ST_TABLE} WHERE trade_date = :d LIMIT 1"
            ), {"d": d}).first())
    except Exception:  # noqa: BLE001
        return False


def _pool_covered(engine: Any, symbols: list[str], start: date, end: date) -> bool:
    """池内 ≥90% 标的的 PIT 行数覆盖其窗口内 K 线交易日 ≥80% → 视为已回填。

    P1-1 修复：旧实现只统计"窗口内出现过（count DISTINCT symbol）"——
    每日快照只写今天一行即可满足覆盖检查，历史 ST/退市期永远缺失。
    现按"PIT 行数 / K 线交易日数"逐股计算真实覆盖率：
        - 窗口内无 K 线的标的（池内无数据/窗口外上市）→ 视为覆盖（无需求）
        - PIT 行数 < 80% K 线交易日 → 未覆盖（历史缺失，需重跑回填源）
    """
    if not symbols:
        return True
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT k.symbol,
                       COUNT(DISTINCT k.trade_date) AS k_days,
                       COUNT(DISTINCT s.trade_date) AS st_days
                FROM stock_daily_kline k
                LEFT JOIN {_ST_TABLE} s
                  ON s.symbol = k.symbol AND s.trade_date = k.trade_date
                WHERE k.symbol = ANY(:syms)
                  AND k.trade_date >= :start AND k.trade_date <= :end
                GROUP BY k.symbol
            """), {"syms": symbols, "start": start, "end": end}).fetchall()
    except Exception:  # noqa: BLE001
        return False
    if not rows:
        return False
    covered = 0
    total = 0
    for _sym, k_days, st_days in rows:
        total += 1
        if k_days == 0 or (st_days or 0) >= int(k_days * 0.8):
            covered += 1
    return covered >= max(1, int(total * 0.9))


# ── PIT 重建逻辑 ──


def _sz_st_periods(
    names_df: pd.DataFrame, pool: set[str]
) -> dict[str, list[tuple[date, date | None, bool, bool]]]:
    """深交所简称变更历史 → 每股 ST/退市整理期时间线。

    Returns:
        {symbol: [(period_start, period_end|None, is_st, is_delisting)]}
        period_end 为下一变更日（不含）；末段为 None（至今）。
    """
    df = names_df.copy()
    if "证券代码" not in df.columns or "证券简称" not in df.columns:
        logger.warning("[ST PIT] 简称变更数据缺列，跳过 SZ 历史回填")
        return {}
    df["证券代码"] = df["证券代码"].astype(str)
    if "变更日期" in df.columns:
        df["变更日期"] = pd.to_datetime(df["变更日期"], errors="coerce").dt.date
        df = df[df["变更日期"].notna()]
    else:
        logger.warning("[ST PIT] 简称变更数据缺变更日期，跳过 SZ 历史回填")
        return {}

    periods: dict[str, list[tuple[date, date | None, bool, bool]]] = {}
    for code, grp in df.groupby("证券代码", sort=False):
        sym = _norm_codes([code])
        if not sym or next(iter(sym)) not in pool:
            continue
        symbol = next(iter(sym))
        grp = grp.sort_values("变更日期")
        cur_start: date | None = None
        cur_flags: tuple[bool, bool] | None = None
        for _, row in grp.iterrows():
            flags = _name_flags(str(row.get("证券简称", "")))
            d = row["变更日期"]
            if cur_start is None:
                cur_start, cur_flags = d, flags
                continue
            if flags == cur_flags:
                continue
            if any(cur_flags):
                periods.setdefault(symbol, []).append((cur_start, d, cur_flags[0], cur_flags[1]))
            cur_start, cur_flags = d, flags
        if cur_start is not None and any(cur_flags):
            periods.setdefault(symbol, []).append((cur_start, None, cur_flags[0], cur_flags[1]))
    return periods


def _expand_period_rows(
    engine: Any, symbol: str, start: date, end: date,
    periods: list[tuple[date, date | None, bool, bool]],
) -> list[dict[str, Any]]:
    """ST/退市整理期 → 实际交易日行（仅展开区间与 K 线重叠的部分）。"""
    rows: list[dict[str, Any]] = []
    for p_start, p_end, is_st, is_del in periods:
        eff_start = max(p_start, start)
        eff_end = p_end if p_end is not None else end
        if eff_start > eff_end:
            continue
        for d in _kline_days(engine, symbol, eff_start, eff_end):
            rows.append({
                "symbol": symbol, "trade_date": d,
                "is_st": bool(is_st), "is_delisting": bool(is_del),
            })
    return rows


# ── 对外 API ──


def sync_st_pit(
    engine: Any,
    symbols: list[str],
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    force: bool = False,
) -> dict[str, int]:
    """回填并增量维护 ST/退市 PIT 逐日状态表（幂等，网络失败优雅降级）。

    Args:
        engine: SQLAlchemy engine。
        symbols: 回测股票池（带前缀或不带均可）。
        start_date: 回填起点（建议 = K 线预热缓冲起点）。
        end_date: 回填终点（建议 = K 线最大日期）。
        force: True 时无视覆盖检查强制重新同步。

    Returns:
        dict: {"archive_today", "sz_st_rows", "delist_rows"} 统计。
    """
    stats: dict[str, int] = {"archive_today": 0, "sz_st_rows": 0, "delist_rows": 0}
    pool = _norm_pool(symbols)
    if not pool:
        return stats
    ensure_st_history_table(engine)
    today = date.today()
    start = pd.to_datetime(start_date).date() if start_date else today - timedelta(days=365 * 5)
    end = pd.to_datetime(end_date).date() if end_date else today

    if not force and _pool_covered(engine, sorted(pool), start, end):
        logger.info(f"[ST PIT] 池内 {len(pool)} 只标的 PIT 数据已覆盖查询窗口 "
                    f"[{start}, {end}]，跳过同步（force=True 可强制）")
        return stats

    # ── 1. 每日增量归档（快照日 = 今天） ──
    if not _has_row_for_date(engine, today):
        st_list = _fetch_st_list()
        st_syms: set[str] = set()
        if st_list is not None:
            st_syms = _norm_codes(st_list.get("代码", st_list.get("证券代码", pd.Series(dtype=str))).astype(str)) & pool
            rows = [{"symbol": s, "trade_date": today, "is_st": True, "is_delisting": False} for s in st_syms]
            _upsert_rows(engine, rows)
            stats["archive_today"] += len(rows)
        spot_list = _fetch_spot_list()
        if spot_list is not None:
            codes = spot_list.get("代码", spot_list.get("证券代码", pd.Series(dtype=str)))
            spot_syms = (_norm_codes(codes.astype(str)) & pool) - st_syms
            rows = [{"symbol": s, "trade_date": today, "is_st": False, "is_delisting": False} for s in spot_syms]
            _upsert_rows(engine, rows)
            stats["archive_today"] += len(rows)

    # ── 2. SZ 简称变更历史 → 历史 ST/退市整理期 PIT ──
    names_df = _fetch_sz_change_names()
    if names_df is not None:
        for symbol, periods in _sz_st_periods(names_df, pool).items():
            rows = _expand_period_rows(engine, symbol, start, end, periods)
            _upsert_rows(engine, rows)
            stats["sz_st_rows"] += len(rows)
        logger.info(f"[ST PIT] SZ 简称变更回填完成: {stats['sz_st_rows']} 行 ST/退市整理期行")

    # ── 3. 终止上市日期 → 退市整理期 PIT（P1-4 修复） ──
    # 退市整理期 = 摘牌日（终止上市日期）前的 N 个交易日（含摘牌日当天，
    # 摘牌日若为交易日则当天仍在交易）：
    #   2020-12-31（退市新规）及以后摘牌 → 15 个交易日
    #   此前摘牌 → 30 个交易日
    # 旧实现"从终止上市日期起向后全部标记 is_delisting"方向错误：
    # 整理期发生在摘牌日之前；摘牌后的 K 线延伸日不是退市整理期。
    delist_map: dict[str, date] = {}
    delist_st_name_syms: set[str] = set()   # 退市时简称仍含 ST 的标的（P1-1 降级可见性）
    for dl_df in (_fetch_sz_delist(), _fetch_sh_delist()):
        if dl_df is None:
            continue
        code_col = "证券代码" if "证券代码" in dl_df.columns else "公司代码"
        date_col = "终止上市日期" if "终止上市日期" in dl_df.columns else "暂停上市日期"
        name_col = "证券简称" if "证券简称" in dl_df.columns else (
            "公司简称" if "公司简称" in dl_df.columns else None)
        if code_col not in dl_df.columns or date_col not in dl_df.columns:
            continue
        for _, row in dl_df.iterrows():
            syms = _norm_codes([str(row[code_col])])
            if not syms:
                continue
            sym = next(iter(syms))
            if sym not in pool:
                continue
            if name_col is not None and pd.notna(row[name_col]):
                name = str(row[name_col]).upper()
                if "ST" in name or "退" in name:
                    delist_st_name_syms.add(sym)
            d = pd.to_datetime(row[date_col], errors="coerce").date() if pd.notna(row[date_col]) else None
            if d is not None and (sym not in delist_map or d < delist_map[sym]):
                delist_map[sym] = d
    for sym, ddate in delist_map.items():
        kdays = _kline_days(engine, sym, start, ddate)
        if not kdays:
            continue
        n = DELISTING_PERIOD_DAYS_OLD if ddate < DELISTING_REFORM_DATE else DELISTING_PERIOD_DAYS_NEW
        period_start = kdays[-min(n, len(kdays))]
        rows = [{
            "symbol": sym, "trade_date": d,
            "is_st": False, "is_delisting": True,
        } for d in kdays if d >= period_start]
        _upsert_rows(engine, rows)
        stats["delist_rows"] += len(rows)
    if delist_map:
        logger.info(
            f"[ST PIT] 退市整理期 PIT 回填完成: {len(delist_map)} 只退市股，"
            f"{stats['delist_rows']} 行（摘牌日前 {DELISTING_PERIOD_DAYS_NEW}/{DELISTING_PERIOD_DAYS_OLD} "
            f"交易日，退市新规 2020-12-31 分界）"
        )
    if delist_st_name_syms:
        logger.warning(
            f"[ST PIT] P1-1 降级提示: {len(delist_st_name_syms)} 只退市股退市时简称仍含 ST/退 "
            f"（{sorted(delist_st_name_syms)[:5]}...），其退市整理期之前的 ST 历史期"
            f"无免费带日期数据源可回填，仅退市整理期 PIT 已写入"
        )

    if stats["archive_today"] == 0 and stats["sz_st_rows"] == 0 and stats["delist_rows"] == 0:
        logger.warning(
            "[ST PIT] 本次同步未写入任何行（网络不可用？）。历史 ST 期缺失将持续影响回测精度，"
            "请在有网环境重跑或外部预灌 stock_st_history 全历史 PIT 数据"
        )
    return stats


def load_st_pit(
    engine: Any, symbols: list[str], start_date: str, end_date: str
) -> dict[str, dict[str, tuple[bool, bool]]]:
    """加载 PIT ST/退市状态（参数化 ANY(:syms)，杜绝 SQL 字符串插值注入）。

    Returns:
        dict: {symbol: {trade_date: (is_st, is_delisting)}}
    """
    if not symbols:
        return {}
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(f"""
                SELECT symbol, trade_date, is_st, is_delisting
                FROM {_ST_TABLE}
                WHERE symbol = ANY(:syms)
                  AND trade_date >= :start_date
                  AND trade_date <= :end_date
            """), {
                "syms": symbols,
                "start_date": start_date,
                "end_date": end_date,
            }).fetchall()

        st_history: dict[str, dict[str, tuple[bool, bool]]] = {}
        for symbol, trade_date, is_st, is_delisting in rows:
            st_history.setdefault(symbol, {})[str(trade_date)[:10]] = (
                bool(is_st), bool(is_delisting),
            )
        logger.info(f"加载 ST 历史状态(PIT): {len(st_history)} 只股票，{len(rows)} 条记录")
        return st_history
    except Exception as e:  # noqa: BLE001
        logger.warning(f"加载 ST 历史失败，将使用静态剔除: {e}")
        return {}