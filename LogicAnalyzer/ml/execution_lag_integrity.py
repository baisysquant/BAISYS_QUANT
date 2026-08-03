"""
信号执行滞后校验(SignalExecutionLag)

1.7 执行滞后时序合规（Signal Execution Lag Integrity）

业务定义：确保回测收益计算符合物理世界的因果律。
  - 策略的「仓位信号」在 T 日收盘后才能确定（使用了 T 日全部收盘数据）；
  - 因此 T 日信号仓位的收益率只能自 T+1 交易日起计，即
      组合收益序列 ≡ 仓位信号(t-1) × 资产收益率(t)
    绝对禁止：当天仓位信号 × 当天资产收益率。
  - 等价判定（成交价语义）：T 日任何一笔成交的成交价只能取「信号日收盘价」
    （收盘成交模型）或「信号次日开盘价」（次日模型）；若成交价取信号日
    开盘/冲高/盘中价，等于白拿 T 日 open→close 收益 —— 引擎会把「当天信号」
    乘上「当天收益」，破坏因果律。

自检内容：
  1. 成交价时点合规 check_execution_price —— 每笔 buy/sell 的成交价必须等于
     信号日收盘价或下一交易日开盘价；当日盘中价（open/high/low）一律禁止。
  2. 当日收益入账归零 check_same_day_contribution —— T 日建仓的仓位，其
     「当日收益乘数」对该仓位的入账量必须为 0
     （收盘模型：close(t)/成交价 - 1 ≈ 0；次日模型：open(t+1)/成交价 - 1 ≈ 0）。
  3. 收益乘数对齐 check_return_multiplier_alignment —— 给定合规公式
     pnl(t) = position(t-1) · asset_returns(t)，与实测/实现对比：若实现写成
     position(t) · asset_returns(t)（同日相乘，超前 1 日）即违规。

模块功能：
  - check_execution_price             —— 成交价时点合规
  - check_same_day_contribution       —— 信号日收益贡献归零
  - check_return_multiplier_alignment —— 收益乘数对齐（序列级通用）
  - run_execution_lag_check           —— 一站式入口
  - SplitReport                       —— PASS/FAIL 自检报告（复用切分报告）

用法:
    from LogicAnalyzer.ml.execution_lag_integrity import run_execution_lag_check
    result = run_execution_lag_check(trade_log, bars, exec_mode="close")
    if not result["passed"]:
        logger.warning(f"[执行滞后合规] 违规: {';'.join(r.details[0] for r in result['reports'] if not r.passed)}")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from LogicAnalyzer.ml.split_integrity import SplitReport

_EXEC_MODES = ("close", "next_open")
# 价格比较容差（绝对/相对），用于吸收浮点舍入
_RTOL = 1e-6
_ATOL = 1e-3


def _norm_date(x: Any) -> str | None:
    """把 str / pd.Timestamp / 时间戳统一为 %Y-%m-%d；无法解析返回 None。"""
    if x is None:
        return None
    if isinstance(x, str) and not x.strip():
        return None
    ts = pd.to_datetime(x, errors="coerce")
    if pd.isna(ts):
        return None
    return str(ts.strftime("%Y-%m-%d"))


def _price_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """抽取价格列并统一 trade_date 为字符串；无 close_adj 时回落 close。

    注意：trade_date 批量转 Timestamp 后格式化，禁止逐元素 pd.to_datetime
    （20 万行量级下逐元素转换耗时数十秒，会拖垮 WFO 全量回测）。
    """
    b = bars.copy()
    if "close_adj" not in b.columns and "close" in b.columns:
        b["close_adj"] = b["close"]
    keep = ["trade_date", "symbol", "close_adj"]
    keep.extend(c for c in ("open", "high", "low", "close") if c in b.columns)
    b = b[keep]
    b["trade_date"] = pd.to_datetime(b["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return b.dropna(subset=["trade_date"])


def _bar_index(frame: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    """{(date): {symbol: {价格列: 值}}} — 仅包含存在数据的价格列。"""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for dt, grp in frame.groupby("trade_date", sort=False):
        d: dict[str, dict[str, float]] = {}
        for r in grp.to_dict("records"):
            sym = str(r.get("symbol", ""))
            if not sym:
                continue
            px: dict[str, float] = {}
            for c in ("close_adj", "open", "high", "low", "close"):
                v = r.get(c)
                if v is None or pd.isna(v):
                    continue
                try:
                    px[c] = float(v)
                except (TypeError, ValueError):
                    continue
            d[sym] = px
        out[str(dt)] = d
    return out


def _same_price(a: float, b: float | None) -> bool:
    if b is None or not np.isfinite(b) or not np.isfinite(a):
        return False
    return abs(a - b) <= _ATOL + _RTOL * abs(b)


def _trade_rows(
    trade_log: Sequence[dict[str, Any]],
) -> list[tuple[str, str, str, float]]:
    """规范化成交记录 → (date, symbol, action, price)；无效记录剔除。"""
    rows: list[tuple[str, str, str, float]] = []
    for t in trade_log:
        dt = _norm_date(t.get("time"))
        sym = str(t.get("symbol", ""))
        action = str(t.get("action", ""))
        px = t.get("price")
        if dt is None or not sym or not action or px is None or pd.isna(px):
            continue
        try:
            rows.append((dt, sym, action, float(px)))
        except (TypeError, ValueError):
            continue
    return rows


def _next_day(dates_sorted: Sequence[str], dt: str) -> str | None:
    """返回 dt 之后的第一个交易日（用于 next_open 模式的次日开盘锚点）。"""
    pos = np.searchsorted(np.asarray(dates_sorted, dtype=str), dt)
    if pos < len(dates_sorted):
        return str(dates_sorted[pos]) if str(dates_sorted[pos]) != dt else (
            str(dates_sorted[pos + 1]) if pos + 1 < len(dates_sorted) else None
        )
    return None


# ── 1. 成交价时点合规 ───────────────────────────────────────

def check_execution_price(
    trade_log: Sequence[dict[str, Any]],
    bars: pd.DataFrame,
    exec_mode: str = "close",
    _log: bool = True,
) -> SplitReport:
    """成交价时点合规（1.7 执行滞后）。

    规则：
      - exec_mode="close"（默认）：成交价必须 ≈ 信号日收盘价 close_adj。
      - exec_mode="next_open"：成交价必须 ≈ 信号下一交易日开盘价 open。
      - 成交价 == 信号日 open/high/low（当日盘中价）= 当天信号吃当天收益，违规。

    Returns:
        SplitReport: n_checked = 可比对的成交笔数，违规数 = 不满足的笔数。
    """
    check = SplitReport(check_name=f"成交价时点合规({exec_mode})", passed=True)
    if exec_mode not in _EXEC_MODES:
        check.passed = False
        check.details.append(f"未知成交模式 {exec_mode}（仅支持 {_EXEC_MODES}）")
        if _log:
            check.log()
        return check

    frame = _price_frame(bars)
    if frame.empty:
        check.details.append("无可比对的 bar 数据")
        check.passed = False
        if _log:
            check.log()
        return check
    idx = _bar_index(frame)

    allowed_col = "close_adj" if exec_mode == "close" else "open"
    intraday_cols = ("open", "high", "low") if exec_mode == "close" else ()
    dates_sorted = sorted(frame["trade_date"].unique().tolist())

    checked = 0
    violations: list[str] = []
    for dt, sym, action, trade_px in _trade_rows(trade_log):
        bar_date = dt
        if exec_mode == "next_open":
            nxt = _next_day(dates_sorted, dt)
            if nxt is None:
                continue  # 信号日为最后一个交易日 → 无可比对的次日 bar
            bar_date = nxt
        sym_bar = idx.get(bar_date, {}).get(sym)
        if sym_bar is None:
            continue  # 当日停牌/数据缺失 → 不可比，不计违规
        allowed_val = sym_bar.get(allowed_col)
        if allowed_val is None:
            continue
        checked += 1
        if _same_price(trade_px, allowed_val):
            continue
        if any(_same_price(trade_px, sym_bar.get(c)) for c in intraday_cols):
            note = "（当日盘中价成交 = 当天信号吃当天收益）"
        else:
            note = ""
        violations.append(f"{action} {sym} {dt}@{trade_px} != {allowed_col}={allowed_val} {note}")

    check.n_checked = checked
    check.n_violations = len(violations)
    check.passed = check.n_violations == 0
    if violations:
        check.details.append(
            f"{check.n_violations}/{check.n_checked} 笔成交价 ≠ {allowed_col}"
            "（含当日盘中价＝当天信号×当天收益）"
        )
        check.details.append("示例: " + "; ".join(violations[:5]))
    if _log:
        check.log()
    return check


# ── 1b. 热路径轻量校验（O(成交笔数)，引擎运行期用） ─────────

def check_price_vs_close_adj(
    trade_log: Sequence[dict[str, Any]],
    _log: bool = True,
) -> SplitReport:
    """基于成交记录自带 close_adj 字段的轻量校验（1.7）。

    引擎在记录每笔成交时已把「同一信号日的复权收盘价」写入 trade["close_adj"]。
    热路径（WFO 中每轮回测）用本函数 O(成交笔数) 完成乘数对齐校验，
    无需对全量面板做 O(N) 扫描。任何 price != close_adj 的成交，意味着
    成交价不是信号日收盘价（可能用了开盘/盘中价）→ 当天信号吃当天收益。

    Args:
        trade_log: 引擎成交记录（必须含 close_adj 字段；缺失该字段的旧日志跳过）。
        _log: 是否输出日志。

    Returns:
        SplitReport。
    """
    check = SplitReport(check_name="成交价=信号日收盘价(热路径)", passed=True)
    checked = 0
    violations: list[str] = []
    for t in trade_log:
        px = t.get("price")
        close_adj = t.get("close_adj")
        if px is None or close_adj is None or pd.isna(px) or pd.isna(close_adj):
            continue
        checked += 1
        if not _same_price(float(px), float(close_adj)):
            violations.append(
                f"{t.get('action', 'trade')} {t.get('symbol')} {t.get('time')}: "
                f"price={px} != close_adj={close_adj}"
            )
    check.n_checked = checked
    check.n_violations = len(violations)
    check.passed = check.n_violations == 0
    if violations:
        check.details.append(
            f"{check.n_violations}/{check.n_checked} 笔成交价 ≠ 信号日收盘价"
            "（疑似开盘/盘中价成交，当天信号吃当天收益）"
        )
        check.details.append("示例: " + "; ".join(violations[:5]))
    if _log:
        check.log()
    return check


# ── 2. 信号日收益贡献归零 ───────────────────────────────────

def check_same_day_contribution(
    trade_log: Sequence[dict[str, Any]],
    bars: pd.DataFrame,
    exec_mode: str = "close",
    _log: bool = True,
) -> SplitReport:
    """信号日收益贡献归零（1.7 执行滞后）。

    对每笔买入：成交当日该资产「当日收益乘数」的入账量必须为 0 ——
      收盘模型：贡献 = close(t)/成交价 - 1（成交价=当日收盘 ⇒ 0）；
      次日模型：贡献 = open(t+1)/成交价 - 1（成交价=次日开盘 ⇒ 0）。
    任何非零贡献说明该仓当天已被「当天收益」乘入（收益乘数错配）。

    Returns:
        SplitReport。
    """
    check = SplitReport(check_name=f"信号日收益贡献归零({exec_mode})", passed=True)
    if exec_mode not in _EXEC_MODES:
        check.passed = False
        check.details.append(f"未知成交模式 {exec_mode}")
        if _log:
            check.log()
        return check

    frame = _price_frame(bars)
    if frame.empty:
        check.details.append("未提供价格数据")
        check.passed = False
        if _log:
            check.log()
        return check

    idx = _bar_index(frame)
    dates_sorted = sorted(frame["trade_date"].unique().tolist())

    checked = 0
    violations: list[str] = []
    for dt, sym, action, trade_px in _trade_rows(trade_log):
        if action != "buy":
            continue
        sym_bar = idx.get(dt, {}).get(sym)
        if sym_bar is None or "close_adj" not in sym_bar:
            continue
        if exec_mode == "close":
            anchor = sym_bar["close_adj"]
        else:
            nxt = _next_day(dates_sorted, dt)
            if nxt is None:
                continue
            nxt_bar = idx.get(nxt, {}).get(sym)
            if nxt_bar is None or "open" not in nxt_bar:
                continue
            anchor = nxt_bar["open"]
        if anchor is None:
            continue
        checked += 1
        contribution = anchor / trade_px - 1.0
        if abs(contribution) > _RTOL * max(1.0, abs(anchor)):
            violations.append(
                f"buy {sym} {dt} 当日收益乘数入账 {contribution:+.6f}（必须为 0）"
            )

    check.n_checked = checked
    check.n_violations = len(violations)
    check.passed = check.n_violations == 0
    check.details.append(
        f"校验 {checked} 笔买入当日收益贡献均为 0（锚点={('当日收盘' if exec_mode == 'close' else '次日开盘')}）"
    )
    if violations:
        check.details.append("示例: " + "; ".join(violations[:5]))
    if _log:
        check.log()
    return check


# ── 3. 收益乘数对齐（序列级总校验） ─────────────────────────

def check_return_multiplier_alignment(
    position: pd.Series,
    asset_returns: pd.Series,
    observed_pnl: pd.Series,
    *,
    tol: float = 1e-9,
    _log: bool = True,
) -> SplitReport:
    """收益乘数对齐（1.7）。

    合规公式：pnl(t) = position(t-1) × asset_returns(t) —— 当日信号的仓位
    只能赚取下一个交易日的收益。若实现提供的是 position × asset_returns
    （当果相乘），预期序列将整体超前 1 日且不可由 shift(1) 重建 → 判 FAIL。

    Args:
        position: 当日收盘确立的仓位信号（index=日期）。
        asset_returns: 资产的日收益率序列。
        observed_pnl: 待检的、声称由该仓位乘出的收益贡献序列。
        tol: 相对容差。

    Returns:
        SplitReport（乘数超前 1 日时违规数>0）。
    """
    check = SplitReport(check_name="收益乘数对齐", passed=True)
    pos = pd.Series(position).astype(float)
    ret = pd.Series(asset_returns).astype(float)
    pnl = pd.Series(observed_pnl).astype(float)
    idx = pnl.dropna().index

    if len(idx) < 2:
        check.details.append("可检样本 < 2 期，跳过")
        check.n_checked = len(idx)
        if _log:
            check.log()
        return check

    pos_r = pos.reindex(idx).fillna(0.0)
    ret_r = ret.reindex(idx).fillna(0.0)
    expected = pos_r.shift(1) * ret_r
    diff = (pnl.reindex(idx) - expected).abs()
    denom = np.maximum(expected.abs(), 1e-9)
    viol_mask = (diff > tol * denom) & (expected.abs() + pnl.reindex(idx).abs() > 0)
    n_viol = int(viol_mask.sum())
    check.n_checked = len(idx)
    check.n_violations = n_viol
    check.passed = n_viol == 0
    if n_viol:
        bad_days = [str(i)[:10] for i in idx[viol_mask].tolist()[:5]]
        check.details.append(
            f"{n_viol}/{len(idx)} 期收益 ≠ position(t-1)×ret(t)"
            "（疑似当天信号乘当天收益，超前 1 日）"
        )
        check.details.append("示例违规日: " + ", ".join(bad_days))
    if _log:
        check.log()
    return check


# ── 一站式入口 ──────────────────────────────────────────────

def run_execution_lag_check(
    trade_log: Sequence[dict[str, Any]],
    bars: pd.DataFrame,
    *,
    exec_mode: str = "close",
    position_series: pd.Series | None = None,
    asset_returns: pd.Series | None = None,
    observed_pnl: pd.Series | None = None,
    _log: bool = True,
) -> dict[str, Any]:
    """信号执行滞后一站式自检（1.7）。

    Args:
        trade_log: 回测成交记录（time / symbol / action / price）。
        bars: 逐日逐标的 K 线。
        exec_mode: "close" 收盘成交 或 "next_open" 次日开盘成交。
        position_series / asset_returns / observed_pnl: 可选完整乘数序列校验。

    Returns:
        {"passed", "reports", "summary"}
    """
    # 性能：按成交涉及日期裁剪 bar 数据，避免对全量面板做 O(N) 拷贝/索引
    #（引擎在 WFO 内会被调用数百次，全量扫描会把单次回测拖慢一个量级）。
    if not trade_log and position_series is None:
        reports = [
            SplitReport(check_name=f"成交价时点合规({exec_mode})", passed=True,
                        n_checked=0),
            SplitReport(check_name=f"信号日收益贡献归零({exec_mode})", passed=True,
                        n_checked=0),
        ]
        return {"passed": True, "reports": reports,
                "summary": pd.DataFrame([r.to_dict() for r in reports])}

    if trade_log:
        dates_needed = {_norm_date(t.get("time")) for t in trade_log}
        dates_needed.discard(None)
        syms_needed = {str(t.get("symbol")) for t in trade_log}
        if dates_needed:
            td = bars["trade_date"]
            sym_arr = bars["symbol"].astype(str).values
            if exec_mode == "next_open":
                # 次日模型还需成交日的下一交易日 bar
                if pd.api.types.is_datetime64_any_dtype(td):
                    all_dates = sorted(set(td.dt.strftime("%Y-%m-%d").tolist()))
                else:
                    all_dates = sorted(set(td.astype(str).tolist()))
                arr_d = np.array(all_dates, dtype=str)
                for d in list(dates_needed):
                    pos = int(np.searchsorted(arr_d, d))
                    if pos < len(arr_d) and arr_d[pos] != d:
                        dates_needed.add(arr_d[pos])
                    elif pos < len(arr_d) and arr_d[pos] == d and pos + 1 < len(arr_d):
                        dates_needed.add(arr_d[pos + 1])
            if pd.api.types.is_datetime64_any_dtype(td):
                dt_arr = td.to_numpy().astype("datetime64[D]")
                date_key = np.array(sorted(dates_needed), dtype="datetime64[D]")
            else:
                dt_arr = td.astype(str).values
                date_key = np.array(sorted(dates_needed), dtype=str)
            # 裁剪到 trade 涉及的 (日期, 标的) 行——O(成交笔数) 而非 O(全面板)
            mask = np.isin(dt_arr, date_key) & np.isin(sym_arr, np.array(sorted(syms_needed), dtype=str))
            bars = bars[mask]

    reports = [
        check_execution_price(trade_log, bars, exec_mode=exec_mode, _log=_log),
        check_same_day_contribution(trade_log, bars, exec_mode=exec_mode, _log=_log),
    ]
    if position_series is not None and asset_returns is not None and observed_pnl is not None:
        reports.append(
            check_return_multiplier_alignment(
                position_series, asset_returns, observed_pnl, _log=_log,
            )
        )
    return {
        "passed": all(r.passed for r in reports),
        "reports": reports,
        "summary": pd.DataFrame([r.to_dict() for r in reports]),
    }
