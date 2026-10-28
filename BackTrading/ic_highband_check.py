# -*- coding: utf-8 -*-
r"""A1b: 高分区间（可选买入样本）IC 补充验证 — 避免全市场 0 分股票稀释横截面 IC。

与 A1 相同数据通路（信号缓存 + DB 后复权收益），但只在
  1) 综合评分 > 0    （有信号）
  2) 综合评分 >= 45  （买入阈值附近，当前 buy_threshold=45）
两个子样本上重算综合/进场评分与 10 个单因子的 IC/ICIR/单调性。

用途: 验证 "高分区间的评分是否真的反向/无 Alpha"，避免 0 分占比 ~50%
      对 Spearman IC 的过度稀释导致误判。
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from BackTrading.prepare import _REV_SIGNAL_COL_MAP  # noqa: E402

from DataManager.DbEngine import get_engine  # noqa: E402
from UtilsManager.ConfigParser import Config  # noqa: E402


def _load(cache_dir: Path) -> pd.DataFrame:
    files = sorted(p for p in cache_dir.glob("*/*.parquet"))
    if not files:
        files = sorted(cache_dir.glob("*.parquet"))
    parts: list[pd.DataFrame] = []
    for f in files:
        try:
            part = pd.read_parquet(f)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {f.name}: {e}")
            continue
        if "exit_strategy" in part.columns:
            part = part.drop(columns=["exit_strategy"])
        _rn = {eng: chn for eng, chn in _REV_SIGNAL_COL_MAP.items()
               if eng in part.columns and eng != chn}
        if _rn:
            part = part.rename(columns=_rn)
        parts.append(part)
        if len(parts) >= 800:
            df = pd.concat(parts, ignore_index=True)
            del parts[:]
            parts.append(df)
            gc.collect()
    if not parts:
        raise SystemExit("no data")
    return pd.concat(parts, ignore_index=True)


def _fut(df: pd.DataFrame, fwd: int) -> pd.DataFrame:
    syms = df["symbol"].unique().tolist()
    SLICE = 500
    from sqlalchemy import text
    out = None
    for i in range(0, len(syms), SLICE):
        batch = syms[i:i + SLICE]
        ph = ",".join(f"'{s}'" for s in batch)
        q = text(f"SELECT symbol, trade_date, close_normal FROM stock_daily_kline "
                 f"WHERE symbol IN ({ph}) ORDER BY symbol, trade_date")
        try:
            part = pd.read_sql(q, get_engine(Config()))
        except Exception:
            q = text(f"SELECT symbol, trade_date, close AS close_normal FROM stock_daily_kline "
                     f"WHERE symbol IN ({ph}) ORDER BY symbol, trade_date")
            part = pd.read_sql(q, get_engine(Config()))
        out = part if out is None else pd.concat([out, part], ignore_index=True)
        del part
        gc.collect()
    out["trade_date"] = out["trade_date"].astype(str)
    out = out.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    grp = out.groupby("symbol")["close_normal"]
    out[f"fwd_ret_{fwd}"] = (grp.shift(-fwd) / out["close_normal"]) - 1.0
    return df.merge(out[["symbol", "trade_date", f"fwd_ret_{fwd}"]],
                    on=["symbol", "trade_date"], how="left")


def _ic_icir(frame: pd.DataFrame, factor: str, fwd_col: str):
    ics = []
    for _dt, g in frame.groupby("trade_date"):
        g = g.dropna(subset=[factor, fwd_col])
        if len(g) < 5:
            continue
        ic = g[factor].corr(g[fwd_col], method="spearman")
        if np.isfinite(ic):
            ics.append(float(ic))
    if not ics:
        return np.nan, np.nan, 0, np.nan
    a = np.asarray(ics)
    m = float(a.mean())
    s = float(a.std(ddof=1)) if len(a) > 1 else np.nan
    icir = m / s * np.sqrt(252) if np.isfinite(s) and s > 0 else np.nan
    return m, icir, len(a), float((a > 0).mean())


def _mono(frame: pd.DataFrame, factor: str, fwd_col: str, q: int = 5):
    f = frame.dropna(subset=[factor, fwd_col])
    if len(f) < 20:
        return None
    try:
        f["_q"] = pd.qcut(f[factor], q, labels=False, duplicates="drop")
    except Exception:
        return None
    means = f.groupby("_q")[fwd_col].mean()
    if means.count() < 2:
        return None
    ms = means.dropna()
    return float(np.corrcoef(np.arange(len(ms)), ms)[0, 1]) if len(ms) > 2 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signal-date", default="2026-10-28")
    ap.add_argument("--forward", type=int, default=10)
    ap.add_argument("--out", default=r"E:\Downloads\CoreNews_Reports\Logs\ic_highband.csv")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    if args.rebuild:
        from BackTrading.prepare import prepare_backtest_data
        prepare_backtest_data(Config())

    from BackTrading.prepare import CACHE_DIR
    cands = sorted(
        (p for p in CACHE_DIR.iterdir()
         if p.is_dir() and p.name.startswith(f"signal_cache_{args.signal_date}")),
        key=lambda p: p.stat().st_mtime_ns, reverse=True,
    )
    if not cands:
        raise SystemExit(f"no cache for {args.signal_date}")
    d = cands[0]
    print(f"[A1b] cache={d.name}")
    frame = _load(d)
    print(f"[A1b] rows={len(frame)}, syms={frame['symbol'].nunique()}, "
          f"dates={frame['trade_date'].min()}~{frame['trade_date'].max()}")

    fwd = args.forward
    frame = _fut(frame, fwd)
    fc = f"fwd_ret_{fwd}"
    factors = ["综合评分", "进场评分", "MACD趋势数值分", "金叉信号分",
               "柱状动能分", "DIF斜评分", "背离信号分", "量价配合分",
               "K线形态分", "波动率压缩分", "价格位置分", "动能加速分"]
    rows = []
    for tag, mask in [("ALL(score>0)", frame["综合评分"] > 0),
                      ("BUY-BAND(score>=45)", frame["综合评分"] >= 45)]:
        sub = frame[mask].copy()
        print(f"\n===== 子样本: {tag}  rows={len(sub)} =====")
        for fac in factors:
            if fac not in sub.columns:
                continue
            m, icir, nd, pos = _ic_icir(sub, fac, fc)
            mono = _mono(sub, fac, fc)
            rows.append({"band": tag, "factor": fac, "ic": m, "icir": icir,
                         "days": nd, "pos_ratio": pos, "mono": mono})
            print(f"  {fac:<10} IC={m:+.4f} ICIR={icir:+.2f} pos={pos:.0%} "
                  f"mono={mono if mono is None else round(mono,2)}")
    out = pd.DataFrame(rows)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\n[A1b] saved: {args.out}")


if __name__ == "__main__":
    main()
