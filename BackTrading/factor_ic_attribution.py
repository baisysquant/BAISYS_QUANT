# -*- coding: utf-8 -*-
r"""
A1: 单因子 IC 归因脚本 — 评分合成修复的第一性诊断

目标（问题①【P0·信号层】评分分档非单调、IC 方向不稳）：
  回测 [信号质量] 报告显示 Q1~Q5 分档收益非单调（Q5 高分反而最差），IC≈0.05 偏弱。
  但该报告只汇总「综合评分」对收益的相关性，无法定位是哪个子因子在拖后腿。
  本脚本对 11 个评分成分逐因子计算：
    1. Spearman IC（因子分 vs 未来 N 日收益，逐日横截面）
    2. ICIR（IC 均值 / IC 标准差 × sqrt(252)，方向稳定性）
    3. 分档单调性（Q1~Q5 平均未来收益 + Q5-Q1 价差 + 单调系数）
    4. 综合评分与各因子的相关性（看谁主导了综合评分）
  输出 CSV + 标准输出摘要，供 A2/A3 据此调整评分权重/方向。

数据通路（零重复计算）：
  因子分直接读取 prepare 阶段已生成的信号缓存
    (signal_cache_{trade_date}_{config_hash}_{param_hash}/*.parquet)，
  未来收益从 DB `stock_daily_kline`（后复权 close_normal）计算。
  若信号缓存不存在，可传 --rebuild 调 prepare_backtest_data 生成（代价高）。

用法:
  python BackTrading/factor_ic_attribution.py \
      --signal-date 2026-10-28 \
      --param-hash <可选，缺省用最新缓存> \
      --forward 3 5 10 20 \
      --min-symbols 30 \
      --out E:\Downloads\CoreNews_Reports\Logs\factor_ic_report.csv

示例（指定某次回测产生的缓存目录）:
  python BackTrading/factor_ic_attribution.py --signal-date 2026-10-28 --forward 3 5 10
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 评分成分（与 vectorized_signal.compute_signals 输出列一致）
# FIX(A1-2026-10-28): "MACD趋势分"列是字符串分类（指标强势/...），数值分为
# "MACD趋势数值分"（macd_trend_value），spearman 计算必须用数值列。
FACTOR_COLS = [
    "MACD趋势数值分", "金叉信号分", "柱状动能分", "DIF斜评分",
    "背离信号分", "量价配合分", "K线形态分", "波动率压缩分",
    "价格位置分", "动能加速分",
]
SCORE_COLS = ["综合评分", "进场评分", "退出评分"]

FWD_COL_PREFIX = "fwd_ret_"


def _latest_cache_dir(signal_date: str) -> Path | None:
    """定位最新信号缓存目录。"""
    from BackTrading.prepare import CACHE_DIR
    cands = sorted(
        (p for p in CACHE_DIR.iterdir() if p.is_dir()
         and p.name.startswith(f"signal_cache_{signal_date}")),
        key=lambda p: p.stat().st_mtime_ns, reverse=True,
    )
    return cands[0] if cands else None


def _load_signals(cache_dir: Path, min_symbols: int) -> pd.DataFrame:
    """读取信号缓存（分块 concat，丢弃 exit_strategy 大列）。

    FIX(A1-2026-10-28): 缓存 parquet 为英文列名（symbol/trade_date/score/
    macd_trend/...，见 prepare._SIGNAL_COL_MAP），此前直接按中文 FACTOR_COLS
    匹配导致 10 个单因子 IC 全部被 continue 跳过，只剩"综合评分"兜底行。
    此处读入后按映射统一重命名为中文列。
    """
    from BackTrading.prepare import _REV_SIGNAL_COL_MAP

    files = sorted(p for p in cache_dir.glob("*/*.parquet"))
    if not files:
        files = sorted(cache_dir.glob("*.parquet"))
    if not files:
        raise SystemExit(f"[A1] 缓存目录 {cache_dir} 下没有 parquet 文件")
    print(f"[A1] 读取 {len(files)} 个信号缓存文件: {cache_dir}")
    parts: list[pd.DataFrame] = []
    _CHUNK = 800
    for idx, f in enumerate(files):
        try:
            part = pd.read_parquet(f)
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠ 跳过损坏缓存 {f.name}: {e}")
            continue
        if "exit_strategy" in part.columns:
            part = part.drop(columns=["exit_strategy"])
        # 英文列名 -> 中文列名（prepare 写入缓存时的反映射）
        _rn = {eng: chn for eng, chn in _REV_SIGNAL_COL_MAP.items()
               if eng in part.columns and eng != chn}
        if _rn:
            part = part.rename(columns=_rn)
        parts.append(part)
        if len(parts) >= _CHUNK:
            df = pd.concat(parts, ignore_index=True)
            del parts[:]
            parts.append(df)
            gc.collect()
    if not parts:
        raise SystemExit("[A1] 无有效信号数据")
    df = pd.concat(parts, ignore_index=True)
    gc.collect()
    for c in ("trade_date", "symbol"):
        if c not in df.columns:
            raise SystemExit(f"[A1] 缓存缺少必需列 {c}，实际列: {list(df.columns)[:20]}")
    if "综合评分" not in df.columns and "score" in df.columns:
        df["综合评分"] = df["score"]
    print(f"[A1] 信号行数={len(df)}, 股票数={df['symbol'].nunique()}, "
          f"日期范围={df['trade_date'].min()}~{df['trade_date'].max()}")
    if df["symbol"].nunique() < min_symbols:
        print(f"  ⚠ 股票数 < {min_symbols}，IC 可能不稳定")
    return df


def _load_future_returns(signal_df: pd.DataFrame, fwd_days: list[int],
                         db_engine) -> pd.DataFrame:
    """从 DB 计算各股票未来 N 日收益（后复权 close_normal），对齐信号日期。

    返回信号表 + fwd_ret_{N} 列。每股按 (trade_date 升序) 组内 shift(-N) 前移收益。
    """
    syms = signal_df["symbol"].unique().tolist()
    print(f"[A1] 从 DB 加载 {len(syms)} 只股票后复权收盘价…")
    df = None
    SLICE = 500
    from sqlalchemy import text
    for i in range(0, len(syms), SLICE):
        batch = syms[i:i + SLICE]
        # 动态 IN 展开，避免 ANY(:syms) 对 psycopg2/psycopg3 数组参数绑定的依赖
        placeholders = ",".join(f"'{s}'" for s in batch)
        q = text(
            f"SELECT symbol, trade_date, close_normal FROM stock_daily_kline "
            f"WHERE symbol IN ({placeholders}) ORDER BY symbol, trade_date"
        )
        try:
            part = pd.read_sql(q, db_engine)
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠ 批量 {i} 读取失败: {e}，尝试 close 列")
            q = text(
                f"SELECT symbol, trade_date, close AS close_normal FROM stock_daily_kline "
                f"WHERE symbol IN ({placeholders}) ORDER BY symbol, trade_date"
            )
            part = pd.read_sql(q, db_engine)
        df = part if df is None else pd.concat([df, part], ignore_index=True)
        del part
        gc.collect()
    if df is None or df.empty:
        raise SystemExit("[A1] DB 中无 K 线数据（stock_daily_kline 空？）")
    df["trade_date"] = df["trade_date"].astype(str)
    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    for n in fwd_days:
        grp = df.groupby("symbol")["close_normal"]
        df[f"fwd_ret_{n}"] = (grp.shift(-n) / df["close_normal"]) - 1.0
    cols = ["symbol", "trade_date"] + [f"fwd_ret_{n}" for n in fwd_days]
    ret_part = df[cols]
    del df
    gc.collect()
    out = signal_df.merge(ret_part, on=["symbol", "trade_date"], how="left")
    print(f"[A1] 对齐后行数={len(out)}, 有未来收益占比="
          f"{(out[f'fwd_ret_{fwd_days[0]}'].notna()).mean():.1%}")
    return out


def _spearman_ic(frame: pd.DataFrame, factor: str, fwd_col: str) -> tuple[float, float]:
    """按日横截面 Spearman IC。返回 (IC均值, ICIR=mean/std*sqrt(252), 天数, IC>0占比)。"""
    ics: list[float] = []
    for _dt, g in frame.groupby("trade_date"):
        g = g.dropna(subset=[factor, fwd_col])
        if len(g) < 5:
            continue
        ic = g[factor].corr(g[fwd_col], method="spearman")
        if np.isfinite(ic):
            ics.append(float(ic))
    if not ics:
        return np.nan, np.nan, 0, np.nan
    arr = np.asarray(ics)
    ic_mean = float(arr.mean())
    ic_std = float(arr.std(ddof=1)) if len(arr) > 1 else np.nan
    icir = ic_mean / ic_std * np.sqrt(252) if np.isfinite(ic_std) and ic_std > 0 else np.nan
    pos_ratio = float((arr > 0).mean())
    return ic_mean, icir, len(arr), pos_ratio


def _quantile_monotonic(frame: pd.DataFrame, factor: str, fwd_col: str,
                        q: int = 5) -> dict:
    """分档单调性：按因子分 Q1~Q5 分组，组内平均未来收益、Q5-Q1、Spearman(rank, mean)。"""
    f = frame.dropna(subset=[factor, fwd_col]).copy()
    if len(f) < 20:
        return {"n": 0}
    try:
        f["_q"] = pd.qcut(f[factor], q, labels=False, duplicates="drop")
    except ValueError:
        return {"n": 0}
    means = f.groupby("_q")[fwd_col].mean()
    out = {f"Q{i+1}_mean": float(means.get(i, np.nan)) for i in range(q)}
    q1 = out["Q1_mean"]
    q5 = out.get(f"Q{q-1}_mean", np.nan)
    spread = (q5 - q1) if (np.isfinite(q1) and np.isfinite(q5)) else np.nan
    out["Q5_Q1_spread"] = spread
    on_top = means.dropna().shape[0]
    if on_top < 2:
        return {"n": int(len(f))}
    _ms = means.dropna().values
    _ranks = np.arange(len(_ms))
    monot = float(np.corrcoef(_ranks, _ms)[0, 1]) if len(_ms) > 2 and np.std(_ms) > 0 else float("nan")
    out["monotonic_corr"] = monot
    out["n"] = int(len(f))
    out["qn"] = int(on_top)
    return out


def run(signal_date: str, param_hash: str | None, fwd_days: list[int],
        min_symbols: int, out_path: Path | None, rebuild: bool = False) -> pd.DataFrame:
    from UtilsManager.ConfigParser import Config
    from DataManager.DbEngine import get_engine

    if rebuild:
        from BackTrading.prepare import prepare_backtest_data
        print("[A1] --rebuild 模式：调用 prepare_backtest_data 生成信号缓存（较慢）…")
        prepare_backtest_data(Config())
    cache_dir = _latest_cache_dir(signal_date) if not param_hash else None
    if cache_dir is None:
        # 尝试指定 param_hash 的目录
        from BackTrading.prepare import CACHE_DIR, _cache_dir_for
        cache_dir = _cache_dir_for(signal_date, param_hash=param_hash)
        if not cache_dir.exists():
            raise SystemExit(
                f"[A1] 未找到 {signal_date} 的信号缓存。\n"
                f"  1) 先跑一次回测/prepare 生成缓存；或\n"
                f"  2) 用 --rebuild 重新生成；或\n"
                f"  3) 检查 signal_cache 目录: {CACHE_DIR}"
            )
    signal_df = _load_signals(cache_dir, min_symbols)
    engine = get_engine(Config())
    frame = _load_future_returns(signal_df, fwd_days, engine)

    rows: list[dict] = []
    factors = FACTOR_COLS + SCORE_COLS
    for fwd in fwd_days:
        fwd_col = f"fwd_ret_{fwd}"
        for fac in factors:
            if fac not in frame.columns:
                continue
            ic_mean, icir, ndays, pos_ratio = _spearman_ic(frame, fac, fwd_col)
            qm = _quantile_monotonic(frame, fac, fwd_col)
            row = {
                "forward_days": fwd, "factor": fac,
                "ic_mean": ic_mean, "icir": icir,
                "ic_days": ndays, "ic_positive_ratio": pos_ratio,
                "q5_q1_spread": qm.get("Q5_Q1_spread", np.nan),
                "monotonic_corr": qm.get("monotonic_corr", np.nan),
                "n": qm.get("n", 0),
            }
            rows.append(row)
    report = pd.DataFrame(rows)

    # ── 综合评分与各因子相关性（看谁主导综合评分） ──
    if "综合评分" in frame.columns and "trade_date" in frame.columns:
        dep_rows = []
        for fac in FACTOR_COLS:
            if fac not in frame.columns:
                continue
            corrs = []
            for _dt, g in frame.groupby("trade_date"):
                g = g.dropna(subset=[fac, "综合评分"])
                if len(g) < 5:
                    continue
                c = g[fac].corr(g["综合评分"], method="spearman")
                if np.isfinite(c):
                    corrs.append(float(c))
            dep_rows.append({
                "factor": fac,
                "corr_with_score": float(np.mean(corrs)) if corrs else np.nan,
            })
        if dep_rows:
            dep_df = pd.DataFrame(dep_rows).sort_values("corr_with_score", ascending=False)
            print("\n[A1] 因子与综合评分相关性（谁主导评分）:")
            for _, r in dep_df.iterrows():
                print(f"  {r['factor']:<12} corr={r['corr_with_score']:+.3f}")

    return frame, report


def main() -> None:
    ap = argparse.ArgumentParser(description="单因子 IC 归因（评分合成修复的第一性诊断）")
    ap.add_argument("--signal-date", default=None, help="信号缓存 trade_date（默认用最新缓存）")
    ap.add_argument("--param-hash", default=None, help="指定 param_hash 缓存子目录（可选）")
    ap.add_argument("--forward", nargs="+", type=int, default=[3, 5, 10],
                    help="未来收益天数（默认 3 5 10）")
    ap.add_argument("--min-symbols", type=int, default=30,
                    help="最低股票数（低于则告警 IC 不稳），默认 30")
    ap.add_argument("--out", default=None, help="输出 CSV 路径（缺省则打印摘要）")
    ap.add_argument("--rebuild", action="store_true",
                    help="先调 prepare_backtest_data 重建信号缓存（慢）")
    args = ap.parse_args()

    frame, report = run(args.signal_date, args.param_hash, args.forward,
                        args.min_symbols, Path(args.out) if args.out else None,
                        rebuild=args.rebuild)

    print("\n" + "=" * 88)
    print("[A1] 单因子 IC 归因摘要（剔除无效行后）")
    print("=" * 88)
    for fwd in args.forward:
        sub = report[report["forward_days"] == fwd].sort_values("ic_mean", ascending=False)
        print(f"\n── 未来 {fwd} 日收益 ────────────────────────────────────────────")
        for _, r in sub.iterrows():
            flag = ""
            if np.isfinite(r["ic_mean"]) and abs(r["ic_mean"]) < 0.01:
                flag = " ← 无预测力"
            elif np.isfinite(r["ic_mean"]) and r["ic_mean"] < 0 and np.isfinite(r["monotonic_corr"]) and r["monotonic_corr"] < 0:
                flag = " ← 方向反了(建议反转)"
            print(f"  {r['factor']:<12} IC={r['ic_mean']:+.4f} ICIR={r['icir']:+.2f} "
                  f"IC+占比={r['ic_positive_ratio']:.0%} Q5-Q1={r['q5_q1_spread']:+.2%} "
                  f"单调={r['monotonic_corr']:+.2f}{flag}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\n[A1] 完整报告已保存: {args.out}")
    print("\n[A1] 使用建议:")
    print("  1) IC≈0 且单调≈0 → 该因子无 Alpha；可在权重里降权或移除")
    print("  2) IC<0 且单调<0 → 评分方向反了，反转该因子（如 10分 - 原始分）")
    print("  3) IC>0 但单调非单调 → 高分档存在尾部反转，需检查权重或阈值")
    print("  4) 按 ICIR 排序找方向最稳定的因子，作为评分权重上调对象")


if __name__ == "__main__":
    main()
