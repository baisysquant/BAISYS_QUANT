"""P2 双路径重构回归验证脚本。

构造两套合成数据，分别用 portfolio_method="score_weighted"（use_sw=True）
和 portfolio_method="equal_weight"（use_sw=False）跑回测，
保存 equity_curve 和 trade_log 作为重构前的黄金标准。

重构后重新运行此脚本，diff 验证数值一致性。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 确保项目根目录在 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from BackTrading.engine import EngineConfig, run_full_backtest
from BackTrading.domain.models import CostModel


def _make_synthetic_kline(n_days: int = 120, n_stocks: int = 30, seed: int = 42) -> pd.DataFrame:
    """生成合成 K 线数据，覆盖涨跌停/停牌/正常交易等场景。"""
    rng = np.random.default_rng(seed)
    # 交易日序列（跳过周末）
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    syms = [f"sh60{i:04d}" for i in range(1, n_stocks + 1)]
    rows = []
    for dt in dates:
        for si, sym in enumerate(syms):
            # 随机停牌（~3% 的天数）
            if rng.random() < 0.03:
                continue
            base = 10.0 + si * 0.5
            ret = rng.normal(0, 0.03)
            close = base * (1 + ret)
            open_p = close * (1 + rng.normal(0, 0.01))
            high = max(open_p, close) * (1 + abs(rng.normal(0, 0.005)))
            low = min(open_p, close) * (1 - abs(rng.normal(0, 0.005)))
            volume = int(rng.lognormal(12, 0.5))
            amount = volume * close
            score = float(rng.integers(0, 60))
            sell_score = float(rng.integers(0, 60))
            risk = rng.choice(["A", "B", "C", "D"], p=[0.3, 0.4, 0.2, 0.1])
            rows.append({
                "trade_date": dt,
                "symbol": sym,
                "open": round(open_p, 3),
                "high": round(high, 3),
                "low": round(low, 3),
                "close": round(close, 3),
                "close_adj": round(close, 3),
                "volume": volume,
                "amount": round(amount, 2),
                "AMOUNT_MA20": round(amount * 20, 2),
                "进场评分": score,
                "退出评分": sell_score,
                "风险等级": risk,
                "止损价": round(close * 0.92, 3),
                "is_trading": 1,
            })
    df = pd.DataFrame(rows)
    # 添加 ATR 列
    df["ATR"] = df.groupby("symbol")["close"].transform(lambda x: x.diff().abs().rolling(14, min_periods=1).mean())
    df["ATR"] = df["ATR"].fillna(1.0)
    return df


def _hash_equity_curve(ec: list[dict]) -> str:
    """对 equity curve 做确定性哈希（忽略浮点微小差异）。"""
    # 提取关键数值，round 到 2 位小数
    parts = []
    for row in ec:
        t = str(row.get("time", ""))
        pv = round(float(row.get("portfolio_value", 0)), 2)
        tv = round(float(row.get("turnover", 0)), 6)
        sv = round(float(row.get("susp_value_ratio", 0)), 6)
        parts.append(f"{t}|{pv}|{tv}|{sv}")
    return hashlib.md5("\n".join(parts).encode()).hexdigest()


def _hash_trade_log(tl: list[dict]) -> str:
    """对 trade log 做确定性哈希。"""
    parts = []
    for row in tl:
        t = str(row.get("time", ""))
        sym = str(row.get("symbol", ""))
        act = str(row.get("action", ""))
        px = round(float(row.get("price", 0)), 4)
        val = round(float(row.get("value", 0)), 2)
        qty = int(row.get("qty", 0))
        parts.append(f"{t}|{sym}|{act}|{px}|{val}|{qty}")
    return hashlib.md5("\n".join(parts).encode()).hexdigest()


def _equity_curve_summary(ec: list[dict]) -> dict:
    """生成 equity curve 数值摘要用于快速比对。"""
    if not ec:
        return {"len": 0}
    vals = [float(r.get("portfolio_value", 0)) for r in ec]
    turns = [float(r.get("turnover", 0)) for r in ec]
    return {
        "len": len(ec),
        "first_pv": round(vals[0], 2),
        "last_pv": round(vals[-1], 2),
        "max_pv": round(max(vals), 2),
        "min_pv": round(min(vals), 2),
        "sum_turnover": round(sum(turns), 6),
        "first_date": str(ec[0].get("time", "")),
        "last_date": str(ec[-1].get("time", "")),
    }


def _trade_log_summary(tl: list[dict]) -> dict:
    """生成 trade log 数值摘要。"""
    if not tl:
        return {"len": 0, "buys": 0, "sells": 0}
    buys = [r for r in tl if r.get("action") == "buy"]
    sells = [r for r in tl if r.get("action") == "sell"]
    return {
        "len": len(tl),
        "buys": len(buys),
        "sells": len(sells),
        "total_buy_value": round(sum(float(r.get("value", 0)) for r in buys), 2),
        "total_sell_value": round(sum(float(r.get("value", 0)) for r in sells), 2),
    }


def run_baseline():
    """跑两条路径的回测，保存基线。

    用例 1: buy_threshold=15（两路径阈值回退等价，验证纯重复消除）
    用例 2: buy_threshold=5 （两路径阈值回退分叉，捕获 bug 差异）
    """
    kline = _make_synthetic_kline(n_days=120, n_stocks=30, seed=42)

    results = {}
    test_configs = [
        # (label, buy_threshold, expect_identical)
        ("bt15", 15, True),   # max(15,10)==15 → 两路径等价
        ("bt5", 5, False),     # max(5,10)==10 vs 5 → 两路径分叉（bug）
    ]
    for label, bt, _expect in test_configs:
        for method in ["score_weighted", "equal_weight"]:
            key = f"{label}_{method}"
            params = {"buy_threshold": bt, "atr_stop_mult": 2.0, "max_position_pct": 0.1}
            ecfg = EngineConfig(
                portfolio_method=method,
                point_in_time=False,
                max_position_pct=0.1,
                buy_threshold=bt,
                atr_stop_mult=2.0,
                cost_model=CostModel(),
            )
            tl, ec = run_full_backtest(kline, params, ecfg)
            results[key] = {
                "equity_hash": _hash_equity_curve(ec),
                "trade_hash": _hash_trade_log(tl),
                "equity_summary": _equity_curve_summary(ec),
                "trade_summary": _trade_log_summary(tl),
                "equity_curve": ec,
                "trade_log": tl,
            }
            s = results[key]["equity_summary"]
            ts = results[key]["trade_summary"]
            print(f"[{key}] EC hash={results[key]['equity_hash'][:12]}  "
                  f"TL hash={results[key]['trade_hash'][:12]}  "
                  f"days={s['len']}  PV[{s['first_pv']}→{s['last_pv']}]  "
                  f"trades={ts['len']} (buy={ts['buys']}, sell={ts['sells']})")

    # 交叉验证：bt15 两路径应一致，bt5 两路径应分叉
    sw15 = results["bt15_score_weighted"]
    ew15 = results["bt15_equal_weight"]
    sw5 = results["bt5_score_weighted"]
    ew5 = results["bt5_equal_weight"]
    print()
    print(f"bt15: score_weighted vs equal_weight → "
          f"EC={'IDENTICAL' if sw15['equity_hash'] == ew15['equity_hash'] else 'DIFFERENT'}  "
          f"TL={'IDENTICAL' if sw15['trade_hash'] == ew15['trade_hash'] else 'DIFFERENT'}")
    print(f"bt5:  score_weighted vs equal_weight → "
          f"EC={'IDENTICAL' if sw5['equity_hash'] == ew5['equity_hash'] else 'DIFFERENT'}  "
          f"TL={'IDENTICAL' if sw5['trade_hash'] == ew5['trade_hash'] else 'DIFFERENT'}")

    return results


if __name__ == "__main__":
    print("=" * 70)
    print("P2 双路径重构回归基线")
    print("=" * 70)
    results = run_baseline()

    # 保存基线到文件
    out_dir = Path(__file__).resolve().parent / "baseline"
    out_dir.mkdir(exist_ok=True)
    for key, data in results.items():
        # 保存摘要（不含完整 curve/log，用于快速比对）
        summary = {
            "equity_hash": data["equity_hash"],
            "trade_hash": data["trade_hash"],
            "equity_summary": data["equity_summary"],
            "trade_summary": data["trade_summary"],
        }
        (out_dir / f"p2_baseline_{key}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        # 保存完整 curve/log（用于详细 diff）
        (out_dir / f"p2_baseline_{key}_ec.json").write_text(
            json.dumps(data["equity_curve"], indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (out_dir / f"p2_baseline_{key}_tl.json").write_text(
            json.dumps(data["trade_log"], indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(f"\n基线已保存到 {out_dir}")
