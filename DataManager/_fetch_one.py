"""Standalone script: fetch kline for one stock via akshare (Sina API).
Called by IncrementalSyncEngine._fetch_kline via subprocess.

Usage: python _fetch_one.py <symbol> <start_yyyymmdd> <end_yyyymmdd>
Output: JSON dict with keys: symbol, trade_date, open, close, high, low, volume, amount, adj_factor, close_normal
"""
import sys, json, time, random, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd

symbol = sys.argv[1]
start = sys.argv[2]
end = sys.argv[3]

time.sleep(random.uniform(0.05, 0.3))

try:
    import akshare as ak
    raw = ak.stock_zh_a_daily(symbol=symbol, start_date=start, end_date=end, adjust="")
    hfq = ak.stock_zh_a_daily(symbol=symbol, start_date=start, end_date=end, adjust="hfq")
    if raw is None or raw.empty or hfq is None or hfq.empty:
        print("null")
        sys.exit(0)
    raw = raw.rename(columns={"date": "trade_date", "close": "close_raw", "volume": "vol_sina"})
    hfq = hfq.rename(columns={"date": "trade_date"})
    merged = raw[["trade_date", "close_raw", "vol_sina", "amount"]].merge(
        hfq[["trade_date", "open", "high", "low", "close"]], on="trade_date", how="inner"
    )
    for col in ("open", "high", "low", "close", "close_raw"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged["volume"] = pd.to_numeric(merged["vol_sina"], errors="coerce").fillna(0).astype("int64")
    merged["amount"] = pd.to_numeric(merged["amount"], errors="coerce").fillna(0)
    merged["close_normal"] = merged["close_raw"]
    merged["symbol"] = symbol
    if hasattr(merged["trade_date"].iloc[0], "isoformat"):
        merged["trade_date"] = merged["trade_date"].apply(lambda d: d.isoformat())
    merged["adj_factor"] = merged["close"] / merged["close_normal"].replace(0, float("nan"))
    merged["adj_factor"] = merged["adj_factor"].replace([float("inf")], 1.0).fillna(1.0)
    out = merged[["symbol", "trade_date", "open", "close", "high", "low", "volume", "amount", "adj_factor", "close_normal"]]
    print(json.dumps(out.to_dict(orient="list")))
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(1)
