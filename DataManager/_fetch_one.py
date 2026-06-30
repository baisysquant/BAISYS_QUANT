"""Standalone script: fetch kline for stocks via akshare (Sina API).
Called by IncrementalSyncEngine._fetch_kline_batch via subprocess.

Input (single JSON arg): {"symbols": [...], "start": "yyyymmdd", "end": "yyyymmdd"}
Output: JSON list of dicts, each with columns:
  symbol, trade_date, open, close, high, low, volume, amount, adj_factor, close_normal
"""
import sys, json, time, random, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd

payload = json.loads(sys.argv[1])
symbols = payload["symbols"]
start = payload["start"]
end = payload["end"]

import akshare as ak
ak.session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://finance.sina.com.cn/",
})

results = []
for symbol in symbols:
    time.sleep(random.uniform(0.5, 1.5))
    try:
        raw = ak.stock_zh_a_daily(symbol=symbol, start_date=start, end_date=end, adjust="")
        hfq = ak.stock_zh_a_daily(symbol=symbol, start_date=start, end_date=end, adjust="hfq")
        if raw is None or raw.empty or hfq is None or hfq.empty:
            continue
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
        results.append(out.to_dict(orient="list"))
    except Exception:
        pass

print(json.dumps(results))
