"""Standalone script: fetch kline for stocks via Tencent API (no V8).
Called by IncrementalSyncEngine._fetch_kline_batch via subprocess.

Input (single JSON arg): {"symbols": [...], "start": "yyyymmdd", "end": "yyyymmdd"}
Output: JSON list of dicts, each with columns:
  symbol, trade_date, open, close, high, low, volume, amount, adj_factor, close_normal
"""
import sys, json, time, random, os, requests

payload = json.loads(sys.argv[1])
symbols = payload["symbols"]
start_date_str = payload["start"]
end_date_str = payload["end"]

# Determine the year range needed
start_year = int(start_date_str[:4])
end_year = int(end_date_str[:4])

TX_URL = "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get"


def _tx_raw(symbol: str, adjust: str = "") -> list[list]:
    """Fetch one stock's adjusted kline from Tencent API for the needed years.
    Returns list of raw rows as lists.
    """
    rows: list[list] = []
    for year in range(start_year, end_year + 1):
        adj_part = adjust
        var_name = f"kline_day{adjust}{year}"
        param_str = f"{symbol},day,{year}-01-01,{year+1}-12-31,640,{adj_part}"
        try:
            r = requests.get(
                TX_URL,
                params={"_var": var_name, "param": param_str, "r": "0.8205"},
                timeout=15,
            )
            text = r.text
            data = json.loads(text[text.find("={") + 1:])
            inner = data["data"][symbol]
            # 不复权 -> "day", 后复权 -> "hfqday", 前复权 -> "qfqday"
            key = "day"
            if adjust == "hfq":
                key = "hfqday"
            elif adjust == "qfq":
                key = "qfqday"
            yr_rows = inner.get(key, [])
            rows.extend(yr_rows)
        except Exception:
            pass
    return rows


results = []
for symbol in symbols:
    time.sleep(random.uniform(0.5, 1.5))
    try:
        raw_rows = _tx_raw(symbol, adjust="")
        hfq_rows = _tx_raw(symbol, adjust="hfq")
        if not raw_rows or not hfq_rows:
            continue

        # Build date -> row dicts
        raw_map: dict[str, list] = {}
        for row in raw_rows:
            d = str(row[0])
            raw_map[d] = row
        hfq_map: dict[str, list] = {}
        for row in hfq_rows:
            d = str(row[0])
            hfq_map[d] = row

        # Intersect on dates and filter by range
        common_dates = sorted(set(raw_map) & set(hfq_map))
        filtered = [d for d in common_dates if start_date_str <= d.replace("-", "") <= end_date_str]
        if not filtered:
            continue

        out_dict: dict[str, list] = {
            "symbol": [],
            "trade_date": [],
            "open": [],
            "close": [],
            "high": [],
            "low": [],
            "volume": [],
            "amount": [],
            "adj_factor": [],
            "close_normal": [],
        }

        for d in filtered:
            raw = raw_map[d]
            hfq = hfq_map[d]
            # raw (11 cols): [0]date, [1]open, [2]close(=close_normal), [3]high, [4]low, [5]vol_hand, [6]{}, [7]pct, [8]amt_wan, [9]?, [10]?
            # hfq (10 cols): [0]date, [1]open, [2]close, [3]high, [4]low, [5]vol_hand, [6]{}, [7]pct, [8]amt_wan, [9]?
            close_raw = float(raw[2])
            close_hfq = float(hfq[2])
            adj_factor = close_hfq / close_raw if close_raw != 0 else 1.0
            out_dict["symbol"].append(symbol)
            out_dict["trade_date"].append(d)
            out_dict["open"].append(float(hfq[1]))
            out_dict["close"].append(close_hfq)
            out_dict["high"].append(float(hfq[3]))
            out_dict["low"].append(float(hfq[4]))
            out_dict["volume"].append(int(float(raw[5]) * 100))
            out_dict["amount"].append(float(raw[8]) * 10000)
            out_dict["adj_factor"].append(adj_factor)
            out_dict["close_normal"].append(close_raw)

        results.append(out_dict)
    except Exception:
        pass

print(json.dumps(results))
