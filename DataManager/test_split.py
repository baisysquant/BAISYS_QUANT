"""Quick test: verify split detection triggers full history rewrite."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from DataManager.IncrementalSyncEngine import IncrementalSyncEngine, _fetch_kline_batch
from sqlalchemy import create_engine
import pandas as pd

# Use the same DB as the main app
engine = create_engine("postgresql://...")  # fill in actual DSN

syncer = IncrementalSyncEngine(engine, default_start="20230101")

# Pick a stock known to have had a recent split
# Force a test: get its latest adj_factor from DB, compare with fresh data
symbol = "sh600519"  # Maotai - check if it had a split

# Fetch window data (same as pipeline does)
batch = _fetch_kline_batch([symbol], "20260531", "20260630")
if batch:
    df = pd.DataFrame(batch[0])
    latest = syncer._get_latest_date(symbol)
    print(f"{symbol}: latest in DB = {latest}")
    if latest:
        has_split = syncer._detect_split_from_adj(symbol, df, latest)
        print(f"  split detected: {has_split}")
        if has_split:
            print("  -> would trigger full history rewrite")
            full = syncer._fetch_full_history(symbol)
            if full is not None:
                print(f"  -> full history fetched: {len(full)} rows, {full['trade_date'].min()} ~ {full['trade_date'].max()}")
