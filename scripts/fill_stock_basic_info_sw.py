"""从 stock_daily_kline 表反向填充 stock_basic_info_sw 表。

当外部管线未运行导致 stock_basic_info_sw 为空时，使用此脚本从已有 K 线数据
提取全市场股票代码 + 最近交易日的收盘信息，构建申万二级表的基础数据。

用法:
    .venv\\Scripts\\python.exe scripts\\fill_stock_basic_info_sw.py
"""
import sys
import re
from datetime import date, datetime

sys.path.insert(0, ".")

from UtilsManager.ConfigParser import Config
from sqlalchemy import create_engine, text
import pandas as pd

cfg = Config()
url = f"postgresql+psycopg2://{cfg.DB_USER}:{cfg.DB_PASSWORD}@{cfg.DB_HOST}:{cfg.DB_PORT}/{cfg.DB_NAME}"
engine = create_engine(url)

print("=== 填充 stock_basic_info_sw 表 ===")

# 1. 检查当前状态
with engine.connect() as conn:
    r = conn.execute(text("SELECT COUNT(*) FROM stock_basic_info_sw"))
    current_count = r.fetchone()[0]
    print(f"当前 stock_basic_info_sw: {current_count} 行")

    r = conn.execute(text("SELECT COUNT(*) FROM stock_daily_kline"))
    kline_count = r.fetchone()[0]
    print(f"K 线表 stock_daily_kline: {kline_count} 行")

if current_count > 0:
    print("表已有数据，跳过填充。如需强制重新填充，请先 TRUNCATE 表。")
    sys.exit(0)

# 2. 从 K 线表提取全市场股票代码 + 最近交易日数据
print("\n从 stock_daily_kline 提取股票信息...")
with engine.connect() as conn:
    # 取最近交易日每只股票的代码
    df = pd.read_sql(text("""
        SELECT DISTINCT symbol
        FROM stock_daily_kline
        ORDER BY symbol
    """), conn)

print(f"K 线表中 distinct symbol: {len(df)} 只")

# 3. 规范化代码 + 生成股票名称
def normalize_code(raw: str) -> str:
    """sz000985 → 000985"""
    s = str(raw).strip()
    for pfx in ("sh", "sz", "bj"):
        if s.lower().startswith(pfx):
            s = s[len(pfx):]
            break
    if "." in s:
        s = s.split(".")[0]
    return s.zfill(6)

def get_exchange(raw: str) -> str:
    """sz000985 → SZ, sh600000 → SH"""
    s = str(raw).strip().lower()
    if s.startswith("sh"):
        return "SH"
    elif s.startswith("sz"):
        return "SZ"
    elif s.startswith("bj"):
        return "BJ"
    return ""

# 过滤主板（60x / 00x）
df["pure_code"] = df["symbol"].apply(normalize_code)
df["exchange"] = df["symbol"].apply(get_exchange)
df = df[df["pure_code"].str.match(r"^(60|00)", na=False)].copy()
print(f"主板过滤后: {len(df)} 只 (60x/00x)")

# 股票名称：K 线表无名称列，用代码作为占位
df["stock_name"] = df["pure_code"]
# 申万二级分类：无外部数据，填占位值
df["industry_code"] = "UNKNOWN"
df["industry_name"] = "未分类"
df["weight"] = 0.0
df["record_date"] = date.today()

# 4. 写入数据库
print(f"\n准备写入 {len(df)} 行到 stock_basic_info_sw...")
rows = [
    {
        "industry_code": r.industry_code,
        "industry_name": r.industry_name,
        "stock_code": r.pure_code,
        "stock_name": r.stock_name,
        "weight": r.weight,
        "record_date": r.record_date,
    }
    for r in df.itertuples(index=False)
]

with engine.connect() as conn:
    # 使用 INSERT ... ON CONFLICT DO NOTHING 避免重复键冲突
    conn.execute(text("""
        INSERT INTO stock_basic_info_sw
            (industry_code, industry_name, stock_code, stock_name, weight, record_date)
        VALUES (:industry_code, :industry_name, :stock_code, :stock_name, :weight, :record_date)
        ON CONFLICT (industry_code, stock_code, record_date) DO NOTHING
    """), rows)
    conn.commit()

    r = conn.execute(text("SELECT COUNT(*) FROM stock_basic_info_sw"))
    final_count = r.fetchone()[0]
    print(f"\n写入完成！stock_basic_info_sw 现有: {final_count} 行")

    # 验证
    r = conn.execute(text("SELECT * FROM stock_basic_info_sw LIMIT 5"))
    print("\n=== 样本数据 ===")
    for row in r:
        print(f"  {row}")

print("\n注意: industry_code/industry_name 为占位值(未分类)，")
print("股票名称为代码占位。待网络恢复后可运行申万行业同步补全行业分类。")
print("但股票池过滤功能已可正常工作。")
