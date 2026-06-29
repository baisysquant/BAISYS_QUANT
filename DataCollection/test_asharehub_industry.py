"""
测试 AShareHub 行业分类接口（申万 SW2021 三层体系）
GET /v2/reference/industries
输入参数不填写 -> 获取全量 A 股行业分类
"""
import sys, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, r"E:\BAISYS_QUANT\BAISYS_QUANT")

from ConfigParser import Config
from asharehub import AShareHub

# 从配置读取 API Key
config = Config()
api_key = config.ASHAREHUB_API_KEY
print(f"API Key: {api_key[:8]}...{api_key[-4:]}")

client = AShareHub(api_key=api_key)

# 不传 symbol 获取全量
print("正在获取全量行业分类数据...")
df = client.industry_list()
print(f"\n成功获取 {len(df)} 条记录")
print(f"列名: {df.columns.tolist()}")
print()

print("前10行:")
print(df.head(10).to_string())
print()

# 统计三级体系
l1_count = df["l1_name"].nunique()
l2_count = df["l2_name"].nunique()
l3_count = df["l3_name"].nunique()
print(f"一级行业: {l1_count} 个")
print(f"二级行业: {l2_count} 个")
print(f"三级行业: {l3_count} 个")
print(f"覆盖股票: {df['symbol'].nunique()} 只")

# 保存到 parquet 看看大小
df.to_parquet(r"E:\BAISYS_QUANT\BAISYS_QUANT\DataCollection\asharehub_industries.parquet", index=False)
print(f"\n已保存到 DataCollection/asharehub_industries.parquet")
