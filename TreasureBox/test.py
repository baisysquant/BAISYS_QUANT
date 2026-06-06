import akshare as ak

# 1. 获取东方财富网-ST股票列表数据
stock_zh_a_st_em_df = ak.stock_zh_a_st_em()

# 2. 修改为保存至 TXT 文件（以制表符 \t 分隔，保留表头，不保留索引）
stock_zh_a_st_em_df.to_csv("stock_zh_a_st_em.txt", sep="\t", index=False, encoding="utf-8")
