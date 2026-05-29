import akshare as ak

stock_info_global_em_df = ak.stock_info_global_em()
print(stock_info_global_em_df)

# 保存到本地 txt 文件
output_file = 'stock_info_global_em.txt'
stock_info_global_em_df.to_csv(output_file, sep='\t', index=False, encoding='utf-8-sig')
print(f"\n数据已保存至: {output_file}")
