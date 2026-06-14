import akshare as ak

stock_zh_a_hist_tx_df = ak.stock_zh_a_hist_tx(symbol="sz000001", start_date="20200101", end_date="20231027", adjust="")
print(stock_zh_a_hist_tx_df)