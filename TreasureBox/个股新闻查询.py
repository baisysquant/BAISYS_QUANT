from __future__ import annotations
from datetime import datetime, timedelta
import akshare as ak
import pandas as pd

def get_period_selection() -> int:
    while True:
        print("\n请选择您要查询的新闻周期：")
        print("  输入 1: 最近 30 天")
        print("  输入 2: 最近 60 天")
        choice = input("请输入您的选择 (1 或 2): ").strip()

        if choice == "1":
            return 30
        elif choice == "2":
            return 60
        else:
            print(f"输入 '{choice}' 无效。请重新输入 1 或 2。")

def query_stock_news() -> None:
    print("--- 东方财富个股新闻查询工具 ---")

    # 循环直到用户输入有效的股票代码
    while True:
        input_str = input("请输入您要查询的股票代码 (例如: 603777)，或输入 'exit' 退出: ").strip()

        if input_str.lower() == "exit":
            print("程序已退出。")
            return

        # 仅保留数字，确保获取的是纯数字字符串
        stock_code = ''.join(filter(str.isdigit, input_str))

        if stock_code.isdigit() and len(stock_code) == 6:
            break
        else:
            print(f"输入 '{input_str}' 无效。股票代码通常是 6 位数字。请重新输入。")

    # 第二个交互问题：选择查询周期
    days_to_query = get_period_selection()

    print(f"\n正在查询股票代码 {stock_code} 在过去 {days_to_query} 天内的最新新闻资讯...")

    try:
        # 调用 akshare 接口获取个股新闻，确保 stock_code 是字符串
        news_df: pd.DataFrame | None = ak.stock_news_em(symbol=stock_code)

        if news_df is None or news_df.empty:
            print(f"\n[提示] 未能获取到股票 {stock_code} 的新闻数据，可能是今日无相关新闻或接口查询失败。")
            return

        # 后续处理逻辑不变...
        if "发布时间" in news_df.columns:
            try:
                cutoff_date = datetime.now() - timedelta(days=days_to_query)
                news_df["发布时间_dt"] = pd.to_datetime(news_df["发布时间"], errors="coerce")
                news_df = news_df[(news_df["发布时间_dt"].notna()) & (news_df["发布时间_dt"] >= cutoff_date)].copy()
                news_df = news_df.sort_values(by="发布时间_dt", ascending=False)
                news_df = news_df.drop(columns=["发布时间_dt"])
            except Exception as e:
                print(f"[警告] 对 '发布时间' 进行 {days_to_query} 天过滤和排序时失败，将显示所有新闻且可能未排序。错误详情: {e}")

        if news_df.empty:
            print(f"\n[提示] 股票 {stock_code} 在过去 {days_to_query} 天内没有新的新闻资讯。")
            return

        required_cols = ["关键词", "新闻标题", "新闻内容", "发布时间"]
        display_df = news_df[[col for col in required_cols if col in news_df.columns]].copy()

        if "发布时间" in display_df.columns:
            try:
                display_df["发布时间"] = pd.to_datetime(display_df["发布时间"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                pass

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"个股新闻_{stock_code}_{timestamp}.xlsx"

        try:
            display_df.to_excel(filename, index=False, sheet_name=stock_code)
            print(f"成功获取 {stock_code} 的 {len(display_df)} 条新闻。")
            print(f"结果已保存到 Excel 文件: {filename}")
            print("您可以在 Excel 中打开此文件，查看完整的新闻内容。")
        except Exception as e:
            print(f"[错误] 保存 Excel 文件失败: {e}")
            print("\n[警告] 文件保存失败，以下是完整数据控制台输出（可能格式错乱）:")
            print(display_df.to_string(index=False))

    except Exception as e:
        print(f"[错误] 查询过程中发生异常，请检查股票代码是否正确或网络连接: {e}")

if __name__ == "__main__":
    query_stock_news()