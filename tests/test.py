"""
申万二级行业成分股字典生成器
从 akshare 获取申万二级行业全部成分股，输出标准化的股票字典文件
"""
import akshare as ak
import pandas as pd
import time
import os
from datetime import datetime


def get_sw_industry_dict():
    """
    获取申万二级行业字典
    返回: DataFrame，包含 行业代码, 行业名称 等信息
    """
    print("[1/3] 正在获取申万二级行业列表...")
    industry_df = ak.sw_index_second_info()
    print(f"  → 获取到 {len(industry_df)} 个申万二级行业")
    print(f"  → 返回列名: {industry_df.columns.tolist()}")
    print(industry_df.head(3).to_string(index=False))
    print()
    return industry_df


def get_all_component_stocks(industry_df, retry=3, sleep_sec=1.5):
    """
    遍历每个申万二级行业，获取其全部成分股
    """
    # 提取行业代码和行业名称
    # 行业代码可能是 "801016" 或 "801016.SI"，统一处理
    code_col = "行业代码"
    name_col = "行业名称"

    all_rows = []
    total = len(industry_df)
    fail_list = []

    print(f"[2/3] 开始遍历 {total} 个行业，逐个获取成分股...\n")

    for idx, row in industry_df.iterrows():
        raw_code = str(row[code_col]).strip()
        ind_name = str(row[name_col]).strip()

        # 核心处理：去掉 .SI 后缀，index_component_sw 只需要纯数字代码
        symbol = raw_code.replace(".SI", "").replace(".si", "")

        seq = idx + 1 if isinstance(idx, int) else len(all_rows) + 1
        print(f"  [{seq:>3d}/{total}] {ind_name:<10s} ({symbol}) ...", end=" ", flush=True)

        component_df = None
        for attempt in range(retry):
            try:
                component_df = ak.index_component_sw(symbol=symbol)
                break  # 成功则跳出重试循环
            except Exception as e:
                if attempt < retry - 1:
                    time.sleep(sleep_sec * (attempt + 1))  # 递增等待
                else:
                    print(f"[FAIL] 失败({e})")
                    fail_list.append((symbol, ind_name, str(e)))

        if component_df is None or component_df.empty:
            print("[FAIL] 无数据")
            continue

        # ---------- 组装结果行 ----------
        # 根据接口文档，返回列: 序号, 证券代码, 证券名称, 最新权重, 计入日期
        for _, stock_row in component_df.iterrows():
            all_rows.append({
                "证券代码": str(stock_row.get("证券代码", "")).strip(),
                "证券名称": str(stock_row.get("证券名称", "")).strip(),
                "行业代码": raw_code,       # 保留原始代码（含.SI）
                "行业名称": ind_name,
                "最新权重": stock_row.get("最新权重", 0),
                "计入日期": str(stock_row.get("计入日期", "")).strip(),
                "更新时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

        print(f"[OK] {len(component_df)} 只")

        # 请求间隔，避免被反爬
        time.sleep(sleep_sec)

    print(f"\n  → 遍历完成，共获取 {len(all_rows)} 条成分股记录")
    if fail_list:
        print(f"  → 失败行业 {len(fail_list)} 个:")
        for s, n, e in fail_list:
            print(f"      {s} {n}: {e}")

    return pd.DataFrame(all_rows)


def save_stock_dict(df, output_dir="output"):
    """
    保存股票字典到 CSV 和 TXT 文件
    """
    os.makedirs(output_dir, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")

    # --- 输出去重后的全量股票列表 ---
    final_cols = ["证券代码", "证券名称", "行业代码", "行业名称", "更新时间"]
    out_df = df[final_cols].copy()

    # 按证券代码去重（同一只股票可能跨行业，保留首次出现的）
    out_df_unique = out_df.drop_duplicates(subset=["证券代码"], keep="first")
    out_df_unique = out_df_unique.sort_values("证券代码").reset_index(drop=True)

    # 保存全量（含重复，一个股票可能属于多个子行业）
    csv_all = os.path.join(output_dir, f"sw_stocks_all_{today}.csv")
    df[final_cols].to_csv(csv_all, index=False, encoding="utf-8-sig")
    print(f"  → 全量记录({len(df)}条): {csv_all}")

    # 保存去重后的股票字典
    csv_unique = os.path.join(output_dir, f"sw_stocks_dict_{today}.csv")
    out_df_unique.to_csv(csv_unique, index=False, encoding="utf-8-sig")
    print(f"  → 去重字典({len(out_df_unique)}只): {csv_unique}")

    # 保存 TXT 版本（制表符分隔，方便查看）
    txt_file = os.path.join(output_dir, f"sw_stocks_dict_{today}.txt")
    out_df_unique.to_csv(txt_file, index=False, sep="\t", encoding="utf-8")
    print(f"  → TXT版本: {txt_file}")

    return out_df_unique


def main():
    print("=" * 65)
    print("  申万二级行业成分股字典生成器")
    print(f"  运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    print()

    # 第1步：获取行业列表
    industry_df = get_sw_industry_dict()

    # 第2步：遍历获取全部成分股
    all_stocks_df = get_all_component_stocks(industry_df)

    if all_stocks_df.empty:
        print("\n[FAIL] 未获取到任何数据，请检查网络或 akshare 版本")
        return

    # 第3步：保存结果
    print(f"\n[3/3] 保存股票字典文件...")
    unique_df = save_stock_dict(all_stocks_df)

    # 统计摘要
    print(f"\n{'=' * 65}")
    print(f"  统计摘要")
    print(f"{'=' * 65}")
    print(f"  行业总数  : {industry_df.shape[0]}")
    print(f"  成分股记录: {len(all_stocks_df)} 条（含重复）")
    print(f"  去重股票数: {len(unique_df)} 只")
    print(f"\n  前10条预览:")
    print(unique_df.head(10).to_string(index=False))
    print()


if __name__ == "__main__":
    main()