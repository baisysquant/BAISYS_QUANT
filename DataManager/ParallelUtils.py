from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd


def _normalize_fund_data(df):
    """
    标准化资金数据：将所有资金相关的列统一转换为数值型（单位：万元）。

    处理逻辑：
    - 字符串类型：识别'亿'、'万'单位并转换
    - 数值类型：假设单位为'元'，转换为'万元'（除以10000）
    """
    if df is None or df.empty:
        return df

    # 创建副本避免视图问题
    df = df.copy()

    # 定义关键词，自动识别需要转换单位的列
    zijin_keywords = ["资金", "流向", "净流入", "净额", "成交额"]
    target_cols = [col for col in df.columns if any(k in col for k in zijin_keywords)]

    if not target_cols:
        return df

    for col in target_cols:
        # 判断是否为字符串类型（object类型或dtype.kind为'O'）
        is_string_type = df[col].dtype.kind == "O" or df[col].dtype == object

        # 处理字符串类型（包含'亿'、'万'单位）
        if is_string_type:

            def convert_to_wan(val):
                if val is None or str(val).strip() in ["", "-", "nan", "NaN", "None"]:
                    return 0.0
                val_str = str(val).strip()
                try:
                    if "亿" in val_str:
                        return float(val_str.replace("亿", "")) * 10000
                    elif "万" in val_str:
                        return float(val_str.replace("万", ""))
                    else:
                        return float(val_str)
                except (ValueError, TypeError):
                    return 0.0

            # 先转换为Series，再赋值，避免pandas string类型的限制
            converted_series = df[col].apply(convert_to_wan)
            df[col] = converted_series.astype(float)

        # 处理数值类型（假设单位为'元'，转换为'万元'）
        elif pd.api.types.is_numeric_dtype(df[col]):
            # 检查是否有异常大的值（可能是'元'为单位）
            max_val = df[col].abs().max()
            if pd.notna(max_val) and max_val > 1000000:  # 大于100万，很可能是'元'
                df.loc[:, col] = df[col] / 10000.0  # 转换为万元

    return df


def run_with_thread_pool(
    items: Iterable[Any], worker_func: Callable[[Any], Any], max_workers: int = 10, desc: str = "任务"
) -> list[Any]:
    """
    通用的多线程执行器。

    :param items: 需要处理的数据列表 (如股票代码列表)
    :param worker_func: 处理单个数据的函数 (输入一个item，返回结果)
    :param max_workers: 最大线程数
    :param desc: 任务描述，用于日志打印
    :return: 包含所有成功结果的列表 (过滤掉 None)
    """
    results = []
    total = len(list(items))  # 注意：如果items是生成器，这里会消耗掉，建议传list

    print(f"\n>>> 开始并发执行: {desc} (数量: {total}, 线程: {max_workers})...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任务
        future_to_item = {executor.submit(worker_func, item): item for item in items}

        for future in as_completed(future_to_item):
            item = future_to_item[future]
            try:
                res = future.result()
                if res is not None:
                    # 如果结果是 DataFrame 且为空，视具体情况决定是否添加
                    # 这里只要不是 None 就添加，由调用方后续处理 (如 concat)
                    results.append(res)
            except (TimeoutError, TypeError, ValueError, KeyError, AttributeError) as e:
                print(f"[ERROR] 处理 {item} 时发生异常: {e}")

    print(f">>> {desc} 执行完毕，成功获取 {len(results)} 条结果。")
    return results
