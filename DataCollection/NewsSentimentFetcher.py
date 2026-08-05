from __future__ import annotations

from typing import Any

import pandas as pd
from loguru import logger

_POSITIVE_WORDS = [
    "利好", "增长", "突破", "盈利", "新高", "涨停", "大涨", "买入", "推荐",
    "增持", "回购", "分红", "龙头", "扩张", "创新", "领先", "优势",
    "提振", "回暖", "复苏", "加速", "放量",
]

_NEGATIVE_WORDS = [
    "利空", "下跌", "亏损", "减持", "跌停", "大跌", "卖出", "回避",
    "风险", "暴雷", "违约", "诉讼", "处罚", "调查", "退市", "ST",
    "下滑", "萎缩", "放缓", "收缩", "预警",
]


class NewsSentimentFetcher:
    """NLP 舆情因子 — 基于 Baidu 新闻关键词情感打分。

    用法:
        fetcher = NewsSentimentFetcher(config)
        df = fetcher.fetch_multi_day(days=20)
        # df 包含 symbol, 情感分, 新闻数 列
    """

    def __init__(self, config: Any = None) -> None:  # noqa: ANN401
        self.config = config

    def fetch_multi_day(self, days: int = 20) -> pd.DataFrame:
        """获取多日新闻情感数据。

        Returns:
            DataFrame with columns: symbol, 情感总分, 正面新闻数, 负面新闻数, 总新闻数, 行业
        """
        try:
            import akshare as ak
        except ImportError:
            logger.warning("[舆情因子] akshare 不可用")
            return pd.DataFrame()

        try:
            df = ak.news_report_time_baidu()
        except Exception:
            logger.warning("[舆情因子] Baidu 新闻获取失败")
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        # 修复列名编码问题（某些版本返回 gbk 编码的列名）
        df.columns = [
            c.encode("gbk", errors="replace").decode("gbk", errors="replace")
            if isinstance(c, str) and any(ord(ch) > 127 for ch in c)
            else c
            for c in df.columns
        ]
        # 清理不可见字符
        df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

        col_map = {
            "股票代码": "symbol",
            "股票名称": "name",
            "文章来源": "source",
            "标题": "title",
            "发布时间": "pub_time",
            "分类": "category",
        }
        known = {k: v for k, v in col_map.items() if k in df.columns}
        if not known:
            return pd.DataFrame()
        df = df.rename(columns=known).copy()

        if "title" not in df.columns:
            return pd.DataFrame()

        df["title"] = df["title"].astype(str)

        sentiments = []
        for _, row in df.iterrows():
            title = row.get("title", "")
            pos = sum(1 for w in _POSITIVE_WORDS if w in title)
            neg = sum(1 for w in _NEGATIVE_WORDS if w in title)
            sentiments.append(max(-1, min(1, (pos - neg) / max(pos + neg, 1))))

        df["情感分"] = sentiments
        df["正面新闻数"] = df["情感分"].apply(lambda s: 1 if s > 0 else 0)
        df["负面新闻数"] = df["情感分"].apply(lambda s: 1 if s < 0 else 0)

        if "symbol" not in df.columns:
            return pd.DataFrame()

        symbols_clean = []
        for s in df["symbol"]:
            ss = str(s).strip().replace("sh", "").replace("sz", "").replace("SH", "").replace("SZ", "")
            if ss.isdigit():
                ss = ss.zfill(6)
            symbols_clean.append(ss)
        df["symbol"] = symbols_clean

        df["pub_date"] = pd.to_datetime(df.get("pub_time", "1900-01-01"), errors="coerce").dt.date.astype(str)

        grouped = df.groupby("symbol").agg(
            情感总分=("情感分", "sum"),
            正面新闻数=("正面新闻数", "sum"),
            负面新闻数=("负面新闻数", "sum"),
            总新闻数=("情感分", "count"),
        ).reset_index()

        grouped.columns = ["symbol", "情感总分", "正面新闻数", "负面新闻数", "总新闻数"]
        grouped["情感总分"] = grouped["情感总分"].fillna(0)
        grouped["正面新闻数"] = grouped["正面新闻数"].fillna(0).astype(int)
        grouped["负面新闻数"] = grouped["负面新闻数"].fillna(0).astype(int)
        grouped["总新闻数"] = grouped["总新闻数"].fillna(0).astype(int)
        grouped["行业"] = "未知"

        return grouped
