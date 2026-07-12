from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
from typing import Any

import httpx
import pandas as pd
from loguru import logger


def _check_v8_works() -> bool:
    """预检 py_mini_racer V8 是否可用，避免崩溃杀死主进程"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "from py_mini_racer import MiniRacer; MiniRacer()"],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def _try_akshare_fetch() -> pd.DataFrame | None:
    """在子进程中调用 ak.stock_comment_em()，隔离 V8 segfault"""
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        pkl_path = f.name
    script = (
        "import akshare as ak; import pickle; import pandas as pd\n"
        "try:\n"
        "    df = ak.stock_comment_em()\n"
        "    pickle.dump(df, open(r'%s', 'wb'))\n"
        "except Exception as e:\n"
        "    pickle.dump(e, open(r'%s', 'wb'))\n"
    ) % (pkl_path, pkl_path)
    try:
        subprocess.run(
            [sys.executable, "-c", script],
            timeout=120, capture_output=True,
        )
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, pd.DataFrame) and not obj.empty:
                return obj
    except Exception:
        pass
    finally:
        if os.path.exists(pkl_path):
            os.unlink(pkl_path)
    return None


class MainCostDataManager:
    """
    主力成本数据管理类
    提供主力成本、机构参与度等相关数据的获取、分析和管理功能
    """

    def __init__(self, config: Any | None = None, cache_enabled: bool = True, cache_dir: str | None = None) -> None:
        if config is None:
            try:
                from UtilsManager.ConfigParser import Config
                config = Config()
            except Exception:
                config = None
        self.config = config
        if cache_dir is None and config is not None:
            cache_dir = os.path.join(config.CACHE_DIRECTORY, "cost_data_cache")
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/Downloads/CoreNews_Reports/cache/cost_data_cache")
        self.cache_enabled = cache_enabled
        self.cache_dir = cache_dir
        if cache_enabled and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _today_str(self) -> str:
        try:
            from DataCollection.CalendarManager import TradingCalendarAnalyzer
            return TradingCalendarAnalyzer().get_last_trading_day().replace("-", "")
        except Exception:
            from datetime import datetime
            return datetime.now().strftime("%Y%m%d")

    def _fetch_from_eastmoney(self) -> pd.DataFrame:
        """直调东方财富数据中心 API，获取主力成本数据"""
        token = (
            self.config.DISTRIBUTION_API_TOKEN
            if self.config is not None
            else "894050c76af8597a853f5b408b759f5d"
        )
        url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
        base_params = {
            "sortColumns": "SECURITY_CODE",
            "sortTypes": "1",
            "pageSize": "500",
            "pageNumber": "1",
            "reportName": "RPT_DMSK_TS_STOCKNEW",
            "quoteColumns": "f2~01~SECURITY_CODE~CLOSE_PRICE,"
                            "f8~01~SECURITY_CODE~TURNOVERRATE,"
                            "f3~01~SECURITY_CODE~CHANGE_RATE,"
                            "f9~01~SECURITY_CODE~PE_DYNAMIC",
            "columns": "ALL",
            "filter": "",
            "token": token,
        }

        frames: list[pd.DataFrame] = []
        client = httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0))

        # 首次请求获取总页数
        resp = client.get(url, params=base_params)
        resp.raise_for_status()
        payload = resp.json()
        total_pages = payload["result"]["pages"]
        frames.append(pd.DataFrame(payload["result"]["data"]))

        # 翻页
        for page in range(2, total_pages + 1):
            params = dict(base_params, pageNumber=str(page))
            try:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()["result"]["data"]
                if data:
                    frames.append(pd.DataFrame(data))
            except Exception:
                continue

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)

        # 映射为中文字段名（与原有 akshare 接口返回一致）
        col_map = {
            "SECURITY_CODE": "代码",
            "SECURITY_NAME_ABBR": "名称",
            "CLOSE_PRICE": "最新价",
            "CHANGE_RATE": "涨跌幅",
            "PE_DYNAMIC": "市盈率",
            "PRIME_COST": "主力成本",
            "ORG_PARTICIPATE": "机构参与度",
            "TOTALSCORE": "综合得分",
        }
        keep_cols = [k for k in col_map if k in df.columns]
        df = df[keep_cols].rename(columns=col_map)

        # 添加序号列并确保值类型
        df.insert(0, "序号", range(1, len(df) + 1))
        for col in ["最新价", "主力成本", "机构参与度", "涨跌幅", "市盈率", "综合得分"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def get_main_cost_data(self) -> pd.DataFrame:
        cache_file = (
            os.path.join(self.cache_dir, f"main_cost_data_{self._today_str()}.csv")
            if self.cache_enabled
            else None
        )

        if self.cache_enabled and cache_file and os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file)
                logger.info(f"从缓存加载主力成本数据: {cache_file}")
                return df
            except Exception as e:
                logger.error(f"读取缓存失败: {e}")

        logger.info("正在获取主力成本数据...")
        df = None
        if _check_v8_works():
            df = _try_akshare_fetch()
        if df is None:
            df = self._fetch_from_eastmoney()

        if self.cache_enabled and cache_file and not df.empty:
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                df.to_csv(cache_file, index=False)
                logger.info(f"主力成本数据已缓存: {cache_file}")
            except Exception as e:
                logger.error(f"缓存数据失败: {e}")

        return df

    def analyze_cost_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        result_df = df.copy()

        if "主力成本" in result_df.columns and "最新价" in result_df.columns:
            result_df["主力成本差价"] = result_df["最新价"] - result_df["主力成本"]

            result_df["主力成本差价百分比"] = (
                (result_df["最新价"] - result_df["主力成本"]) / result_df["主力成本"]
            ) * 100

            def cost_position(row: pd.Series) -> str:
                if row["最新价"] > row["主力成本"]:
                    if row["主力成本差价百分比"] > 10:
                        return "大幅高于成本"
                    else:
                        return "略高于成本"
                elif row["最新价"] < row["主力成本"]:
                    if row["主力成本差价百分比"] < -10:
                        return "大幅低于成本"
                    else:
                        return "略低于成本"
                else:
                    return "等于成本"

            result_df["成本位置"] = result_df.apply(cost_position, axis=1)

        if "机构参与度" in result_df.columns:
            result_df["机构参与度等级"] = pd.cut(
                result_df["机构参与度"],
                bins=[-1, 20, 50, 80, 101],
                labels=["低", "中低", "中高", "高"],
                include_lowest=True,
            ).astype(str)

        if "主力成本" in result_df.columns and "机构参与度" in result_df.columns:

            def control_strength(row: pd.Series) -> str:
                if row["机构参与度"] >= 80 and abs(row["主力成本差价百分比"]) <= 10:
                    return "高度控盘"
                elif row["机构参与度"] >= 50 and abs(row["主力成本差价百分比"]) <= 15:
                    return "中度控盘"
                elif row["机构参与度"] >= 20:
                    return "轻度控盘"
                else:
                    return "低度控盘"

            result_df["主力控盘强度"] = result_df.apply(control_strength, axis=1)

        return result_df

    def print_cost_summary(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            logger.warning("主力成本数据为空")
            return

        logger.info("主力成本数据分析摘要")

        logger.info(f"总股票数量: {len(df)}")

        if "主力成本" in df.columns:
            logger.info(f"主力成本有效数量: {df['主力成本'].notna().sum()}")
            logger.info(f"主力成本平均值: {df['主力成本'].mean():.2f}")
            logger.info(f"主力成本中位数: {df['主力成本'].median():.2f}")
            logger.info(f"主力成本最高值: {df['主力成本'].max():.2f}")
            logger.info(f"主力成本最低值: {df['主力成本'].min():.2f}")

        if "成本位置" in df.columns:
            logger.info("\n成本位置分布:")
            position_counts = df["成本位置"].value_counts()
            for pos, count in position_counts.items():
                logger.info(f"  {pos}: {count} 只 ({count / len(df) * 100:.1f}%)")

        if "机构参与度等级" in df.columns:
            logger.info("\n机构参与度等级分布:")
            level_counts = df["机构参与度等级"].value_counts()
            for level, count in level_counts.items():
                logger.info(f"  {level}: {count} 只 ({count / len(df) * 100:.1f}%)")

        if "主力控盘强度" in df.columns:
            logger.info("\n主力控盘强度分布:")
            strength_counts = df["主力控盘强度"].value_counts()
            for strength, count in strength_counts.items():
                logger.info(f"  {strength}: {count} 只 ({count / len(df) * 100:.1f}%)")
