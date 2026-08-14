from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import Any

import pandas as pd
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from UtilsManager.ConfigParser import Config


# 申万一级行业 → 宏观敏感度分类
_SW1_MACRO_CLASS = {
    "银行": "金融", "非银金融": "金融", "房地产": "金融",
    "钢铁": "周期", "有色金属": "周期", "煤炭": "周期",
    "基础化工": "周期", "建筑材料": "周期", "石油石化": "周期",
    "电子": "科技", "计算机": "科技", "通信": "科技", "传媒": "科技",
    "电力设备": "科技", "机械设备": "科技", "国防军工": "科技",
    "食品饮料": "消费", "家用电器": "消费", "纺织服饰": "消费",
    "轻工制造": "消费", "农林牧渔": "消费", "商贸零售": "消费",
    "汽车": "消费", "美容护理": "消费",
    "医药生物": "医药",
    "公用事业": "防御", "交通运输": "防御", "建筑装饰": "防御",
    "环保": "防御", "社会服务": "防御", "综合": "防御",
}

# 宏观状态 → 行业类别偏好（正值 = 超配，负值 = 低配）
REGIME_INDUSTRY_TILT: dict[str, dict[str, float]] = {
    "boom": {
        "金融": 0.8, "周期": 0.8, "科技": 0.6,
        "消费": 0.0, "医药": -0.2, "防御": -0.6,
    },
    "normal": {
        "金融": 0.0, "周期": 0.0, "科技": 0.0,
        "消费": 0.0, "医药": 0.0, "防御": 0.0,
    },
    "recession": {
        "防御": 0.8, "医药": 0.6, "消费": 0.4,
        "科技": -0.2, "金融": -0.4, "周期": -0.8,
    },
}


class MacroFactorFetcher:
    """宏观因子获取器 — PMI / M2 / CPI。

    用于判断当前经济周期状态（扩张/平稳/收缩），
    并据此为不同行业计算偏好得分（宏观 tilt）。
    """

    CACHE_DIR: str | None = None

    def __init__(self, config: Config) -> None:
        self.config = config
        if hasattr(config, "TEMP_DATA_DIRECTORY"):
            self.CACHE_DIR = config.TEMP_DATA_DIRECTORY
        else:
            self.CACHE_DIR = os.path.expanduser("~/Downloads/CoreNews_Reports/cache")
        os.makedirs(self.CACHE_DIR, exist_ok=True)

    def _cache_path(self, name: str) -> str:
        return os.path.join(self.CACHE_DIR, f"macro_{name}.csv")

    def fetch_pmi(self) -> float | None:
        """获取最新制造业 PMI。"""
        path = self._cache_path("pmi")
        try:
            import akshare as ak
            df = ak.macro_china_pmi()
            if df is None or df.empty:
                return self._read_cached_fallback(path)
            val_col = [c for c in df.columns if "制造业" in c or "PMI" in c or "pmi" in c.lower() or "当月" in c]
            if not val_col:
                val_col = df.select_dtypes(include="number").columns
            if not val_col:
                return self._read_cached_fallback(path)
            latest = float(pd.to_numeric(df[val_col[0]], errors="coerce").dropna().iloc[-1])
            self._write_cache(path, latest)
            return latest
        except Exception:
            return self._read_cached_fallback(path)

    def fetch_m2(self) -> float | None:
        """获取最新 M2 同比增速(%)。"""
        path = self._cache_path("m2")
        try:
            import akshare as ak
            df = ak.macro_china_money_supply()
            if df is None or df.empty:
                return self._read_cached_fallback(path)
            val_col = [c for c in df.columns if "M2" in c and ("同比" in c or "yoy" in c.lower())]
            if not val_col:
                val_col = df.select_dtypes(include="number").columns
            if not val_col:
                return self._read_cached_fallback(path)
            latest = float(pd.to_numeric(df[val_col[0]], errors="coerce").dropna().iloc[-1])
            self._write_cache(path, latest)
            return latest
        except Exception:
            return self._read_cached_fallback(path)

    def fetch_cpi(self) -> float | None:
        """获取最新 CPI 同比增速(%)。"""
        path = self._cache_path("cpi")
        try:
            import akshare as ak
            df = ak.macro_china_cpi_monthly()
            if df is None or df.empty:
                return self._read_cached_fallback(path)
            val_col = [c for c in df.columns if "当月同比" in c or "cpi" in c.lower() or "全国" in c]
            if not val_col:
                val_col = df.select_dtypes(include="number").columns
            if not val_col:
                return self._read_cached_fallback(path)
            latest = float(pd.to_numeric(df[val_col[0]], errors="coerce").dropna().iloc[-1])
            self._write_cache(path, latest)
            return latest
        except Exception:
            return self._read_cached_fallback(path)

    def classify_regime(self, pmi: float | None, m2: float | None, cpi: float | None) -> str:
        """综合 PMI + M2 + CPI 判断宏观状态。"""
        score = 0.0
        if pmi is not None:
            if pmi > 50.5:
                score += 1.0
            elif pmi < 49.5:
                score -= 1.0
        if m2 is not None:
            if m2 > 10.0:
                score += 0.5
            elif m2 < 8.0:
                score -= 0.5
        if cpi is not None:
            if cpi > 3.0:
                score -= 0.5  # 通胀过热
            elif cpi < 0.5:
                score -= 0.5  # 通缩风险
            else:
                score += 0.3
        if score > 0.5:
            return "boom"
        if score < -0.5:
            return "recession"
        return "normal"

    def get_industry_tilts(self) -> dict[str, float]:
        """返回 {申万一级行业: tilt_score} 映射。

        tilt_score 范围 [-1, 1]，正值=超配，负值=低配。
        """
        pmi = self.fetch_pmi()
        m2 = self.fetch_m2()
        cpi = self.fetch_cpi()
        regime = self.classify_regime(pmi, m2, cpi)

        class_tilts = REGIME_INDUSTRY_TILT.get(regime, REGIME_INDUSTRY_TILT["normal"])
        result: dict[str, float] = {}
        for sw1, cls in _SW1_MACRO_CLASS.items():
            result[sw1] = class_tilts.get(cls, 0.0)
        # "未知"行业默认 0
        result["未知"] = 0.0
        logger.info(f"[宏观] PMI={pmi}, M2={m2}%, CPI={cpi}% → 状态={regime}, 行业tilt已生成")
        return result

    def _write_cache(self, path: str, val: float) -> None:
        try:
            pd.DataFrame({"date": [datetime.now().strftime("%Y%m%d")], "value": [val]}).to_csv(
                path, index=False, encoding="utf-8-sig"
            )
        except Exception:
            pass

    def _read_cached_fallback(self, path: str) -> float | None:
        if not os.path.exists(path):
            logger.warning(f"[宏观] 缓存不存在: {path}")
            return None
        try:
            df = pd.read_csv(path)
            return float(df["value"].iloc[-1])
        except Exception:
            return None
