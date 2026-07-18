"""
因子注册中心 — YAML 驱动的因子定义管理。

功能：
  - 集中管理所有因子定义（名称、权重、类别、参数）
  - 支持动态调整权重（无需修改代码）
  - 提供因子元数据查询接口
  - 支持 IC 衰减监控的自动配置

用法：
  registry = FactorRegistry()
  weights = registry.weights
  factor_def = registry.get("momentum")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml


DEFAULT_CONFIG = """
# ============================================================
# 多因子 Alpha 注册中心
# ============================================================
# 研究效率提升：修改此文件即可调整因子定义，无需修改 Python 代码。
# 每项因子可配置：权重、计算参数、行业中性化开关、IC 衰减参数。

factors:
  macd:
    name: "MACD评分"
    category: "technical"
    weight: 0.25
    weight_min: 0.0
    weight_max: 0.5
    description: "MACD 七维度综合评分（行业百分位归一化）"
    industry_neutral: true
    column: "MACD评分"

  momentum:
    name: "动量评分"
    category: "momentum"
    weight: 0.25
    weight_min: 0.0
    weight_max: 0.5
    description: "21 日动量因子（行业内 Z-Score）"
    industry_neutral: true
    column: "动量评分"
    params:
      lookback: 21

  moneyflow:
    name: "资金流评分"
    category: "moneyflow"
    weight: 0.20
    weight_min: 0.0
    weight_max: 0.4
    description: "资金流加权评分（行业内 Z-Score）"
    industry_neutral: true
    column: "资金流评分"
    params:
      column_weights:
        "3日资金流入万元": 0.3
        "5日资金流入万元": 0.4
        "10日资金流入万元": 0.2
        "20日资金流入万元": 0.1

  quality:
    name: "基本面评分"
    category: "fundamental"
    weight: 0.15
    weight_min: 0.0
    weight_max: 0.3
    description: "质量因子：ROE×0.4 + 毛利率×0.3 + 净利率×0.3"
    industry_neutral: true
    column: "基本面评分"
    params:
      components:
        roe: 0.4
        gross_profit_margin: 0.3
        net_profit_margin: 0.3

  valuation:
    name: "估值评分"
    category: "fundamental"
    weight: 0.15
    weight_min: 0.0
    weight_max: 0.3
    description: "估值因子：-(PE_TTM_z + PB_z) / 2"
    industry_neutral: true
    column: "估值评分"
    params:
      pe_range: [0, 200]
      pb_range: [0, 100]
"""


@dataclass
class FactorDef:
    """单个因子定义。"""
    key: str
    name: str
    category: str
    weight: float
    weight_min: float = 0.0
    weight_max: float = 1.0
    description: str = ""
    industry_neutral: bool = True
    column: str = ""
    params: dict[str, Any] = field(default_factory=dict)


class FactorRegistry:
    """因子注册中心 — 集中管理所有因子定义。"""

    _instance: FactorRegistry | None = None
    _config_path: str | None = None

    def __new__(cls, config_path: str | None = None) -> FactorRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: str | None = None) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._factors: dict[str, FactorDef] = {}
        self._config_path = config_path
        self._load_config()

    # ── 加载 ───────────────────────────────────────────────

    def _load_config(self) -> None:
        """从 YAML 文件加载因子定义，文件不存在时使用默认配置。"""
        raw: dict[str, Any] = {}
        if self._config_path and os.path.exists(self._config_path):
            with open(self._config_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
        else:
            raw = yaml.safe_load(DEFAULT_CONFIG) or {}
            # 写出默认配置供用户编辑
            if self._config_path:
                os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
                with open(self._config_path, "w", encoding="utf-8") as f:
                    f.write(DEFAULT_CONFIG)

        factors_raw = raw.get("factors", {})
        for key, cfg in factors_raw.items():
            self._factors[key] = FactorDef(
                key=key,
                name=cfg.get("name", key),
                category=cfg.get("category", ""),
                weight=float(cfg.get("weight", 0)),
                weight_min=float(cfg.get("weight_min", 0.0)),
                weight_max=float(cfg.get("weight_max", 1.0)),
                description=cfg.get("description", ""),
                industry_neutral=bool(cfg.get("industry_neutral", True)),
                column=cfg.get("column", ""),
                params=cfg.get("params", {}),
            )

    def reload(self) -> None:
        """热重载配置（config.ini 变化后调用）。"""
        self._load_config()

    # ── 查询 ───────────────────────────────────────────────

    @property
    def weights(self) -> dict[str, float]:
        """{factor_key: weight} 字典，用于加权融合。"""
        return {k: f.weight for k, f in self._factors.items()}

    @property
    def factor_keys(self) -> list[str]:
        return list(self._factors.keys())

    @property
    def factor_columns(self) -> dict[str, str]:
        """{factor_key: column_name_in_report} 映射。"""
        return {k: f.column for k, f in self._factors.items() if f.column}

    def get(self, key: str) -> FactorDef | None:
        return self._factors.get(key)

    def by_category(self, category: str) -> dict[str, FactorDef]:
        return {k: v for k, v in self._factors.items() if v.category == category}

    @property
    def descriptions(self) -> list[str]:
        return [f"{f.key}: {f.description} (权重={f.weight})" for f in self._factors.values()]

    def validate_weights(self) -> list[str]:
        """校验权重是否在合理范围内，返回警告列表。"""
        warnings: list[str] = []
        for f in self._factors.values():
            if f.weight < f.weight_min or f.weight > f.weight_max:
                warnings.append(
                    f"因子 '{f.key}' 权重 {f.weight} 超出范围 [{f.weight_min}, {f.weight_max}]"
                )
        total = sum(f.weight for f in self._factors.values())
        if abs(total - 1.0) > 0.01:
            warnings.append(f"因子权重之和为 {total:.3f}，建议归一到 1.0")
        return warnings

    def adjust_weight(self, key: str, new_weight: float) -> None:
        """动态调整单因子权重。"""
        f = self._factors.get(key)
        if f is None:
            raise KeyError(f"未知因子: {key}")
        f.weight = max(f.weight_min, min(f.weight_max, new_weight))

    def normalize_weights(self) -> None:
        """权重归一化到总和为 1.0。"""
        total = sum(f.weight for f in self._factors.values())
        if total == 0:
            return
        for f in self._factors.values():
            f.weight /= total
