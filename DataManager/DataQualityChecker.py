"""
数据质量检查器

在流水线中间步骤执行数据质量校验，提前发现数据异常。
校验项：
  - 股票覆盖率（sync 数量 vs 全市场预期）
  - 关键列空值率
  - 数据新鲜度（trade_date 是否匹配交易日）
  - 评分分布合理性
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text as sql_text


class DataQualityChecker:
    """流水线数据质量检查，每步结果写入 dash_quality_log。"""

    TABLE_NAME = "dash_quality_log"

    # 需要监控空值率的关键列
    KEY_COLUMNS = [
        "股票代码", "股票简称", "行业", "最新价",
        "综合分析评分", "建议仓位比例",
    ]

    def __init__(self, db_engine: Any = None) -> None:
        self._engine = db_engine

    # ── 表结构 ─────────────────────────────────────────────

    def ensure_table(self) -> None:
        if self._engine is None:
            return
        ddl = f"""
        CREATE TABLE IF NOT EXISTS public.{self.TABLE_NAME} (
            id SERIAL PRIMARY KEY,
            trade_date VARCHAR(16) NOT NULL,
            step_name VARCHAR(64) NOT NULL,
            check_name VARCHAR(64) NOT NULL,
            status VARCHAR(16) NOT NULL DEFAULT 'pass',
            metric FLOAT,
            threshold FLOAT,
            detail TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_ql_date
            ON public.{self.TABLE_NAME} (trade_date);
        CREATE INDEX IF NOT EXISTS idx_ql_step
            ON public.{self.TABLE_NAME} (step_name);
        """
        with self._engine.begin() as conn:
            for stmt in ddl.split(";"):
                s = stmt.strip()
                if s:
                    conn.execute(sql_text(s))

    def _log(self, trade_date: str, step_name: str, check_name: str,
             status: str, metric: float = 0.0, threshold: float = 0.0,
             detail: str = "") -> bool:
        if self._engine is None:
            return False
        try:
            sql = sql_text(f"""
            INSERT INTO public.{self.TABLE_NAME}
                (trade_date, step_name, check_name, status, metric, threshold, detail)
            VALUES (:td, :sn, :cn, :st, :m, :th, :dt)
            """)
            with self._engine.begin() as conn:
                conn.execute(sql, {
                    "td": trade_date, "sn": step_name, "cn": check_name,
                    "st": status, "m": float(metric), "th": float(threshold),
                    "dt": detail[:500],
                })
            return True
        except Exception:
            return False

    # ── 检查项 ─────────────────────────────────────────────

    def check_coverage(self, df: pd.DataFrame, trade_date: str,
                       expected_min: int = 3000) -> bool:
        """检查股票覆盖率。"""
        n = len(df)
        status = "pass" if n >= expected_min else "warn"
        detail = f"股票数量: {n}, 预期最低: {expected_min}"
        self._log(trade_date, "数据质量", "股票覆盖率", status, float(n), float(expected_min), detail)
        if status == "warn":
            logger.warning(f"[数据质量] {detail}")
        else:
            logger.info(f"[数据质量] {detail}")
        return status == "pass"

    def check_null_rates(self, df: pd.DataFrame, trade_date: str,
                         max_null_pct: float = 30.0) -> bool:
        """检查关键列空值率。"""
        all_pass = True
        for col in self.KEY_COLUMNS:
            if col not in df.columns:
                self._log(trade_date, "数据质量", f"列缺失:{col}",
                          "warn", 100.0, max_null_pct, f"列 '{col}' 不存在")
                logger.warning(f"[数据质量] 关键列缺失: {col}")
                all_pass = False
                continue
            null_pct = df[col].isna().mean() * 100
            status = "pass" if null_pct <= max_null_pct else "warn"
            detail = f"'{col}' 空值率: {null_pct:.1f}%, 阈值: {max_null_pct}%"
            self._log(trade_date, "数据质量", f"空值率:{col}",
                      status, null_pct, max_null_pct, detail)
            if status != "pass":
                logger.warning(f"[数据质量] {detail}")
                all_pass = False
        return all_pass

    def check_score_distribution(self, df: pd.DataFrame, trade_date: str,
                                 score_col: str = "综合分析评分") -> bool:
        """检查评分分布是否正常（避免全部相同或全部极端值）。"""
        if score_col not in df.columns:
            return True
        scores = pd.to_numeric(df[score_col], errors="coerce").dropna()
        if len(scores) == 0:
            self._log(trade_date, "数据质量", "评分分布", "warn", 0, 0, "评分列为空")
            logger.warning("[数据质量] 评分列为空")
            return False

        unique_pct = scores.nunique() / len(scores) * 100
        all_same = unique_pct < 1.0
        mean_val = scores.mean()
        std_val = scores.std()

        status = "warn" if all_same else "pass"
        detail = f"评分均值={mean_val:.1f}, 标准差={std_val:.2f}, 唯一值比例={unique_pct:.1f}%"
        self._log(trade_date, "数据质量", "评分分布", status, unique_pct, 1.0, detail)
        if all_same:
            logger.warning(f"[数据质量] 评分异常: 全部相同 ({detail})")
        else:
            logger.info(f"[数据质量] {detail}")
        return not all_same

    def check_trade_date_freshness(self, trade_date: str,
                                   df: pd.DataFrame | None = None,
                                   date_col: str = "trade_date") -> bool:
        """检查数据新鲜度（DataFrame 中的日期是否匹配当前交易日）。"""
        if df is not None and date_col in df.columns:
            dates = df[date_col].astype(str).unique()
            if trade_date not in dates:
                detail = f"DataFrame 中无交易日 {trade_date}, 仅有: {sorted(dates)[-3:]}"
                self._log(trade_date, "数据质量", "数据新鲜度", "warn", 0, 0, detail)
                logger.warning(f"[数据质量] {detail}")
                return False
        return True

    def log_suspension_suspects(self, suspects: list[dict[str, Any]],
                                trade_date: str) -> int:
        """把"停牌-疑似漏采"清单写入质量日志（check_name=停牌-疑似漏采）。

        缺失日可能来自数据源漏采而非真实停牌，逐一落库便于 /pipeline/quality-log
        查询与人工复核。

        Args:
            suspects: BackTrading.precheck.suspension_suspects() 输出。
            trade_date: 数据/回测日期。

        Returns:
            写入条数（无引擎时 0）。
        """
        n = 0
        for r in suspects:
            detail = (
                f"{r['symbol']} 缺失占比 {r['ratio']:.2%}（{r['days']}天）"
                f"tail={len(r.get('tail_days') or [])}天 "
                f"interior={len(r.get('interior_days') or [])}天"
            )
            if r.get("cross_validated"):
                detail += (
                    f" 确认停牌={len(r.get('confirmed_days') or [])}天 "
                    f"漏采嫌疑={len(r.get('under_collected_days') or [])}天"
                )
            md = r.get("missing_days") or []
            detail += f" 缺失日={','.join(md[:10])}{'…' if len(md) > 10 else ''}"
            if self._log(trade_date, "数据质量", "停牌-疑似漏采", "warn",
                         float(r.get("ratio", 0.0)), 0.0, detail):
                n += 1
        return n

    # ── 批量运行 ───────────────────────────────────────────

    def run_all(self, df: pd.DataFrame, trade_date: str,
                step_name: str = "数据质量") -> dict[str, Any]:
        """执行所有检查项，返回结果摘要。"""
        self.ensure_table()

        results: dict[str, Any] = {
            "trade_date": trade_date,
            "step_name": step_name,
            "checks": {},
            "all_pass": True,
        }

        checks = [
            ("覆盖率", lambda: self.check_coverage(df, trade_date)),
            ("空值率", lambda: self.check_null_rates(df, trade_date)),
            ("评分分布", lambda: self.check_score_distribution(df, trade_date)),
        ]

        for name, fn in checks:
            ok = fn()
            results["checks"][name] = "pass" if ok else "warn"
            if not ok:
                results["all_pass"] = False

        if results["all_pass"]:
            logger.info(f"[数据质量] 全部 {len(checks)} 项检查通过 ✓")
        else:
            failed = [k for k, v in results["checks"].items() if v != "pass"]
            logger.warning(f"[数据质量] {len(failed)} 项检查未通过: {failed}")

        return results
