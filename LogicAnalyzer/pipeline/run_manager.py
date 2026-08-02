"""
运行版本管理器 — 为每次管线执行分配唯一 run_id，
记录 config_hash、stock_pool_hash 等元数据，
支持多版本对比和结果追溯。
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime
from typing import Any

from loguru import logger
from sqlalchemy import text as sql_text


class RunManager:
    """管线运行版本管理器。

    每次 run() 分配 UUID，记录：
      - run_id：唯一标识
      - timestamp：启动和完成时间
      - config_hash：config.ini 的 SHA256
      - stock_pool_hash：股票池筛选条件的哈希
      - status：running / completed / failed
      - summary：执行摘要
    """

    TABLE_NAME = "dash_run_log"

    def __init__(self, db_engine: Any, config_path: str = "config.ini") -> None:
        self._engine = db_engine
        self._config_path = config_path
        self._run_id: str = ""
        self._ensure_table()

    # ── 属性 ───────────────────────────────────────────────

    @property
    def run_id(self) -> str:
        return self._run_id

    # ── 表 ─────────────────────────────────────────────────

    def _ensure_table(self) -> None:
        if self._engine is None:
            return
        ddl = f"""
        CREATE TABLE IF NOT EXISTS public.{self.TABLE_NAME} (
            id SERIAL PRIMARY KEY,
            run_id VARCHAR(64) NOT NULL UNIQUE,
            trade_date VARCHAR(16) NOT NULL,
            pipeline_name VARCHAR(64) NOT NULL DEFAULT '',
            status VARCHAR(16) NOT NULL DEFAULT 'running',
            started_at TIMESTAMP NOT NULL DEFAULT NOW(),
            finished_at TIMESTAMP,
            duration_seconds FLOAT,
            config_hash VARCHAR(64),
            stock_pool_hash VARCHAR(64),
            stock_count INT DEFAULT 0,
            score_summary JSONB DEFAULT '{{}}'::jsonb,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_runlog_date
            ON public.{self.TABLE_NAME} (trade_date DESC);
        CREATE INDEX IF NOT EXISTS idx_runlog_status
            ON public.{self.TABLE_NAME} (status);
        """
        with self._engine.begin() as conn:
            for stmt in ddl.split(";"):
                s = stmt.strip()
                if s:
                    conn.execute(sql_text(s))

    # ── Hash 计算 ──────────────────────────────────────────

    def _config_hash(self) -> str:
        """计算 config.ini 的 SHA256。"""
        if not os.path.exists(self._config_path):
            return ""
        with open(self._config_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]

    @staticmethod
    def _stock_pool_hash(codes: list[str] | None = None) -> str:
        """计算股票池的哈希（用于追溯筛选条件变化）。"""
        if not codes:
            return ""
        raw = ",".join(sorted(codes))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # ── 生命周期 ───────────────────────────────────────────

    def start(self, trade_date: str, pipeline_name: str = "",
              stock_codes: list[str] | None = None) -> str:
        """启动一次新运行，返回 run_id。"""
        self._run_id = uuid.uuid4().hex[:12]

        config_hash = self._config_hash()
        pool_hash = self._stock_pool_hash(stock_codes)

        if self._engine is not None:
            try:
                sql = sql_text(f"""
                INSERT INTO public.{self.TABLE_NAME}
                    (run_id, trade_date, pipeline_name, status,
                     started_at, config_hash, stock_pool_hash, stock_count)
                VALUES
                    (:rid, :td, :pn, 'running',
                     NOW(), :ch, :ph, :sc)
                """)
                with self._engine.begin() as conn:
                    conn.execute(sql, {
                        "rid": self._run_id,
                        "td": trade_date,
                        "pn": pipeline_name,
                        "ch": config_hash,
                        "ph": pool_hash,
                        "sc": len(stock_codes) if stock_codes else 0,
                    })
            except Exception as e:
                logger.warning(f"[RunManager] 写入 start 记录失败: {e}")

        logger.info(f"[RunManager] 启动 run_id={self._run_id} 交易日={trade_date}")
        return self._run_id

    def finish(self, status: str = "completed",
               error_message: str | None = None) -> None:
        """完成一次运行，记录结束时间和状态。"""
        if self._engine is None or not self._run_id:
            return
        try:
            sql = sql_text(f"""
            UPDATE public.{self.TABLE_NAME}
            SET status = :st,
                finished_at = NOW(),
                duration_seconds = EXTRACT(EPOCH FROM NOW() - started_at),
                error_message = :err
            WHERE run_id = :rid
            """)
            with self._engine.begin() as conn:
                conn.execute(sql, {
                    "rid": self._run_id,
                    "st": status,
                    "err": error_message,
                })
            logger.info(f"[RunManager] 完成 run_id={self._run_id} status={status}")
        except Exception as e:
            logger.warning(f"[RunManager] 写入 finish 记录失败: {e}")


    # ── 查询 ───────────────────────────────────────────────
