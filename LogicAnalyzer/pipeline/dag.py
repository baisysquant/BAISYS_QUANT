from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime
from typing import Any, Callable

import networkx as nx
import pandas as pd
from sqlalchemy import text as sql_text


class PipelineStep:
    """单个流水线步骤定义。"""

    __slots__ = ("name", "fn", "depends_on", "is_fatal", "timeout", "description")

    def __init__(
        self,
        name: str,
        fn: Callable,
        depends_on: list[str] | None = None,
        is_fatal: bool = False,
        timeout: int = 600,
        description: str = "",
    ) -> None:
        self.name = name
        self.fn = fn
        self.depends_on = depends_on or []
        self.is_fatal = is_fatal
        self.timeout = timeout
        self.description = description


class DagPipeline:
    """
    基于 DAG 的流水线引擎，支持断点续跑和版本管理。

    使用 networkx 构建有向无环图，按拓扑序执行步骤。
    每步执行状态持久化到 PostgreSQL，失败后可断点续跑。
    每次运行分配唯一 run_id，记录 config_hash 和股票池哈希。
    """

    CHECKPOINT_TABLE = "dash_pipeline_checkpoint"

    def __init__(
        self,
        name: str,
        db_engine: Any = None,
        cache_dir: str | None = None,
        run_id: str | None = None,
        config_path: str | None = None,
    ) -> None:
        self.name = name
        self.db_engine = db_engine
        self.cache_dir = cache_dir
        self.config_path = config_path
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self._steps: dict[str, PipelineStep] = {}
        self._graph = nx.DiGraph()
        self._results: dict[str, bool] = {}
        self._started_at: datetime | None = None
        self._run_manager: Any = None
        self._intermediate_overrides: dict[str, str] = {}

    # ── 步骤注册 ─────────────────────────────────────────────

    def add_step(self, step: PipelineStep) -> "DagPipeline":
        self._steps[step.name] = step
        self._graph.add_node(step.name)
        for dep in step.depends_on:
            self._graph.add_edge(dep, step.name)
        return self

    def _check_dag(self) -> None:
        if not nx.is_directed_acyclic_graph(self._graph):
            raise ValueError("Pipeline steps contain a cycle!")

    def _get_execution_order(self) -> list[str]:
        return list(nx.topological_sort(self._graph))

    # ── checkpoint 持久化 ─────────────────────────────────────

    def _ensure_checkpoint_table(self) -> None:
        if self.db_engine is None:
            return
        ddl = f"""
        CREATE TABLE IF NOT EXISTS public.{self.CHECKPOINT_TABLE} (
            id SERIAL PRIMARY KEY,
            run_id VARCHAR(64) NOT NULL,
            pipeline_name VARCHAR(64) NOT NULL,
            trade_date VARCHAR(16) NOT NULL,
            step_name VARCHAR(64) NOT NULL,
            status VARCHAR(16) NOT NULL DEFAULT 'pending',
            started_at TIMESTAMP,
            finished_at TIMESTAMP,
            duration_seconds FLOAT,
            error_message TEXT,
            ctx_json JSONB DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMP DEFAULT NOW(),
            UNIQUE (run_id, step_name)
        );
        CREATE INDEX IF NOT EXISTS idx_ckpt_run
            ON public.{self.CHECKPOINT_TABLE} (run_id);
        CREATE INDEX IF NOT EXISTS idx_ckpt_date
            ON public.{self.CHECKPOINT_TABLE} (trade_date);
        CREATE INDEX IF NOT EXISTS idx_ckpt_pname_date
            ON public.{self.CHECKPOINT_TABLE} (pipeline_name, trade_date);
        """
        with self.db_engine.begin() as conn:
            for stmt in ddl.split(";"):
                s = stmt.strip()
                if s:
                    conn.execute(sql_text(s))

    def _get_checkpoints(self, trade_date: str) -> dict[str, dict[str, Any]]:
        """加载该交易日当前 run_id 的所有 checkpoint。"""
        if self.db_engine is None:
            return {}
        sql = sql_text(f"""
        SELECT step_name, status, error_message, ctx_json
        FROM public.{self.CHECKPOINT_TABLE}
        WHERE pipeline_name = :pname AND trade_date = :tdate
          AND run_id = :run_id
        ORDER BY id
        """)
        with self.db_engine.connect() as conn:
            rows = conn.execute(sql, {"pname": self.name, "tdate": trade_date, "run_id": self.run_id}).fetchall()
        cp: dict[str, dict[str, Any]] = {}
        for r in rows:
            cp[r[0]] = {
                "status": r[1],
                "error": r[2],
                "ctx_json": r[3] or {},
            }
        return cp

    def _save_checkpoint(
        self,
        step_name: str,
        status: str,
        trade_date: str,
        error: str | None = None,
        duration: float | None = None,
        ctx_json: dict[str, Any] | None = None,
    ) -> None:
        if self.db_engine is None:
            return
        import json

        sql = sql_text(f"""
        INSERT INTO public.{self.CHECKPOINT_TABLE}
            (run_id, pipeline_name, trade_date, step_name, status,
             started_at, finished_at, duration_seconds, error_message, ctx_json)
        VALUES
            (:run_id, :pname, :tdate, :sname, :status,
             :started, :finished, :dur, :err, CAST(:ctx_json AS jsonb))
        ON CONFLICT (run_id, step_name) DO UPDATE SET
            status = EXCLUDED.status,
            finished_at = EXCLUDED.finished_at,
            duration_seconds = EXCLUDED.duration_seconds,
            error_message = EXCLUDED.error_message,
            ctx_json = CASE WHEN CAST(EXCLUDED.ctx_json AS jsonb) != '{{}}'::jsonb
                            THEN CAST(EXCLUDED.ctx_json AS jsonb) ELSE {self.CHECKPOINT_TABLE}.ctx_json END
        """)
        now = datetime.now()
        with self.db_engine.begin() as conn:
            conn.execute(
                sql,
                {
                    "run_id": self.run_id,
                    "pname": self.name,
                    "tdate": trade_date,
                    "sname": step_name,
                    "status": status,
                    "started": self._started_at,
                    "finished": now,
                    "dur": duration,
                    "err": error,
                    "ctx_json": json.dumps(ctx_json or {}),
                },
            )

    # ── 中间数据持久化（parquet） ────────────────────────────

    def _save_intermediate(self, step_name: str, ctx: Any, trade_date: str) -> None:
        """保存对恢复有用的中间数据。"""
        if self.cache_dir is None:
            return
        safe = step_name.replace(" ", "_")
        key = self._intermediate_key(step_name)
        if key and ctx.has(key):
            df = ctx.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                path = os.path.join(self.cache_dir, f"ckpt_{safe}_{trade_date}.parquet")
                df = df.replace(r'^\s*$', pd.NA, regex=True)
                df.to_parquet(path, index=False)
        # 保存上下文，供跨进程恢复使用
        ctx_path = os.path.join(self.cache_dir, f"ctx_{safe}_{trade_date}.json")
        try:
            save_data = {}
            for k, v in ctx.data.items():
                if not isinstance(v, pd.DataFrame):
                    try:
                        json.dumps({k: str(v)[:200]})
                        save_data[k] = str(v)
                    except Exception:
                        save_data[k] = f"<{type(v).__name__}>"
            with open(ctx_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load_intermediate(self, step_name: str, trade_date: str) -> pd.DataFrame | None:
        if self.cache_dir is None:
            return None
        safe = step_name.replace(" ", "_")
        path = os.path.join(self.cache_dir, f"ckpt_{safe}_{trade_date}.parquet")
        if os.path.exists(path):
            try:
                return pd.read_parquet(path)
            except Exception:
                return None
        return None

    def register_intermediate(self, step_name: str, ctx_key: str) -> None:
        """运行时注册中间数据键。"""
        self._intermediate_overrides[step_name] = ctx_key

    def _intermediate_key(self, step_name: str) -> str | None:
        mapping = {
            "合并处理数据": "consolidated_report",
            "多因子Alpha评分": "consolidated_report",
            "组合构建": "consolidated_report",
        }
        if step_name in mapping:
            return mapping[step_name]
        return self._intermediate_overrides.get(step_name)

    # ── 核心执行逻辑 ─────────────────────────────────────────

    def run(
        self,
        ctx: Any,
        trade_date: str,
        force_rerun: bool = False,
    ) -> bool:
        """执行流水线，支持断点续跑。

        Args:
            ctx: PipelineContext 实例。
            trade_date: 交易日字符串（YYYY-MM-DD）。
            force_rerun: 强制重跑所有步骤。

        Returns:
            True 表示所有步骤执行成功，False 表示有致命步骤失败。
        """
        self._check_dag()
        self._ensure_checkpoint_table()
        self._started_at = datetime.now()

        # 初始化运行版本管理器
        if self.db_engine is not None:
            from LogicAnalyzer.pipeline.run_manager import RunManager
            self._run_manager = RunManager(self.db_engine, config_path=self.config_path or "")
            stock_codes: list[str] = ctx.get("stock_codes_pure", [])
            self.run_id = self._run_manager.start(trade_date, self.name, stock_codes)

        checkpoint = {} if force_rerun else self._get_checkpoints(trade_date)
        execution_order = self._get_execution_order()
        completed_count = sum(1 for v in checkpoint.values() if v["status"] == "completed")

        print(f"\n  DAG 流水线: {self.name} | 交易日: {trade_date} | run_id: {self.run_id}")
        print(f"  步骤数: {len(execution_order)}, 已完成: {completed_count}")
        if force_rerun:
            print(f"  强制重跑: 所有步骤重新执行")

        overall_ok = True
        for step_name in execution_order:
            step = self._steps[step_name]

            # 前置依赖检查
            missing = [d for d in step.depends_on if d not in self._results]
            if missing:
                print(f"  [{step_name}] 前置依赖未完成: {missing}，跳过")
                continue

            # 断点续跑：已完成且数据可恢复的直接跳过
            cp = checkpoint.get(step_name, {})
            if cp.get("status") == "completed" and not force_rerun:
                key = self._intermediate_key(step_name)
                if key:
                    restored = self._load_intermediate(step_name, trade_date)
                    if restored is not None:
                        ctx.set(key, restored)
                        print(f"  ✓ {step_name} (从 checkpoint 恢复)", flush=True)
                        self._results[step_name] = True
                        continue
                # 非数据生产者 -> 检查上下文是否完备，否则重跑
                if key is None:
                    if ctx.data:
                        print(f"  ✓ {step_name} (已跳过)", flush=True)
                        self._results[step_name] = True
                        continue
                    # 跨进程恢复：ctx 为空但 checkpoint 存在 → 需要重跑
                    print(f"  [{step_name}] 上下文不完整，重新执行")
                else:
                    # 数据生产者但无缓存 -> 需要重跑
                    print(f"  [{step_name}] checkpoint 存在但数据未缓存，重新执行")

            # 执行步骤
            idx = execution_order.index(step_name) + 1
            total = len(execution_order)
            print(f"  [{idx}/{total}] {step_name}...", end="", flush=True)
            step_start = time.time()

            try:
                ok = step.fn(ctx)
                elapsed = time.time() - step_start

                if ok:
                    print(f" ✓ ({elapsed:.1f}s)", flush=True)
                    self._save_checkpoint(step_name, "completed", trade_date, duration=elapsed)
                    self._save_intermediate(step_name, ctx, trade_date)
                    self._results[step_name] = True
                else:
                    print(f" ✗ ({elapsed:.1f}s)", flush=True)
                    self._save_checkpoint(
                        step_name, "failed", trade_date, error="Step returned False", duration=elapsed
                    )
                    self._results[step_name] = False
                    if step.is_fatal:
                        print(f"\n  ⚠ 致命步骤失败，流水线终止。")
                        overall_ok = False
                        break

            except Exception as e:
                elapsed = time.time() - step_start
                err_msg = f"{type(e).__name__}: {e}"
                print(f" ✗ ({elapsed:.1f}s) {err_msg}", flush=True)
                self._save_checkpoint(step_name, "failed", trade_date, error=err_msg, duration=elapsed)
                self._results[step_name] = False
                if step.is_fatal:
                    print(f"\n  ⚠ 致命步骤失败，流水线终止。")
                    overall_ok = False
                    break

        # 记录运行版本
        if self._run_manager is not None:
            error_msgs = [v.get("error", "") for v in checkpoint.values() if v.get("status") == "failed"]
            status = "completed" if overall_ok else "failed"
            self._run_manager.finish(status=status, error_message="; ".join(error_msgs) if error_msgs else None)

        return overall_ok
