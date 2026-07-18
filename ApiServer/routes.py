"""
FastAPI 路由 — BAISYS_QUANT 管线 REST API。

提供管线触发、状态查询、因子监控、数据质量等接口。
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel
from sqlalchemy import text as sql_text

router = APIRouter()


# ── 依赖（在 app.py 中注入） ──────────────────────────────

class AppDeps:
    """全局依赖，由 app.py 启动时注入。"""
    db_engine: Any = None
    config: Any = None
    alert_sender: Any = None
    pipeline_task: Any = None  # 后台运行中的管线任务
    pipeline_status: dict[str, Any] = {}


deps = AppDeps()


# ── 请求/响应模型 ─────────────────────────────────────────

class PipelineRunRequest(BaseModel):
    force: bool = False
    pipeline_only: bool = True


class PipelineRunResponse(BaseModel):
    run_id: str
    status: str
    message: str


class AlertConfigRequest(BaseModel):
    webhook_url: str = ""
    alert_on_failure: bool = True
    alert_on_success: bool = False
    alert_channel: str = "generic"


# ── 辅助 ──────────────────────────────────────────────────

def _json_safe(val: Any) -> Any:
    if isinstance(val, (datetime,)):
        return val.isoformat()
    if isinstance(val, (pd.Timestamp,)):
        return val.isoformat()
    return val


def _row_to_dict(row: Any) -> dict[str, Any]:
    return {k: _json_safe(v) for k, v in dict(row._mapping).items()}


# ══════════════════════════════════════════════════════════
# 健康检查
# ══════════════════════════════════════════════════════════


@router.get("/health")
def health_check() -> dict[str, Any]:
    db_ok = False
    if deps.db_engine is not None:
        try:
            with deps.db_engine.connect() as conn:
                conn.execute(sql_text("SELECT 1"))
            db_ok = True
        except Exception:
            pass
    return {
        "status": "ok" if db_ok else "degraded",
        "database": "connected" if db_ok else "disconnected",
        "timestamp": datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════
# 管线执行
# ══════════════════════════════════════════════════════════


def _run_pipeline_in_thread(force: bool = False) -> None:
    """在后台线程中执行管线。"""
    try:
        deps.pipeline_status = {"status": "running", "started_at": datetime.now().isoformat(), "run_id": ""}

        # 导入并运行
        sys.path.insert(0, os.getcwd())
        from Review.coordinator import StockAnalysisCoordinatorFactory

        coordinator = StockAnalysisCoordinatorFactory.create(
            config_file="config.ini",
            force_rerun=force,
        )

        # 获取 run_id（DAG pipeline 会在内部生成）
        deps.pipeline_status["run_id"] = coordinator.dag.run_id if hasattr(coordinator, 'dag') else ""
        deps.pipeline_status["trade_date"] = coordinator.today_str

        # 执行
        coordinator.run()

        deps.pipeline_status["status"] = "completed"
        deps.pipeline_status["finished_at"] = datetime.now().isoformat()

        # 发送告警
        if deps.alert_sender is not None:
            deps.alert_sender.notify_pipeline_result(
                run_id=deps.pipeline_status.get("run_id", ""),
                trade_date=deps.pipeline_status.get("trade_date", ""),
                status="completed",
            )

    except Exception as e:
        deps.pipeline_status["status"] = "failed"
        deps.pipeline_status["error"] = str(e)
        deps.pipeline_status["finished_at"] = datetime.now().isoformat()

        if deps.alert_sender is not None:
            deps.alert_sender.notify_pipeline_result(
                run_id=deps.pipeline_status.get("run_id", ""),
                trade_date=deps.pipeline_status.get("trade_date", ""),
                status="failed",
                summary=str(e),
            )


@router.post("/pipeline/run", response_model=PipelineRunResponse)
def trigger_pipeline(req: PipelineRunRequest) -> PipelineRunResponse:
    """触发管线执行（后台异步运行）。"""
    if deps.pipeline_status.get("status") == "running":
        raise HTTPException(409, detail="管线正在运行中，请等待完成")

    import threading
    t = threading.Thread(target=_run_pipeline_in_thread, args=(req.force,), daemon=True)
    t.start()

    return PipelineRunResponse(
        run_id=deps.pipeline_status.get("run_id", "pending"),
        status="started",
        message="管线已在后台启动",
    )


@router.get("/pipeline/status")
def get_pipeline_status() -> dict[str, Any]:
    """获取当前管线执行状态。"""
    return {
        **deps.pipeline_status,
        "timestamp": datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════
# 运行历史
# ══════════════════════════════════════════════════════════


@router.get("/pipeline/runs")
def list_runs(days: int = 7, limit: int = 20) -> list[dict[str, Any]]:
    """查询最近运行记录。"""
    if deps.db_engine is None:
        raise HTTPException(503, detail="数据库未连接")
    sql = sql_text("""
        SELECT run_id, trade_date, pipeline_name, status,
               started_at, finished_at, duration_seconds,
               config_hash, stock_pool_hash, stock_count, error_message
        FROM dash_run_log
        WHERE started_at >= NOW() - :days::INTERVAL
        ORDER BY started_at DESC
        LIMIT :lim
    """)
    with deps.db_engine.connect() as conn:
        rows = conn.execute(sql, {"days": timedelta(days=days), "lim": limit}).fetchall()
    return [_row_to_dict(r) for r in rows]


@router.get("/pipeline/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    """查询指定运行的详情。"""
    if deps.db_engine is None:
        raise HTTPException(503, detail="数据库未连接")
    sql = sql_text("SELECT * FROM dash_run_log WHERE run_id = :rid")
    with deps.db_engine.connect() as conn:
        row = conn.execute(sql, {"rid": run_id}).fetchone()
    if row is None:
        raise HTTPException(404, detail=f"run_id {run_id} 未找到")
    return _row_to_dict(row)


@router.get("/pipeline/steps/{run_id}")
def list_step_checkpoints(run_id: str) -> list[dict[str, Any]]:
    """查询指定运行的步骤级 checkpoint 详情。"""
    if deps.db_engine is None:
        raise HTTPException(503, detail="数据库未连接")
    sql = sql_text("""
        SELECT step_name, status, started_at, finished_at,
               duration_seconds, error_message
        FROM dash_pipeline_checkpoint
        WHERE run_id = :rid
        ORDER BY id
    """)
    with deps.db_engine.connect() as conn:
        rows = conn.execute(sql, {"rid": run_id}).fetchall()
    if not rows:
        raise HTTPException(404, detail=f"run_id {run_id} 无 checkpoint 记录")
    return [_row_to_dict(r) for r in rows]


# ══════════════════════════════════════════════════════════
# 因子监控
# ══════════════════════════════════════════════════════════


@router.get("/factors/registry")
def get_factor_registry() -> dict[str, Any]:
    """获取因子注册表定义。"""
    try:
        from LogicAnalyzer.scoring.factor_registry import FactorRegistry
        r = FactorRegistry("config/factor_registry.yaml")
        return {
            "factors": [
                {"key": k, "name": f.name, "category": f.category,
                 "weight": f.weight, "description": f.description,
                 "industry_neutral": f.industry_neutral}
                for k, f in r._factors.items()
            ],
            "warnings": r.validate_weights(),
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))


@router.get("/factors/ic-history")
def get_ic_history(days: int = 60, factor: str = "") -> list[dict[str, Any]]:
    """查询因子 IC 历史。"""
    if deps.db_engine is None:
        raise HTTPException(503, detail="数据库未连接")
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    if factor:
        sql = sql_text("""
            SELECT * FROM ods_factor_ic_history
            WHERE check_date >= :since AND factor_name = :fn
            ORDER BY check_date
        """)
        params = {"since": since, "fn": factor}
    else:
        sql = sql_text("""
            SELECT * FROM ods_factor_ic_history
            WHERE check_date >= :since
            ORDER BY check_date, factor_name
        """)
        params = {"since": since}
    with deps.db_engine.connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_dict(r) for r in rows]


# ══════════════════════════════════════════════════════════
# 数据质量
# ══════════════════════════════════════════════════════════


@router.get("/pipeline/quality-log")
def get_quality_log(days: int = 7, status: str = "") -> list[dict[str, Any]]:
    """查询数据质量检查日志。"""
    if deps.db_engine is None:
        raise HTTPException(503, detail="数据库未连接")
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    if status:
        sql = sql_text("""
            SELECT * FROM dash_quality_log
            WHERE created_at >= :since::timestamp AND status = :st
            ORDER BY created_at DESC
        """)
        params = {"since": since, "st": status}
    else:
        sql = sql_text("""
            SELECT * FROM dash_quality_log
            WHERE created_at >= :since::timestamp
            ORDER BY created_at DESC
        """)
        params = {"since": since}
    with deps.db_engine.connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_dict(r) for r in rows]


# ══════════════════════════════════════════════════════════
# 告警配置
# ══════════════════════════════════════════════════════════


@router.get("/alert/config")
def get_alert_config() -> dict[str, Any]:
    """获取当前告警配置。"""
    if deps.alert_sender is None:
        return {"enabled": False}
    return {
        "enabled": deps.alert_sender.enabled,
        "channel": deps.alert_sender._channel,
        "alert_on_failure": deps.alert_sender._alert_on_failure,
        "alert_on_success": deps.alert_sender._alert_on_success,
    }


@router.post("/alert/config")
def update_alert_config(req: AlertConfigRequest) -> dict[str, str]:
    """更新告警配置（运行时动态调整）。"""
    from ApiServer.alert import AlertSender
    deps.alert_sender = AlertSender(
        webhook_url=req.webhook_url,
        alert_on_failure=req.alert_on_failure,
        alert_on_success=req.alert_on_success,
        alert_channel=req.alert_channel,
    )
    return {"status": "updated", "enabled": str(deps.alert_sender.enabled)}


@router.post("/alert/test")
def test_alert() -> dict[str, str]:
    """发送测试告警消息。"""
    if deps.alert_sender is None or not deps.alert_sender.enabled:
        raise HTTPException(400, detail="告警未配置，请先设置 webhook_url")
    ok = deps.alert_sender.send(
        title="BAISYS_QUANT 测试告警",
        message="这是一条来自 API 服务器的测试消息",
        level="info",
    )
    return {"status": "sent" if ok else "failed"}
