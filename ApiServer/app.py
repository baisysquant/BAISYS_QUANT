"""
BAISYS_QUANT FastAPI 应用入口。

启动方式：
    python -m ApiServer.app [--port 8000] [--reload]

或：
    uvicorn ApiServer.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

app = FastAPI(
    title="BAISYS_QUANT API",
    description="A股量化复盘分析系统 REST API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS：允许前后端分离
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def init_deps(config_path: str = "config.ini",
              webhook_url: str = "",
              alert_channel: str = "generic",
              alert_on_failure: bool = True,
              alert_on_success: bool = False) -> None:
    """初始化全局依赖（数据库引擎、告警发送器等）。"""
    from ApiServer.routes import deps
    from ApiServer.alert import AlertSender
    from UtilsManager.ConfigParser import Config
    from DataManager.DbEngine import get_engine

    # 配置
    cfg = Config(config_file=config_path)
    deps.config = cfg

    # 数据库
    try:
        deps.db_engine = get_engine(cfg)
        logger.info("[API] 数据库连接成功")
    except Exception as e:
        logger.warning(f"[API] 数据库连接失败: {e}")
        deps.db_engine = None

    # 告警
    api_cfg = getattr(cfg, "app_config", None)
    api_cfg = getattr(api_cfg, "api", None) if api_cfg else None
    wh = webhook_url or (api_cfg.ALERT_WEBHOOK_URL if api_cfg else "")
    ac = alert_channel if webhook_url else (api_cfg.ALERT_CHANNEL if api_cfg else "generic")
    aof = alert_on_failure if webhook_url else (api_cfg.ALERT_ON_FAILURE if api_cfg else True)
    aos = alert_on_success if webhook_url else (api_cfg.ALERT_ON_SUCCESS if api_cfg else False)
    deps.alert_sender = AlertSender(
        webhook_url=wh,
        alert_on_failure=aof,
        alert_on_success=aos,
        alert_channel=ac,
    )
    if deps.alert_sender.enabled:
        logger.info(f"[API] 告警已配置: {alert_channel}")
    else:
        logger.info("[API] 告警未配置")


# 启动时自动初始化
@app.on_event("startup")
def startup() -> None:
    # 确保工作目录是项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    os.chdir(project_root)

    from ApiServer.routes import router
    app.include_router(router, prefix="/api/v1")

    init_deps()
    logger.info("[API] BAISYS_QUANT API 服务已启动")


@app.on_event("shutdown")
def shutdown() -> None:
    from ApiServer.routes import deps
    if deps.pipeline_status.get("status") == "running":
        logger.warning("[API] 服务关闭，后台管线任务可能中断")
    logger.info("[API] BAISYS_QUANT API 服务已停止")


# ── 命令行入口 ────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="BAISYS_QUANT API 服务")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="监听地址")
    parser.add_argument("--reload", action="store_true", help="热重载（开发模式）")
    parser.add_argument("--config", type=str, default="config.ini", help="配置文件路径")
    parser.add_argument("--webhook", type=str, default="", help="告警 Webhook URL")
    parser.add_argument("--alert-channel", type=str, default="generic",
                        choices=["generic", "wecom", "feishu", "dingtalk"],
                        help="告警渠道")
    args = parser.parse_args()

    # 初始化后再启动 uvicorn
    init_deps(
        config_path=args.config,
        webhook_url=args.webhook,
        alert_channel=args.alert_channel,
    )

    import uvicorn
    uvicorn.run(
        "ApiServer.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
