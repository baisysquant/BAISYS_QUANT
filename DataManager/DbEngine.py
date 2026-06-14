"""
单例数据库引擎提供者

统一管理 SQLAlchemy Engine 生命周期，全局仅创建一个连接池。
所有需要数据库访问的模块通过 get_engine(config) 获取共享引擎。
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import URL, create_engine
from sqlalchemy.engine import Engine

_engine: Engine | None = None


def get_engine(config: Any) -> Engine:  # noqa: ANN401
    """获取全局单例数据库引擎（惰性创建）。"""
    global _engine
    if _engine is None:
        url_object = URL.create(
            "postgresql+psycopg2",
            username=config.DB_USER,
            password=config.DB_PASSWORD,
            host=config.DB_HOST,
            port=config.DB_PORT,
            database=config.DB_NAME,
        )
        _engine = create_engine(
            url_object,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False,
            client_encoding="utf8",
        )
    return _engine


def dispose_engine() -> None:
    """释放全局连接池（仅在进程退出时调用）。"""
    global _engine
    if _engine is not None:
        _engine.dispose()
        _engine = None
