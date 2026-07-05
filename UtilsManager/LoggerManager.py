
"""
专业日志管理器（基于 Loguru）
支持控制台 + 文件双输出，多级别控制，线程/进程安全，日志自动轮转
"""

from __future__ import annotations

import os
import sys
import threading
from datetime import datetime
from typing import Any

from loguru import logger


def setup_logger(log_dir: str | None = None, log_filename: str | None = None, level: str = "INFO") -> loguru.Logger:
    """
    配置并初始化 Loguru 日志系统

    Args:
        log_dir: 日志目录
        log_filename: 日志文件名
        level: 日志级别 (DEBUG/INFO/WARNING/ERROR/CRITICAL)

    Returns:
        logger: Loguru logger 实例
    """
    # 配置参数
    log_dir = log_dir or os.path.join(os.path.expanduser("~"), "Downloads", "CoreNews_Reports", "Logs")
    if log_filename is None:
        try:
            from DataCollection.CalendarManager import TradingCalendarAnalyzer
            _trade_day = TradingCalendarAnalyzer().get_last_trading_day().replace("-", "")
        except Exception:
            _trade_day = datetime.now().strftime('%Y%m%d')
        log_filename = f"Corenews_Main_{_trade_day}.log"
    log_path = os.path.join(log_dir, log_filename)

    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)

    # 日志级别
    log_level = level.upper()

    # 先移除默认的 handler
    logger.remove()

    # 定义格式函数 - 根据级别显示不同的信息
    def format_console(record: Any) -> str:  # noqa: ANN401
        if record["level"].name in ["ERROR", "CRITICAL"]:
            return (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>\n"
            )
        else:
            return (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level}</level> | "
                "<level>{message}</level>\n"
            )

    def format_file(record: Any) -> str:  # noqa: ANN401
        if record["level"].name in ["ERROR", "CRITICAL"]:
            return "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}\n"
        else:
            return "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}\n"

    # 添加控制台 handler
    logger.add(
        sys.stdout,
        format=format_console,
        level=log_level,
        colorize=True,
        enqueue=True,
    )

    # 手动轮转旧日志（避免自动轮转时 Windows 文件锁冲突）
    if os.path.exists(log_path):
        try:
            base, ext = os.path.splitext(log_path)
            rotated_name = f"{base}.{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}{ext}"
            os.rename(log_path, rotated_name)
        except PermissionError:
            pass

    # 添加文件 handler
    logger.add(
        log_path,
        format=format_file,
        level="DEBUG",
        encoding="utf-8",
        enqueue=True,
    )

    # 清理 30 天前的旧日志
    try:
        now = datetime.now()
        for f in os.listdir(log_dir):
            fpath = os.path.join(log_dir, f)
            if os.path.isfile(fpath) and f != os.path.basename(log_path):
                age = now - datetime.fromtimestamp(os.path.getmtime(fpath))
                if age.days > 30:
                    os.remove(fpath)
    except Exception:
        pass

    return logger


# 全局 logger 实例（单例模式，线程安全）
_global_logger = None
_global_log_path = None
_init_lock = threading.Lock()


def get_logger(log_dir: str | None = None, log_filename: str | None = None, level: str = "INFO") -> loguru.Logger:
    """
    获取全局 logger 实例（单例模式，线程安全）

    Args:
        log_dir: 日志目录
        log_filename: 日志文件名
        level: 日志级别

    Returns:
        logger: Loguru logger 实例
    """
    global _global_logger
    global _global_log_path
    if _global_logger is None:
        with _init_lock:
            if _global_logger is None:
                _global_logger = setup_logger(log_dir, log_filename, level)
                _global_log_path = os.path.join(
                    log_dir or os.path.join(os.path.expanduser("~"), "Downloads", "CoreNews_Reports", "Logs"),
                    log_filename or f"Corenews_Main_{datetime.now().strftime('%Y%m%d')}.log",
                )
    return _global_logger


def get_log_path() -> str | None:
    """获取当前日志文件路径"""
    global _global_log_path
    return _global_log_path or ""

