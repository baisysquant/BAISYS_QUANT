# LoggerManager.py

import logging
import os
import sys
from datetime import datetime
from typing import Optional


class LoggerManager:
    """
    专业日志管理器，支持控制台 + 文件双输出，多级别控制，线程安全。
    专为量化分析系统设计，提供统一、清晰、可配置的日志输出。
    """

    _instance = None  # 单例模式，避免重复初始化

    def __new__(cls, log_dir: str = None, log_filename: str = None, level: str = "INFO"):
        if cls._instance is None:
            cls._instance = super(LoggerManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, log_dir: str = None, log_filename: str = None, level: str = "INFO"):
        if self._initialized:
            return

        # 配置参数
        self.log_dir = log_dir or os.path.join(os.path.expanduser("~"), "Downloads", "CoreNews_Reports", "Logs")
        self.log_filename = log_filename or f"Corenews_Main_{datetime.now().strftime('%Y%m%d')}.log"
        self.log_path = os.path.join(self.log_dir, self.log_filename)

        # 确保日志目录存在
        os.makedirs(self.log_dir, exist_ok=True)

        # 设置日志级别映射
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        self.level = level_map.get(level.upper(), logging.INFO)

        # 创建 logger
        self.logger = logging.getLogger("Corenews_Main")
        self.logger.setLevel(self.level)
        self.logger.propagate = False  # 防止重复输出到 root logger

        # 清除已有处理器（避免重复）
        self.logger.handlers.clear()

        # 创建格式化器
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # 控制台处理器（INFO及以上）
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # 文件处理器（所有级别）
        file_handler = logging.FileHandler(self.log_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self._initialized = True

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def critical(self, message: str):
        self.logger.critical(message)

    def debug(self, message: str):
        self.logger.debug(message)

    def get_log_path(self) -> str:
        return self.log_path