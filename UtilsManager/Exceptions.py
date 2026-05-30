"""
量化系统自定义异常类

提供分层的异常体系，便于统一处理和错误恢复。
"""


class QuantBaseException(Exception):
    """量化系统基础异常类"""

    def __init__(self, message: str, error_code: str = "UNKNOWN", recoverable: bool = False):
        """
        Args:
            message: 错误描述信息
            error_code: 错误代码（便于日志追踪和问题定位）
            recoverable: 是否可恢复（True=可以继续执行，False=需要终止流程）
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.recoverable = recoverable

    def __str__(self):
        return f"[{self.error_code}] {self.message}"


# ==================== 数据获取层异常 ====================


class DataFetchError(QuantBaseException):
    """数据获取失败异常"""

    def __init__(self, source: str, message: str):
        super().__init__(
            message=f"从 {source} 获取数据失败: {message}",
            error_code="DATA_FETCH_ERROR",
            recoverable=True,  # 单个数据源失败不影响整体流程
        )
        self.source = source


class DataValidationError(QuantBaseException):
    """数据验证失败异常"""

    def __init__(self, data_name: str, reason: str):
        super().__init__(
            message=f"数据验证失败 [{data_name}]: {reason}", error_code="DATA_VALIDATION_ERROR", recoverable=True
        )
        self.data_name = data_name


class CacheError(QuantBaseException):
    """缓存操作异常"""

    def __init__(self, operation: str, message: str):
        super().__init__(
            message=f"缓存{operation}失败: {message}",
            error_code="CACHE_ERROR",
            recoverable=True,  # 缓存失败可以降级为重新获取
        )
        self.operation = operation


# ==================== 数据处理层异常 ====================


class DataProcessingError(QuantBaseException):
    """数据处理异常"""

    def __init__(self, step: str, message: str):
        super().__init__(
            message=f"数据处理步骤 [{step}] 失败: {message}",
            error_code="DATA_PROCESSING_ERROR",
            recoverable=False,  # 数据处理失败通常不可恢复
        )
        self.step = step


class CalculationError(QuantBaseException):
    """计算异常（如指标计算、评分计算等）"""

    def __init__(self, calculation_type: str, message: str):
        super().__init__(
            message=f"{calculation_type} 计算失败: {message}",
            error_code="CALCULATION_ERROR",
            recoverable=True,  # 单个股票计算失败不影响其他股票
        )
        self.calculation_type = calculation_type


# ==================== 配置层异常 ====================


class ConfigError(QuantBaseException):
    """配置错误异常"""

    def __init__(self, config_key: str, message: str):
        super().__init__(
            message=f"配置项 [{config_key}] 错误: {message}",
            error_code="CONFIG_ERROR",
            recoverable=False,  # 配置错误必须修复后才能继续
        )
        self.config_key = config_key


# ==================== 数据库层异常 ====================


class DatabaseError(QuantBaseException):
    """数据库操作异常"""

    def __init__(self, operation: str, message: str):
        super().__init__(
            message=f"数据库{operation}失败: {message}",
            error_code="DATABASE_ERROR",
            recoverable=False,  # 数据库失败通常需要终止
        )
        self.operation = operation


class DatabaseConnectionError(DatabaseError):
    """数据库连接异常"""

    def __init__(self, message: str):
        super().__init__(operation="连接", message=message)
        self.error_code = "DB_CONNECTION_ERROR"


# ==================== 报告生成层异常 ====================


class ReportGenerationError(QuantBaseException):
    """报告生成异常"""

    def __init__(self, report_type: str, message: str):
        super().__init__(
            message=f"生成{report_type}报告失败: {message}", error_code="REPORT_GENERATION_ERROR", recoverable=False
        )
        self.report_type = report_type


# ==================== 工具函数 ====================


def handle_exception_with_recovery(
    exception: Exception, logger, context: str, default_value=None, raise_on_critical: bool = True
):
    """
    统一的异常处理工具函数

    Args:
        exception: 捕获的异常对象
        logger: 日志记录器
        context: 异常发生的上下文描述
        default_value: 可恢复时的默认返回值
        raise_on_critical: 对于不可恢复异常是否重新抛出

    Returns:
        如果异常可恢复，返回 default_value
        如果异常不可恢复且 raise_on_critical=True，重新抛出异常
        否则返回 None

    Example:
        try:
            data = fetch_data()
        except Exception as e:
            result = handle_exception_with_recovery(
                e, logger, "获取资金流数据",
                default_value=pd.DataFrame(),
                raise_on_critical=False
            )
            if result is None:
                return  # 终止当前流程
            data = result
    """
    if isinstance(exception, QuantBaseException):
        if exception.recoverable:
            logger.warning(f"[{context}] 可恢复错误: {exception}")
            return default_value
        else:
            logger.error(f"[{context}] 不可恢复错误: {exception}")
            if raise_on_critical:
                raise
            return None
    else:
        # 未知异常，默认为不可恢复
        logger.critical(f"[{context}] 未预期的异常: {type(exception).__name__}: {exception}")
        if raise_on_critical:
            raise
        return None
