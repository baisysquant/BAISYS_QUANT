from UtilsManager.CodeNormalizer import CodeNormalizer


def format_stock_code(code: str) -> str:
    """已迁移至 CodeNormalizer.add_market_prefix（保留此别名保持向后兼容）"""
    return CodeNormalizer.add_market_prefix(code)
