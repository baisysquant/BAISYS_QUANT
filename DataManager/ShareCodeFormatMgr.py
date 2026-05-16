# FormatManager.py

def format_stock_code(code: str) -> str:
    """
    根据股票代码的开头数字，添加市场前缀。
    兼容 '000001' 和 'sz000001' 两种输入，并统一输出 'sz000001' 格式。
    """
    code_str = str(code)

    if code_str.startswith(('sh', 'sz', 'bj')):
        return code_str

    code_str = code_str.zfill(6)
    if code_str.startswith('6'):
        return 'sh' + code_str
    elif code_str.startswith(('0', '3')):
        return 'sz' + code_str
    elif code_str.startswith(('4', '8', '9')):
        return 'bj' + code_str
    return code_str