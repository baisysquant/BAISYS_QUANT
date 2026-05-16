import pandas as pd
import re


class Parse_Currency:
    """
    专门处理金融数据清洗的工具类
    """

    @staticmethod
    def parse_money_str(val):
        """
        统一处理含有 '亿'、'万'、'%' 或非法字符的金融数据
        """
        # 1. 处理空值
        if pd.isna(val) or val == '' or val == '--' or val == '-':
            return 0.0

        # 2. 转换为字符串并去除首尾空格
        s = str(val).strip()

        try:
            # 3. 处理百分比 (如果是 5.2% 这种格式)
            if '%' in s:
                return float(s.replace('%', '')) / 100.0

            # 4. 处理单位换算
            multiplier = 1.0
            if '亿' in s:
                multiplier = 100000000.0
                s = s.replace('亿', '')
            elif '万' in s:
                multiplier = 10000.0
                s = s.replace('万', '')

            # 5. 最终转换
            return float(s) * multiplier

        except (ValueError, TypeError):
            # 兜底：如果还是转换失败（比如非数字字符串），返回 0
            return 0.0
