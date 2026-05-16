"""
列名常量定义

统一管理所有DataFrame列名，避免魔法字符串散落在代码中。
"""


class ColumnNames:
    """
    股票分析报告列名常量
    
    按功能分组：
    - 基础信息列
    - 技术指标列
    - 资金流列
    - 信号列
    - 主力成本列
    - 均线突破列
    """
    
    # ==================== 基础信息列 ====================
    STOCK_CODE = "股票代码"
    STOCK_NAME = "股票简称"
    INDUSTRY = "行业"
    INDUSTRY_SIGNAL = "所属行业信号"
    LATEST_PRICE = "最新价"
    STOCK_LINK = "股票链接"
    
    # akshare原始列名（数据清洗阶段使用）
    AKSHARE_CODE_RAW = "代码"  # 主力成本等接口返回的原始列名
    AKSHARE_INDUSTRY_BOARD_NAME = "板块名称"
    AKSHARE_INDUSTRY_BOARD_CODE = "板块代码"
    
    # ==================== 技术指标列 - MACD ====================
    MACD_12269 = "MACD_12269"
    MACD_12269_DIF = "MACD_12269_DIF"
    MACD_12269_MOMENTUM = "MACD_12269_动能"
    
    MACD_COMBINED_DIVERGENCE = "MACD_组合背离"
    
    # MACD第二周期（动态生成，这里提供模板）
    # 格式: f"MACD_{fast}{slow}{signal}"
    # 例如: MACD_5345, MACD_215526 等
    MACD_SECOND_TEMPLATE = "MACD_{}"
    MACD_SECOND_DIF_TEMPLATE = "MACD_{}_DIF"
    MACD_SECOND_MOMENTUM_TEMPLATE = "MACD_{}_动能"
    
    # ==================== 技术指标列 - 其他 ====================
    KDJ_SIGNAL = "KDJ_Signal"
    CCI_SIGNAL = "CCI_Signal"
    RSI_SIGNAL = "RSI_Signal"
    BOLL_SIGNAL = "BOLL_Signal"
    
    # ==================== 资金流列 ====================
    # 注意：单位统一为"万元"
    FUND_FLOW_3D = "3日资金流入万元"
    FUND_FLOW_5D = "5日资金流入万元"
    FUND_FLOW_10D = "10日资金流入万元"
    FUND_FLOW_20D = "20日资金流入万元"
    
    # 资金流动能
    FUND_MOMENTUM = "资金动能"
    FUND_MOMENTUM_SCORE = "资金动能评分"
    
    # ==================== 信号列 ====================
    STRONG_STOCK = "强势股"
    PRICE_VOLUME_RISE = "量价齐升"
    CONSECUTIVE_RISE_DAYS = "连涨天数"
    VOLUME_INCREASE_DAYS = "放量天数"
    TOP10_INDUSTRY = "TOP10行业"
    
    # ==================== 主力成本列 ====================
    MAIN_COST = "主力成本"
    MAIN_COST_DIFF = "主力成本差价"
    COST_POSITION = "成本位置"
    MAIN_CONTROL_STRENGTH = "主力控盘强度"
    INSTITUTION_PARTICIPATION = "机构参与度"
    INSTITUTION_LEVEL = "机构参与度等级"
    MAIN_COST_DIFF_PERCENT = "主力成本差价百分比"
    
    # ==================== 均线突破列 ====================
    PERFECT_BULL_ARRANGEMENT = "完全多头排列"
    CURRENT_PRICE = "当前价格"
    MA10_PRICE = "10日均线价"
    MA30_PRICE = "30日均线价"
    MA60_PRICE = "60日均线价"
    
    # ==================== 趋势评分列 ====================
    BULL_TREND = "多头排列趋势"
    FULL_BULL_SCORE = "FullBull_Score"  # 内部使用，可能不在最终报告中
    
    # ==================== akshare原始列名映射 ====================
    # akshare接口返回的原始列名（用于数据提取阶段）
    AKSHARE_NET_FLOW = "净流入"
    AKSHARE_FUND_FLOW_NET = "资金流入净额"
    AKSHARE_MAIN_NET_FLOW = "今日主力净流入-净额"
    
    # akshare资金流原始列名候选列表
    AKSHARE_FLOW_COLUMNS = [
        AKSHARE_NET_FLOW,
        AKSHARE_FUND_FLOW_NET,
        AKSHARE_MAIN_NET_FLOW,
    ]
    
    # ==================== 列分组（用于报告生成）====================
    @classmethod
    def get_base_columns(cls) -> list:
        """获取基础信息列"""
        return [
            cls.STOCK_CODE,
            cls.STOCK_NAME,
            cls.INDUSTRY,
            cls.INDUSTRY_SIGNAL,
            cls.LATEST_PRICE,
            cls.MAIN_COST,
            cls.MAIN_COST_DIFF,
            cls.COST_POSITION,
            cls.MAIN_CONTROL_STRENGTH,
        ]
    
    @classmethod
    def get_signal_columns(cls, second_period_name: str = None) -> list:
        """
        获取信号列
        
        Args:
            second_period_name: 第二周期名称（如"5345"），如果为None则不包含第二周期列
            
        Returns:
            信号列列表
        """
        cols = [
            cls.STRONG_STOCK,
            cls.PRICE_VOLUME_RISE,
            cls.CONSECUTIVE_RISE_DAYS,
            cls.VOLUME_INCREASE_DAYS,
            cls.TOP10_INDUSTRY,
            cls.MACD_12269,
            cls.MACD_12269_MOMENTUM,
            cls.MACD_12269_DIF,
        ]
        
        if second_period_name:
            cols.extend([
                cls.MACD_SECOND_TEMPLATE.format(second_period_name),
                cls.MACD_SECOND_MOMENTUM_TEMPLATE.format(second_period_name),
                cls.MACD_SECOND_DIF_TEMPLATE.format(second_period_name),
            ])
        
        cols.extend([
            cls.MACD_COMBINED_DIVERGENCE,
            cls.KDJ_SIGNAL,
            cls.CCI_SIGNAL,
            cls.RSI_SIGNAL,
            cls.BOLL_SIGNAL,
        ])
        
        return cols
    
    @classmethod
    def get_report_columns(cls, fund_flow_periods: list = None) -> list:
        """
        获取报告列（趋势、动能、资金流）
        
        Args:
            fund_flow_periods: 资金流周期列表，如 [5, 10, 20]
            
        Returns:
            报告列列表
        """
        cols = [
            cls.BULL_TREND,
            cls.FUND_MOMENTUM,
        ]
        
        if fund_flow_periods:
            period_map = {
                3: cls.FUND_FLOW_3D,
                5: cls.FUND_FLOW_5D,
                10: cls.FUND_FLOW_10D,
                20: cls.FUND_FLOW_20D,
            }
            for period in fund_flow_periods:
                if period in period_map:
                    cols.append(period_map[period])
        
        return cols
    
    @classmethod
    def get_all_technical_signal_columns(cls, second_period_name: str = None) -> list:
        """
        获取所有技术指标信号列（用于筛选有信号的股票）
        
        Args:
            second_period_name: 第二周期名称
            
        Returns:
            技术指标信号列列表
        """
        cols = [
            cls.MACD_12269,
            cls.MACD_COMBINED_DIVERGENCE,
            cls.KDJ_SIGNAL,
            cls.CCI_SIGNAL,
            cls.RSI_SIGNAL,
            cls.BOLL_SIGNAL,
        ]
        
        if second_period_name:
            cols.append(cls.MACD_SECOND_TEMPLATE.format(second_period_name))
        
        return cols
    
    @classmethod
    def get_final_column_order(cls, second_period_name: str = None, 
                                fund_flow_periods: list = None) -> list:
        """
        获取最终报告的列顺序
        
        Args:
            second_period_name: 第二周期名称
            fund_flow_periods: 资金流周期列表
            
        Returns:
            完整的列顺序列表
        """
        base_cols = cls.get_base_columns()
        signal_cols = cls.get_signal_columns(second_period_name)
        report_cols = cls.get_report_columns(fund_flow_periods)
        
        return base_cols + signal_cols + report_cols + [cls.STOCK_LINK]
