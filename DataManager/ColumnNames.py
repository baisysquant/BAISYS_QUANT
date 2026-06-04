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
    MACD_FULL_BULL_LABEL = "MACD_FULL_BULL_Label"
    MACD_FULL_BULL_SIGNAL = "MACD_FULL_BULL_Signals"
    FULL_BULL_SCORE = "FullBull_Score"
    FULL_BULL_SCORE_BASE = "FullBull_Score_Base"

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
    KLINE_PATTERN_SIGNAL = "K线形态信号"

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

    # 研报相关
    RESEARCH_REPORT_COUNT = "研报买入次数"
