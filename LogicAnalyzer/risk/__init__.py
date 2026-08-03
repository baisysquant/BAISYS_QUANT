"""
风险分析与业绩归因模块

提供组合层面的风险度量与收益归因工具：
  - HistoricalVaR         —— 历史模拟法 VaR(95%/99%) / ES（期望损失），组合日频
  - FactorRiskModel        —— 正交化因子暴露风险归因（因子风险 vs 个股特质风险）
  - BrinsonDecomposition   —— Brinson 行业配置 / 个股选择收益归因，支持 Excel 输出
"""
