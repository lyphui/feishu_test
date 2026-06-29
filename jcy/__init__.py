"""JCY 数据准备流水线包（Step 1-3）。

- config   常量 / 环境变量 / prompt / logging
- store    单一真值源读写、复合键、跳过判断
- feishu   Step 1 飞书采集
- advice   Step 2 Perplexity 投资建议
- extract  Step 3 LLM 结构化提取
- pipeline main 编排
"""
