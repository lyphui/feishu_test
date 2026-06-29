"""JCY 数据准备一体化流水线入口（Step 1-3）。

实现已拆分到 `jcy/` 包：
- jcy.config   常量 / 环境变量 / prompt / logging
- jcy.store    单一真值源读写、复合键、跳过判断
- jcy.feishu   Step 1 飞书采集
- jcy.advice   Step 2 Perplexity 投资建议
- jcy.extract  Step 3 LLM 结构化提取
- jcy.pipeline main 编排

本文件保留为薄入口，便于 `python prepare_jcy_data.py` 直接运行。

超时保护：Step 2/3 中 API 连续超时/失败 MAX_CONSECUTIVE_TIMEOUTS 次时终止当前步骤，
         不将无效响应写入 data/ 目录。
"""

from jcy.feishu import run_step1
from jcy.advice import run_step2
from jcy.extract import run_step3
from jcy.pipeline import main

__all__ = ["run_step1", "run_step2", "run_step3", "main"]


if __name__ == "__main__":
    main()
