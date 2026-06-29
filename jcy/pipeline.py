"""JCY 数据准备流水线编排（Step 1 → 2 → 3）。"""

import argparse

from jcy.lib.common import load_docs
from jcy import config, store
from jcy.feishu import run_step1
from jcy.advice import run_step2
from jcy.extract import run_step3


def main():
    parser = argparse.ArgumentParser(description="JCY 数据准备流水线（Step 1-3）")
    parser.add_argument("--strict", action="store_true",
                        help="Step 1 采集失败时硬中止（默认继续用旧缓存）")
    parser.add_argument("--log-file", type=str, default="",
                        help="日志同时写入该文件")
    args = parser.parse_args()
    config.setup_logging(args.log_file or None)
    log = config.log

    log.info("\n" + "=" * 60)
    log.info("  JCY 数据准备流水线（Step 1 → 2 → 3）")
    log.info("=" * 60)

    log.info("\n" + "─" * 60 + "\n  Step 1: 飞书数据采集\n" + "─" * 60)
    step1_ok = run_step1()
    if not step1_ok:
        log.warning("⚠️ Step 1 采集失败，后续步骤将使用已有旧缓存（数据可能不是最新）")
        if args.strict:
            log.error("--strict 已启用，终止流水线")
            return

    try:
        docs = load_docs()
        log.info(f"\n📂 读取到 {len(docs)} 篇文档")
    except FileNotFoundError:
        log.error("❌ 未找到文档缓存（jcy_docs.yaml），无法执行 Step 2/3")
        return

    # Step 1 按 URL 去重、Step 2/3 按 (date,title) 复合键去重；提前暴露键冲突，
    # 避免 upsert 静默互相覆盖（同一逻辑文章多 URL 或标题日期漂移时会发生）。
    collisions = store.detect_doc_key_collisions(docs)
    if collisions:
        log.warning(f"⚠️ 检测到 {len(collisions)} 组文档折叠到同一复合键（Step 2/3 可能互相覆盖）：")
        for key, titles in collisions.items():
            log.warning(f"   键 {key}：{titles}")

    log.info("\n" + "─" * 60 + "\n  Step 2: Perplexity 投资建议生成\n" + "─" * 60)
    run_step2(docs)
    log.info("\n" + "─" * 60 + "\n  Step 3: LLM 结构化提取\n" + "─" * 60)
    run_step3(docs)

    log.info("\n" + "=" * 60)
    log.info(f"  全部流程完成（Step 1 采集：{'成功' if step1_ok else '失败-用旧缓存'}）")
    log.info("=" * 60)
