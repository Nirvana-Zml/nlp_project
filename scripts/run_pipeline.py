"""一条命令完成数据校验、词典基线、BERT 训练、Qwen 推理和指标重算。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.local_resources import preflight

FULL_STEPS = (
    ("数据校验与词典基线", "scripts/run_step1.py"),
    ("BERT 训练", "scripts/train_bert_bio.py"),
    ("Qwen 零样本与三样本推理", "scripts/run_qwen_3b.py"),
    ("Qwen 提示数量推理", "scripts/run_qwen_shot_count.py"),
    ("词典指标重算", "scripts/recompute_dictionary_metrics.py"),
    ("BERT 指标重算", "scripts/recompute_bert_bio_metrics.py"),
    ("Qwen 主测试指标重算", "scripts/recompute_qwen_3b_metrics.py"),
    ("Qwen 提示数量指标重算", "scripts/recompute_qwen_shot_count.py"),
)
RECOMPUTE_STEPS = FULL_STEPS[4:]
RECOMPUTE_FILES = (
    "outputs/predictions/dictionary_all.json",
    "outputs/predictions/dictionary_longest.json",
    "outputs/predictions/bert_bio.json",
    "outputs/predictions/qwen_3b_zero.json",
    "outputs/predictions/qwen_3b_three.json",
    "outputs/predictions/qwen_3b_one.json",
    "outputs/predictions/qwen_shot_0_tune.json",
    "outputs/predictions/qwen_shot_1_tune.json",
    "outputs/predictions/qwen_shot_3_tune.json",
    "outputs/metrics/dictionary_baseline.csv",
    "outputs/metrics/bert_bio.csv",
    "outputs/metrics/qwen_3b.csv",
    "outputs/metrics/qwen_3b_cost.json",
    "outputs/metrics/qwen_3b_one.csv",
    "outputs/metrics/qwen_3b_one_cost.json",
    "outputs/metrics/qwen_shot_count_tune.csv",
    "outputs/metrics/qwen_shot_count_cost.json",
    "outputs/metrics/qwen_shot_count_selection.json",
)


def _run_step(title: str, script: str) -> None:
    print(f"STEP {title}: {script}", flush=True)
    completed = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"PIPELINE_STOP {script} 退出码 {completed.returncode}")


def _require_recompute_files() -> None:
    missing = [name for name in RECOMPUTE_FILES if not (ROOT / name).is_file()]
    if not missing:
        return
    listed = "\n".join(missing)
    raise SystemExit(
        "缺少预测或指标文件，无法重算：\n"
        f"{listed}\n"
        "这些文件由完整流水线生成。python scripts/run_pipeline.py --recompute 不训练、不推理，也不需要模型和 GPU。"
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="复现 CMeEE-V2 词典基线、BERT-BIO 和 Qwen2.5-3B 提示抽取")
    parser.add_argument("--recompute", action="store_true", help="只按已有预测重算指标，不训练、不推理")
    args = parser.parse_args()
    if args.recompute:
        _require_recompute_files()
        steps = RECOMPUTE_STEPS
    else:
        preflight()
        print(
            "完整流水线会训练 BERT 并运行 Qwen 推理。"
            "按已记录的生成时间，Qwen 部分约 21.8 小时，不含模型加载。",
            flush=True,
        )
        steps = FULL_STEPS
    for title, script in steps:
        _run_step(title, script)
    print("PIPELINE_OK", flush=True)


if __name__ == "__main__":
    main()
