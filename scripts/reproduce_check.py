"""只读核验已有预测和指标。不训练，不调用生成模型。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = (
    "scripts/recompute_dictionary_metrics.py",
    "scripts/recompute_bert_bio_metrics.py",
    "scripts/recompute_qwen_3b_metrics.py",
    "scripts/recompute_qwen_shot_count.py",
    "scripts/analyze_report_gaps.py",
    "scripts/analyze_report_diagnostics.py",
)


def main() -> None:
    for relative in COMMANDS:
        print(f"RUN {relative}", flush=True)
        completed = subprocess.run([sys.executable, str(ROOT / relative)], cwd=ROOT, check=False)
        if completed.returncode != 0:
            raise SystemExit(f"{relative} 退出码 {completed.returncode}")
    print("CHECK_OK")


if __name__ == "__main__":
    main()
