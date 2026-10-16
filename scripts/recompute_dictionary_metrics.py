"""只从预测文件重算词典基线，并与 CSV 对照。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import read_metrics_csv, read_predictions, score_documents


def main() -> None:
    outputs = ROOT / "outputs"
    specs = [
        ("dictionary_all", outputs / "predictions" / "dictionary_all.json"),
        ("dictionary_longest", outputs / "predictions" / "dictionary_longest.json"),
    ]
    saved = {row["method"]: row for row in read_metrics_csv(outputs / "metrics" / "dictionary_baseline.csv")}
    mismatches = []
    for method, path in specs:
        scored = score_documents(read_predictions(path))
        row = saved[method]
        comparable = {
            "precision": f"{scored['precision']:.6f}",
            "recall": f"{scored['recall']:.6f}",
            "micro_f1": f"{scored['micro_f1']:.6f}",
            "tp": str(scored["tp"]),
            "fp": str(scored["fp"]),
            "fn": str(scored["fn"]),
        }
        for field, value in comparable.items():
            if row[field] != value:
                mismatches.append(f"{method}.{field}: csv={row[field]} recompute={value}")
        print(
            f"{method} precision={comparable['precision']} recall={comparable['recall']} "
            f"micro_f1={comparable['micro_f1']} tp={comparable['tp']} fp={comparable['fp']} fn={comparable['fn']}"
        )
    if mismatches:
        print("MISMATCH")
        print("\n".join(mismatches))
        sys.exit(1)
    print("RECOMPUTE_OK")


if __name__ == "__main__":
    main()
