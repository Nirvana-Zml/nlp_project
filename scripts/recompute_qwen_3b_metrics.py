"""只从 Qwen 预测文件重算 CSV，并与已保存的 CSV、成本文件对照。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import read_metrics_csv, read_predictions, score_documents

METHODS = ("qwen_3b_zero", "qwen_3b_three")
EXPECTED_GOLD = 4877


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    outputs = ROOT / "outputs"
    csv_rows = read_metrics_csv(outputs / "metrics" / "qwen_3b.csv")
    saved = {row["method"]: row for row in csv_rows}
    cost = json.loads((outputs / "metrics" / "qwen_3b_cost.json").read_text(encoding="utf-8"))
    if list(saved) != list(METHODS):
        raise SystemExit(f"CSV 方法顺序不是 {METHODS}")
    for name in METHODS:
        records = read_predictions(outputs / "predictions" / f"{name}.json")
        if len(records) != 1000:
            raise SystemExit(f"{name} 不是 1000 条")
        score = score_documents(records)
        if score["tp"] + score["fn"] != EXPECTED_GOLD:
            raise SystemExit(f"{name} 的 TP+FN={score['tp'] + score['fn']}")
        row = saved[name]
        compared = {
            "precision": f"{score['precision']:.6f}",
            "recall": f"{score['recall']:.6f}",
            "micro_f1": f"{score['micro_f1']:.6f}",
            "tp": str(score["tp"]),
            "fp": str(score["fp"]),
            "fn": str(score["fn"]),
        }
        for field, value in compared.items():
            if row[field] != value:
                raise SystemExit(f"{name} 的 {field} 重算值 {value} 与 CSV {row[field]} 不一致")
        fresh_cost = _cost_row(records)
        if fresh_cost != cost[name]:
            raise SystemExit(f"{name} 的成本重算与 qwen_3b_cost.json 不一致")
        missing_raw = sum(1 for record in records if "raw_output" not in record)
        if missing_raw:
            raise SystemExit(f"{name} 有 {missing_raw} 条没有原始输出")
    print("RECOMPUTE_OK")


def _cost_row(records: list[dict]) -> dict:
    count = len(records)
    return {
        "documents": count,
        "mean_latency_sec": round(sum(row["latency_sec"] for row in records) / count, 6),
        "mean_input_tokens": round(sum(row["input_tokens"] for row in records) / count, 6),
        "mean_output_tokens": round(sum(row["output_tokens"] for row in records) / count, 6),
        "parse_failures": sum(1 for row in records if not row["parse_ok"]),
        "invalid_entities": sum(row["invalid_entities"] for row in records),
        "surface_mismatches": sum(row["surface_mismatches"] for row in records),
    }


if __name__ == "__main__":
    main()
