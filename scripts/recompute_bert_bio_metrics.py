"""只从 BERT-BIO 预测文件重算 CSV，并与已保存的 CSV 对照。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import read_metrics_csv, read_predictions, score_documents
from src.models.bio import score_token_sequences


def main() -> None:
    outputs = ROOT / "outputs"
    predictions = read_predictions(outputs / "predictions" / "bert_bio.json")
    entity_score = score_documents(predictions)
    token_score = score_token_sequences(
        [sequence for row in predictions for sequence in row["token_gold"]],
        [sequence for row in predictions for sequence in row["token_pred"]],
    )
    scored = {
        "bert_bio_token": token_score,
        "bert_bio_entity": entity_score,
    }
    saved = {row["method"]: row for row in read_metrics_csv(outputs / "metrics" / "bert_bio.csv")}
    mismatches = []
    if set(saved) != set(scored):
        mismatches.append(f"methods csv={sorted(saved)} recompute={sorted(scored)}")
    for method, score in scored.items():
        comparable = {
            "precision": f"{score['precision']:.6f}",
            "recall": f"{score['recall']:.6f}",
            "micro_f1": f"{score['micro_f1']:.6f}",
            "tp": str(score["tp"]),
            "fp": str(score["fp"]),
            "fn": str(score["fn"]),
        }
        row = saved.get(method)
        if row is None:
            continue
        for field, value in comparable.items():
            if row[field] != value:
                mismatches.append(f"{method}.{field}: csv={row[field]} recompute={value}")
        print(
            f"{method} precision={comparable['precision']} recall={comparable['recall']} "
            f"micro_f1={comparable['micro_f1']} tp={comparable['tp']} fp={comparable['fp']} fn={comparable['fn']}"
        )
    entity_total = entity_score["tp"] + entity_score["fn"]
    print(f"entity_tp_plus_fn={entity_total}")
    if entity_total != 4877:
        mismatches.append(f"entity TP+FN={entity_total} expected 4877")
    if mismatches:
        print("MISMATCH")
        print("\n".join(mismatches))
        sys.exit(1)
    print("RECOMPUTE_OK")


if __name__ == "__main__":
    main()
