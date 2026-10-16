"""严格实体级 Micro P/R/F1。正确条件：start_idx、end_idx、type 都相同。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.data.schema import entity_key


def _as_set(entities: list[dict]) -> set[tuple]:
    return {entity_key(entity) for entity in entities if _complete(entity)}


def _complete(entity: dict) -> bool:
    return all(field in entity for field in ("start_idx", "end_idx", "type"))


def score_documents(documents: list[dict], gold_field: str = "gold", pred_field: str = "pred") -> dict:
    true_positive = false_positive = false_negative = 0
    for doc in documents:
        gold = _as_set(doc.get(gold_field) or [])
        pred = _as_set(doc.get(pred_field) or [])
        true_positive += len(gold & pred)
        false_positive += len(pred - gold)
        false_negative += len(gold - pred)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    micro_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "micro_f1": micro_f1,
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
    }


def read_predictions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} 的顶层不是列表")
    return data


def write_metrics_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "precision", "recall", "micro_f1", "tp", "fp", "fn"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": row["method"],
                    "precision": f"{row['precision']:.6f}",
                    "recall": f"{row['recall']:.6f}",
                    "micro_f1": f"{row['micro_f1']:.6f}",
                    "tp": row["tp"],
                    "fp": row["fp"],
                    "fn": row["fn"],
                }
            )


def read_metrics_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
