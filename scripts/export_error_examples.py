"""从主测试预测只读抽出 3 条失败样例。不调用模型，不改预测文件。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import read_predictions
from src.models.qwen_prompt import _first_json_value, _legal_span

PREDICTIONS = ROOT / "outputs" / "predictions"
OUTPUT = ROOT / "outputs" / "errors" / "examples.json"


def main() -> None:
    selected: list[dict] = []
    used: set[tuple[str, int]] = set()
    cache: dict[str, list[dict]] = {}

    def rows(name: str) -> list[dict]:
        if name not in cache:
            cache[name] = read_predictions(PREDICTIONS / name)
        return cache[name]

    def take(category: str, method: str, filename: str, predicate) -> dict | None:
        matches = []
        for row in rows(filename):
            key = (method, row["source_index"])
            if key in used:
                continue
            if predicate(row):
                matches.append(row)
        if not matches:
            return None
        chosen = min(matches, key=lambda row: row["source_index"])
        used.add((method, chosen["source_index"]))
        return example(category, method, filename, chosen)

    primaries = [
        ("解析失败", "qwen_3b_one", "qwen_3b_one.json", parse_failure),
        ("边界错误", "bert_bio", "bert_bio.json", bert_boundary),
        ("类型错误", "bert_bio", "bert_bio.json", type_error),
    ]
    fallbacks = [
        ("无效实体", "qwen_3b_one", "qwen_3b_one.json", invalid_entity),
        ("表面不一致", "qwen_3b_one", "qwen_3b_one.json", surface_mismatch),
        ("边界错误", "dictionary_longest", "dictionary_longest.json", boundary_error),
        ("类型错误", "dictionary_longest", "dictionary_longest.json", type_error),
        ("解析失败", "qwen_3b_zero", "qwen_3b_zero.json", parse_failure),
        ("解析失败", "qwen_3b_three", "qwen_3b_three.json", parse_failure),
    ]
    for category, method, filename, predicate in primaries:
        item = take(category, method, filename, predicate)
        if item is None:
            for fallback in fallbacks:
                if fallback[0] in {row["category"] for row in selected}:
                    continue
                item = take(*fallback)
                if item is not None:
                    break
        if item is not None:
            selected.append(item)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"FOUND={len(selected)}")
    for item in selected:
        print(
            f"category={item['category']} method={item['method']} "
            f"source_index={item['source_index']} file={item['prediction_file']}"
        )
    if len(selected) < 3:
        sys.exit(1)
    print("EXPORT_OK")


def example(category: str, method: str, filename: str, row: dict) -> dict:
    item = {
        "method": method,
        "prediction_file": f"outputs/predictions/{filename}",
        "source_index": row["source_index"],
        "category": category,
        "text": row["text"],
        "gold": row["gold"],
        "pred": row["pred"],
        "attribution": "",
        "improvement": "",
    }
    if filename.startswith("qwen_"):
        item["raw_output"] = row["raw_output"]
        item["discarded"] = discarded_objects(row["raw_output"], row["text"])
    elif filename == "bert_bio.json":
        item["token_pred"] = row["token_pred"]
    return item


def triples(entities: list[dict]) -> list[tuple]:
    return [(entity["start_idx"], entity["end_idx"], entity["type"]) for entity in entities]


def overlaps(left: tuple, right: tuple) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def has_boundary(row: dict) -> bool:
    for gold in triples(row["gold"]):
        for pred in triples(row["pred"]):
            if gold[2] == pred[2] and overlaps(gold, pred) and (gold[0] != pred[0] or gold[1] != pred[1]):
                return True
    return False


def has_type(row: dict) -> bool:
    for gold in triples(row["gold"]):
        for pred in triples(row["pred"]):
            if gold[0] == pred[0] and gold[1] == pred[1] and gold[2] != pred[2]:
                return True
    return False


def parse_failure(row: dict) -> bool:
    return (not row["parse_ok"]) and bool(row["gold"])


def bert_boundary(row: dict) -> bool:
    return has_boundary(row) and not has_type(row)


def boundary_error(row: dict) -> bool:
    return has_boundary(row)


def type_error(row: dict) -> bool:
    return has_type(row)


def invalid_entity(row: dict) -> bool:
    return bool(row["parse_ok"]) and row["invalid_entities"] > 0


def surface_mismatch(row: dict) -> bool:
    return row["surface_mismatches"] > 0 and set(triples(row["gold"])) != set(triples(row["pred"]))


def discarded_objects(raw: str, text: str) -> list:
    payload = _first_json_value(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("entities"), list):
        return []
    dropped = []
    seen: set[tuple] = set()
    for item in payload["entities"]:
        if not isinstance(item, dict):
            dropped.append(item)
            continue
        entity_type = item.get("type")
        start = item.get("start_idx")
        end = item.get("end_idx")
        if not _legal_span(entity_type, start, end, text):
            dropped.append(item)
            continue
        key = (start, end, entity_type)
        if key in seen:
            dropped.append(item)
            continue
        seen.add(key)
    return dropped


if __name__ == "__main__":
    main()
