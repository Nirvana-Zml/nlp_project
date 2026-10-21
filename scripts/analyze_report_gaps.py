"""只读已有预测，统计报告里缺少的错误构成。不改预测、不加载模型。"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.bio import flatten_entities

OUT = ROOT / "outputs" / "metrics" / "report_gap_stats.json"
MAX_NEW = 1536


def key(entity: dict) -> tuple:
    return (entity["start_idx"], entity["end_idx"], entity["type"])


def overlaps(left: dict, right: dict) -> bool:
    return left["start_idx"] < right["end_idx"] and right["start_idx"] < left["end_idx"]


def classify_gold(gold: list[dict], pred: list[dict]) -> Counter:
    """每条金标只进一类。精确命中优先，其次同跨度异类，再同类型相交异边界，其余为漏检。"""
    counts = Counter()
    pred_keys = {key(item) for item in pred}
    for entity in gold:
        if key(entity) in pred_keys:
            counts["exact"] += 1
            continue
        same_span = [
            item
            for item in pred
            if item["start_idx"] == entity["start_idx"] and item["end_idx"] == entity["end_idx"]
        ]
        if same_span:
            counts["same_span_wrong_type"] += 1
            continue
        same_type_overlap = [
            item
            for item in pred
            if item["type"] == entity["type"] and overlaps(item, entity)
        ]
        if same_type_overlap:
            counts["same_type_boundary"] += 1
            continue
        any_overlap = [item for item in pred if overlaps(item, entity)]
        if any_overlap:
            counts["overlap_wrong_type"] += 1
            continue
        counts["miss"] += 1
    return counts


def repeated_surface(text: str, entities: list[dict]) -> int:
    total = 0
    for entity in entities:
        surface = text[entity["start_idx"] : entity["end_idx"]]
        if surface and text.count(surface) > 1:
            total += 1
    return total


def summarize_qwen(path: Path) -> dict:
    docs = json.loads(path.read_text(encoding="utf-8"))
    gold_counts = Counter()
    parse_fail_docs = 0
    parse_fail_gold = 0
    hit_cap = 0
    hit_cap_and_fail = 0
    output_tokens = []
    fail_output_tokens = []
    repeat_gold = 0
    example = None
    for doc in docs:
        gold = doc.get("gold") or []
        pred = doc.get("pred") or []
        tokens = int(doc["output_tokens"])
        output_tokens.append(tokens)
        if tokens >= MAX_NEW:
            hit_cap += 1
        if not doc.get("parse_ok"):
            parse_fail_docs += 1
            parse_fail_gold += len(gold)
            fail_output_tokens.append(tokens)
            if tokens >= MAX_NEW:
                hit_cap_and_fail += 1
        gold_counts.update(classify_gold(gold, pred))
        repeat_gold += repeated_surface(doc["text"], gold)
        if doc.get("source_index") == 95:
            raw = doc.get("raw_output") or ""
            example = {
                "source_index": 95,
                "text_chars": len(doc["text"]),
                "output_tokens": tokens,
                "max_new_tokens": MAX_NEW,
                "reached_max_new_tokens": tokens >= MAX_NEW,
                "parse_ok": doc.get("parse_ok"),
                "raw_chars": len(raw),
                "raw_closed": raw.rstrip().endswith("}"),
                "pred_count": len(pred),
            }
    return {
        "file": str(path.relative_to(ROOT)).replace("\\", "/"),
        "documents": len(docs),
        "parse_fail_docs": parse_fail_docs,
        "parse_fail_gold_entities": parse_fail_gold,
        "output_tokens_ge_1536": hit_cap,
        "parse_fail_and_output_tokens_ge_1536": hit_cap_and_fail,
        "parse_fail_output_tokens_max": max(fail_output_tokens) if fail_output_tokens else None,
        "parse_fail_output_tokens_min": min(fail_output_tokens) if fail_output_tokens else None,
        "mean_output_tokens": sum(output_tokens) / len(output_tokens),
        "gold_outcome": dict(gold_counts),
        "gold_entities": sum(gold_counts.values()),
        "gold_with_repeated_surface_in_text": repeat_gold,
        "source_index_95": example,
    }


def summarize_bert(path: Path) -> dict:
    docs = json.loads(path.read_text(encoding="utf-8"))
    gold_counts = Counter()
    dropped_outcome = Counter()
    kept_outcome = Counter()
    repeat_gold = 0
    for doc in docs:
        gold = doc.get("gold") or []
        pred = doc.get("pred") or []
        kept, dropped = flatten_entities(gold)
        dropped_keys = {key(item) for item in dropped}
        gold_counts.update(classify_gold(gold, pred))
        dropped_outcome.update(classify_gold(dropped, pred))
        kept_outcome.update(classify_gold(kept, pred))
        repeat_gold += repeated_surface(doc["text"], gold)
    return {
        "file": str(path.relative_to(ROOT)).replace("\\", "/"),
        "documents": len(docs),
        "gold_outcome": dict(gold_counts),
        "kept_after_flatten": dict(kept_outcome),
        "dropped_by_flatten": dict(dropped_outcome),
        "gold_entities": sum(gold_counts.values()),
        "gold_with_repeated_surface_in_text": repeat_gold,
    }


def damage_labels() -> dict:
    """含「损害」的金标表面在 train/tune/test 上的类型计数。只说明本数据里两种标签都出现。"""
    counts = {}
    for name in ("train", "tune", "test"):
        docs = json.loads((ROOT / "data" / "processed" / f"{name}.json").read_text(encoding="utf-8"))
        by_type = Counter()
        surfaces = Counter()
        for doc in docs:
            for entity in doc.get("entities") or []:
                surface = doc["text"][entity["start_idx"] : entity["end_idx"]]
                if "损害" not in surface:
                    continue
                by_type[entity["type"]] += 1
                surfaces[surface] += 1
        counts[name] = {"by_type": dict(by_type), "surfaces": dict(surfaces.most_common(12))}
    return counts


def main() -> None:
    payload = {
        "qwen_3b_one": summarize_qwen(ROOT / "outputs" / "predictions" / "qwen_3b_one.json"),
        "qwen_3b_zero": summarize_qwen(ROOT / "outputs" / "predictions" / "qwen_3b_zero.json"),
        "qwen_3b_three": summarize_qwen(ROOT / "outputs" / "predictions" / "qwen_3b_three.json"),
        "bert_bio": summarize_bert(ROOT / "outputs" / "predictions" / "bert_bio.json"),
        "damage_surface_labels": damage_labels(),
        "rules": {
            "gold_outcome": "exact, same_span_wrong_type, same_type_boundary, overlap_wrong_type, miss",
            "max_new_tokens": MAX_NEW,
            "flatten": "src.models.bio.flatten_entities",
        },
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    one = payload["qwen_3b_one"]
    bert = payload["bert_bio"]
    print("ANALYZE_OK")
    print("qwen_one_gold", one["gold_outcome"])
    print("qwen_one_cap", one["output_tokens_ge_1536"], "fail_and_cap", one["parse_fail_and_output_tokens_ge_1536"])
    print("index95", one["source_index_95"])
    print("bert_kept", bert["kept_after_flatten"])
    print("bert_dropped", bert["dropped_by_flatten"])


if __name__ == "__main__":
    main()
