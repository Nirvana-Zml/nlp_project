"""只读已有预测、日志和划分，补报告所需的诊断计数。不改预测、不加载生成模型。"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines.dictionary_ner import build_lexicon, compile_lexicon, find_all, longest_match
from src.data.schema import ENTITY_TYPES
from src.evaluation.metrics import score_documents
from src.models.bio import flatten_entities
from src.models.qwen_prompt import _first_json_value

OUT = ROOT / "outputs" / "metrics" / "report_diagnostics.json"
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 42


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def entity_key(entity: dict) -> tuple:
    return (entity["start_idx"], entity["end_idx"], entity["type"])


def overlaps(left: dict, right: dict) -> bool:
    return left["start_idx"] < right["end_idx"] and right["start_idx"] < left["end_idx"]


def strict_contains(outer: dict, inner: dict) -> bool:
    return (
        outer["start_idx"] <= inner["start_idx"]
        and inner["end_idx"] <= outer["end_idx"]
        and (outer["start_idx"], outer["end_idx"]) != (inner["start_idx"], inner["end_idx"])
    )


def occurrences(text: str, surface: str) -> list[tuple[int, int]]:
    if not surface:
        return []
    found = []
    start = 0
    while True:
        index = text.find(surface, start)
        if index < 0:
            return found
        found.append((index, index + len(surface)))
        start = index + 1


def legal_span(entity_type, start, end, text: str) -> bool:
    if entity_type not in ENTITY_TYPES:
        return False
    if type(start) is not int or type(end) is not int:
        return False
    return 0 <= start < end <= len(text)


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "micro_f1": f1, "tp": tp, "fp": fp, "fn": fn}


def per_type(docs: list[dict], pred_field: str = "pred") -> dict:
    rows = {}
    for name in ENTITY_TYPES:
        tp = fp = fn = 0
        for doc in docs:
            gold = {entity_key(item) for item in doc.get("gold") or [] if item.get("type") == name}
            pred = {entity_key(item) for item in doc.get(pred_field) or [] if item.get("type") == name}
            tp += len(gold & pred)
            fp += len(pred - gold)
            fn += len(gold - pred)
        rows[name] = prf(tp, fp, fn)
    return rows


def nested_flags(gold: list[dict]) -> list[bool]:
    flags = []
    for entity in gold:
        flags.append(any(strict_contains(entity, other) or strict_contains(other, entity) for other in gold if other is not entity))
    return flags


def recall_by_nesting(docs: list[dict], pred_field: str = "pred") -> dict:
    counts = {
        "nested": {"tp": 0, "fn": 0, "gold": 0},
        "non_nested": {"tp": 0, "fn": 0, "gold": 0},
    }
    for doc in docs:
        gold = doc.get("gold") or []
        pred = {entity_key(item) for item in doc.get(pred_field) or []}
        for entity, nested in zip(gold, nested_flags(gold)):
            bucket = "nested" if nested else "non_nested"
            counts[bucket]["gold"] += 1
            if entity_key(entity) in pred:
                counts[bucket]["tp"] += 1
            else:
                counts[bucket]["fn"] += 1
    for bucket in counts.values():
        gold = bucket["gold"]
        bucket["recall"] = bucket["tp"] / gold if gold else 0.0
    return counts


def classify_fp(pred: dict, gold: list[dict], accidental: bool) -> str:
    """一条假正例只进一类。优先级：错下标切片、同跨度异类、外扩、内缩、同类部分相交、异类相交、不相交。"""
    if accidental:
        return "accidental_slice"
    same_span = [
        item
        for item in gold
        if item["start_idx"] == pred["start_idx"] and item["end_idx"] == pred["end_idx"]
    ]
    if same_span:
        return "same_span_wrong_type"
    same_type = [item for item in gold if item["type"] == pred["type"] and overlaps(item, pred)]
    if any(strict_contains(pred, item) for item in same_type):
        return "boundary_expand"
    if any(strict_contains(item, pred) for item in same_type):
        return "boundary_shrink"
    if same_type:
        return "boundary_partial"
    if any(overlaps(item, pred) for item in gold):
        return "overlap_wrong_type"
    return "no_overlap"


def gold_outcome_one(entity: dict, pred: list[dict]) -> str:
    pred_keys = {entity_key(item) for item in pred}
    if entity_key(entity) in pred_keys:
        return "exact"
    same_span = [
        item
        for item in pred
        if item["start_idx"] == entity["start_idx"] and item["end_idx"] == entity["end_idx"]
    ]
    if same_span:
        return "same_span_wrong_type"
    same_type = [item for item in pred if item["type"] == entity["type"] and overlaps(item, entity)]
    if same_type:
        return "same_type_boundary"
    if any(overlaps(item, entity) for item in pred):
        return "overlap_wrong_type"
    return "miss"


def kept_miss_breakdown(docs: list[dict]) -> dict:
    outcome = Counter()
    by_type = {name: Counter() for name in ENTITY_TYPES}
    for doc in docs:
        gold = doc.get("gold") or []
        pred = doc.get("pred") or []
        kept, _dropped = flatten_entities(gold)
        for entity in kept:
            label = gold_outcome_one(entity, pred)
            if label == "exact":
                continue
            outcome[label] += 1
            by_type[entity["type"]][label] += 1
    return {"total": sum(outcome.values()), "outcome": dict(outcome), "by_type": {name: dict(counts) for name, counts in by_type.items()}}


def qwen_surface_and_fp(path: Path) -> dict:
    docs = load_json(path)
    surface = Counter()
    fp_counts = Counter()
    tp_mismatch = 0
    tp_match = 0
    parse_fail_docs = 0
    named_gold: set[tuple] = set()
    index_hit_gold: set[tuple] = set()
    unique_recoverable_gold: set[tuple] = set()
    for doc_index, doc in enumerate(docs):
        text = doc["text"]
        gold = doc.get("gold") or []
        gold_keys = {entity_key(item) for item in gold}
        gold_spans = defaultdict(list)
        for item in gold:
            surface_text = text[item["start_idx"] : item["end_idx"]]
            gold_spans[(surface_text, item["type"])].append((item["start_idx"], item["end_idx"]))
        payload = _first_json_value(doc.get("raw_output") or "")
        reported = []
        if not isinstance(payload, dict) or not isinstance(payload.get("entities"), list):
            parse_fail_docs += 1
        else:
            reported = payload["entities"]
        raw_for_key = defaultdict(list)
        for item in reported:
            surface["reported_items"] += 1
            if not isinstance(item, dict):
                surface["not_object"] += 1
                continue
            surface["reported_objects"] += 1
            name = item.get("entity")
            entity_type = item.get("type")
            start = item.get("start_idx")
            end = item.get("end_idx")
            if not isinstance(name, str) or name == "":
                surface["missing_surface"] += 1
                continue
            surface["has_surface"] += 1
            found = occurrences(text, name)
            if not found:
                surface["surface_absent"] += 1
                continue
            surface["surface_present"] += 1
            if len(found) > 1:
                surface["surface_present_repeated"] += 1
            span_ok = legal_span(entity_type, start, end, text)
            if span_ok:
                raw_for_key[(start, end, entity_type)].append(name)
            gold_hit_spans = set(gold_spans.get((name, entity_type), []))
            if not gold_hit_spans:
                if any(name == text[g["start_idx"] : g["end_idx"]] for g in gold):
                    surface["present_gold_surface_wrong_type_only"] += 1
                else:
                    surface["present_not_gold_surface"] += 1
                continue
            surface["present_and_gold_surface_type"] += 1
            if len(found) > 1:
                surface["gold_surface_type_repeated"] += 1
            index_hit = span_ok and (start, end, entity_type) in gold_keys and text[start:end] == name
            for span in gold_hit_spans:
                named_gold.add((doc_index, span[0], span[1], entity_type))
            if index_hit:
                surface["gold_surface_type_index_hit"] += 1
                index_hit_gold.add((doc_index, start, end, entity_type))
                continue
            if any(span in gold_hit_spans for span in found):
                surface["index_error_recoverable"] += 1
                if len(found) == 1:
                    surface["index_error_unique"] += 1
                    span = found[0]
                    unique_recoverable_gold.add((doc_index, span[0], span[1], entity_type))
                else:
                    surface["index_error_repeated"] += 1
            else:
                surface["gold_surface_type_without_aligned_span"] += 1
        pred_keys = []
        seen = set()
        for item in doc.get("pred") or []:
            key = entity_key(item)
            if key in seen:
                continue
            seen.add(key)
            pred_keys.append(item)
        for item in pred_keys:
            key = entity_key(item)
            names = raw_for_key.get(key, [])
            slice_text = text[item["start_idx"] : item["end_idx"]]
            mismatched = [name for name in names if name != slice_text]
            matched = [name for name in names if name == slice_text]
            accidental = bool(mismatched) and not matched
            if key in gold_keys:
                if accidental:
                    tp_mismatch += 1
                elif matched:
                    tp_match += 1
                continue
            fp_counts[classify_fp(item, gold, accidental)] += 1
    scored = score_documents(docs)
    return {
        "file": str(path.relative_to(ROOT)).replace("\\", "/"),
        "documents": len(docs),
        "parse_fail_docs": parse_fail_docs,
        "scored": scored,
        "reported_surface": dict(surface),
        "false_positive_class": dict(fp_counts),
        "false_positive_total": sum(fp_counts.values()),
        "true_positive_surface_equals_slice": tp_match,
        "true_positive_surface_replaced": tp_mismatch,
        "distinct_gold_named_by_surface_and_type": len(named_gold),
        "distinct_gold_hit_by_submitted_index": len(index_hit_gold),
        "distinct_gold_unique_surface_wrong_index": len(unique_recoverable_gold),
        "rules": {
            "surface_present": "自报表面是非空字符串，且在输入句中至少出现一次（允许重叠查找）",
            "present_and_gold_surface_type": "表面出现在句中，且至少一条金标的切片与类型都相同",
            "index_error_recoverable": "表面和类型能对上某条金标，句中也存在该金标跨度，但模型提交的下标不是这条金标",
            "index_error_unique": "上述情况里表面只出现一次，正确下标唯一",
            "fp_priority": "accidental_slice, same_span_wrong_type, boundary_expand, boundary_shrink, boundary_partial, overlap_wrong_type, no_overlap",
            "accidental_slice": "该计分三元组的全部自报表面都与原文切片不同；解析器用切片替换表面后仍未命中金标",
            "one_prediction_one_class": True,
            "prediction_may_overlap_multiple_gold": True,
        },
    }


def geometric_fp(path: Path, pred_field: str = "pred") -> dict:
    docs = load_json(path)
    counts = Counter()
    for doc in docs:
        gold = doc.get("gold") or []
        gold_keys = {entity_key(item) for item in gold}
        seen = set()
        for item in doc.get(pred_field) or []:
            key = entity_key(item)
            if key in seen or key in gold_keys:
                seen.add(key)
                continue
            seen.add(key)
            counts[classify_fp(item, gold, False)] += 1
    return {"file": str(path.relative_to(ROOT)).replace("\\", "/"), "false_positive_class": dict(counts), "false_positive_total": sum(counts.values()), "scored": score_documents(docs, pred_field=pred_field)}


def micro_from_sums(rows: list[tuple[int, int, int]]) -> float:
    tp = sum(item[0] for item in rows)
    fp = sum(item[1] for item in rows)
    fn = sum(item[2] for item in rows)
    return prf(tp, fp, fn)["micro_f1"]


def doc_counts(docs: list[dict]) -> list[tuple[int, int, int]]:
    rows = []
    for doc in docs:
        gold = {entity_key(item) for item in doc.get("gold") or []}
        pred = {entity_key(item) for item in doc.get("pred") or []}
        rows.append((len(gold & pred), len(pred - gold), len(gold - pred)))
    return rows


def bootstrap_f1_diff(left_docs: list[dict], right_docs: list[dict]) -> dict:
    if [doc["source_index"] for doc in left_docs] != [doc["source_index"] for doc in right_docs]:
        raise SystemExit("调参预测的 source_index 顺序不一致")
    left = doc_counts(left_docs)
    right = doc_counts(right_docs)
    point = micro_from_sums(left) - micro_from_sums(right)
    rng = random.Random(BOOTSTRAP_SEED)
    count = len(left)
    diffs = []
    positive = 0
    for _ in range(BOOTSTRAP_DRAWS):
        chosen = [rng.randrange(count) for _ in range(count)]
        diff = micro_from_sums([left[index] for index in chosen]) - micro_from_sums([right[index] for index in chosen])
        diffs.append(diff)
        if diff > 0:
            positive += 1
    diffs.sort()
    low_index = int(0.025 * (BOOTSTRAP_DRAWS - 1))
    high_index = int(0.975 * (BOOTSTRAP_DRAWS - 1))
    return {
        "documents": count,
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "unit": "document",
        "statistic": "micro_f1(shot1) - micro_f1(shot3)",
        "point_estimate": point,
        "percentile_2_5": diffs[low_index],
        "percentile_97_5": diffs[high_index],
        "draws_with_shot1_higher": positive,
        "note": "对文档有放回抽样后重算微平均，不是逐篇 F1 的平均。只描述这一组固定示例。",
    }


def duplicate_texts() -> dict:
    grouped = defaultdict(list)
    within = {}
    for name in ("train", "tune", "test"):
        docs = load_json(ROOT / "data" / "processed" / f"{name}.json")
        seen = Counter(doc["text"] for doc in docs)
        within[name] = {
            "documents": len(docs),
            "unique_texts": len(seen),
            "texts_repeated_inside_split": sum(1 for count in seen.values() if count > 1),
        }
        for doc in docs:
            grouped[doc["text"]].append(name)
    cross = Counter()
    for names in grouped.values():
        key = tuple(sorted(set(names)))
        if len(key) > 1:
            cross["+".join(key)] += 1
    return {"within_split": within, "texts_shared_across_splits": dict(cross)}


def flatten_counts() -> dict:
    payload = {}
    for name in ("train", "tune", "test"):
        docs = load_json(ROOT / "data" / "processed" / f"{name}.json")
        gold = dropped = 0
        for doc in docs:
            kept, lost = flatten_entities(doc.get("entities") or [])
            gold += len(kept) + len(lost)
            dropped += len(lost)
        payload[name] = {"gold": gold, "dropped_when_building_flat_reference": dropped, "kept": gold - dropped}
    return payload


def dictionary_latency() -> dict:
    train_docs = load_json(ROOT / "data" / "processed" / "train.json")
    test_docs = load_json(ROOT / "data" / "processed" / "test.json")
    started = time.perf_counter()
    lexicon = build_lexicon(train_docs)
    root = compile_lexicon(lexicon)
    build_sec = time.perf_counter() - started
    started = time.perf_counter()
    rows = []
    for doc in test_docs:
        matches = find_all(doc["text"], lexicon, root)
        rows.append({"gold": doc.get("entities") or [], "pred": longest_match(matches)})
    match_sec = time.perf_counter() - started
    score = score_documents(rows)
    return {
        "device": "cpu",
        "includes_model_load": False,
        "lexicon_build_sec": build_sec,
        "match_sec": match_sec,
        "documents": len(test_docs),
        "mean_match_sec": match_sec / len(test_docs),
        "batch_size": None,
        "score_matches_longest": score,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(relative: str) -> dict:
    path = ROOT / relative
    if not path.is_file():
        return {"path": relative, "exists": False}
    return {"path": relative, "exists": True, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def prompt_check() -> dict:
    import yaml

    log = (ROOT / "outputs" / "logs" / "qwen_shot_count.log").read_text(encoding="utf-8")
    early = (ROOT / "outputs" / "logs" / "qwen_3b.log").read_text(encoding="utf-8")
    logged = []
    for line in log.splitlines():
        if line.startswith("prompt_template="):
            logged.append(json.loads(line.split("=", 1)[1]))
    current = yaml.safe_load((ROOT / "configs" / "qwen_3b.yaml").read_text(encoding="utf-8"))["prompt_template"]
    return {
        "shot_log_prompt_copies": len(logged),
        "shot_log_prompts_identical": len(set(logged)) == 1 and len(logged) > 0,
        "shot_log_prompt_equals_current_yaml": bool(logged) and all(item == current for item in logged),
        "early_main_log_contains_prompt_template": any(line.startswith("prompt_template=") for line in early.splitlines()),
    }


def log_timeline() -> dict:
    early = (ROOT / "outputs" / "logs" / "qwen_3b.log").read_text(encoding="utf-8").splitlines()
    shot = (ROOT / "outputs" / "logs" / "qwen_shot_count.log").read_text(encoding="utf-8").splitlines()
    train = (ROOT / "outputs" / "logs" / "train_bio.log").read_text(encoding="utf-8").splitlines()
    step = (ROOT / "outputs" / "logs" / "step1.log").read_text(encoding="utf-8").splitlines()

    def pick(lines: list[str], prefix: str) -> list[str]:
        return [line.split("=", 1)[1] for line in lines if line.startswith(prefix)]

    return {
        "step1_created_at": pick(step, "created_at=")[0],
        "bert_created_at": pick(train, "created_at=")[0],
        "qwen_0_and_3_main_created_at": pick(early, "created_at=")[0],
        "qwen_0_and_3_main_results_in_same_log": sum(line.startswith("RESULT method=qwen_3b_") for line in early),
        "shot_log_created_at": pick(shot, "created_at="),
        "shot_log_flags": [line for line in shot if line.startswith(("selection_", "selected_shots", "regenerated_", "generated_", "test_score", "start method"))],
    }


def main() -> None:
    one = qwen_surface_and_fp(ROOT / "outputs" / "predictions" / "qwen_3b_one.json")
    bert_fp = geometric_fp(ROOT / "outputs" / "predictions" / "bert_bio.json")
    longest_fp = geometric_fp(ROOT / "outputs" / "predictions" / "dictionary_longest.json", "pred")
    dictionary_docs = load_json(ROOT / "outputs" / "predictions" / "dictionary_longest.json")
    bert_docs = load_json(ROOT / "outputs" / "predictions" / "bert_bio.json")
    qwen_docs = load_json(ROOT / "outputs" / "predictions" / "qwen_3b_one.json")
    for doc in qwen_docs:
        doc.pop("raw_output", None)
    shot1 = load_json(ROOT / "outputs" / "predictions" / "qwen_shot_1_tune.json")
    shot3 = load_json(ROOT / "outputs" / "predictions" / "qwen_shot_3_tune.json")
    for doc in shot1 + shot3:
        doc.pop("raw_output", None)
    payload = {
        "timeline": log_timeline(),
        "prompt_check": prompt_check(),
        "flatten_reference_labels": flatten_counts(),
        "duplicate_texts": duplicate_texts(),
        "qwen_3b_one": one,
        "bert_bio_false_positive": bert_fp,
        "dictionary_longest_false_positive": longest_fp,
        "bert_kept_miss": kept_miss_breakdown(bert_docs),
        "per_type": {
            "dictionary_longest": per_type(dictionary_docs),
            "bert_bio": per_type(bert_docs),
            "qwen_3b_one": per_type(qwen_docs),
        },
        "nesting_recall": {
            "dictionary_longest": recall_by_nesting(dictionary_docs),
            "bert_bio": recall_by_nesting(bert_docs),
            "qwen_3b_one": recall_by_nesting(qwen_docs),
            "definition": "金标与同句另一条金标存在严格包含关系（边界相同不算）。只分解召回，假正例不按嵌套归类。",
        },
        "shot_bootstrap": bootstrap_f1_diff(shot1, shot3),
        "dictionary_latency": dictionary_latency(),
        "files": [
            file_record(relative)
            for relative in (
                "outputs/metrics/dictionary_baseline.csv",
                "outputs/metrics/bert_bio.csv",
                "outputs/metrics/qwen_3b.csv",
                "outputs/metrics/qwen_3b_one.csv",
                "outputs/metrics/qwen_shot_count_tune.csv",
                "outputs/metrics/qwen_shot_count_selection.json",
                "outputs/predictions/qwen_3b_one.json",
                "outputs/predictions/bert_bio.json",
                "outputs/predictions/dictionary_longest.json",
                "outputs/checkpoints/bert_bio/model.safetensors",
            )
        ],
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("DIAGNOSTICS_OK")
    print("surface", one["reported_surface"])
    print("qwen_fp", one["false_positive_class"], one["false_positive_total"])
    print("bert_fp", bert_fp["false_positive_class"], bert_fp["false_positive_total"])
    print("kept_miss", payload["bert_kept_miss"]["outcome"], payload["bert_kept_miss"]["total"])
    print("bootstrap", payload["shot_bootstrap"])
    print("duplicates", payload["duplicate_texts"]["texts_shared_across_splits"])
    print("latency", payload["dictionary_latency"]["match_sec"], payload["dictionary_latency"]["score_matches_longest"]["micro_f1"])


if __name__ == "__main__":
    main()
