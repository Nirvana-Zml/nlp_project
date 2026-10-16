"""BERT-BIO 的压平、切窗、标签和 token 级计数。

训练标签用压平后的 BIO。实体级评测仍使用原始金标。
"""

from __future__ import annotations

from src.data.schema import ENTITY_TYPES

LABELS: tuple[str, ...] = ("O", *(f"{prefix}-{name}" for name in ENTITY_TYPES for prefix in ("B", "I")))
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}

# 2026-10-14 对冻结划分套用本文件的压平规则得到的计数。实现必须与此一致。
EXPECTED_DROPS = {
    "train": {
        "gold": 74502,
        "total": 13702,
        "by_type": {
            "dis": 393,
            "sym": 10,
            "dru": 58,
            "equ": 55,
            "pro": 583,
            "bod": 10706,
            "ite": 1743,
            "mic": 153,
            "dep": 1,
        },
    },
    "tune": {
        "gold": 8147,
        "total": 1458,
        "by_type": {
            "dis": 31,
            "sym": 1,
            "dru": 9,
            "equ": 9,
            "pro": 51,
            "bod": 1154,
            "ite": 186,
            "mic": 17,
            "dep": 0,
        },
    },
    "test": {
        "gold": 4877,
        "total": 839,
        "by_type": {
            "dis": 25,
            "sym": 0,
            "dru": 2,
            "equ": 2,
            "pro": 26,
            "bod": 657,
            "ite": 123,
            "mic": 4,
            "dep": 0,
        },
    },
}


def overlaps(left: dict, right: dict) -> bool:
    """左闭右开区间至少共享一个字符。首尾相接不算冲突。"""
    return left["start_idx"] < right["end_idx"] and right["start_idx"] < left["end_idx"]


def flatten_entities(entities: list[dict]) -> tuple[list[dict], list[dict]]:
    """长度降序、原始顺序升序，贪心保留互不重叠的实体。"""
    ranked = sorted(
        enumerate(entities),
        key=lambda item: (-(item[1]["end_idx"] - item[1]["start_idx"]), item[0]),
    )
    kept: list[dict] = []
    dropped: list[dict] = []
    for _, entity in ranked:
        if any(overlaps(entity, chosen) for chosen in kept):
            dropped.append(entity)
        else:
            kept.append(entity)
    return kept, dropped


def drop_report(documents: list[dict]) -> dict:
    by_type = {name: 0 for name in ENTITY_TYPES}
    gold = 0
    total = 0
    for document in documents:
        entities = document.get("entities") or []
        gold += len(entities)
        _, dropped = flatten_entities(entities)
        total += len(dropped)
        for entity in dropped:
            by_type[entity["type"]] += 1
    return {"gold": gold, "total": total, "by_type": by_type}


def assert_expected_drops(reports: dict[str, dict]) -> None:
    for split, expected in EXPECTED_DROPS.items():
        actual = reports[split]
        if actual["gold"] != expected["gold"] or actual["total"] != expected["total"]:
            raise ValueError(f"{split} 舍弃总数不符: {actual} != {expected}")
        if actual["by_type"] != expected["by_type"]:
            raise ValueError(f"{split} 类别计数不符: {actual['by_type']} != {expected['by_type']}")


def _content_encoding(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )
    token_ids = [int(token_id) for token_id in encoded["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in encoded["offset_mapping"]]
    if len(token_ids) != len(offsets):
        raise ValueError("input_ids 与 offset 长度不一致")
    return token_ids, offsets


def _window_spans(num_tokens: int, content_tokens: int, stride: int) -> list[tuple[int, int]]:
    if num_tokens <= content_tokens:
        return [(0, num_tokens)] if num_tokens else []
    spans: list[tuple[int, int]] = []
    start = 0
    while start < num_tokens:
        end = min(start + content_tokens, num_tokens)
        spans.append((start, end))
        if end >= num_tokens:
            break
        start += stride
    return spans


def _token_owner(start: int, end: int, entities: list[dict]) -> dict | None:
    """一个 WordPiece 只属于一个保留实体。起点落在实体内的优先。"""
    overlapping = [
        entity
        for entity in entities
        if start < entity["end_idx"] and entity["start_idx"] < end
    ]
    if not overlapping:
        return None
    containing = [entity for entity in overlapping if entity["start_idx"] <= start < entity["end_idx"]]
    if containing:
        return min(containing, key=lambda entity: (entity["start_idx"], entity["type"]))

    def overlap_amount(entity: dict) -> int:
        return min(end, entity["end_idx"]) - max(start, entity["start_idx"])

    return min(overlapping, key=lambda entity: (-overlap_amount(entity), entity["start_idx"], entity["type"]))


def build_document_windows(
    tokenizer,
    text: str,
    entities: list[dict],
    max_length: int = 256,
    stride: int = 64,
) -> dict:
    """切正文 token。实体完整落在窗口内时，相交的第一个 WordPiece 标 B-，其余标 I-。"""
    if max_length < 3:
        raise ValueError("max_length 至少要容下 [CLS]、一个正文 token 和 [SEP]")
    content_tokens = max_length - 2
    token_ids, offsets = _content_encoding(tokenizer, text)
    kept, _dropped = flatten_entities(entities)
    windows = []
    supervised: set[tuple] = set()
    token_spans: set[tuple] = set()
    token_span_mismatch = 0
    no_token: list[dict] = []
    window_cut: list[dict] = []
    for token_start, token_end in _window_spans(len(token_ids), content_tokens, stride):
        piece_ids = token_ids[token_start:token_end]
        piece_offsets = offsets[token_start:token_end]
        if not piece_offsets:
            continue
        window_start = piece_offsets[0][0]
        window_end = piece_offsets[-1][1]
        active = [
            entity
            for entity in kept
            if entity["start_idx"] >= window_start and entity["end_idx"] <= window_end
        ]
        labels = ["O"] * len(piece_offsets)
        groups: dict[tuple, list[int]] = {}
        for index, (start, end) in enumerate(piece_offsets):
            if end <= start:
                continue
            owner = _token_owner(start, end, active)
            if owner is None:
                continue
            key = (owner["start_idx"], owner["end_idx"], owner["type"])
            groups.setdefault(key, []).append(index)
        for key, indexes in groups.items():
            if indexes != list(range(indexes[0], indexes[-1] + 1)):
                raise ValueError(f"实体 {key} 的 WordPiece 不连续，无法标成一段 BIO")
            labels[indexes[0]] = f"B-{key[2]}"
            for index in indexes[1:]:
                labels[index] = f"I-{key[2]}"
            span = (piece_offsets[indexes[0]][0], piece_offsets[indexes[-1]][1], key[2])
            token_spans.add(span)
            supervised.add(key)
        input_ids = [tokenizer.cls_token_id, *piece_ids, tokenizer.sep_token_id]
        label_ids = [-100, *[LABEL_TO_ID[label] for label in labels], -100]
        windows.append(
            {
                "input_ids": input_ids,
                "label_ids": label_ids,
                "content_labels": labels,
                "offsets": piece_offsets,
            }
        )
    for key in supervised:
        if (key[0], key[1], key[2]) not in token_spans:
            token_span_mismatch += 1
    for entity in kept:
        key = (entity["start_idx"], entity["end_idx"], entity["type"])
        if key in supervised:
            continue
        inside_some_window = False
        for window in windows:
            if not window["offsets"]:
                continue
            window_start = window["offsets"][0][0]
            window_end = window["offsets"][-1][1]
            if entity["start_idx"] >= window_start and entity["end_idx"] <= window_end:
                inside_some_window = True
                break
        if inside_some_window:
            no_token.append(entity)
        else:
            window_cut.append(entity)
    return {
        "windows": windows,
        "supervised": supervised,
        "token_spans": token_spans,
        "token_span_mismatch": token_span_mismatch,
        "no_token": no_token,
        "window_cut": window_cut,
    }


def decode_labels(labels: list[str], offsets: list[tuple[int, int]], text: str) -> list[dict]:
    """B- 开始，同类型 I- 延续。还原到原文字符下标。"""
    found = []
    index = 0
    while index < len(labels):
        label = labels[index]
        if not label.startswith("B-"):
            index += 1
            continue
        entity_type = label[2:]
        start = offsets[index][0]
        end = offsets[index][1]
        index += 1
        while index < len(labels) and labels[index] == f"I-{entity_type}":
            end = offsets[index][1]
            index += 1
        if 0 <= start < end <= len(text):
            found.append(
                {
                    "start_idx": start,
                    "end_idx": end,
                    "type": entity_type,
                    "entity": text[start:end],
                }
            )
    return found


def decode_windows(windows: list[dict], label_key: str, text: str) -> list[dict]:
    seen: set[tuple] = set()
    predictions = []
    for window in windows:
        for entity in decode_labels(window[label_key], window["offsets"], text):
            key = (entity["start_idx"], entity["end_idx"], entity["type"])
            if key in seen:
                continue
            seen.add(key)
            predictions.append(entity)
    predictions.sort(key=lambda entity: (entity["start_idx"], entity["end_idx"], entity["type"]))
    return predictions


def ids_to_labels(label_ids: list[int]) -> list[str]:
    return [LABELS[label_id] for label_id in label_ids]


def score_token_sequences(gold_sequences: list[list[str]], pred_sequences: list[list[str]]) -> dict:
    """只统计金标或预测属于 B-/I- 的正文 token。"""
    true_positive = false_positive = false_negative = 0
    if len(gold_sequences) != len(pred_sequences):
        raise ValueError("token 金标窗口数与预测窗口数不一致")
    for gold, pred in zip(gold_sequences, pred_sequences):
        if len(gold) != len(pred):
            raise ValueError("token 金标长度与预测长度不一致")
        for gold_label, pred_label in zip(gold, pred):
            same = gold_label == pred_label
            gold_entity = gold_label != "O"
            pred_entity = pred_label != "O"
            if same and gold_entity:
                true_positive += 1
            elif pred_entity and not same:
                false_positive += 1
            if gold_entity and not same:
                false_negative += 1
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


def assert_label_roundtrip(tokenizer, documents: list[dict], max_length: int, stride: int) -> dict:
    """金标 BIO 解回的实体必须等于实际标上的实体。"""
    unaligned = 0
    mismatch = 0
    window_cut = 0
    windows = 0
    multi = 0
    for document in documents:
        built = build_document_windows(
            tokenizer,
            document.get("text") or "",
            document.get("entities") or [],
            max_length,
            stride,
        )
        decoded = {
            (entity["start_idx"], entity["end_idx"], entity["type"])
            for entity in decode_windows(built["windows"], "content_labels", document.get("text") or "")
        }
        if decoded != built["token_spans"]:
            raise ValueError("BIO 解码与标注的 token 跨度不一致")
        unaligned += len(built["no_token"])
        mismatch += built["token_span_mismatch"]
        window_cut += len(built["window_cut"])
        windows += len(built["windows"])
        if len(built["windows"]) > 1:
            multi += 1
    return {
        "documents": len(documents),
        "windows": windows,
        "multi_window_documents": multi,
        "unaligned": unaligned,
        "token_span_mismatch": mismatch,
        "window_cut": window_cut,
    }
