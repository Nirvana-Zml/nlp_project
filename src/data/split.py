"""按种子 42 固定划分。划分规则写死后不再随指标改动。"""

from __future__ import annotations

import random
import re

from src.data.schema import ENTITY_TYPES
from src.data.validate import is_nested

_ASCII = re.compile(r"[A-Za-z0-9]")


def document_features(doc: dict) -> set[str]:
    features = set()
    text = doc.get("text") or ""
    entities = doc.get("entities") or []
    if is_nested(entities):
        features.add("nested")
    else:
        features.add("non_nested")
    if _ASCII.search(text):
        features.add("has_ascii")
    for entity in entities:
        entity_type = entity.get("type")
        if entity_type in ENTITY_TYPES:
            features.add(f"type:{entity_type}")
        start, end = entity.get("start_idx"), entity.get("end_idx")
        if isinstance(start, int) and isinstance(end, int) and end >= start:
            length = end - start
            if length <= 2:
                features.add("len:1-2")
            elif length <= 5:
                features.add("len:3-5")
            else:
                features.add("len:6+")
    return features


def required_features() -> set[str]:
    features = {f"type:{name}" for name in ENTITY_TYPES}
    features.update({"nested", "non_nested", "has_ascii", "len:1-2", "len:3-5", "len:6+"})
    return features


def split_train_tune(documents: list, train_size: int, tune_size: int, seed: int) -> tuple[list[int], list[int]]:
    if train_size + tune_size != len(documents):
        raise ValueError(f"train {train_size} + tune {tune_size} != {len(documents)}")
    indices = list(range(len(documents)))
    random.Random(seed).shuffle(indices)
    return indices[:train_size], indices[train_size:]


def split_main_test(documents: list, test_size: int, seed: int) -> tuple[list[int], list[str]]:
    """独立的 Random(seed)。先按洗牌顺序补齐覆盖条件，再补满条数。"""
    if test_size > len(documents):
        raise ValueError(f"test_size {test_size} > {len(documents)}")
    order = list(range(len(documents)))
    random.Random(seed).shuffle(order)
    needed = required_features()
    selected: list[int] = []
    selected_set: set[int] = set()
    for index in order:
        if not needed:
            break
        features = document_features(documents[index])
        hit = features & needed
        if hit:
            selected.append(index)
            selected_set.add(index)
            needed -= hit
    for index in order:
        if len(selected) >= test_size:
            break
        if index not in selected_set:
            selected.append(index)
            selected_set.add(index)
    uncovered = sorted(needed)
    return selected, uncovered


def take(documents: list, indices: list[int]) -> list[dict]:
    chosen = []
    for index in indices:
        doc = dict(documents[index])
        doc["source_index"] = index
        chosen.append(doc)
    return chosen
