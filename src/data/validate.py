"""校验 CMeEE-V2 原始 json。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from src.data.schema import ENTITY_TYPES, REQUIRED_ENTITY_FIELDS


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> list:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} 的顶层不是列表")
    return data


def is_nested(entities: list[dict]) -> bool:
    """是否存在严格包含关系。下标为左闭右开。"""
    spans = []
    for entity in entities:
        start, end = entity.get("start_idx"), entity.get("end_idx")
        if isinstance(start, int) and isinstance(end, int) and end > start:
            spans.append((start, end))
    spans.sort()
    for index, (start, end) in enumerate(spans):
        for other_start, other_end in spans[index + 1 :]:
            if other_start >= end:
                break
            if start <= other_start and other_end <= end and (start, end) != (other_start, other_end):
                return True
    return False


def validate_documents(documents: list, filename: str) -> tuple[dict, list[dict]]:
    type_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    errors: list[dict] = []
    entity_total = 0
    empty_docs = 0
    nested_docs = 0

    for doc_index, doc in enumerate(documents):
        if not isinstance(doc, dict):
            error_counts["doc_not_object"] += 1
            errors.append({"file": filename, "doc_index": doc_index, "reason": "doc_not_object"})
            continue
        text = doc.get("text")
        if not isinstance(text, str):
            error_counts["missing_text"] += 1
            errors.append({"file": filename, "doc_index": doc_index, "reason": "missing_text"})
            text = ""
        entities = doc.get("entities")
        if not isinstance(entities, list):
            error_counts["entities_not_list"] += 1
            errors.append({"file": filename, "doc_index": doc_index, "reason": "entities_not_list"})
            continue
        if len(entities) == 0:
            empty_docs += 1
        seen: set[tuple] = set()
        valid_for_nesting: list[dict] = []
        for entity_index, entity in enumerate(entities):
            entity_total += 1
            if not isinstance(entity, dict):
                error_counts["entity_not_object"] += 1
                errors.append(
                    {
                        "file": filename,
                        "doc_index": doc_index,
                        "entity_index": entity_index,
                        "reason": "entity_not_object",
                    }
                )
                continue
            missing = [field for field in REQUIRED_ENTITY_FIELDS if field not in entity]
            if missing:
                error_counts["missing_field"] += 1
                errors.append(
                    {
                        "file": filename,
                        "doc_index": doc_index,
                        "entity_index": entity_index,
                        "reason": "missing_field",
                        "fields": missing,
                    }
                )
                continue
            start, end = entity["start_idx"], entity["end_idx"]
            entity_type = entity["type"]
            surface = entity["entity"]
            if entity_type not in ENTITY_TYPES:
                error_counts["illegal_type"] += 1
                errors.append(
                    {
                        "file": filename,
                        "doc_index": doc_index,
                        "entity_index": entity_index,
                        "reason": "illegal_type",
                        "type": entity_type,
                    }
                )
            identity = (start, end, entity_type, surface)
            if identity in seen:
                error_counts["duplicate_entity"] += 1
                errors.append(
                    {
                        "file": filename,
                        "doc_index": doc_index,
                        "entity_index": entity_index,
                        "reason": "duplicate_entity",
                        "start_idx": start,
                        "end_idx": end,
                        "type": entity_type,
                    }
                )
            seen.add(identity)
            if not isinstance(start, int) or not isinstance(end, int) or not isinstance(surface, str):
                error_counts["bad_offset_type"] += 1
                errors.append(
                    {
                        "file": filename,
                        "doc_index": doc_index,
                        "entity_index": entity_index,
                        "reason": "bad_offset_type",
                    }
                )
                continue
            if not (0 <= start <= end <= len(text)) or text[start:end] != surface:
                error_counts["offset_mismatch"] += 1
                errors.append(
                    {
                        "file": filename,
                        "doc_index": doc_index,
                        "entity_index": entity_index,
                        "reason": "offset_mismatch",
                        "start_idx": start,
                        "end_idx": end,
                        "entity": surface,
                    }
                )
                continue
            if entity_type in ENTITY_TYPES:
                type_counts[entity_type] += 1
                valid_for_nesting.append(entity)
        if is_nested(valid_for_nesting):
            nested_docs += 1

    summary = {
        "docs": len(documents),
        "entities": entity_total,
        "empty_docs": empty_docs,
        "nested_docs": nested_docs,
        "type_counts": {name: type_counts[name] for name in ENTITY_TYPES},
        "error_counts": dict(error_counts),
    }
    return summary, errors
