"""Qwen 提示、示例选择和输出解析。解析失败不改写成正确答案。"""

from __future__ import annotations

import json
import random

from src.data.schema import ENTITY_TYPES


def select_train_examples(documents: list[dict], seed: int = 42, count: int = 3) -> list[dict]:
    eligible = [doc for doc in documents if doc.get("entities")]
    eligible.sort(key=lambda doc: doc["source_index"])
    order = eligible.copy()
    random.Random(seed).shuffle(order)
    return order[:count]


def fill_prompt(template: str, examples: str, text: str) -> str:
    head, rest = template.split("{examples}", 1)
    middle, tail = rest.split("{text}", 1)
    return head + examples + middle + text + tail


def format_examples(documents: list[dict]) -> str:
    if not documents:
        return ""
    blocks = []
    for index, doc in enumerate(documents, start=1):
        entities = [
            {
                "entity": item["entity"],
                "type": item["type"],
                "start_idx": item["start_idx"],
                "end_idx": item["end_idx"],
            }
            for item in doc["entities"]
        ]
        payload = json.dumps({"entities": entities}, ensure_ascii=False, separators=(",", ":"))
        blocks.append(f"示例{index}\n医学文本：\n{doc['text']}\n输出：\n{payload}")
    return "\n\n".join(blocks) + "\n\n"


def parse_model_output(raw: str, text: str) -> dict:
    payload = _first_json_value(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("entities"), list):
        return {
            "pred": [],
            "parse_ok": False,
            "invalid_entities": 0,
            "surface_mismatches": 0,
        }
    pred = []
    seen: set[tuple] = set()
    invalid = 0
    mismatches = 0
    for item in payload["entities"]:
        if not isinstance(item, dict):
            invalid += 1
            continue
        entity_type = item.get("type")
        start = item.get("start_idx")
        end = item.get("end_idx")
        if not _legal_span(entity_type, start, end, text):
            invalid += 1
            continue
        surface = item.get("entity")
        if isinstance(surface, str) and text[start:end] != surface:
            mismatches += 1
        key = (start, end, entity_type)
        if key in seen:
            continue
        seen.add(key)
        pred.append(
            {
                "entity": text[start:end],
                "type": entity_type,
                "start_idx": start,
                "end_idx": end,
            }
        )
    return {
        "pred": pred,
        "parse_ok": True,
        "invalid_entities": invalid,
        "surface_mismatches": mismatches,
    }


def _legal_span(entity_type, start, end, text: str) -> bool:
    if entity_type not in ENTITY_TYPES:
        return False
    if type(start) is not int or type(end) is not int:
        return False
    return 0 <= start < end <= len(text)


def _first_json_value(raw: str):
    stripped = _strip_fence(raw.strip())
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    if start < 0:
        return None
    fragment = _matching_object(stripped, start)
    if fragment is None:
        return None
    try:
        return json.loads(fragment)
    except json.JSONDecodeError:
        return None


def _strip_fence(raw: str) -> str:
    if not raw.startswith("```"):
        return raw
    lines = raw.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _matching_object(raw: str, start: int) -> str | None:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(raw)):
        char = raw[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return raw[start : index + 1]
    return None
