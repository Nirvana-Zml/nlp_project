"""训练集词典匹配。类别并列时取出现次数最多者，次数相同则取类型名字典序较小者。"""

from __future__ import annotations

from collections import Counter, defaultdict


class _Node:
    __slots__ = ("children", "entity_type")

    def __init__(self) -> None:
        self.children: dict[str, _Node] = {}
        self.entity_type: str | None = None


def build_lexicon(train_docs: list[dict]) -> dict[str, str]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for doc in train_docs:
        for entity in doc.get("entities") or []:
            surface = entity.get("entity")
            entity_type = entity.get("type")
            if isinstance(surface, str) and surface and isinstance(entity_type, str):
                counts[surface][entity_type] += 1
    lexicon: dict[str, str] = {}
    for surface, type_counts in counts.items():
        best_count = max(type_counts.values())
        candidates = sorted(name for name, count in type_counts.items() if count == best_count)
        lexicon[surface] = candidates[0]
    return lexicon


def lexicon_stats(train_docs: list[dict], lexicon: dict[str, str]) -> dict:
    counts: dict[str, Counter] = defaultdict(Counter)
    for doc in train_docs:
        for entity in doc.get("entities") or []:
            surface = entity.get("entity")
            entity_type = entity.get("type")
            if isinstance(surface, str) and surface and isinstance(entity_type, str):
                counts[surface][entity_type] += 1
    ties = 0
    for surface, type_counts in counts.items():
        best_count = max(type_counts.values())
        winners = [name for name, count in type_counts.items() if count == best_count]
        if len(winners) > 1:
            ties += 1
    return {"surfaces": len(lexicon), "tie_surfaces": ties}


def compile_lexicon(lexicon: dict[str, str]) -> _Node:
    root = _Node()
    for surface, entity_type in lexicon.items():
        node = root
        for char in surface:
            node = node.children.setdefault(char, _Node())
        node.entity_type = entity_type
    return root


def find_all(text: str, lexicon: dict[str, str], root: _Node | None = None) -> list[dict]:
    """全部匹配：保留所有命中，允许重叠。root 由调用方复用，避免重复建树。"""
    root = root or compile_lexicon(lexicon)
    found: list[dict] = []
    seen: set[tuple] = set()
    for start, _char in enumerate(text):
        node = root
        for offset, char in enumerate(text[start:], start=start):
            child = node.children.get(char)
            if child is None:
                break
            node = child
            if node.entity_type is None:
                continue
            end = offset + 1
            key = (start, end, node.entity_type)
            if key in seen:
                continue
            seen.add(key)
            found.append(
                {
                    "start_idx": start,
                    "end_idx": end,
                    "type": node.entity_type,
                    "entity": text[start:end],
                }
            )
    found.sort(key=lambda item: (item["start_idx"], item["end_idx"], item["type"]))
    return found


def longest_match(matches: list[dict]) -> list[dict]:
    """位置冲突时保留更长实体；长度相同则保留起点更早者。"""
    ordered = sorted(
        matches,
        key=lambda item: (
            -(item["end_idx"] - item["start_idx"]),
            item["start_idx"],
            item["type"],
        ),
    )
    kept: list[dict] = []
    occupied: list[tuple[int, int]] = []
    for item in ordered:
        start, end = item["start_idx"], item["end_idx"]
        if any(not (end <= left or start >= right) for left, right in occupied):
            continue
        kept.append(item)
        occupied.append((start, end))
    kept.sort(key=lambda item: (item["start_idx"], item["end_idx"], item["type"]))
    return kept
