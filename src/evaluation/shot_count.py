"""提示条数的选择规则。选择只看调参集 CSV 的 6 位小数 Micro-F1。"""

from __future__ import annotations

import re
from decimal import Decimal

TUNE_METHODS = ("qwen_shot_0", "qwen_shot_1", "qwen_shot_3")
SHOTS = (0, 1, 3)
SCORE_FIELDS = ("method", "precision", "recall", "micro_f1", "tp", "fp", "fn")
PRELIMINARY_KEYS = (
    "selected_shots",
    "tune_scores",
    "test_score_source",
    "test_score_method",
    "selection_written_at",
)
FILLED_KEYS = PRELIMINARY_KEYS + ("test_score", "test_score_filled_at")
TEST_SOURCE = {
    0: ("outputs/metrics/qwen_3b.csv", "qwen_3b_zero"),
    1: ("outputs/metrics/qwen_3b_one.csv", "qwen_3b_one"),
    3: ("outputs/metrics/qwen_3b.csv", "qwen_3b_three"),
}
_SIX = re.compile(r"\d+\.\d{6}")


def six_decimal(text: str) -> Decimal:
    if not isinstance(text, str) or _SIX.fullmatch(text) is None:
        raise ValueError(f"Micro-F1 不是 6 位小数字符串：{text!r}")
    return Decimal(text)


def select_shots(rows: list[dict]) -> int:
    if [row.get("method") for row in rows] != list(TUNE_METHODS):
        raise ValueError("调参集 CSV 必须按 qwen_shot_0、qwen_shot_1、qwen_shot_3 排列")
    best_key = None
    best_shots = None
    for row in rows:
        shots = int(str(row["method"]).removeprefix("qwen_shot_"))
        key = (-six_decimal(row["micro_f1"]), shots)
        if best_key is None or key < best_key:
            best_key = key
            best_shots = shots
    if best_shots not in SHOTS:
        raise ValueError("没有选出条数")
    return best_shots


def preliminary_selection(selected_shots: int, tune_rows: list[dict], written_at: str) -> dict:
    if type(selected_shots) is not int or selected_shots not in SHOTS:
        raise ValueError(f"入选条数非法：{selected_shots!r}")
    if [row.get("method") for row in tune_rows] != list(TUNE_METHODS):
        raise ValueError("调参集分数行不完整")
    source, method = TEST_SOURCE[selected_shots]
    document = {
        "selected_shots": selected_shots,
        "tune_scores": [{field: row[field] for field in SCORE_FIELDS} for row in tune_rows],
        "test_score_source": source,
        "test_score_method": method,
        "selection_written_at": written_at,
    }
    if set(document) != set(PRELIMINARY_KEYS) or "test_score" in document:
        raise ValueError("选择文件初稿包含主测试数字")
    return document


def fill_test_score(selection: dict, test_row: dict, filled_at: str) -> dict:
    if "test_score" in selection:
        raise ValueError("主测试数字已经填入")
    shots = selection["selected_shots"]
    if type(shots) is not int or test_row.get("method") != selection["test_score_method"]:
        raise ValueError("主测试行与入选条数不一致")
    updated = {
        "selected_shots": shots,
        "tune_scores": selection["tune_scores"],
        "test_score_source": selection["test_score_source"],
        "test_score_method": selection["test_score_method"],
        "selection_written_at": selection["selection_written_at"],
        "test_score": {field: test_row[field] for field in SCORE_FIELDS},
        "test_score_filled_at": filled_at,
    }
    if updated["selected_shots"] != shots or set(updated) != set(FILLED_KEYS):
        raise ValueError("填入主测试数字时入选条数发生变化")
    return updated


def assert_filled_matches(selection: dict, test_row: dict) -> None:
    if "test_score" not in selection:
        raise ValueError("选择文件还没有主测试数字")
    if selection["test_score_method"] != test_row["method"]:
        raise ValueError("主测试方法与出处不一致")
    for field in SCORE_FIELDS:
        if selection["test_score"][field] != test_row[field]:
            raise ValueError(f"选择文件的主测试 {field} 与出处不一致")
    expected_source, expected_method = TEST_SOURCE[selection["selected_shots"]]
    if selection["test_score_source"] != expected_source or expected_method != test_row["method"]:
        raise ValueError("主测试出处与入选条数不一致")


def assert_tune_selection(selection: dict, tune_rows: list[dict]) -> int:
    choice = select_shots(tune_rows)
    if selection.get("selected_shots") != choice:
        raise ValueError(f"入选条数 {selection.get('selected_shots')} 与调参集 CSV 的 {choice} 不一致")
    stored = selection.get("tune_scores")
    expected = [{field: row[field] for field in SCORE_FIELDS} for row in tune_rows]
    if stored != expected:
        raise ValueError("选择文件中的调参集分数与 CSV 不一致")
    source, method = TEST_SOURCE[choice]
    if selection.get("test_score_source") != source or selection.get("test_score_method") != method:
        raise ValueError("主测试出处与入选条数不一致")
    return choice


def formatted_score(method: str, score: dict) -> dict:
    return {
        "method": method,
        "precision": f"{score['precision']:.6f}",
        "recall": f"{score['recall']:.6f}",
        "micro_f1": f"{score['micro_f1']:.6f}",
        "tp": str(score["tp"]),
        "fp": str(score["fp"]),
        "fn": str(score["fn"]),
    }


def cost_row(records: list[dict]) -> dict:
    count = len(records)
    if count == 0:
        raise ValueError("没有预测可汇总")
    return {
        "documents": count,
        "mean_latency_sec": round(sum(row["latency_sec"] for row in records) / count, 6),
        "mean_input_tokens": round(sum(row["input_tokens"] for row in records) / count, 6),
        "mean_output_tokens": round(sum(row["output_tokens"] for row in records) / count, 6),
        "parse_failures": sum(1 for row in records if not row["parse_ok"]),
        "invalid_entities": sum(row["invalid_entities"] for row in records),
        "surface_mismatches": sum(row["surface_mismatches"] for row in records),
    }


def self_check() -> None:
    higher = _rows("0.100000", "0.200000", "0.150000")
    if select_shots(higher) != 1:
        raise SystemExit("选择自检失败：更高的 6 位小数应入选。")
    if select_shots(_rows("0.300000", "0.200000", "0.100000")) != 0:
        raise SystemExit("选择自检失败：零样本更高时应入选。")
    if select_shots(_rows("0.100000", "0.200000", "0.300000")) != 3:
        raise SystemExit("选择自检失败：三样本更高时应入选。")
    if select_shots(_rows("0.200000", "0.200000", "0.200000")) != 0:
        raise SystemExit("选择自检失败：并列时应取更少示例。")
    if select_shots(_rows("0.100000", "0.200000", "0.200000")) != 1:
        raise SystemExit("选择自检失败：单样本与三样本并列时应取单样本。")
    if select_shots(_rows("0.200000", "0.100000", "0.200000")) != 0:
        raise SystemExit("选择自检失败：零样本与三样本并列时应取零样本。")
    try:
        six_decimal("0.2000000")
    except ValueError:
        pass
    else:
        raise SystemExit("选择自检失败：超过 6 位小数应拒绝。")
    for shots, source, method in (
        (0, "outputs/metrics/qwen_3b.csv", "qwen_3b_zero"),
        (1, "outputs/metrics/qwen_3b_one.csv", "qwen_3b_one"),
        (3, "outputs/metrics/qwen_3b.csv", "qwen_3b_three"),
    ):
        document = preliminary_selection(shots, higher, "2026-10-15T00:00:00+00:00")
        if set(document) != set(PRELIMINARY_KEYS) or document["test_score_source"] != source:
            raise SystemExit("选择文件初稿自检失败。")
        if document["test_score_method"] != method or "test_score" in document:
            raise SystemExit("选择文件初稿不应含主测试数字。")
    draft = preliminary_selection(1, higher, "2026-10-15T00:00:00+00:00")
    one_row = {
        "method": "qwen_3b_one",
        "precision": "0.200000",
        "recall": "0.200000",
        "micro_f1": "0.200000",
        "tp": "2",
        "fp": "3",
        "fn": "4",
    }
    filled = fill_test_score(draft, one_row, "later")
    if filled["selected_shots"] != 1 or filled["test_score"] != one_row:
        raise SystemExit("填入主测试数字时改动了入选条数。")
    if "test_score" in draft:
        raise SystemExit("填入主测试数字时改写了选择文件初稿。")
    assert_filled_matches(filled, one_row)
    if assert_tune_selection(draft, higher) != 1:
        raise SystemExit("调参集核对自检失败。")


def _rows(first: str, second: str, third: str) -> list[dict]:
    rows = []
    for method, micro in zip(TUNE_METHODS, (first, second, third), strict=True):
        rows.append(
            {
                "method": method,
                "precision": micro,
                "recall": micro,
                "micro_f1": micro,
                "tp": "1",
                "fp": "1",
                "fn": "1",
            }
        )
    return rows
