"""只从预测文件重算提示数量的 CSV 和成本，不重新生成。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import read_metrics_csv, read_predictions, score_documents
from src.evaluation.shot_count import (
    TUNE_METHODS,
    assert_filled_matches,
    assert_tune_selection,
    cost_row,
    formatted_score,
    self_check,
)

TUNE_FILES = {
    "qwen_shot_0": "qwen_shot_0_tune.json",
    "qwen_shot_1": "qwen_shot_1_tune.json",
    "qwen_shot_3": "qwen_shot_3_tune.json",
}
TUNE_GOLD = 8147
TEST_GOLD = 4877


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    self_check()
    outputs = ROOT / "outputs"
    tune_rows = read_metrics_csv(outputs / "metrics" / "qwen_shot_count_tune.csv")
    if [row["method"] for row in tune_rows] != list(TUNE_METHODS):
        raise SystemExit(f"tune CSV 方法顺序不是 {TUNE_METHODS}")
    saved_cost = _read_json(outputs / "metrics" / "qwen_shot_count_cost.json")
    orders = []
    for method, filename in TUNE_FILES.items():
        records = read_predictions(outputs / "predictions" / filename)
        if len(records) != 1500:
            raise SystemExit(f"{method} 不是 1500 条")
        _assert_raw(records, method)
        orders.append([row["source_index"] for row in records])
        score = score_documents(records)
        if score["tp"] + score["fn"] != TUNE_GOLD:
            raise SystemExit(f"{method} 的 TP+FN={score['tp'] + score['fn']}")
        fresh = formatted_score(method, score)
        saved = tune_rows[list(TUNE_METHODS).index(method)]
        _assert_same_row(method, fresh, saved)
        if cost_row(records) != saved_cost[method]:
            raise SystemExit(f"{method} 的成本重算与 qwen_shot_count_cost.json 不一致")
    if orders[0] != orders[1] or orders[1] != orders[2]:
        raise SystemExit("三种调参集预测的样本顺序不一致")
    selection = _read_json(outputs / "metrics" / "qwen_shot_count_selection.json")
    if "test_score" not in selection:
        raise SystemExit("选择文件还没有主测试数字")
    try:
        choice = assert_tune_selection(selection, tune_rows)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if choice == 1:
        _recompute_one(outputs, selection)
    else:
        _recompute_saved(outputs, selection)
    print("RECOMPUTE_OK")


def _recompute_one(outputs: Path, selection: dict) -> None:
    prediction = outputs / "predictions" / "qwen_3b_one.json"
    csv_path = outputs / "metrics" / "qwen_3b_one.csv"
    cost_path = outputs / "metrics" / "qwen_3b_one_cost.json"
    if not prediction.is_file() or not csv_path.is_file() or not cost_path.is_file():
        raise SystemExit("单样本入选，但主测试预测、CSV 或成本文件缺失")
    records = read_predictions(prediction)
    if len(records) != 1000:
        raise SystemExit("qwen_3b_one 不是 1000 条")
    _assert_raw(records, "qwen_3b_one")
    score = score_documents(records)
    if score["tp"] + score["fn"] != TEST_GOLD:
        raise SystemExit(f"qwen_3b_one 的 TP+FN={score['tp'] + score['fn']}")
    fresh = formatted_score("qwen_3b_one", score)
    saved_rows = read_metrics_csv(csv_path)
    if len(saved_rows) != 1:
        raise SystemExit("qwen_3b_one.csv 不是一行")
    _assert_same_row("qwen_3b_one", fresh, saved_rows[0])
    if {"qwen_3b_one": cost_row(records)} != _read_json(cost_path):
        raise SystemExit("qwen_3b_one 的成本重算与 qwen_3b_one_cost.json 不一致")
    try:
        assert_filled_matches(selection, saved_rows[0])
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _recompute_saved(outputs: Path, selection: dict) -> None:
    stray = [
        outputs / "predictions" / "qwen_3b_one.json",
        outputs / "metrics" / "qwen_3b_one.csv",
        outputs / "metrics" / "qwen_3b_one_cost.json",
    ]
    present = [path for path in stray if path.is_file()]
    if present:
        raise SystemExit("未入选单样本，但存在单样本主测试文件")
    saved_rows = read_metrics_csv(outputs / "metrics" / "qwen_3b.csv")
    matched = [row for row in saved_rows if row["method"] == selection["test_score_method"]]
    if len(matched) != 1:
        raise SystemExit("qwen_3b.csv 中没有唯一的入选主测试行")
    try:
        assert_filled_matches(selection, matched[0])
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    cost = _read_json(outputs / "metrics" / "qwen_3b_cost.json")
    if selection["test_score_method"] not in cost:
        raise SystemExit("qwen_3b_cost.json 中没有入选方法的主测试成本")


def _assert_same_row(method: str, fresh: dict, saved: dict) -> None:
    for field, value in fresh.items():
        if saved[field] != value:
            raise SystemExit(f"{method} 的 {field} 重算值 {value} 与 CSV {saved[field]} 不一致")


def _assert_raw(records: list[dict], method: str) -> None:
    missing = sum(1 for record in records if "raw_output" not in record)
    if missing:
        raise SystemExit(f"{method} 有 {missing} 条没有原始输出")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
