"""在调参集上比较 0、1、3 条示例。三种预测完整后才选择，主测试不重跑零样本和三样本。"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import read_metrics_csv, score_documents, write_metrics_csv
from src.evaluation.shot_count import (
    SHOTS,
    assert_filled_matches,
    assert_tune_selection,
    cost_row,
    fill_test_score,
    preliminary_selection,
    select_shots,
    self_check,
)
from src.models.qwen_prompt import select_train_examples

SHOT_CONFIG = {
    "shot_counts": [0, 1, 3],
    "one_shot_source_index": [3949],
    "three_shot_source_index": [3949, 5340, 1237],
    "tune_path": "data/processed/tune.json",
    "expected_tune_documents": 1500,
    "expected_tune_gold_entities": 8147,
}
TUNE_FILES = {
    0: "qwen_shot_0_tune",
    1: "qwen_shot_1_tune",
    3: "qwen_shot_3_tune",
}
METHOD_BY_SHOT = {0: "qwen_shot_0", 1: "qwen_shot_1", 3: "qwen_shot_3"}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    self_check()
    step3 = _load_step3()
    step3._self_check_parser()
    base = yaml.safe_load((ROOT / "configs" / "qwen_3b.yaml").read_text(encoding="utf-8"))
    shot = yaml.safe_load((ROOT / "configs" / "qwen_shot_count.yaml").read_text(encoding="utf-8"))
    step3._assert_plan(base)
    _assert_shot_config(shot)

    outputs = ROOT / "outputs"
    prediction_dir = outputs / "predictions"
    metrics_dir = outputs / "metrics"
    log_path = outputs / "logs" / "qwen_shot_count.log"
    selection_path = metrics_dir / "qwen_shot_count_selection.json"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _refuse_live_lock(outputs / "logs" / "qwen_3b.lock", step3)
    step3._acquire_lock(outputs / "logs" / "qwen_shot_count.lock")
    try:
        _append_log(log_path, _header(base))
        train_docs = _read_json(ROOT / base["train_path"])
        tune_docs = _read_json(ROOT / shot["tune_path"])
        manifest = _read_json(ROOT / base["manifest_path"])
        examples_one, examples_three = _checked_examples(train_docs, manifest, base, shot)
        _checked_tune(tune_docs, manifest, shot)
        protected = _snapshot(_protected_paths(base))
        examples = {0: [], 1: examples_one, 3: examples_three}
        pending = [
            shots
            for shots in SHOTS
            if not step3._is_complete(prediction_dir / f"{TUNE_FILES[shots]}.json", tune_docs)
        ]
        model = None
        tokenizer = None
        if pending:
            _require_model(base)
            model, tokenizer = step3._load_model(base)
            for shots in pending:
                step3._run_method(
                    name=TUNE_FILES[shots],
                    docs=tune_docs,
                    examples=examples[shots],
                    template=base["prompt_template"],
                    config=base,
                    model=model,
                    tokenizer=tokenizer,
                    prediction_dir=prediction_dir,
                    log_path=log_path,
                )
                _assert_tune_file(prediction_dir / f"{TUNE_FILES[shots]}.json", tune_docs, shot, step3)
        for shots in SHOTS:
            path = prediction_dir / f"{TUNE_FILES[shots]}.json"
            if not step3._is_complete(path, tune_docs):
                raise SystemExit("三种调参集预测未完整，不做选择。")
        rows = []
        cost = {}
        for shots in SHOTS:
            records = _read_json(prediction_dir / f"{TUNE_FILES[shots]}.json")
            _assert_tune_records(records, tune_docs, shot)
            score = score_documents(records)
            if score["tp"] + score["fn"] != shot["expected_tune_gold_entities"]:
                raise SystemExit(f"{METHOD_BY_SHOT[shots]} 的 TP+FN={score['tp'] + score['fn']}，不是 8147。")
            rows.append({"method": METHOD_BY_SHOT[shots], **score})
            cost[METHOD_BY_SHOT[shots]] = cost_row(records)
            _append_log(log_path, [_format_result(METHOD_BY_SHOT[shots], score, cost[METHOD_BY_SHOT[shots]])])
        tune_csv = metrics_dir / "qwen_shot_count_tune.csv"
        write_metrics_csv(tune_csv, rows)
        cost_path = metrics_dir / "qwen_shot_count_cost.json"
        _write_json(cost_path, cost)
        tune_rows = read_metrics_csv(tune_csv)
        try:
            choice = select_shots(tune_rows)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        selection = _commit_selection(selection_path, choice, tune_rows, log_path)
        _append_log(
            log_path,
            [
                f"selected_shots={selection['selected_shots']}",
                "selection_written_before_main_test=true",
                f"test_score_source={selection['test_score_source']}",
                "regenerated_zero_or_three=false",
            ],
        )
        opened_main_test = True
        if selection["selected_shots"] in (0, 3):
            _finish_saved_main_test(selection, selection_path, metrics_dir, log_path)
        elif selection["selected_shots"] == 1:
            _finish_one_shot_main_test(
                selection,
                selection_path,
                base,
                examples_one,
                manifest,
                prediction_dir,
                metrics_dir,
                log_path,
                step3,
                model,
                tokenizer,
            )
        else:
            raise SystemExit("入选条数不是 0、1 或 3，停止。")
        _assert_unchanged(protected)
        if selection["selected_shots"] != 1 and (prediction_dir / "qwen_3b_one.json").is_file():
            raise SystemExit("未入选单样本，但写出了主测试预测，停止。")
        print("RUN_OK", flush=True)
    finally:
        step3._release_lock(outputs / "logs" / "qwen_shot_count.lock")


def _finish_saved_main_test(selection: dict, selection_path: Path, metrics_dir: Path, log_path: Path) -> None:
    rows = read_metrics_csv(metrics_dir / "qwen_3b.csv")
    matched = [row for row in rows if row["method"] == selection["test_score_method"]]
    if len(matched) != 1:
        raise SystemExit("qwen_3b.csv 中没有唯一的入选主测试行，停止。")
    cost = _read_json(metrics_dir / "qwen_3b_cost.json")
    if selection["test_score_method"] not in cost:
        raise SystemExit("qwen_3b_cost.json 中没有入选方法的主测试成本，停止。")
    _store_test_score(selection, selection_path, matched[0])
    _append_log(
        log_path,
        [
            "generated_one_shot_main_test=false",
            "main_test_prediction_reused=true",
            f"test_cost_source=outputs/metrics/qwen_3b_cost.json method={selection['test_score_method']}",
        ],
    )


def _finish_one_shot_main_test(
    selection,
    selection_path,
    base,
    examples_one,
    manifest,
    prediction_dir,
    metrics_dir,
    log_path,
    step3,
    model,
    tokenizer,
) -> None:
    test_docs = _read_json(ROOT / base["test_path"])
    step3._checked_test(test_docs, manifest, base)
    prediction_path = prediction_dir / "qwen_3b_one.json"
    already_complete = step3._is_complete(prediction_path, test_docs)
    if not already_complete:
        if "test_score" in selection:
            raise SystemExit("主测试数字已填入但单样本预测未完成，停止，不改入选条数。")
        if model is None or tokenizer is None:
            _require_model(base)
            model, tokenizer = step3._load_model(base)
        step3._run_method(
            name="qwen_3b_one",
            docs=test_docs,
            examples=examples_one,
            template=base["prompt_template"],
            config=base,
            model=model,
            tokenizer=tokenizer,
            prediction_dir=prediction_dir,
            log_path=log_path,
        )
    records = _read_json(prediction_path)
    step3._assert_records(records, test_docs, base)
    for record, doc in zip(records, test_docs, strict=True):
        if record["text"] != doc["text"] or record["gold"] != doc["entities"]:
            raise SystemExit(f"source_index={doc['source_index']} 的金标与主测试集不一致，停止。")
    score = score_documents(records)
    if score["tp"] + score["fn"] != base["expected_gold_entities"]:
        raise SystemExit(f"qwen_3b_one 的 TP+FN={score['tp'] + score['fn']}，不是 4877。")
    write_metrics_csv(metrics_dir / "qwen_3b_one.csv", [{"method": "qwen_3b_one", **score}])
    _write_json(metrics_dir / "qwen_3b_one_cost.json", {"qwen_3b_one": cost_row(records)})
    saved = read_metrics_csv(metrics_dir / "qwen_3b_one.csv")
    if len(saved) != 1 or saved[0]["method"] != "qwen_3b_one":
        raise SystemExit("qwen_3b_one.csv 不是一行主测试指标，停止。")
    _store_test_score(selection, selection_path, saved[0])
    _append_log(log_path, [_format_result("qwen_3b_one", score, cost_row(records))])
    _append_log(
        log_path,
        [
            f"generated_one_shot_main_test={str(not already_complete).lower()}",
            "main_test_prediction_reused=false",
            "test_cost_source=outputs/metrics/qwen_3b_one_cost.json method=qwen_3b_one",
        ],
    )


def _store_test_score(selection: dict, path: Path, row: dict) -> None:
    current = _read_json(path)
    if current["selected_shots"] != selection["selected_shots"]:
        raise SystemExit("选择文件中的入选条数与本次选择不一致，停止。")
    if "test_score" in current:
        try:
            assert_filled_matches(current, row)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        return
    try:
        filled = fill_test_score(current, row, datetime.now(timezone.utc).isoformat())
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if filled["selected_shots"] != current["selected_shots"]:
        raise SystemExit("填入主测试数字时入选条数发生变化，停止。")
    _write_json(path, filled)
    reloaded = _read_json(path)
    if reloaded["selected_shots"] != current["selected_shots"]:
        raise SystemExit("写回选择文件后入选条数发生变化，停止。")
    try:
        assert_filled_matches(reloaded, row)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _commit_selection(path: Path, choice: int, tune_rows: list[dict], log_path: Path) -> dict:
    if path.is_file():
        existing = _read_json(path)
        try:
            assert_tune_selection(existing, tune_rows)
        except ValueError as exc:
            raise SystemExit(f"{exc}。不改入选条数，停止。") from exc
        _append_log(log_path, ["selection_already_present=true", "selected_shots_unchanged=true"])
        return existing
    document = preliminary_selection(choice, tune_rows, datetime.now(timezone.utc).isoformat())
    _write_json(path, document)
    loaded = _read_json(path)
    if "test_score" in loaded or loaded["selected_shots"] != choice:
        raise SystemExit("选择文件初稿含有主测试数字或条数不一致，停止。")
    try:
        assert_tune_selection(loaded, tune_rows)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    _append_log(log_path, ["selection_already_present=false", "test_score_in_selection=false"])
    return loaded


def _assert_shot_config(config: dict) -> None:
    if set(config) != set(SHOT_CONFIG):
        raise SystemExit(f"configs/qwen_shot_count.yaml 的字段与计划不一致：{sorted(config)}，停止。")
    for key, expected in SHOT_CONFIG.items():
        if config.get(key) != expected:
            raise SystemExit(f"配置 {key}={config.get(key)!r} 与计划 {expected!r} 不一致，停止。")


def _checked_examples(train_docs, manifest, base, shot) -> tuple[list[dict], list[dict]]:
    chosen_three = select_train_examples(train_docs, seed=base["seed"], count=3)
    chosen_one = select_train_examples(train_docs, seed=base["seed"], count=1)
    found_three = [doc["source_index"] for doc in chosen_three]
    found_one = [doc["source_index"] for doc in chosen_one]
    if found_three != shot["three_shot_source_index"] or found_three != base["example_source_index"]:
        raise SystemExit(f"三样本编号是 {found_three}，与计划不一致，停止。")
    if found_one != shot["one_shot_source_index"] or found_one != found_three[:1]:
        raise SystemExit(f"单样本编号是 {found_one}，与计划不一致，停止。")
    blocked = set(manifest["test"]["indices"]) | set(manifest["tune"]["indices"])
    train_ids = set(manifest["train"]["indices"])
    by_index = {doc["source_index"]: doc for doc in train_docs}
    examples = []
    for source_index in found_three:
        if source_index not in train_ids or source_index in blocked:
            raise SystemExit(f"示例 {source_index} 不在 train，停止。")
        doc = by_index[source_index]
        for entity in doc["entities"]:
            if doc["text"][entity["start_idx"] : entity["end_idx"]] != entity["entity"]:
                raise SystemExit(f"示例 {source_index} 的下标与表面不一致，停止。")
        examples.append(doc)
    return [examples[0]], examples


def _checked_tune(tune_docs, manifest, shot) -> None:
    indices = [doc["source_index"] for doc in tune_docs]
    if indices != manifest["tune"]["indices"]:
        raise SystemExit("调参集与 split_manifest.json 不一致，停止。")
    if len(tune_docs) != shot["expected_tune_documents"]:
        raise SystemExit("调参集不是 1500 条，停止。")
    score = score_documents([{"gold": doc["entities"], "pred": []} for doc in tune_docs])
    if score["tp"] != 0 or score["fn"] != shot["expected_tune_gold_entities"]:
        raise SystemExit(f"调参集金标唯一三元组是 {score['fn']}，不是 8147，停止。")


def _assert_tune_file(path: Path, tune_docs, shot, step3) -> None:
    if not step3._is_complete(path, tune_docs):
        raise SystemExit(f"{path.name} 未完整，停止。")
    _assert_tune_records(_read_json(path), tune_docs, shot)


def _assert_tune_records(records: list[dict], tune_docs, shot) -> None:
    if [row["source_index"] for row in records] != [doc["source_index"] for doc in tune_docs]:
        raise SystemExit("预测文件的样本顺序与调参集不一致。")
    for record, doc in zip(records, tune_docs, strict=True):
        if record["text"] != doc["text"] or record["gold"] != doc["entities"]:
            raise SystemExit(f"source_index={doc['source_index']} 的金标与调参集不一致，停止。")
    if len(records) != shot["expected_tune_documents"]:
        raise SystemExit("预测文件不是 1500 条。")
    score = score_documents(records)
    if score["tp"] + score["fn"] != shot["expected_tune_gold_entities"]:
        raise SystemExit(f"TP+FN={score['tp'] + score['fn']}，不是 8147。")


def _header(base: dict) -> list[str]:
    return [
        f"created_at={datetime.now(timezone.utc).isoformat()}",
        "base_config=configs/qwen_3b.yaml",
        f"model_path={base['model_path']}",
        "quantization=4bit_nf4 double_quant=true compute_dtype=float16",
        f"temperature={base['temperature']}",
        f"do_sample={str(base['do_sample']).lower()}",
        f"max_new_tokens={base['max_new_tokens']}",
        f"repetition_penalty={base['repetition_penalty']}",
        f"batch_size={base['batch_size']}",
        f"seed={base['seed']}",
        "shot_counts=0,1,3",
        "one_shot_source_index=3949",
        "three_shot_source_index=3949,5340,1237",
        "example_split=train",
        "tune_documents=1500",
        "subset=false",
        "api_unit_price=none",
        "prompt_source=configs/qwen_3b.yaml",
        "prompt_template=" + json.dumps(base["prompt_template"], ensure_ascii=False),
        "on_load_failure=stop_no_precision_change_no_subset",
    ]


def _format_result(name: str, score: dict, cost: dict) -> str:
    return (
        f"RESULT method={name} precision={score['precision']:.6f} recall={score['recall']:.6f} "
        f"micro_f1={score['micro_f1']:.6f} tp={score['tp']} fp={score['fp']} fn={score['fn']} "
        f"mean_latency_sec={cost['mean_latency_sec']} mean_input_tokens={cost['mean_input_tokens']} "
        f"mean_output_tokens={cost['mean_output_tokens']} parse_failures={cost['parse_failures']} "
        f"invalid_entities={cost['invalid_entities']} surface_mismatches={cost['surface_mismatches']}"
    )


def _protected_paths(base: dict) -> list[Path]:
    return [
        ROOT / "outputs" / "predictions" / "qwen_3b_zero.json",
        ROOT / "outputs" / "predictions" / "qwen_3b_three.json",
        ROOT / "outputs" / "metrics" / "qwen_3b.csv",
        ROOT / "outputs" / "metrics" / "qwen_3b_cost.json",
        ROOT / "data" / "processed" / "split_manifest.json",
        ROOT / "configs" / "qwen_3b.yaml",
        ROOT / base["train_path"],
        ROOT / base["test_path"],
        ROOT / "data" / "processed" / "tune.json",
    ]


def _snapshot(paths: list[Path]) -> dict[Path, tuple[int, int] | None]:
    snapshot = {}
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"缺少不得改动的文件 {path}，停止。")
        stat = path.stat()
        snapshot[path] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def _assert_unchanged(snapshot: dict[Path, tuple[int, int] | None]) -> None:
    for path, old in snapshot.items():
        stat = path.stat()
        if (stat.st_mtime_ns, stat.st_size) != old:
            raise SystemExit(f"{path} 被改动，停止。")


def _require_model(config: dict) -> None:
    model_path = Path(config["model_path"])
    if not (model_path / "config.json").is_file():
        raise SystemExit(f"模型目录不可用：{model_path}。按计划停止，不改模型、不换精度、不缩小调参集。")


def _refuse_live_lock(path: Path, step3) -> None:
    if not path.is_file():
        return
    pid_text = path.read_text(encoding="utf-8").strip()
    if pid_text.isdigit() and step3._pid_alive(int(pid_text)):
        raise SystemExit(f"已有推理进程 {pid_text}，不并行启动第二次。")


def _load_step3():
    path = ROOT / "scripts" / "run_qwen_3b.py"
    spec = importlib.util.spec_from_file_location("run_qwen_3b_step3", path)
    if spec is None or spec.loader is None:
        raise SystemExit("无法加载第 3 步推理程序，停止。")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_log(path: Path, lines: list[str]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
