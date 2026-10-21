"""Qwen2.5-3B 零样本和三样本。每种方法在主测试集上只生成一次。"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.schema import ENTITY_TYPES
from src.evaluation.metrics import score_documents, write_metrics_csv
from src.local_resources import cuda_issue, qwen_directory, qwen_issues
from src.models.qwen_prompt import fill_prompt, format_examples, parse_model_output, select_train_examples

METHODS = ("qwen_3b_zero", "qwen_3b_three")
PLAN = {
    "quantization": "4bit_nf4",
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "float16",
    "temperature": 0,
    "do_sample": False,
    "max_new_tokens": 1536,
    "repetition_penalty": 1.0,
    "batch_size": 1,
    "seed": 42,
    "example_source_index": [3949, 5340, 1237],
    "expected_test_documents": 1000,
    "expected_gold_entities": 4877,
}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    config = yaml.safe_load((ROOT / "configs" / "qwen_3b.yaml").read_text(encoding="utf-8"))
    _assert_plan(config)
    _self_check_parser()
    issues = qwen_issues()
    if issues:
        raise SystemExit("\n\n".join(issues) + "\n按计划停止，不改模型、不换精度。")

    train_docs = _read_json(ROOT / config["train_path"])
    test_docs = _read_json(ROOT / config["test_path"])
    manifest = _read_json(ROOT / config["manifest_path"])
    examples = _checked_examples(train_docs, test_docs, manifest, config)
    _checked_test(test_docs, manifest, config)

    outputs = ROOT / "outputs"
    prediction_dir = outputs / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    log_path = outputs / "logs" / "qwen_3b.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _acquire_lock(outputs / "logs" / "qwen_3b.lock")
    try:
        _append_log(
            log_path,
            [
                f"created_at={datetime.now(timezone.utc).isoformat()}",
                f"model_path={config['model_path']}",
                "quantization=4bit_nf4 double_quant=true compute_dtype=float16",
                f"temperature={config['temperature']}",
                f"do_sample={str(config['do_sample']).lower()}",
                f"max_new_tokens={config['max_new_tokens']}",
                f"repetition_penalty={config['repetition_penalty']}",
                f"batch_size={config['batch_size']}",
                f"seed={config['seed']}",
                "example_source_index=3949,5340,1237",
                "example_split=train",
                f"test_documents={len(test_docs)}",
                "api_unit_price=none",
            ],
        )
        pending = [name for name in METHODS if not _is_complete(prediction_dir / f"{name}.json", test_docs)]
        if pending:
            model, tokenizer = _load_model(config)
            for name in pending:
                use_examples = name == "qwen_3b_three"
                _run_method(
                    name=name,
                    docs=test_docs,
                    examples=examples if use_examples else [],
                    template=config["prompt_template"],
                    config=config,
                    model=model,
                    tokenizer=tokenizer,
                    prediction_dir=prediction_dir,
                    log_path=log_path,
                )
        rows = []
        cost = {}
        for name in METHODS:
            records = _read_json(prediction_dir / f"{name}.json")
            _assert_records(records, test_docs, config)
            score = score_documents(records)
            if score["tp"] + score["fn"] != config["expected_gold_entities"]:
                raise SystemExit(f"{name} 的 TP+FN={score['tp'] + score['fn']}，不是 4877。")
            rows.append({"method": name, **score})
            cost[name] = _cost_row(records)
            _append_log(log_path, [_format_result(name, score, cost[name])])
        write_metrics_csv(outputs / "metrics" / "qwen_3b.csv", rows)
        cost_path = outputs / "metrics" / "qwen_3b_cost.json"
        cost_path.write_text(json.dumps(cost, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("RUN_OK")
    finally:
        _release_lock(outputs / "logs" / "qwen_3b.lock")


def _assert_plan(config: dict) -> None:
    for key, expected in PLAN.items():
        if config.get(key) != expected:
            raise SystemExit(f"配置 {key}={config.get(key)!r} 与计划 {expected!r} 不一致，停止。")
    template = config["prompt_template"]
    if "{examples}" not in template or "{text}" not in template:
        raise SystemExit("提示模板缺少占位符，停止。")
    for name in ENTITY_TYPES:
        if name not in template:
            raise SystemExit(f"提示模板缺少类别 {name}，停止。")
    if "左闭右开" not in template:
        raise SystemExit("提示模板没有写左闭右开，停止。")


def _self_check_parser() -> None:
    text = "患者发热。"
    raw = (
        '{"entities":['
        '{"entity":"发热","type":"sym","start_idx":2,"end_idx":4},'
        '{"entity":"错","type":"nope","start_idx":0,"end_idx":1},'
        '{"entity":"发热","type":"sym","start_idx":2,"end_idx":4},'
        '{"entity":"X","type":"sym","start_idx":2,"end_idx":4}'
        "]}"
    )
    parsed = parse_model_output(raw, text)
    if not parsed["parse_ok"] or len(parsed["pred"]) != 1:
        raise SystemExit("解析自检失败。")
    if parsed["invalid_entities"] != 1 or parsed["surface_mismatches"] != 1:
        raise SystemExit("无效实体或表面不一致计数自检失败。")
    if parse_model_output("不是 JSON", text)["parse_ok"]:
        raise SystemExit("解析失败自检没有拦住非法输出。")
    fenced = '说明\n```json\n{"entities":[{"entity":"发热","type":"sym","start_idx":2,"end_idx":4}]}\n```'
    if not parse_model_output(fenced, text)["parse_ok"]:
        raise SystemExit("代码块 JSON 自检失败。")


def _checked_examples(train_docs, test_docs, manifest, config) -> list[dict]:
    test_ids = set(manifest["test"]["indices"])
    tune_ids = set(manifest["tune"]["indices"])
    chosen = select_train_examples(train_docs, seed=config["seed"], count=3)
    found = [doc["source_index"] for doc in chosen]
    if found != config["example_source_index"]:
        raise SystemExit(f"三样本编号是 {found}，与计划不一致，停止。")
    by_index = {doc["source_index"]: doc for doc in train_docs}
    examples = []
    for source_index in found:
        doc = by_index[source_index]
        if source_index in test_ids or source_index in tune_ids:
            raise SystemExit(f"示例 {source_index} 不在 train，停止。")
        for entity in doc["entities"]:
            if doc["text"][entity["start_idx"] : entity["end_idx"]] != entity["entity"]:
                raise SystemExit(f"示例 {source_index} 的下标与表面不一致，停止。")
        examples.append(doc)
    del test_docs
    return examples


def _checked_test(test_docs, manifest, config) -> None:
    indices = [doc["source_index"] for doc in test_docs]
    if indices != manifest["test"]["indices"]:
        raise SystemExit("主测试集与 split_manifest.json 不一致，停止。")
    if len(test_docs) != config["expected_test_documents"]:
        raise SystemExit("主测试集不是 1000 条，停止。")
    empty = [{"gold": doc["entities"], "pred": []} for doc in test_docs]
    score = score_documents(empty)
    if score["tp"] != 0 or score["fn"] != config["expected_gold_entities"]:
        raise SystemExit(f"主测试金标唯一三元组是 {score['fn']}，不是 4877，停止。")


def _load_model(config: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    issues = cuda_issue()
    if issues:
        raise SystemExit("\n\n".join(issues))
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=config["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=torch.float16,
    )
    path = str(qwen_directory())
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            quantization_config=quant,
            device_map="auto",
            local_files_only=True,
        )
    except Exception as exc:
        raise SystemExit(f"4bit 加载失败，按计划停止，不换精度、不缩小测试集：{exc}") from exc
    model.eval()
    model.generation_config.do_sample = False
    model.generation_config.repetition_penalty = float(config["repetition_penalty"])
    return model, tokenizer


def _run_method(name, docs, examples, template, config, model, tokenizer, prediction_dir, log_path) -> None:
    import torch

    final_path = prediction_dir / f"{name}.json"
    partial_path = prediction_dir / f"{name}.partial.jsonl"
    done = _read_partial(partial_path)
    example_text = format_examples(examples)
    _append_log(log_path, [f"start method={name} already_done={len(done)} remaining={len(docs) - len(done)}"])
    print(f"START {name} done={len(done)} total={len(docs)}", flush=True)
    with partial_path.open("a", encoding="utf-8") as handle:
        for position, doc in enumerate(docs, start=1):
            source_index = doc["source_index"]
            if source_index in done:
                continue
            prompt = fill_prompt(template, example_text, doc["text"])
            record = _generate_one(model, tokenizer, prompt, doc, config)
            line = json.dumps(record, ensure_ascii=False)
            handle.write(line + "\n")
            handle.flush()
            done[source_index] = record
            if position == 1 or position % 10 == 0 or position == len(docs):
                print(
                    f"PROGRESS {name} {len(done)}/{len(docs)} "
                    f"latency_sec={record['latency_sec']:.3f} "
                    f"output_tokens={record['output_tokens']} parse_ok={record['parse_ok']}",
                    flush=True,
                )
            del prompt
            if torch.cuda.is_available() and len(done) % 20 == 0:
                torch.cuda.empty_cache()
    ordered = []
    for doc in docs:
        record = done.get(doc["source_index"])
        if record is None:
            raise SystemExit(f"{name} 缺少 source_index={doc['source_index']}，停止。")
        ordered.append(record)
    final_path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    partial_path.unlink(missing_ok=True)
    print(f"WROTE {final_path}", flush=True)


def _generate_one(model, tokenizer, prompt: str, doc: dict, config: dict) -> dict:
    import torch

    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    input_tokens = int(inputs["input_ids"].shape[1])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=int(config["max_new_tokens"]),
            do_sample=False,
            repetition_penalty=float(config["repetition_penalty"]),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time.perf_counter() - started
    new_ids = output_ids[0, input_tokens:]
    raw_output = tokenizer.decode(new_ids, skip_special_tokens=True)
    parsed = parse_model_output(raw_output, doc["text"])
    return {
        "source_index": doc["source_index"],
        "text": doc["text"],
        "gold": doc["entities"],
        "pred": parsed["pred"],
        "raw_output": raw_output,
        "latency_sec": latency,
        "input_tokens": input_tokens,
        "output_tokens": int(new_ids.shape[0]),
        "parse_ok": parsed["parse_ok"],
        "invalid_entities": parsed["invalid_entities"],
        "surface_mismatches": parsed["surface_mismatches"],
    }


def _read_partial(path: Path) -> dict[int, dict]:
    records = {}
    if not path.is_file():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        records.setdefault(record["source_index"], record)
    return records


def _is_complete(path: Path, docs: list[dict]) -> bool:
    if not path.is_file():
        return False
    records = _read_json(path)
    return [row["source_index"] for row in records] == [doc["source_index"] for doc in docs]


def _assert_records(records: list[dict], docs: list[dict], config: dict) -> None:
    if [row["source_index"] for row in records] != [doc["source_index"] for doc in docs]:
        raise SystemExit("预测文件的样本顺序与主测试集不一致。")
    if len(records) != config["expected_test_documents"]:
        raise SystemExit("预测文件不是 1000 条。")


def _cost_row(records: list[dict]) -> dict:
    count = len(records)
    return {
        "documents": count,
        "mean_latency_sec": round(sum(row["latency_sec"] for row in records) / count, 6),
        "mean_input_tokens": round(sum(row["input_tokens"] for row in records) / count, 6),
        "mean_output_tokens": round(sum(row["output_tokens"] for row in records) / count, 6),
        "parse_failures": sum(1 for row in records if not row["parse_ok"]),
        "invalid_entities": sum(row["invalid_entities"] for row in records),
        "surface_mismatches": sum(row["surface_mismatches"] for row in records),
    }


def _format_result(name: str, score: dict, cost: dict) -> str:
    return (
        f"RESULT method={name} precision={score['precision']:.6f} recall={score['recall']:.6f} "
        f"micro_f1={score['micro_f1']:.6f} tp={score['tp']} fp={score['fp']} fn={score['fn']} "
        f"mean_latency_sec={cost['mean_latency_sec']} mean_input_tokens={cost['mean_input_tokens']} "
        f"mean_output_tokens={cost['mean_output_tokens']} parse_failures={cost['parse_failures']} "
        f"invalid_entities={cost['invalid_entities']} surface_mismatches={cost['surface_mismatches']}"
    )


def _append_log(path: Path, lines: list[str]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _acquire_lock(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        pid_text = path.read_text(encoding="utf-8").strip()
        if pid_text.isdigit() and _pid_alive(int(pid_text)):
            raise SystemExit(f"已有推理进程 {pid_text}，不并行启动第二次。")
    path.write_text(str(os.getpid()), encoding="utf-8")


def _release_lock(path: Path) -> None:
    if path.is_file() and path.read_text(encoding="utf-8").strip() == str(os.getpid()):
        path.unlink()


def _pid_alive(pid: int) -> bool:
    import ctypes

    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


if __name__ == "__main__":
    main()
