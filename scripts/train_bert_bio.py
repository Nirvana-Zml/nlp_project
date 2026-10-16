"""按 configs/bert_bio.yaml 训练 BERT-BIO，主测试集只评一次。"""

from __future__ import annotations

import gc
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertForTokenClassification, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import score_documents, write_metrics_csv
from src.models.bio import (
    LABELS,
    LABEL_TO_ID,
    assert_expected_drops,
    build_document_windows,
    decode_windows,
    drop_report,
    ids_to_labels,
    score_token_sequences,
)

PLAN_LEARNING_RATE = 3.0e-5


class WindowDataset(Dataset):
    def __init__(self, examples: list[dict]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_documents(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} 的顶层不是列表")
    return data


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_oom(exc: BaseException) -> bool:
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    return "out of memory" in str(exc).lower()


def assert_plan_config(config: dict) -> None:
    expected = {
        "max_length": 256,
        "content_tokens": 254,
        "window_stride": 64,
        "batch_size": 8,
        "gradient_accumulation_steps": 2,
        "effective_batch_size": 16,
        "epochs": 5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "fp16": True,
        "seed": 42,
        "early_stopping_patience": 2,
        "model_path": "D:/models/bert-base-chinese",
    }
    for key, value in expected.items():
        if config[key] != value:
            raise ValueError(f"配置 {key}={config[key]!r} 与计划 {value!r} 不一致")
    if abs(float(config["learning_rate"]) - PLAN_LEARNING_RATE) > 1e-15:
        raise ValueError(f"学习率 {config['learning_rate']} 不是计划中的 3e-5")
    if int(config["max_length"]) - 2 != int(config["content_tokens"]):
        raise ValueError("正文 token 数必须是最大长度减去 [CLS] 和 [SEP]")


def prepare_split(tokenizer, documents: list[dict], max_length: int, stride: int) -> tuple[list[list[dict]], list[dict], dict]:
    windows_by_doc: list[list[dict]] = []
    examples: list[dict] = []
    stats = {
        "documents": len(documents),
        "windows": 0,
        "multi_window_documents": 0,
        "token_span_mismatch": 0,
        "no_token": 0,
        "window_cut": 0,
    }
    for doc_index, document in enumerate(documents):
        text = document.get("text") or ""
        built = build_document_windows(
            tokenizer,
            text,
            document.get("entities") or [],
            max_length=max_length,
            stride=stride,
        )
        decoded = {
            (entity["start_idx"], entity["end_idx"], entity["type"])
            for entity in decode_windows(built["windows"], "content_labels", text)
        }
        if decoded != built["token_spans"]:
            raise ValueError(f"样本 source_index={document.get('source_index')} 的 BIO 解码与 token 跨度不一致")
        windows_by_doc.append(built["windows"])
        stats["windows"] += len(built["windows"])
        stats["token_span_mismatch"] += built["token_span_mismatch"]
        stats["no_token"] += len(built["no_token"])
        stats["window_cut"] += len(built["window_cut"])
        if len(built["windows"]) > 1:
            stats["multi_window_documents"] += 1
        for window_index, window in enumerate(built["windows"]):
            window["doc_index"] = doc_index
            window["window_index"] = window_index
            examples.append(window)
    return windows_by_doc, examples, stats


def make_collate(pad_id: int):
    def collate(batch: list[dict]):
        width = max(len(item["input_ids"]) for item in batch)
        input_ids = []
        mask = []
        labels = []
        for item in batch:
            pad = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * pad)
            mask.append([1] * len(item["input_ids"]) + [0] * pad)
            labels.append(item["label_ids"] + [-100] * pad)
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    return collate


def move_batch(batch, device: torch.device):
    return tuple(tensor.to(device, non_blocking=True) for tensor in batch)


def predict_documents(
    model,
    documents: list[dict],
    windows_by_doc: list[list[dict]],
    batch_size: int,
    device: torch.device,
    fp16: bool,
    pad_id: int,
) -> list[dict]:
    flat = [window for windows in windows_by_doc for window in windows]
    predicted: dict[tuple[int, int], list[str]] = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(flat), batch_size):
            batch = flat[start : start + batch_size]
            input_ids, mask, _labels = move_batch(make_collate(pad_id)(batch), device)
            with torch.amp.autocast(device_type=device.type, enabled=fp16 and device.type == "cuda"):
                logits = model(input_ids=input_ids, attention_mask=mask).logits
            chosen = logits.argmax(dim=-1).detach().cpu()
            for row, window in enumerate(batch):
                content_len = len(window["content_labels"])
                label_ids = chosen[row, 1 : 1 + content_len].tolist()
                if len(label_ids) != content_len:
                    raise ValueError("预测长度与正文 token 数不一致")
                predicted[(window["doc_index"], window["window_index"])] = ids_to_labels(label_ids)
    rows = []
    for doc_index, document in enumerate(documents):
        text = document.get("text") or ""
        token_gold = []
        token_pred = []
        decode_windows_input = []
        for window in windows_by_doc[doc_index]:
            pred_labels = predicted[(doc_index, window["window_index"])]
            token_gold.append(window["content_labels"])
            token_pred.append(pred_labels)
            decode_windows_input.append({"content_labels": pred_labels, "offsets": window["offsets"]})
        rows.append(
            {
                "source_index": document["source_index"],
                "text": text,
                "gold": document.get("entities") or [],
                "pred": decode_windows(decode_windows_input, "content_labels", text),
                "token_gold": token_gold,
                "token_pred": token_pred,
            }
        )
    return rows


def predict_with_batch_retry(
    model,
    documents,
    windows_by_doc,
    batch_size: int,
    device,
    fp16: bool,
    pad_id: int,
) -> tuple[list[dict], int]:
    current = batch_size
    while True:
        try:
            return predict_documents(model, documents, windows_by_doc, current, device, fp16, pad_id), current
        except Exception as exc:
            if not is_oom(exc) or current == 1:
                raise
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            current = 1


def save_checkpoint(model, tokenizer, path: Path, epoch: int, score: dict) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    dump_json(
        path / "selection.json",
        {
            "epoch": epoch,
            "metric": "tune_original_gold_strict_entity_micro_f1",
            "tie_break": "earlier_epoch",
            "precision": score["precision"],
            "recall": score["recall"],
            "micro_f1": score["micro_f1"],
            "tp": score["tp"],
            "fp": score["fp"],
            "fn": score["fn"],
        },
    )


def load_checkpoint(path: Path, device: torch.device):
    model = BertForTokenClassification.from_pretrained(path)
    mapping = {int(key): value for key, value in model.config.id2label.items()}
    loaded = [mapping[index] for index in range(len(LABELS))]
    if loaded != list(LABELS):
        raise ValueError(f"检查点标签顺序与计划不一致: {loaded}")
    return model.to(device)


def train_once(
    config: dict,
    tokenizer,
    splits: dict,
    batch_size: int,
    accum: int,
    log_path: Path,
    history_path: Path,
    prediction_path: Path,
    metrics_path: Path,
    checkpoint_dir: Path,
    fallback_note: str,
) -> None:
    if batch_size * accum != int(config["effective_batch_size"]):
        raise ValueError("有效 batch 必须保持 16")
    for path in (history_path, prediction_path, metrics_path):
        if path.exists():
            path.unlink()
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    device = torch.device("cuda")
    fp16 = bool(config["fp16"])
    seed = int(config["seed"])
    epochs = int(config["epochs"])
    patience = int(config["early_stopping_patience"])
    set_seed(seed)
    model = BertForTokenClassification.from_pretrained(
        config["model_path"],
        num_labels=len(LABELS),
        id2label={index: label for index, label in enumerate(LABELS)},
        label2id=dict(LABEL_TO_ID),
    ).to(device)
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("bias") or "LayerNorm.weight" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(config["weight_decay"])},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(config["learning_rate"]),
    )
    if abs(optimizer.param_groups[0]["lr"] - PLAN_LEARNING_RATE) > 1e-15:
        raise ValueError("优化器学习率不是 3e-5")
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        WindowDataset(splits["train"]["examples"]),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=make_collate(tokenizer.pad_token_id),
    )
    updates_per_epoch = math.ceil(len(loader) / accum)
    total_updates = updates_per_epoch * epochs
    warmup_steps = int(total_updates * float(config["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)
    scaler = torch.amp.GradScaler("cuda", enabled=fp16)

    lines = [
        f"created_at={datetime.now(timezone.utc).isoformat()}",
        "model_path=D:/models/bert-base-chinese",
        f"seed={seed}",
        f"learning_rate={float(config['learning_rate'])}",
        "scheduler=linear_warmup_decay",
        f"warmup_steps={warmup_steps}",
        f"total_optimizer_steps_planned={total_updates}",
        f"weight_decay={float(config['weight_decay'])}",
        "weight_decay_excluded=bias,LayerNorm.weight",
        f"fp16={fp16}",
        f"max_length={int(config['max_length'])}",
        f"content_tokens={int(config['content_tokens'])}",
        f"window_stride={int(config['window_stride'])}",
        f"batch_size={batch_size}",
        f"gradient_accumulation_steps={accum}",
        f"effective_batch_size={batch_size * accum}",
        f"epochs_planned={epochs}",
        f"early_stopping_patience={patience}",
        "early_stopping_metric=tune_original_gold_strict_entity_micro_f1",
        "tie_break=earlier_epoch",
        "token_count_rule=sum_over_windows",
        f"oom_fallback={fallback_note}",
    ]
    for name in ("train", "tune", "test"):
        stats = splits[name]["stats"]
        lines.append(
            f"{name}_documents={stats['documents']} {name}_windows={stats['windows']} "
            f"{name}_multi_window_documents={stats['multi_window_documents']} "
            f"{name}_token_span_mismatch={stats['token_span_mismatch']} "
            f"{name}_no_token={stats['no_token']} {name}_window_cut={stats['window_cut']}"
        )
        report = splits[name]["drops"]
        lines.append(f"{name}_gold={report['gold']} {name}_dropped={report['total']}")

    def write_log() -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def log(message: str) -> None:
        lines.append(message)
        print(message, flush=True)
        write_log()

    write_log()
    for line in lines:
        print(line, flush=True)

    history = {
        "learning_rate": float(config["learning_rate"]),
        "batch_size": batch_size,
        "gradient_accumulation_steps": accum,
        "effective_batch_size": batch_size * accum,
        "seed": seed,
        "fp16": fp16,
        "oom_fallback": fallback_note,
        "epochs": [],
    }
    best_f1 = None
    best_epoch = None
    stale = 0
    stopped_early = False
    tune_gold = splits["tune"]["drops"]["gold"]

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        loss_sum = 0.0
        token_sum = 0
        pending = 0
        optimizer_steps = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, start=1):
            input_ids, mask, labels = move_batch(batch, device)
            with torch.amp.autocast(device_type="cuda", enabled=fp16):
                raw_loss = model(input_ids=input_ids, attention_mask=mask, labels=labels).loss
            labeled = int((labels != -100).sum().item())
            loss_sum += float(raw_loss.detach().float().item()) * labeled
            token_sum += labeled
            scaler.scale(raw_loss / accum).backward()
            pending += 1
            if pending == accum or step == len(loader):
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
                optimizer_steps += 1
                pending = 0
            if step % 200 == 0:
                print(f"epoch={epoch} step={step}/{len(loader)}", flush=True)
        train_loss = loss_sum / token_sum if token_sum else 0.0
        tune_rows, eval_batch = predict_with_batch_retry(
            model,
            splits["tune"]["documents"],
            splits["tune"]["windows"],
            batch_size,
            device,
            fp16,
            tokenizer.pad_token_id,
        )
        tune_score = score_documents(tune_rows)
        if tune_score["tp"] + tune_score["fn"] != tune_gold:
            raise ValueError(f"tune 实体级 TP+FN={tune_score['tp'] + tune_score['fn']}，应为 {tune_gold}")
        improved = best_f1 is None or tune_score["micro_f1"] > best_f1
        if improved:
            best_f1 = tune_score["micro_f1"]
            best_epoch = epoch
            save_checkpoint(model, tokenizer, checkpoint_dir, epoch, tune_score)
            stale = 0
        else:
            stale += 1
        elapsed = time.perf_counter() - started
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "tune_precision": tune_score["precision"],
            "tune_recall": tune_score["recall"],
            "tune_micro_f1": tune_score["micro_f1"],
            "tune_tp": tune_score["tp"],
            "tune_fp": tune_score["fp"],
            "tune_fn": tune_score["fn"],
            "saved": improved,
            "optimizer_steps": optimizer_steps,
            "eval_batch_size": eval_batch,
            "elapsed_sec": elapsed,
        }
        history["epochs"].append(record)
        history["best_epoch"] = best_epoch
        history["stopped_early"] = False
        dump_json(history_path, history)
        log(
            "epoch={epoch} train_loss={loss:.6f} tune_precision={precision:.6f} "
            "tune_recall={recall:.6f} tune_micro_f1={f1:.6f} tune_tp={tp} tune_fp={fp} "
            "tune_fn={fn} saved={saved} eval_batch_size={eval_batch} elapsed_sec={elapsed:.1f}".format(
                epoch=epoch,
                loss=train_loss,
                precision=tune_score["precision"],
                recall=tune_score["recall"],
                f1=tune_score["micro_f1"],
                tp=tune_score["tp"],
                fp=tune_score["fp"],
                fn=tune_score["fn"],
                saved=int(improved),
                eval_batch=eval_batch,
                elapsed=elapsed,
            )
        )
        if not improved and stale >= patience:
            stopped_early = True
            history["stopped_early"] = True
            dump_json(history_path, history)
            log(f"early_stop=1 best_epoch={best_epoch}")
            break

    if best_epoch is None:
        raise RuntimeError("没有写出检查点")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    model = load_checkpoint(checkpoint_dir, device)
    test_rows, test_eval_batch = predict_with_batch_retry(
        model,
        splits["test"]["documents"],
        splits["test"]["windows"],
        batch_size,
        device,
        fp16,
        tokenizer.pad_token_id,
    )
    if len(test_rows) != 1000:
        raise ValueError(f"主测试集 {len(test_rows)} 条，应为 1000")
    got_indices = [row["source_index"] for row in test_rows]
    if got_indices != splits["test_indices"]:
        raise ValueError("主测试集顺序或索引与 split_manifest.json 不一致")
    entity_score = score_documents(test_rows)
    if entity_score["tp"] + entity_score["fn"] != splits["test"]["drops"]["gold"]:
        raise ValueError("主测试实体级 TP+FN 与原始金标数不一致")
    token_score = score_token_sequences(
        [sequence for row in test_rows for sequence in row["token_gold"]],
        [sequence for row in test_rows for sequence in row["token_pred"]],
    )
    dump_json(prediction_path, test_rows)
    write_metrics_csv(
        metrics_path,
        [
            {"method": "bert_bio_token", **token_score},
            {"method": "bert_bio_entity", **entity_score},
        ],
    )
    history["stopped_early"] = stopped_early
    history["best_epoch"] = best_epoch
    history["test_eval_count"] = 1
    history["test_checkpoint_epoch"] = best_epoch
    dump_json(history_path, history)
    log(f"best_epoch={best_epoch}")
    log("test_eval_count=1")
    log(f"test_checkpoint_epoch={best_epoch}")
    log(f"test_eval_batch_size={test_eval_batch}")
    log(
        "test_entity precision={precision:.6f} recall={recall:.6f} micro_f1={f1:.6f} "
        "tp={tp} fp={fp} fn={fn}".format(
            precision=entity_score["precision"],
            recall=entity_score["recall"],
            f1=entity_score["micro_f1"],
            tp=entity_score["tp"],
            fp=entity_score["fp"],
            fn=entity_score["fn"],
        )
    )
    log(
        "test_token precision={precision:.6f} recall={recall:.6f} micro_f1={f1:.6f} "
        "tp={tp} fp={fp} fn={fn}".format(
            precision=token_score["precision"],
            recall=token_score["recall"],
            f1=token_score["micro_f1"],
            tp=token_score["tp"],
            fp=token_score["fp"],
            fn=token_score["fn"],
        )
    )
    log(f"predictions={prediction_path.relative_to(ROOT).as_posix()}")
    log(f"metrics={metrics_path.relative_to(ROOT).as_posix()}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    config = yaml.safe_load((ROOT / "configs" / "bert_bio.yaml").read_text(encoding="utf-8"))
    assert_plan_config(config)
    if not torch.cuda.is_available():
        raise RuntimeError("计划中的 fp16 训练需要 CUDA")
    set_seed(int(config["seed"]))
    documents = {
        name: load_documents(ROOT / config[f"{name}_path"])
        for name in ("train", "tune", "test")
    }
    reports = {name: drop_report(docs) for name, docs in documents.items()}
    assert_expected_drops(reports)
    dump_json(ROOT / "outputs" / "metrics" / "bio_dropped_entities.json", reports)
    manifest = json.loads((ROOT / config["manifest_path"]).read_text(encoding="utf-8"))
    test_indices = manifest["test"]["indices"]
    if [doc["source_index"] for doc in documents["test"]] != test_indices:
        raise ValueError("test.json 的 source_index 与 split_manifest.json 不一致")
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("需要 fast tokenizer 提供 offset")
    splits = {"test_indices": test_indices}
    for name, docs in documents.items():
        windows, examples, stats = prepare_split(
            tokenizer,
            docs,
            max_length=int(config["max_length"]),
            stride=int(config["window_stride"]),
        )
        print(
            f"prepared {name} windows={stats['windows']} multi={stats['multi_window_documents']} "
            f"window_cut={stats['window_cut']} token_span_mismatch={stats['token_span_mismatch']} "
            f"no_token={stats['no_token']}",
            flush=True,
        )
        splits[name] = {
            "documents": docs,
            "windows": windows,
            "examples": examples,
            "stats": stats,
            "drops": reports[name],
        }

    log_path = ROOT / "outputs" / "logs" / "train_bio.log"
    history_path = ROOT / "outputs" / "metrics" / "bert_bio_history.json"
    prediction_path = ROOT / "outputs" / "predictions" / "bert_bio.json"
    metrics_path = ROOT / "outputs" / "metrics" / "bert_bio.csv"
    checkpoint_dir = ROOT / config["checkpoint_dir"]
    attempts = [
        (int(config["batch_size"]), int(config["gradient_accumulation_steps"]), "none"),
        (4, 4, "batch_size:8->4,gradient_accumulation_steps:2->4"),
    ]
    for index, (batch_size, accum, note) in enumerate(attempts):
        try:
            train_once(
                config,
                tokenizer,
                splits,
                batch_size,
                accum,
                log_path,
                history_path,
                prediction_path,
                metrics_path,
                checkpoint_dir,
                note,
            )
            return
        except Exception as exc:
            gc.collect()
            torch.cuda.empty_cache()
            if index == len(attempts) - 1 or not is_oom(exc):
                raise
            print(f"batch {batch_size} 显存不足，改用 batch_size=4 gradient_accumulation_steps=4", flush=True)


if __name__ == "__main__":
    main()
