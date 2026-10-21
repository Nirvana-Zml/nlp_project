"""测量已有 BERT 检查点在主测试上的前向时间。不写预测，不改检查点。"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import score_documents
from src.local_resources import bert_directory


def _load_train_module():
    path = ROOT / "scripts" / "train_bert_bio.py"
    spec = importlib.util.spec_from_file_location("train_bert_bio_latency", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

OUT = ROOT / "outputs" / "metrics" / "bert_inference_latency.json"


def main() -> None:
    train_bio = _load_train_module()
    config = yaml.safe_load((ROOT / "configs" / "bert_bio.yaml").read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise SystemExit("没有 CUDA，停止。不改用 CPU 数字充当同口径延迟。")
    device = torch.device("cuda")
    test_docs = train_bio.load_documents(ROOT / config["test_path"])
    started = time.perf_counter()
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(bert_directory(), use_fast=True)
    tokenizer_sec = time.perf_counter() - started
    windows, _examples, stats = train_bio.prepare_split(
        tokenizer,
        test_docs,
        max_length=int(config["max_length"]),
        stride=int(config["window_stride"]),
    )
    checkpoint = ROOT / config["checkpoint_dir"]
    started = time.perf_counter()
    model = train_bio.load_checkpoint(checkpoint, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    load_sec = time.perf_counter() - started
    model.eval()
    torch.cuda.synchronize()
    started = time.perf_counter()
    rows = train_bio.predict_documents(
        model,
        test_docs,
        windows,
        batch_size=int(config["batch_size"]),
        device=device,
        fp16=bool(config["fp16"]),
        pad_id=tokenizer.pad_token_id,
    )
    torch.cuda.synchronize()
    forward_sec = time.perf_counter() - started
    score = score_documents(rows)
    if score["tp"] != 2873 or abs(score["micro_f1"] - 0.6347768448961555) > 1e-12:
        raise SystemExit(f"检查点推理与已保存主结果不一致：{score}")
    payload = {
        "device": torch.cuda.get_device_name(0),
        "batch_size": int(config["batch_size"]),
        "fp16": bool(config["fp16"]),
        "documents": len(rows),
        "windows": stats["windows"],
        "includes_model_load": False,
        "tokenizer_load_sec": tokenizer_sec,
        "checkpoint_load_sec": load_sec,
        "forward_and_decode_sec": forward_sec,
        "mean_forward_and_decode_sec": forward_sec / len(rows),
        "timer": "模型已加载、窗口已准备。cuda.synchronize 包住 predict_documents，含前向和跨度解码，不含权重加载。",
        "qwen_timer_difference": "Qwen 的 mean_latency_sec 只包住 generate，批量是 1。两边都不含权重加载，批量不同。",
        "score": score,
        "checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("BERT_LATENCY_OK")
    print(json.dumps({key: payload[key] for key in ("device", "forward_and_decode_sec", "mean_forward_and_decode_sec", "checkpoint_load_sec")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
