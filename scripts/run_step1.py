"""执行已确认的第 1 步：校验、固定划分、词典基线。"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines.dictionary_ner import (
    build_lexicon,
    compile_lexicon,
    find_all,
    lexicon_stats,
    longest_match,
)
from src.data.split import split_main_test, split_train_tune, take
from src.data.validate import file_sha256, load_json, validate_documents
from src.evaluation.metrics import score_documents, write_metrics_csv


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    config = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
    raw_dir = ROOT / config["raw_dir"]
    processed_dir = ROOT / config["processed_dir"]
    outputs_dir = ROOT / config["outputs_dir"]
    seed = int(config["seed"])

    file_map = {
        "train": raw_dir / config["files"]["train"],
        "dev": raw_dir / config["files"]["dev"],
        "test": raw_dir / config["files"]["test"],
    }
    loaded = {name: load_json(path) for name, path in file_map.items()}
    hashes = {name: file_sha256(path) for name, path in file_map.items()}

    summaries = {}
    errors = []
    for name, documents in loaded.items():
        summary, file_errors = validate_documents(documents, file_map[name].name)
        summary["sha256"] = hashes[name]
        summary["path"] = str(file_map[name].relative_to(ROOT)).replace("\\", "/")
        summaries[file_map[name].name] = summary
        errors.extend(file_errors)

    validation_summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "offset_rule": "text[start_idx:end_idx] == entity",
        "files": summaries,
    }
    dump_json(outputs_dir / "data_validation_summary.json", validation_summary)
    dump_json(outputs_dir / "data_validation_errors.json", errors)

    train_indices, tune_indices = split_train_tune(
        loaded["train"], int(config["train_size"]), int(config["tune_size"]), seed
    )
    test_indices, uncovered = split_main_test(loaded["dev"], int(config["main_test_size"]), seed)
    train_docs = take(loaded["train"], train_indices)
    tune_docs = take(loaded["train"], tune_indices)
    test_docs = take(loaded["dev"], test_indices)
    dev_full = take(loaded["dev"], list(range(len(loaded["dev"]))))

    dump_json(processed_dir / "train.json", train_docs)
    dump_json(processed_dir / "tune.json", tune_docs)
    dump_json(processed_dir / "test.json", test_docs)
    dump_json(processed_dir / "dev_full.json", dev_full)
    manifest = {
        "seed": seed,
        "offset_rule": "text[start_idx:end_idx] == entity",
        "official_test_excluded": config["files"]["test"],
        "sources": {
            "train": {"file": config["files"]["train"], "sha256": hashes["train"], "docs": len(loaded["train"])},
            "dev": {"file": config["files"]["dev"], "sha256": hashes["dev"], "docs": len(loaded["dev"])},
        },
        "train": {
            "method": "random.Random(seed).shuffle on official train indices; first train_size, rest tune",
            "size": len(train_indices),
            "indices": train_indices,
        },
        "tune": {"size": len(tune_indices), "indices": tune_indices},
        "test": {
            "method": (
                "separate random.Random(seed).shuffle on official dev indices; "
                "take documents that cover missing required features in shuffle order, then fill to main_test_size"
            ),
            "required_features": [
                "nine entity types",
                "nested",
                "non_nested",
                "len:1-2",
                "len:3-5",
                "len:6+",
                "has_ascii",
            ],
            "uncovered_features": uncovered,
            "size": len(test_indices),
            "indices": test_indices,
            "note": "dev_full contains this main test split and is not an independent score set",
        },
    }
    dump_json(processed_dir / "split_manifest.json", manifest)

    lexicon = build_lexicon(train_docs)
    stats = lexicon_stats(train_docs, lexicon)
    lexicon_root = compile_lexicon(lexicon)
    predictions = []
    for doc in test_docs:
        matches = find_all(doc["text"], lexicon, lexicon_root)
        predictions.append(
            {
                "source_index": doc["source_index"],
                "text": doc["text"],
                "gold": doc.get("entities") or [],
                "pred_all": matches,
                "pred_longest": longest_match(matches),
            }
        )
    all_docs = [
        {"gold": item["gold"], "pred": item["pred_all"], "source_index": item["source_index"], "text": item["text"]}
        for item in predictions
    ]
    longest_docs = [
        {
            "gold": item["gold"],
            "pred": item["pred_longest"],
            "source_index": item["source_index"],
            "text": item["text"],
        }
        for item in predictions
    ]
    dump_json(outputs_dir / "predictions" / "dictionary_all.json", all_docs)
    dump_json(outputs_dir / "predictions" / "dictionary_longest.json", longest_docs)

    rows = [
        {"method": "dictionary_all", **score_documents(all_docs)},
        {"method": "dictionary_longest", **score_documents(longest_docs)},
    ]
    metrics_path = outputs_dir / "metrics" / "dictionary_baseline.csv"
    write_metrics_csv(metrics_path, rows)
    dump_json(outputs_dir / "metrics" / "dictionary_stats.json", stats)

    log_path = outputs_dir / "logs" / "step1.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                f"created_at={validation_summary['created_at']}",
                f"seed={seed}",
                f"train={len(train_docs)}",
                f"tune={len(tune_docs)}",
                f"test={len(test_docs)}",
                f"uncovered={uncovered}",
                f"lexicon_surfaces={stats['surfaces']}",
                f"lexicon_ties={stats['tie_surfaces']}",
                f"metrics={metrics_path.relative_to(ROOT).as_posix()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(log_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
