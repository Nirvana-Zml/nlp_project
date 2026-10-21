"""仓库相对路径下的数据和模型检查。配置文件不写本机绝对路径。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

BERT_ENV = "BERT_MODEL_PATH"
QWEN_ENV = "QWEN_MODEL_PATH"
BERT_RELATIVE = "models/bert-base-chinese"
QWEN_RELATIVE = "models/Qwen2.5-3B-Instruct"
BERT_URL = "https://huggingface.co/google-bert/bert-base-chinese"
QWEN_URL = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct"
QWEN_MODELSCOPE_URL = "https://modelscope.cn/models/Qwen/Qwen2.5-3B-Instruct"
DATA_URL = "https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414"
RAW_FILES = (
    "data/raw/CMeEE-V2/CMeEE-V2_train.json",
    "data/raw/CMeEE-V2/CMeEE-V2_dev.json",
    "data/raw/CMeEE-V2/CMeEE-V2_test.json",
)

CUDA_MESSAGE = (
    "未检测到 CUDA。BERT-BIO 的 fp16 训练和 Qwen2.5-3B 的 4bit 推理都需要 NVIDIA GPU。\n"
    "本程序不会改用 CPU，不会降低精度，也不会缩小测试集。\n"
    "只重算已有预测时请运行：python scripts/run_pipeline.py --recompute"
)
TORCH_MESSAGE = (
    "未安装 torch。请先在仓库根目录安装 requirements.txt 中的依赖。\n"
    "torch==2.6.0+cu124 需要与本机 CUDA 匹配的轮子，只执行 pip install -r requirements.txt 不一定能装上。"
)


def show_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _configured_model_path(filename: str) -> str:
    config = yaml.safe_load((ROOT / "configs" / filename).read_text(encoding="utf-8"))
    value = config.get("model_path")
    if not isinstance(value, str) or not value.strip():
        return ""
    return value.strip()


def _located_dir(configured: str, env_name: str) -> Path:
    override = os.environ.get(env_name, "").strip()
    raw = override or configured
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _absolute_config_issue(configured: str, config_file: str, env_name: str, example: str) -> str | None:
    if os.environ.get(env_name, "").strip():
        return None
    if configured and Path(configured).is_absolute():
        return (
            f"{config_file} 的 model_path 写成了本机绝对路径。"
            f"请改成相对仓库根目录的路径，例如 {example}。"
            f"权重若在仓库外，设置环境变量 {env_name}，不要把本机路径写进配置文件。"
        )
    return None


def _read_model_config(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def data_issues() -> list[str]:
    missing = [name for name in RAW_FILES if not (ROOT / name).is_file()]
    if not missing:
        return []
    listed = "\n".join(missing)
    return [
        "缺少原始数据。请把 CMeEE-V2 的 JSON 放到下列相对路径：\n"
        f"{listed}\n"
        f"下载页：{DATA_URL}\n"
        "官方测试集没有标签，不参与评分，但校验需要这份文件。"
    ]


def bert_issues() -> list[str]:
    configured = _configured_model_path("bert_bio.yaml")
    if not configured:
        return ["configs/bert_bio.yaml 缺少 model_path。"]
    absolute = _absolute_config_issue(configured, "configs/bert_bio.yaml", BERT_ENV, BERT_RELATIVE)
    if absolute:
        return [absolute]
    directory = _located_dir(configured, BERT_ENV)
    shown = show_path(directory)
    if not directory.is_dir() or not (directory / "config.json").is_file() or not (directory / "model.safetensors").is_file():
        return [
            f"缺少 BERT 模型。需要目录 {shown}，其中要有 config.json 和 model.safetensors。\n"
            f"官方权重：{BERT_URL}\n"
            "请放到 models/bert-base-chinese（相对仓库根目录）。本程序不会自动下载，也不会改用其他中文 BERT。\n"
            f"权重若在仓库外，设置环境变量 {BERT_ENV}，不要把本机路径写进配置文件。"
        ]
    config = _read_model_config(directory / "config.json")
    if config is None:
        return [f"{shown} 的 config.json 无法解析。"]
    if config.get("model_type") != "bert" or config.get("vocab_size") != 21128:
        return [
            f"{shown} 不是 bert-base-chinese（需要 model_type=bert 且 vocab_size=21128）。"
            "本程序不会改用其他模型。\n"
            f"官方权重：{BERT_URL}"
        ]
    return []


def qwen_issues() -> list[str]:
    configured = _configured_model_path("qwen_3b.yaml")
    if not configured:
        return ["configs/qwen_3b.yaml 缺少 model_path。"]
    absolute = _absolute_config_issue(configured, "configs/qwen_3b.yaml", QWEN_ENV, QWEN_RELATIVE)
    if absolute:
        return [absolute]
    directory = _located_dir(configured, QWEN_ENV)
    shown = show_path(directory)
    weights = list(directory.glob("*.safetensors")) if directory.is_dir() else []
    if not directory.is_dir() or not (directory / "config.json").is_file() or not weights:
        return [
            f"缺少 Qwen 模型。需要目录 {shown}，其中要有 config.json 和 safetensors 权重。\n"
            f"官方权重：{QWEN_URL}\n"
            f"ModelScope：{QWEN_MODELSCOPE_URL}\n"
            "请放到 models/Qwen2.5-3B-Instruct（相对仓库根目录）。"
            "本程序不会自动下载，也不会改用 1.5B 或其他尺寸。\n"
            f"权重若在仓库外，设置环境变量 {QWEN_ENV}，不要把本机路径写进配置文件。"
        ]
    config = _read_model_config(directory / "config.json")
    if config is None:
        return [f"{shown} 的 config.json 无法解析。"]
    if config.get("model_type") != "qwen2" or config.get("hidden_size") != 2048:
        return [
            f"{shown} 不是 Qwen2.5-3B（需要 model_type=qwen2 且 hidden_size=2048）。"
            "本程序不会改用 1.5B 或其他尺寸。\n"
            f"官方权重：{QWEN_URL}"
        ]
    return []


def bert_directory() -> Path:
    issues = bert_issues()
    if issues:
        raise SystemExit("\n\n".join(issues))
    return _located_dir(_configured_model_path("bert_bio.yaml"), BERT_ENV)


def qwen_directory() -> Path:
    issues = qwen_issues()
    if issues:
        raise SystemExit("\n\n".join(issues))
    return _located_dir(_configured_model_path("qwen_3b.yaml"), QWEN_ENV)


def cuda_issue(available: bool | None = None) -> list[str]:
    if available is None:
        try:
            import torch
        except ImportError:
            return [TORCH_MESSAGE]
        available = bool(torch.cuda.is_available())
    if available:
        return []
    return [CUDA_MESSAGE]


def preflight() -> None:
    issues = data_issues() + bert_issues() + qwen_issues() + cuda_issue()
    if issues:
        raise SystemExit("\n\n".join(issues))
