"""中文医疗实体抽取演示页。只做抽取展示，不回写预测或指标。"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 权重若在仓库外（本机常见布局），未设置环境变量时再指向 D:/models。
if not os.environ.get("BERT_MODEL_PATH", "").strip():
    external_bert = Path("D:/models/bert-base-chinese")
    if external_bert.is_dir() and not (ROOT / "models/bert-base-chinese").is_dir():
        os.environ["BERT_MODEL_PATH"] = str(external_bert)
if not os.environ.get("QWEN_MODEL_PATH", "").strip():
    external_qwen = Path("D:/models/Qwen2.5-3B-Instruct")
    if external_qwen.is_dir() and not (ROOT / "models/Qwen2.5-3B-Instruct").is_dir():
        os.environ["QWEN_MODEL_PATH"] = str(external_qwen)

from src.baselines.dictionary_ner import build_lexicon, compile_lexicon, find_all, longest_match
from src.evaluation.metrics import read_metrics_csv
from src.models.qwen_prompt import fill_prompt, format_examples, select_train_examples

EXAMPLES_PATH = ROOT / "outputs" / "errors" / "examples.json"
BOUNDARY_CATEGORY = "边界错误"
DEFAULT_TEXT = "对儿童SARST细胞亚群的研究表明，与成人SARS相比，儿童细胞下降不明显，证明上述推测成立。"

TYPE_ORDER = ["bod", "dis", "sym", "pro", "ite", "dru", "mic", "equ", "dep"]
TYPE_LABEL = {
    "bod": "bod 身体",
    "dis": "dis 疾病",
    "sym": "sym 临床表现",
    "pro": "pro 医疗程序",
    "ite": "ite 检验项目",
    "dru": "dru 药物",
    "mic": "mic 微生物",
    "equ": "equ 医疗设备",
    "dep": "dep 科室",
}
TYPE_COLOR = {
    "bod": "#4DBBD5",
    "dis": "#E64B35",
    "sym": "#00A087",
    "pro": "#3C5488",
    "ite": "#F39B7F",
    "dru": "#8491B4",
    "mic": "#91D1C2",
    "equ": "#DC0000",
    "dep": "#7E6148",
}

METHOD_OPTIONS = {
    "词典最长匹配": "dictionary_longest",
    "BERT-BIO": "bert_bio",
    "Qwen 零样本": "qwen_zero",
    "Qwen 单样本": "qwen_one",
    "Qwen 三样本": "qwen_three",
}

METRIC_FILES = [
    ("词典基线", ROOT / "outputs/metrics/dictionary_baseline.csv"),
    ("BERT-BIO", ROOT / "outputs/metrics/bert_bio.csv"),
    ("Qwen 零/三样本", ROOT / "outputs/metrics/qwen_3b.csv"),
    ("Qwen 单样本", ROOT / "outputs/metrics/qwen_3b_one.csv"),
]


CUSTOM_SAMPLE = "自定义文本（可任意输入，无绑定 gold）"


def load_examples() -> list[dict]:
    fallback = {
        "category": BOUNDARY_CATEGORY,
        "text": DEFAULT_TEXT,
        "gold": [],
        "pred": [],
        "method": "",
        "source_index": None,
        "prediction_file": "",
    }
    if not EXAMPLES_PATH.is_file():
        return [fallback]
    rows = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))
    return rows if isinstance(rows, list) and rows else [fallback]


def sample_label(ex: dict) -> str:
    return (
        f"{ex.get('category', '?')} · {ex.get('method', '?')} · "
        f"source_index={ex.get('source_index', '?')}"
    )


def find_example_by_label(examples: list[dict], label: str) -> dict | None:
    for ex in examples:
        if sample_label(ex) == label:
            return ex
    return None


def overlaps(left: tuple, right: tuple) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def boundary_pairs(gold: list[dict], pred: list[dict]) -> list[tuple[dict, dict]]:
    pairs = []
    for g in gold:
        g_key = (g["start_idx"], g["end_idx"], g["type"])
        for p in pred:
            p_key = (p["start_idx"], p["end_idx"], p["type"])
            if (
                g_key[2] == p_key[2]
                and overlaps(g_key, p_key)
                and (g_key[0] != p_key[0] or g_key[1] != p_key[1])
            ):
                pairs.append((g, p))
    return pairs


def type_pairs(gold: list[dict], pred: list[dict]) -> list[tuple[dict, dict]]:
    pairs = []
    for g in gold:
        for p in pred:
            if (
                g["start_idx"] == p["start_idx"]
                and g["end_idx"] == p["end_idx"]
                and g["type"] != p["type"]
            ):
                pairs.append((g, p))
    return pairs


def metrics_overview_rows() -> list[dict]:
    preferred = {
        "dictionary_longest",
        "bert_bio_entity",
        "qwen_3b_zero",
        "qwen_3b_one",
        "qwen_3b_three",
    }
    rows: list[dict] = []
    for _title, path in METRIC_FILES:
        if not path.is_file():
            continue
        for row in read_metrics_csv(path):
            if row["method"] in preferred:
                rows.append(
                    {
                        "method": row["method"],
                        "precision": row["precision"],
                        "recall": row["recall"],
                        "micro_f1": row["micro_f1"],
                        "source_file": path.relative_to(ROOT).as_posix(),
                    }
                )
    return rows


def render_highlighted(text: str, entities: list[dict]) -> str:
    if not entities:
        return f'<span class="ner-plain">{_escape(text)}</span>'
    ordered = sorted(entities, key=lambda e: (e["start_idx"], -(e["end_idx"] - e["start_idx"])))
    parts: list[str] = []
    cursor = 0
    covered_until = -1
    for ent in ordered:
        start = int(ent["start_idx"])
        end = int(ent["end_idx"])
        if start < covered_until:
            continue
        if start < cursor:
            continue
        if start > cursor:
            parts.append(_escape(text[cursor:start]))
        color = TYPE_COLOR.get(ent.get("type", ""), "#94A3B8")
        label = TYPE_LABEL.get(ent.get("type", ""), ent.get("type", "?"))
        span = _escape(text[start:end])
        parts.append(
            f'<mark class="ner-mark" style="--mark:{color};" '
            f'title="{_escape(label)} [{start},{end})">{span}</mark>'
        )
        cursor = end
        covered_until = end
    if cursor < len(text):
        parts.append(_escape(text[cursor:]))
    return "".join(parts)


def panel_html(title: str, subtitle: str, body_html: str, kind: str = "pred") -> str:
    return (
        f'<div class="ner-panel ner-panel-{kind}">'
        f'<div class="ner-panel-head"><span class="ner-panel-title">{_escape(title)}</span>'
        f'<span class="ner-panel-sub">{_escape(subtitle)}</span></div>'
        f'<div class="ner-panel-body">{body_html}</div></div>'
    )


def inject_styles() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&family=Noto+Serif+SC:wght@600;700&display=swap');

:root {
  --ink: #13253a;
  --teal: #0a7a72;
  --teal-deep: #065f5a;
  --mint: #e6f3f1;
  --sand: #f4f0e8;
  --line: rgba(19, 37, 58, 0.12);
  --warn: #9a5b1a;
  --ok: #1f6b4a;
}

html, body, [class*="css"]  {
  font-family: "Noto Sans SC", "Source Han Sans SC", sans-serif;
  color: var(--ink);
}

.stApp {
  background:
    radial-gradient(1200px 500px at 8% -10%, rgba(10, 122, 114, 0.16), transparent 55%),
    radial-gradient(900px 420px at 100% 0%, rgba(19, 37, 58, 0.08), transparent 50%),
    linear-gradient(165deg, #f7f5f0 0%, #eef5f4 48%, #f3f1eb 100%);
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0f2a33 0%, #163a42 55%, #1a4548 100%);
  border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stCaption {
  color: #e8f2f1 !important;
}
/* 输入框保持浅底深字：勿用 * 把字色刷成白色，否则会白字白底看不见 */
[data-testid="stSidebar"] .stTextInput input {
  background: #f7faf9 !important;
  border: 1px solid rgba(255,255,255,0.35) !important;
  color: #13253a !important;
  caret-color: #13253a !important;
  border-radius: 8px !important;
}
[data-testid="stSidebar"] .stTextInput input::placeholder {
  color: #6b7f8c !important;
  opacity: 1 !important;
}

.hero {
  margin: 0.2rem 0 1.4rem 0;
  padding: 1.35rem 1.5rem 1.25rem;
  border-radius: 14px;
  background:
    linear-gradient(120deg, rgba(10,122,114,0.12), transparent 42%),
    linear-gradient(180deg, #ffffffcc, #f3faf8cc);
  border: 1px solid var(--line);
  animation: rise 0.55s ease-out;
}
.hero h1 {
  font-family: "Noto Serif SC", "Source Han Serif SC", serif;
  font-size: 2.05rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  color: var(--ink);
  margin: 0 0 0.35rem 0;
  line-height: 1.25;
}
.hero .tagline {
  margin: 0;
  color: #3d5568;
  font-size: 0.98rem;
  line-height: 1.55;
}
.hero .chips {
  margin-top: 0.85rem;
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
}
.hero .chip {
  display: inline-block;
  padding: 0.22rem 0.65rem;
  border-radius: 6px;
  border: 1px solid rgba(10,122,114,0.28);
  background: rgba(10,122,114,0.08);
  color: var(--teal-deep);
  font-size: 0.82rem;
  font-weight: 500;
}

.section-title {
  font-family: "Noto Serif SC", serif;
  font-size: 1.28rem;
  font-weight: 700;
  color: var(--ink);
  margin: 1.35rem 0 0.35rem;
  padding-left: 0.7rem;
  border-left: 4px solid var(--teal);
  animation: rise 0.45s ease-out;
}
.section-note {
  color: #4a6070;
  font-size: 0.9rem;
  margin: 0 0 0.85rem 0.75rem;
  line-height: 1.5;
}

.ner-panel {
  margin: 0.55rem 0 0.9rem;
  border-radius: 12px;
  border: 1px solid var(--line);
  overflow: hidden;
  background: rgba(255,255,255,0.78);
  animation: rise 0.5s ease-out;
}
.ner-panel-pred { border-top: 3px solid var(--teal); }
.ner-panel-gold { border-top: 3px solid #c45c26; }
.ner-panel-head {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem 0.85rem;
  align-items: baseline;
  padding: 0.65rem 0.9rem;
  background: linear-gradient(90deg, var(--mint), transparent 70%);
  border-bottom: 1px solid var(--line);
}
.ner-panel-gold .ner-panel-head {
  background: linear-gradient(90deg, #f8ebe2, transparent 70%);
}
.ner-panel-title {
  font-weight: 700;
  font-size: 0.95rem;
  color: var(--ink);
}
.ner-panel-sub {
  font-size: 0.82rem;
  color: #5a7080;
}
.ner-panel-body {
  padding: 0.95rem 1rem 1.05rem;
  font-size: 1.08rem;
  line-height: 1.85;
  letter-spacing: 0.02em;
}

.ner-mark {
  background: color-mix(in srgb, var(--mark) 88%, white);
  color: #fff;
  padding: 0.08em 0.22em;
  margin: 0 0.02em;
  border-radius: 4px;
  box-decoration-break: clone;
  -webkit-box-decoration-break: clone;
  font-weight: 500;
}
.ner-plain { color: var(--ink); }

.legend-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin: 0.35rem 0 0.9rem;
}
.legend-item {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.2rem 0.55rem;
  border-radius: 6px;
  background: rgba(255,255,255,0.7);
  border: 1px solid var(--line);
  font-size: 0.8rem;
  color: #33485a;
}
.legend-dot {
  width: 0.65rem;
  height: 0.65rem;
  border-radius: 2px;
  flex-shrink: 0;
}

.sidebar-badge {
  margin: 0.4rem 0 1rem;
  padding: 0.85rem 0.9rem;
  border-radius: 10px;
  background: rgba(10, 122, 114, 0.22);
  border: 1px solid rgba(255,255,255,0.14);
  color: #e8f2f1;
}
.sidebar-badge .k { font-size: 0.75rem; opacity: 0.75; color: #e8f2f1; }
.sidebar-badge .v { font-size: 1.15rem; font-weight: 700; letter-spacing: 0.03em; color: #fff; }

div.stButton > button[kind="primary"] {
  background: linear-gradient(180deg, #0d8a81, var(--teal-deep)) !important;
  border: none !important;
  color: #fff !important;
  font-weight: 600 !important;
  border-radius: 8px !important;
  padding: 0.45rem 1.35rem !important;
  transition: transform 0.15s ease, filter 0.15s ease;
}
div.stButton > button[kind="primary"]:hover {
  filter: brightness(1.06);
  transform: translateY(-1px);
}

@keyframes rise {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

[data-testid="stMetric"] {
  background: rgba(255,255,255,0.72);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 0.55rem 0.75rem;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def entity_table(entities: list[dict]) -> list[dict]:
    rows = []
    for ent in entities:
        rows.append(
            {
                "entity": ent.get("entity", ""),
                "type": TYPE_LABEL.get(ent.get("type", ""), ent.get("type", "")),
                "start_idx": ent.get("start_idx"),
                "end_idx": ent.get("end_idx"),
            }
        )
    return rows


@st.cache_resource(show_spinner=False)
def get_lexicon():
    train_docs = json.loads((ROOT / "data/processed/train.json").read_text(encoding="utf-8"))
    lexicon = build_lexicon(train_docs)
    root = compile_lexicon(lexicon)
    return lexicon, root


def _load_module(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def ensure_bert():
    if "bert_bundle" in st.session_state:
        return st.session_state["bert_bundle"]
    import torch
    from transformers import AutoTokenizer
    from src.local_resources import bert_directory

    train_bio = _load_module("train_bert_bio_app", "scripts/train_bert_bio.py")
    config = yaml.safe_load((ROOT / "configs/bert_bio.yaml").read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("没有 CUDA，按计划停止，不改用 CPU。")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(bert_directory(), use_fast=True)
    model = train_bio.load_checkpoint(ROOT / config["checkpoint_dir"], device)
    model.eval()
    bundle = {
        "module": train_bio,
        "config": config,
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
    }
    st.session_state["bert_bundle"] = bundle
    return bundle


def ensure_qwen():
    if "qwen_bundle" in st.session_state:
        return st.session_state["qwen_bundle"]
    run_qwen = _load_module("run_qwen_3b_app", "scripts/run_qwen_3b.py")
    config = yaml.safe_load((ROOT / "configs/qwen_3b.yaml").read_text(encoding="utf-8"))
    model, tokenizer = run_qwen._load_model(config)
    train_docs = json.loads((ROOT / config["train_path"]).read_text(encoding="utf-8"))
    bundle = {
        "module": run_qwen,
        "config": config,
        "model": model,
        "tokenizer": tokenizer,
        "train_docs": train_docs,
    }
    st.session_state["qwen_bundle"] = bundle
    return bundle


def run_dictionary(text: str) -> tuple[list[dict], float]:
    lexicon, root = get_lexicon()
    started = time.perf_counter()
    pred = longest_match(find_all(text, lexicon, root))
    return pred, time.perf_counter() - started


def run_bert(text: str) -> tuple[list[dict], float]:
    import torch

    bundle = ensure_bert()
    train_bio = bundle["module"]
    config = bundle["config"]
    doc = {"source_index": 0, "text": text, "entities": []}
    windows_by_doc, _, _ = train_bio.prepare_split(
        bundle["tokenizer"],
        [doc],
        max_length=int(config["max_length"]),
        stride=int(config["window_stride"]),
    )
    if bundle["device"].type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    rows = train_bio.predict_documents(
        bundle["model"],
        [doc],
        windows_by_doc,
        batch_size=int(config["batch_size"]),
        device=bundle["device"],
        fp16=bool(config["fp16"]),
        pad_id=bundle["tokenizer"].pad_token_id,
    )
    if bundle["device"].type == "cuda":
        torch.cuda.synchronize()
    return rows[0]["pred"], time.perf_counter() - started


def run_qwen(text: str, shots: int) -> tuple[list[dict], float]:
    bundle = ensure_qwen()
    config = bundle["config"]
    if shots == 0:
        examples = []
    else:
        examples = select_train_examples(bundle["train_docs"], seed=config["seed"], count=shots)
    prompt = fill_prompt(config["prompt_template"], format_examples(examples), text)
    doc = {"source_index": -1, "text": text, "entities": []}
    record = bundle["module"]._generate_one(bundle["model"], bundle["tokenizer"], prompt, doc, config)
    return record["pred"], float(record["latency_sec"])


def extract(method_key: str, text: str) -> tuple[list[dict], float]:
    if method_key == "dictionary_longest":
        return run_dictionary(text)
    if method_key == "bert_bio":
        return run_bert(text)
    if method_key == "qwen_zero":
        return run_qwen(text, 0)
    if method_key == "qwen_one":
        return run_qwen(text, 1)
    if method_key == "qwen_three":
        return run_qwen(text, 3)
    raise ValueError(f"未知方法: {method_key}")


def render_compare_section(
    *,
    text: str,
    gold: list[dict],
    show_pred: list[dict],
    show_text: str,
    pred_source: str,
    active: dict | None,
    bound_to_sample: bool,
) -> None:
    st.markdown('<div class="section-title">结果对照（gold 标注 vs pred 预测）</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-note">gold = 标准答案实体；pred = 所选方法抽出的实体。区间均为左闭右开 [start_idx, end_idx)。</p>',
        unsafe_allow_html=True,
    )
    if bound_to_sample and active:
        st.info(
            f"【当前样例】类别：{active.get('category')} ｜ method={active.get('method')} ｜ "
            f"source_index={active.get('source_index')} ｜ 文件 {active.get('prediction_file', '')}"
        )
    elif not bound_to_sample:
        st.caption("当前为自定义文本或已改写原文：仅展示 pred；无绑定 gold（除非文本与某条样例完全一致）。")

    st.markdown(
        panel_html("① 原文按 pred 着色", pred_source, render_highlighted(show_text, show_pred), "pred"),
        unsafe_allow_html=True,
    )
    if gold:
        st.markdown(
            panel_html(
                "② 原文按 gold 着色",
                "样例标准答案（与当前抽取方法无关）",
                render_highlighted(text, gold),
                "gold",
            ),
            unsafe_allow_html=True,
        )

    legend_items = "".join(
        f'<span class="legend-item"><span class="legend-dot" style="background:{TYPE_COLOR[t]};"></span>'
        f"{_escape(TYPE_LABEL[t])}</span>"
        for t in TYPE_ORDER
    )
    st.markdown(f'<div class="legend-row">{legend_items}</div>', unsafe_allow_html=True)

    col_g, col_p = st.columns(2)
    with col_g:
        st.markdown("**③ gold 实体表** — 标准答案")
        st.dataframe(entity_table(gold), width="stretch", hide_index=True)
    with col_p:
        st.markdown(f"**④ pred 实体表** — {pred_source}")
        st.dataframe(entity_table(show_pred), width="stretch", hide_index=True)

    category = (active or {}).get("category") if bound_to_sample else None
    if gold and show_pred:
        if category == BOUNDARY_CATEGORY or (not category and boundary_pairs(gold, show_pred)):
            pairs = boundary_pairs(gold, show_pred)
            st.markdown("**⑤ 边界差异对** — 同类型、区间相交但起止不完全相同（左闭右开）")
            if pairs:
                st.dataframe(
                    [
                        {
                            "type": g["type"],
                            "gold（标准答案）": f'{g.get("entity","")} [{g["start_idx"]},{g["end_idx"]})',
                            "pred（当前预测）": f'{p.get("entity","")} [{p["start_idx"]},{p["end_idx"]})',
                        }
                        for g, p in pairs
                    ],
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.caption("未检出边界差异对。")
        if category == "类型错误" or (not category and type_pairs(gold, show_pred)):
            pairs = type_pairs(gold, show_pred)
            st.markdown("**⑤ 类型差异对** — 起止相同但 type 不同")
            if pairs:
                st.dataframe(
                    [
                        {
                            "span": f'[{g["start_idx"]},{g["end_idx"]}) {g.get("entity","")}',
                            "gold type": g["type"],
                            "pred type": p["type"],
                        }
                        for g, p in pairs
                    ],
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.caption("未检出类型差异对。")
        if category == "解析失败":
            st.markdown("**⑤ 解析失败说明** — 历史记录中 pred 往往为空或与 gold 无法对齐")
            raw = (active or {}).get("raw_output") or ""
            if raw:
                with st.expander("查看该样例记录的 raw_output（截断展示，只读）"):
                    st.code(raw[:1200] + ("…" if len(raw) > 1200 else ""), language="json")
            discarded = (active or {}).get("discarded") or []
            if discarded:
                st.caption(f"记录中 discarded 对象数：{len(discarded)}")


def main() -> None:
    st.set_page_config(page_title="中文医疗实体抽取", layout="wide", initial_sidebar_state="expanded")
    inject_styles()
    examples = load_examples()
    labels = [sample_label(ex) for ex in examples]
    boundary_idx = next((i for i, ex in enumerate(examples) if ex.get("category") == BOUNDARY_CATEGORY), 0)
    default_boundary = examples[boundary_idx]

    with st.sidebar:
        st.markdown("### 演示信息")
        student_id = st.text_input("学号", value="", placeholder="录屏时填写")
        student_name = st.text_input("姓名", value="", placeholder="录屏时填写")
        st.caption("录屏时学号与姓名须留在画面中。页面只做实体抽取，不做诊断、治疗或用药建议。")
        st.markdown(
            f'<div class="sidebar-badge">'
            f'<div class="k">画面中的身份</div>'
            f'<div class="v">学号　{_escape(student_id) or "（未填）"}</div>'
            f'<div class="v">姓名　{_escape(student_name) or "（未填）"}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown("### 系统能力")
        st.caption(
            "词典最长匹配 · BERT-BIO · Qwen 零/单/三样本\n\n"
            "三条错误样例 + 自定义文本 · 主测试指标只读"
        )

    st.markdown(
        """
<div class="hero">
  <h1>中文医疗实体抽取</h1>
  <p class="tagline">CMeEE-V2 全方法现场抽取 · 三类错误样例与自定义文本 · gold / pred 对照 · 主测试指标</p>
  <div class="chips">
    <span class="chip">下标左闭右开</span>
    <span class="chip">录屏默认：边界错误</span>
    <span class="chip">五种抽取方法</span>
    <span class="chip">不做诊疗建议</span>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    tab_live, tab_gallery, tab_system = st.tabs(["现场抽取", "错误样例库", "系统总览与评测"])

    with tab_live:
        st.markdown('<div class="section-title">现场抽取</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-note">可选全部错误样例或自定义文本；五种方法均可现场跑。'
            "</p>",
            unsafe_allow_html=True,
        )
        sample_options = labels + [CUSTOM_SAMPLE]
        choice = st.selectbox(
            "演示样例",
            sample_options,
            index=boundary_idx,
            help="切换样例会填入对应原文与 gold；自定义可任意输入。",
        )
        if st.session_state.get("_sample_choice") != choice:
            st.session_state["_sample_choice"] = choice
            st.session_state.pop("last_extract", None)
            if choice != CUSTOM_SAMPLE:
                ex = find_example_by_label(examples, choice)
                if ex:
                    st.session_state["input_text"] = ex.get("text") or ""
            elif "input_text" not in st.session_state:
                st.session_state["input_text"] = default_boundary.get("text") or DEFAULT_TEXT

        if "input_text" not in st.session_state:
            st.session_state["input_text"] = default_boundary.get("text") or DEFAULT_TEXT

        text = st.text_area("输入文本", key="input_text", height=140)
        active = None if choice == CUSTOM_SAMPLE else find_example_by_label(examples, choice)
        # 若用户改字后仍与某条样例原文完全一致，仍绑定该条 gold
        if active is None or (active and text != active.get("text")):
            matched = next((ex for ex in examples if ex.get("text") == text), None)
            active = matched
        bound_to_sample = active is not None and text == active.get("text")

        col_m, col_b, col_fast = st.columns([2.4, 0.9, 1.2])
        with col_m:
            method_label = st.selectbox("方法（决定本次 pred）", list(METHOD_OPTIONS.keys()), index=0)
        with col_b:
            st.write("")
            st.write("")
            do_extract = st.button("抽取", type="primary", use_container_width=True)
        with col_fast:
            st.write("")
            st.write("")
            do_fast = st.button("词典+BERT 对照", use_container_width=True, help="对当前文本连续跑词典与 BERT，便于快速对比")
        method_key = METHOD_OPTIONS[method_label]

        if do_extract:
            with st.spinner(f"正在用「{method_label}」抽取…"):
                try:
                    pred, latency = extract(method_key, text)
                except SystemExit as exc:
                    st.error(str(exc) or "加载失败，按计划停止。")
                    st.stop()
                except Exception as exc:
                    st.error(f"抽取失败，按计划停止：{exc}")
                    st.stop()
            st.session_state["last_extract"] = {
                "method_label": method_label,
                "method_key": method_key,
                "text": text,
                "pred": pred,
                "latency": latency,
            }
            st.session_state.pop("fast_compare", None)

        if do_fast:
            with st.spinner("正在连续抽取：词典最长匹配 → BERT-BIO…"):
                try:
                    d_pred, d_lat = extract("dictionary_longest", text)
                    b_pred, b_lat = extract("bert_bio", text)
                except SystemExit as exc:
                    st.error(str(exc) or "加载失败，按计划停止。")
                    st.stop()
                except Exception as exc:
                    st.error(f"对照抽取失败：{exc}")
                    st.stop()
            st.session_state["fast_compare"] = {
                "text": text,
                "dictionary": {"pred": d_pred, "latency": d_lat},
                "bert_bio": {"pred": b_pred, "latency": b_lat},
            }
            st.session_state["last_extract"] = {
                "method_label": "BERT-BIO",
                "method_key": "bert_bio",
                "text": text,
                "pred": b_pred,
                "latency": b_lat,
            }

        result = st.session_state.get("last_extract")
        if result and result.get("text") == text:
            m1, m2, m3 = st.columns(3)
            m1.metric("本次方法", result["method_label"])
            m2.metric("延迟（秒）", f"{result['latency']:.3f}")
            m3.metric("pred 实体数", len(result["pred"]))
            st.success(
                f"【本次现场抽取结果】方法「{result['method_label']}」已完成；"
                "下方对照区的 pred 即该方法输出。"
            )
            live_pred = result["pred"]
            live_text = result["text"]
            has_live = True
        else:
            live_pred = []
            live_text = text
            has_live = False
            if bound_to_sample:
                st.warning("尚未对本条文本点击「抽取」。下方 pred 若有内容，来自样例文件历史记录。")
            else:
                st.warning("尚未点击「抽取」。自定义文本需先抽取才有 pred。")

        fast = st.session_state.get("fast_compare")
        if fast and fast.get("text") == text:
            st.markdown("**快速对照摘要（词典 vs BERT）**")
            c1, c2 = st.columns(2)
            with c1:
                st.caption(f"词典最长匹配 · {fast['dictionary']['latency']:.3f}s · {len(fast['dictionary']['pred'])} 实体")
                st.markdown(render_highlighted(text, fast["dictionary"]["pred"]), unsafe_allow_html=True)
            with c2:
                st.caption(f"BERT-BIO · {fast['bert_bio']['latency']:.3f}s · {len(fast['bert_bio']['pred'])} 实体")
                st.markdown(render_highlighted(text, fast["bert_bio"]["pred"]), unsafe_allow_html=True)

        gold = (active.get("gold") or []) if bound_to_sample and active else []
        recorded_pred = (active.get("pred") or []) if bound_to_sample and active else []
        show_pred = live_pred if has_live else recorded_pred
        show_text = live_text if has_live else text
        pred_source = (
            f"本次现场抽取（方法：{result['method_label']}）"
            if has_live and result
            else (
                f"样例文件历史 pred（原方法：{active.get('method', '?')}），请点「抽取」刷新"
                if bound_to_sample and active
                else "尚无 pred"
            )
        )
        render_compare_section(
            text=active.get("text") if bound_to_sample and active else text,
            gold=gold,
            show_pred=show_pred,
            show_text=show_text,
            pred_source=pred_source,
            active=active if bound_to_sample else None,
            bound_to_sample=bound_to_sample,
        )

    with tab_gallery:
        st.markdown('<div class="section-title">错误样例库（只读）</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-note">来自 outputs/errors/examples.json 的三类真实错误；'
            "此处展示历史 gold / pred，不调用模型。完整现场抽取请回「现场抽取」页。</p>",
            unsafe_allow_html=True,
        )
        for ex in examples:
            with st.expander(
                f"{ex.get('category')} · {ex.get('method')} · source_index={ex.get('source_index')}",
                expanded=(ex.get("category") == BOUNDARY_CATEGORY),
            ):
                st.write(ex.get("text", ""))
                g, p = ex.get("gold") or [], ex.get("pred") or []
                st.markdown("**gold 着色**")
                st.markdown(render_highlighted(ex.get("text", ""), g), unsafe_allow_html=True)
                st.markdown("**历史 pred 着色**")
                st.markdown(render_highlighted(ex.get("text", ""), p), unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.dataframe(entity_table(g), width="stretch", hide_index=True)
                with c2:
                    st.dataframe(entity_table(p), width="stretch", hide_index=True)
                if ex.get("category") == BOUNDARY_CATEGORY:
                    pairs = boundary_pairs(g, p)
                    if pairs:
                        st.dataframe(
                            [
                                {
                                    "type": a["type"],
                                    "gold": f'{a.get("entity","")} [{a["start_idx"]},{a["end_idx"]})',
                                    "pred": f'{b.get("entity","")} [{b["start_idx"]},{b["end_idx"]})',
                                }
                                for a, b in pairs
                            ],
                            width="stretch",
                            hide_index=True,
                        )
                if ex.get("category") == "类型错误":
                    pairs = type_pairs(g, p)
                    if pairs:
                        st.dataframe(
                            [
                                {
                                    "span": f'[{a["start_idx"]},{a["end_idx"]})',
                                    "gold type": a["type"],
                                    "pred type": b["type"],
                                }
                                for a, b in pairs
                            ],
                            width="stretch",
                            hide_index=True,
                        )

    with tab_system:
        st.markdown('<div class="section-title">系统总览</div>', unsafe_allow_html=True)
        st.markdown(
            """
| 模块 | 现场行为 | 说明 |
| --- | --- | --- |
| 词典最长匹配 | 不加载神经网络 | 训练集词表最长匹配 |
| BERT-BIO | 加载已有 checkpoint | token / 实体两级指标均已评测 |
| Qwen 零 / 单 / 三样本 | 4bit NF4 本地推理 | 提示条数不同 |
| 数据划分 | 种子 42 | train 13500 · tune 1500 · 主测试 1000 |
| 评分 | 严格实体级 | start、end、type 全相同才算对 |
            """
        )
        st.markdown('<div class="section-title">主测试评测（历史 CSV，只读）</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="section-note">与单条现场抽取无关；页面不比较哪种方法更好。</p>',
            unsafe_allow_html=True,
        )
        overview = metrics_overview_rows()
        if overview:
            st.markdown("**常用行汇总**")
            st.dataframe(overview, width="stretch", hide_index=True)
        for title, path in METRIC_FILES:
            st.markdown(f"**完整表 · {title}**（`{path.relative_to(ROOT).as_posix()}`）")
            if not path.is_file():
                st.warning(f"缺少文件：{path}")
                continue
            rows = read_metrics_csv(path)
            st.dataframe(
                [
                    {
                        "method": row["method"],
                        "precision": row["precision"],
                        "recall": row["recall"],
                        "micro_f1": row["micro_f1"],
                    }
                    for row in rows
                ],
                width="stretch",
                hide_index=True,
            )


if __name__ == "__main__":
    main()
