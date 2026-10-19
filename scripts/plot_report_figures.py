"""Draw report figures from existing metric files.

Paper conventions used here: white background, 300 dpi, SimSun for Chinese and
Times New Roman for Latin text, inward ticks on all four spines, subfigure
labels (a)(b)(c), and no in-figure title or source line. Those stay in the
caption. Does not train, predict, or edit metrics.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"

# Nature Publishing Group palette (ggsci npg). The training curve keeps its own colors.
NAVY = "#3C5488"
CYAN = "#4DBBD5"
TEAL = "#00A087"
RED = "#E64B35"
PEACH = "#F39B7F"
STEEL = "#8491B4"
BLACK = "#000000"
BLUE = CYAN

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
SPLIT_LABEL = {"train": "训练", "tune": "调参", "test": "主测试"}
METRIC_STYLE = (
    ("精确率", "precision", NAVY, ""),
    ("召回率", "recall", RED, ""),
    ("Micro-F1", "micro_f1", TEAL, ""),
)
COUNT_STYLE = (
    ("tp", "tp", TEAL, ""),
    ("fp", "fp", RED, ""),
    ("fn", "fn", STEEL, ""),
)


def setup_font() -> None:
    simsun = Path(r"C:\Windows\Fonts\simsun.ttc")
    times = Path(r"C:\Windows\Fonts\times.ttf")
    missing = [str(path) for path in (simsun, times) if not path.exists()]
    if missing:
        raise SystemExit("missing font: " + ", ".join(missing))
    for path in (simsun, times):
        font_manager.fontManager.addfont(str(path))
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "SimSun"],
            "axes.unicode_minus": False,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": BLACK,
            "axes.labelcolor": BLACK,
            "axes.edgecolor": BLACK,
            "xtick.color": BLACK,
            "ytick.color": BLACK,
            "axes.linewidth": 0.6,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.top": True,
            "ytick.right": True,
            "hatch.linewidth": 0.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    parsed: dict[str, dict] = {}
    for row in rows:
        item: dict = {"method": row["method"]}
        for key, value in row.items():
            if key == "method":
                continue
            item[key] = float(value) if "." in value else int(value)
        parsed[row["method"]] = item
    return parsed


def parse_log_ints(text: str, keys: list[str]) -> dict[str, int]:
    found: dict[str, int] = {}
    for key in keys:
        match = re.search(rf"{re.escape(key)}=(\d+)", text)
        if match is None:
            raise SystemExit(f"missing {key}")
        found[key] = int(match.group(1))
    return found


def style_ax(ax) -> None:
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(BLACK)
        spine.set_linewidth(0.6)
    ax.tick_params(direction="in", length=3, width=0.6, top=True, right=True, labelsize=8)
    ax.grid(False)


def panel_title(ax, letter: str, text: str) -> None:
    ax.set_title(f"({letter}) {text}", loc="left", fontsize=9, pad=3, color=BLACK)


def save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    print(path)


def new_figure(figsize: tuple[float, float]):
    return plt.figure(figsize=figsize, layout="constrained", facecolor="white")


def fmt_int(value: float) -> str:
    return str(int(round(value)))


def label_vbars(ax, bars, values) -> None:
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * 0.02
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            fmt_int(value),
            ha="center",
            va="bottom",
            fontsize=7,
            color=BLACK,
            clip_on=False,
        )


def label_hbars(ax, bars, values) -> None:
    xmax = ax.get_xlim()[1]
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + xmax * 0.015,
            bar.get_y() + bar.get_height() / 2,
            fmt_int(value),
            va="center",
            ha="left",
            fontsize=7,
            color=BLACK,
            clip_on=True,
        )


def grouped_vbars(ax, labels, series: list[tuple[str, list, str, str]]):
    count = len(series)
    width = min(0.76 / count, 0.24)
    x = np.arange(len(labels))
    containers = []
    for index, (name, values, color, hatch) in enumerate(series):
        offset = (index - (count - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=name,
            color=color,
            hatch=hatch,
            zorder=3,
            edgecolor="white",
            linewidth=0.6,
        )
        containers.append(bars)
    ax.set_xticks(x, labels)
    return containers


def single_bars(ax, labels, values, color=BLUE, hatch=""):
    bars = ax.bar(
        labels,
        values,
        width=0.62,
        color=color,
        hatch=hatch,
        zorder=3,
        edgecolor="white",
        linewidth=0.6,
    )
    return bars


def style_legend(legend) -> None:
    frame = legend.get_frame()
    frame.set_linewidth(0.4)
    frame.set_edgecolor(BLACK)
    frame.set_facecolor("white")
    frame.set_alpha(1)


def npg_cmap():
    cmap = LinearSegmentedColormap.from_list(
        "npg_seq", ["#F7FBFC", "#D7F0F3", "#4DBBD5", "#3C5488"]
    )
    cmap.set_bad("#F3F3F3")
    return cmap


def draw_heatmap(ax, matrix, row_labels, col_labels) -> object:
    display = np.array(matrix, dtype=float)
    positive = display[display > 0]
    vmax = float(positive.max()) if positive.size else 1.0
    shown = display.copy()
    shown[shown <= 0] = np.nan
    image = ax.imshow(shown, norm=LogNorm(vmin=1, vmax=vmax), cmap=npg_cmap(), aspect="auto")
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.tick_params(length=0, top=False, right=False, labelsize=7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    threshold = float(np.sqrt(vmax))
    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            raw = display[i, j]
            ax.text(
                j,
                i,
                fmt_int(raw),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if raw >= threshold else BLACK,
            )
    return image


def add_log_colorbar(fig, image, ax) -> None:
    colorbar = fig.colorbar(image, ax=ax, shrink=0.86, pad=0.02)
    colorbar.ax.tick_params(labelsize=7, length=2, width=0.4)
    colorbar.outline.set_linewidth(0.4)
    colorbar.set_label("计数", fontsize=8)


def fig_split_scale(summary: dict, profile: dict) -> None:
    fig = new_figure((6.4, 3.5))
    ax = fig.subplots()
    style_ax(ax)
    official = summary["files"]
    train_docs = official["CMeEE-V2_train.json"]["docs"]
    dev_docs = official["CMeEE-V2_dev.json"]["docs"]
    test_docs = official["CMeEE-V2_test.json"]["docs"]
    project_docs = {key: profile["sizes"][f"{key}_indices"] for key in ("train", "tune", "test")}
    project_entities = {key: profile["splits"][key]["entities"] for key in ("train", "tune", "test")}
    dev_rest = dev_docs - project_docs["test"]
    segments = [
        ("训练", [project_docs["train"], 0, 0], NAVY, project_entities["train"]),
        ("调参", [project_docs["tune"], 0, 0], TEAL, project_entities["tune"]),
        ("主测试", [0, project_docs["test"], 0], RED, project_entities["test"]),
        ("验证集其余", [0, dev_rest, 0], STEEL, None),
        ("无标签测试集", [0, 0, test_docs], PEACH, None),
    ]
    xs = np.arange(3)
    bottoms = np.zeros(3)
    for name, heights, color, entities in segments:
        label = f"{name}  {fmt_int(sum(heights))} 篇"
        if entities is not None:
            label += f"，{fmt_int(entities)} 条"
        ax.bar(
            xs,
            heights,
            bottom=bottoms,
            width=0.58,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=label,
            zorder=3,
        )
        bottoms = bottoms + np.array(heights, dtype=float)
    for x, total in zip(xs, (train_docs, dev_docs, test_docs)):
        ax.text(x, total + 350, fmt_int(total), ha="center", va="bottom", fontsize=7, color=BLACK)
    ax.set_xticks(xs, ["官方训练集", "官方验证集", "无标签测试集"])
    ax.set_ylabel("文档数")
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0, 18000)
    ax.yaxis.set_major_locator(plt.MultipleLocator(5000))
    legend = ax.legend(loc="upper right", frameon=True)
    style_legend(legend)
    save(fig, "01_split_scale.png")


def fig_type_counts(profile: dict) -> None:
    fig = new_figure((5.6, 3.7))
    ax = fig.subplots()
    matrix = [
        [profile["splits"][split]["type_counts"][key] for split in ("train", "tune", "test")]
        for key in TYPE_ORDER
    ]
    image = draw_heatmap(
        ax,
        matrix,
        [TYPE_LABEL[key] for key in TYPE_ORDER],
        [SPLIT_LABEL[key] for key in ("train", "tune", "test")],
    )
    add_log_colorbar(fig, image, ax)
    save(fig, "02_type_counts.png")


def fig_length_nested(profile: dict) -> None:
    fig = new_figure((7.2, 2.85))
    left, right = fig.subplots(1, 2)
    splits = ("train", "tune", "test")
    xs = np.arange(len(splits))
    length_keys = ("1-2", "3-5", "6+")
    length_style = (
        ("1–2 字", PEACH, "o"),
        ("3–5 字", CYAN, "s"),
        ("6 字及以上", NAVY, "D"),
    )
    for ax in (left, right):
        style_ax(ax)
        ax.set_xticks(xs, [SPLIT_LABEL[key] for key in splits])
    panel_title(left, "a", "实体长度")
    panel_title(right, "b", "嵌套文档")
    for (name, color, marker), key in zip(length_style, length_keys):
        shares = []
        for split in splits:
            block = profile["splits"][split]
            shares.append(block["entity_length_counts"][key] / block["entities"] * 100)
        left.plot(xs, shares, color=color, marker=marker, ms=4.5, lw=1.4, label=name)
    nested = [
        profile["splits"][split]["nested_docs"] / profile["splits"][split]["docs"] * 100 for split in splits
    ]
    right.plot(xs, nested, color=RED, marker="o", ms=5, lw=1.4)
    left.set_ylim(0, 60)
    right.set_ylim(0, 40)
    left.set_ylabel("占实体数 / %")
    right.set_ylabel("占文档数 / %")
    legend = left.legend(frameon=True, loc="upper right")
    style_legend(legend)
    save(fig, "03_length_nested.png")


def fig_span_mismatch(log_text: str) -> None:
    keys = parse_log_ints(
        log_text,
        [
            "train_token_span_mismatch",
            "tune_token_span_mismatch",
            "test_token_span_mismatch",
        ],
    )
    fig = new_figure((3.8, 2.7))
    ax = fig.subplots()
    style_ax(ax)
    labels = ["训练", "调参", "主测试"]
    values = [
        keys["train_token_span_mismatch"],
        keys["tune_token_span_mismatch"],
        keys["test_token_span_mismatch"],
    ]
    colors = [NAVY, TEAL, RED]
    xs = np.arange(len(labels))
    bars = ax.bar(xs, values, width=0.55, color=colors, edgecolor="white", linewidth=0.6, zorder=3)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.03,
            fmt_int(value),
            ha="center",
            va="bottom",
            fontsize=7,
            color=BLACK,
        )
    ax.set_xticks(xs, labels)
    ax.set_ylabel("未对齐实体数")
    ax.set_ylim(0, max(values) * 1.22)
    save(fig, "04_span_mismatch.png")


def _radar(ax, series: list[tuple[str, list[float], str]], rmax: float, grids: list[float]) -> None:
    spokes = ["精确率", "召回率", "Micro-F1"]
    angles = np.linspace(0, 2 * np.pi, len(spokes), endpoint=False)
    closed = np.concatenate([angles, angles[:1]])
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles), spokes, fontsize=8)
    ax.set_ylim(0, rmax)
    ax.set_rgrids(grids, [f"{value:g}" for value in grids], fontsize=7, angle=18)
    for name, values, color in series:
        points = np.concatenate([values, values[:1]])
        ax.plot(closed, points, color=color, lw=1.5, marker="o", ms=3.5, label=name)
        ax.fill(closed, points, color=color, alpha=0.14)
    ax.spines["polar"].set_color("#C8C8C8")
    ax.grid(color="#D9D9D9", lw=0.5)
    ax.set_facecolor("white")
    ax.tick_params(labelsize=8)


def fig_main_prf(dictionary: dict, bert: dict, qwen: dict, qwen_one: dict) -> None:
    fig = new_figure((7.2, 3.35))
    left, right = fig.subplots(1, 2, subplot_kw={"projection": "polar"})
    fields = ["precision", "recall", "micro_f1"]
    _radar(
        left,
        [
            ("全匹配", [dictionary["dictionary_all"][field] for field in fields], RED),
            ("最长匹配", [dictionary["dictionary_longest"][field] for field in fields], CYAN),
            ("BERT", [bert["bert_bio_entity"][field] for field in fields], NAVY),
        ],
        0.9,
        [0.2, 0.4, 0.6, 0.8],
    )
    _radar(
        right,
        [
            ("0 条", [qwen["qwen_3b_zero"][field] for field in fields], PEACH),
            ("1 条", [qwen_one["qwen_3b_one"][field] for field in fields], TEAL),
            ("3 条", [qwen["qwen_3b_three"][field] for field in fields], NAVY),
        ],
        0.05,
        [0.01, 0.02, 0.03, 0.04],
    )
    panel_title(left, "a", "词典与 BERT-BIO")
    panel_title(right, "b", "Qwen 提示")
    left.legend(frameon=True, loc="upper right", bbox_to_anchor=(1.18, 1.12))
    right.legend(frameon=True, loc="upper right", bbox_to_anchor=(1.18, 1.12))
    style_legend(left.get_legend())
    style_legend(right.get_legend())
    save(fig, "05_main_entity_prf.png")


def _ternary_xy(tp: float, fp: float, fn: float) -> tuple[float, float]:
    total = tp + fp + fn
    tp, fp, fn = tp / total, fp / total, fn / total
    height = np.sqrt(3) / 2
    return fn + 0.5 * tp, tp * height


def fig_tp_fp_fn(dictionary: dict, bert: dict, qwen: dict, qwen_one: dict) -> None:
    fig = new_figure((4.8, 4.35))
    ax = fig.subplots()
    height = np.sqrt(3) / 2
    corners = np.array([[0.5, height], [0.0, 0.0], [1.0, 0.0]])
    ax.add_patch(plt.Polygon(corners, closed=True, fill=False, edgecolor=BLACK, lw=0.8, zorder=2))
    for level in (0.2, 0.4, 0.6, 0.8):
        ax.plot([0.5 * level, 1 - 0.5 * level], [level * height, level * height], color="#D5D5D5", lw=0.4, zorder=1)
        ax.plot(
            [0.5 * (1 - level), 1 - level],
            [(1 - level) * height, 0],
            color="#D5D5D5",
            lw=0.4,
            zorder=1,
        )
        ax.plot(
            [level + 0.5 * (1 - level), level],
            [(1 - level) * height, 0],
            color="#D5D5D5",
            lw=0.4,
            zorder=1,
        )
    points = [
        ("全匹配", dictionary["dictionary_all"], NAVY, "o"),
        ("最长匹配", dictionary["dictionary_longest"], CYAN, "s"),
        ("BERT", bert["bert_bio_entity"], TEAL, "D"),
        ("Qwen 0", qwen["qwen_3b_zero"], PEACH, "^"),
        ("Qwen 1", qwen_one["qwen_3b_one"], RED, "v"),
        ("Qwen 3", qwen["qwen_3b_three"], STEEL, "P"),
    ]
    offsets = {
        "全匹配": (6, 6),
        "最长匹配": (6, 6),
        "BERT": (6, 4),
        "Qwen 0": (6, 6),
        "Qwen 1": (-8, 8),
        "Qwen 3": (-34, 6),
    }
    for name, row, color, marker in points:
        x, y = _ternary_xy(row["tp"], row["fp"], row["fn"])
        ax.scatter([x], [y], s=42, color=color, marker=marker, zorder=3, label=name)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=offsets[name], fontsize=7, color=BLACK)
    ax.text(0.5, height + 0.035, "tp", ha="center", va="bottom", fontsize=8)
    ax.text(-0.02, -0.045, "fp", ha="right", va="top", fontsize=8)
    ax.text(1.02, -0.045, "fn", ha="left", va="top", fontsize=8)
    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.12, height + 0.1)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "06_main_tp_fp_fn.png")


def fig_bert_granularity(bert: dict) -> None:
    fig = new_figure((4.3, 3.05))
    ax = fig.subplots()
    style_ax(ax)
    rows = [bert["bert_bio_token"], bert["bert_bio_entity"]]
    markers = {"precision": "o", "recall": "s", "micro_f1": "D"}
    for name, field, color, _hatch in METRIC_STYLE:
        values = [row[field] for row in rows]
        ax.plot([0, 1], values, color=color, marker=markers[field], ms=5, lw=1.4, label=name)
    ax.set_xticks([0, 1], ["token 级", "实体级"])
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("分数")
    legend = ax.legend(frameon=True, loc="lower left")
    style_legend(legend)
    save(fig, "07_bert_granularity.png")


def fig_qwen_cost(cost: dict, one_cost: dict) -> None:
    fig = new_figure((4.6, 3.35))
    ax = fig.subplots()
    style_ax(ax)
    order = ["qwen_3b_zero", "qwen_3b_one", "qwen_3b_three"]
    labels = ["0 条", "1 条", "3 条"]
    colors = [PEACH, TEAL, NAVY]
    rows = [cost[key] if key in cost else one_cost[key] for key in order]
    inputs = [row["mean_input_tokens"] for row in rows]
    latency = [row["mean_latency_sec"] for row in rows]
    outputs = [row["mean_output_tokens"] for row in rows]
    ax.plot(inputs, latency, color=STEEL, lw=1.0, zorder=2)
    sizes = [(value / min(outputs)) ** 2 * 80 for value in outputs]
    ax.scatter(inputs, latency, s=sizes, c=colors, zorder=3, edgecolor="white", linewidth=0.6)
    for x, y, label in zip(inputs, latency, labels):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=7)
    ax.set_xlabel("平均输入 token")
    ax.set_ylabel("平均延迟 / s")
    ax.set_xlim(min(inputs) - 80, max(inputs) + 90)
    ax.set_ylim(0, max(latency) * 1.25)
    save(fig, "08_qwen_cost.png")


def fig_qwen_parse(cost: dict, one_cost: dict) -> None:
    fig = new_figure((5.2, 2.7))
    ax = fig.subplots()
    order = ["qwen_3b_zero", "qwen_3b_one", "qwen_3b_three"]
    rows = [cost[key] if key in cost else one_cost[key] for key in order]
    fields = ["parse_failures", "invalid_entities", "surface_mismatches"]
    matrix = [[row[field] for field in fields] for row in rows]
    image = draw_heatmap(ax, matrix, ["0 条", "1 条", "3 条"], ["解析失败", "无效实体", "表面不一致"])
    add_log_colorbar(fig, image, ax)
    save(fig, "09_qwen_parse.png")


def fig_bio_dropped(dropped: dict) -> None:
    fig = new_figure((7.2, 4.9))
    grid = fig.add_gridspec(2, 1, height_ratios=[0.85, 1.45])
    top = fig.add_subplot(grid[0])
    bottom = fig.add_subplot(grid[1])
    style_ax(top)
    panel_title(top, "a", "gold 与 total")
    splits = ("train", "tune", "test")
    y = np.arange(len(splits))[::-1]
    gold = [dropped[key]["gold"] for key in splits]
    total = [dropped[key]["total"] for key in splits]
    for yi, g, t in zip(y, gold, total):
        top.plot([t, g], [yi, yi], color=STEEL, lw=1.3, zorder=2)
        top.annotate(fmt_int(t), (t, yi), textcoords="offset points", xytext=(4, 5), ha="left", fontsize=7, color=RED)
        top.annotate(fmt_int(g), (g, yi), textcoords="offset points", xytext=(4, 5), ha="left", fontsize=7, color=NAVY)
    top.scatter(gold, y, s=32, color=NAVY, zorder=3, label="gold")
    top.scatter(total, y, s=32, color=RED, zorder=3, label="total")
    top.set_yticks(y, [SPLIT_LABEL[key] for key in splits])
    top.set_xscale("log")
    top.set_xlim(500, 160000)
    top.set_xlabel("实体数")
    legend = top.legend(frameon=True, loc="upper right")
    style_legend(legend)

    panel_title(bottom, "b", "by_type")
    matrix = [
        [dropped[split]["by_type"][key] for split in splits] for key in TYPE_ORDER
    ]
    image = draw_heatmap(
        bottom,
        matrix,
        [TYPE_LABEL[key] for key in TYPE_ORDER],
        [SPLIT_LABEL[key] for key in splits],
    )
    add_log_colorbar(fig, image, bottom)
    save(fig, "10_bio_dropped.png")


def fig_shot_tune(tune: dict, cost: dict, selection: dict) -> None:
    fig = new_figure((7.2, 4.55))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])
    top = fig.add_subplot(grid[0, :])
    latency_ax = fig.add_subplot(grid[1, 0])
    token_ax = fig.add_subplot(grid[1, 1])
    order = ["qwen_shot_0", "qwen_shot_1", "qwen_shot_3"]
    selected = int(selection["selected_shots"])
    xs = [0, 1, 3]
    tick_labels = ["0", "1*", "3"] if selected == 1 else ["0", "1", "3"]
    rows = [tune[key] for key in order]
    costs = [cost[key] for key in order]
    markers = {"precision": "o", "recall": "s", "micro_f1": "D"}

    style_ax(top)
    panel_title(top, "a", "精确率、召回率与 Micro-F1")
    for name, field, color, _hatch in METRIC_STYLE:
        top.plot(xs, [row[field] for row in rows], color=color, marker=markers[field], ms=5, lw=1.4, label=name)
    top.axvline(selected, color="#D0D0D0", lw=0.8, zorder=0)
    top.set_xticks(xs, tick_labels)
    top.set_ylim(0, 0.045)
    top.set_ylabel("分数")
    top.set_xlabel("提示条数")
    legend = top.legend(frameon=True, ncol=3, loc="upper left")
    style_legend(legend)

    style_ax(latency_ax)
    panel_title(latency_ax, "b", "平均延迟")
    latency_ax.plot(xs, [row["mean_latency_sec"] for row in costs], color=CYAN, marker="o", ms=5, lw=1.4)
    latency_ax.axvline(selected, color="#D0D0D0", lw=0.8, zorder=0)
    latency_ax.set_xticks(xs, tick_labels)
    latency_ax.set_ylim(0, max(row["mean_latency_sec"] for row in costs) * 1.2)
    latency_ax.set_ylabel("延迟 / s")
    latency_ax.set_xlabel("提示条数")

    style_ax(token_ax)
    panel_title(token_ax, "c", "平均 token")
    token_ax.plot(xs, [row["mean_input_tokens"] for row in costs], color=NAVY, marker="o", ms=5, lw=1.4, label="输入")
    token_ax.plot(xs, [row["mean_output_tokens"] for row in costs], color=PEACH, marker="s", ms=5, lw=1.4, label="输出")
    token_ax.axvline(selected, color="#D0D0D0", lw=0.8, zorder=0)
    token_ax.set_xticks(xs, tick_labels)
    token_ax.set_ylim(0, max(row["mean_input_tokens"] for row in costs) * 1.2)
    token_legend = token_ax.legend(frameon=True, loc="upper left")
    style_legend(token_legend)
    token_ax.set_ylabel("token 数")
    token_ax.set_xlabel("提示条数")
    save(fig, "11_shot_tune.png")


def fig_train_curve(history: dict) -> None:
    fig = new_figure((4.8, 3.45))
    ax = fig.subplots()
    style_ax(ax)
    epochs = history["epochs"]
    xs = [item["epoch"] for item in epochs]
    series = (
        ("train_loss", [item["train_loss"] for item in epochs], "#000000", "-", "o"),
        ("tune_precision", [item["tune_precision"] for item in epochs], "#0072B2", "--", "s"),
        ("tune_recall", [item["tune_recall"] for item in epochs], "#D55E00", ":", "^"),
        ("tune_micro_f1", [item["tune_micro_f1"] for item in epochs], "#009E73", "-.", "D"),
    )
    for name, values, color, linestyle, marker in series:
        ax.plot(
            xs,
            values,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markersize=4,
            linewidth=1.2,
            label=name,
            zorder=3,
        )
    ax.set_xlim(0.7, 5.3)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(xs)
    ax.set_xlabel("epoch")
    ax.set_ylabel("损失 / 调参集分数")
    best = history["best_epoch"]
    ax.axvline(best, color="#6E6E6E", linewidth=0.7, linestyle="--", zorder=1)
    ax.text(best - 0.12, 0.93, f"best_epoch={best}", ha="right", va="top", fontsize=8, color=BLACK)
    handles, labels = ax.get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=True)
    style_legend(legend)
    save(fig, "12_train_curve.png")


def main() -> None:
    setup_font()
    OUT.mkdir(parents=True, exist_ok=True)
    summary = load_json(ROOT / "outputs" / "data_validation_summary.json")
    profile = load_json(ROOT / "outputs" / "metrics" / "split_profile.json")
    dropped = load_json(ROOT / "outputs" / "metrics" / "bio_dropped_entities.json")
    history = load_json(ROOT / "outputs" / "metrics" / "bert_bio_history.json")
    qwen_cost = load_json(ROOT / "outputs" / "metrics" / "qwen_3b_cost.json")
    one_cost = load_json(ROOT / "outputs" / "metrics" / "qwen_3b_one_cost.json")
    shot_cost = load_json(ROOT / "outputs" / "metrics" / "qwen_shot_count_cost.json")
    selection = load_json(ROOT / "outputs" / "metrics" / "qwen_shot_count_selection.json")
    dictionary = load_csv(ROOT / "outputs" / "metrics" / "dictionary_baseline.csv")
    bert = load_csv(ROOT / "outputs" / "metrics" / "bert_bio.csv")
    qwen = load_csv(ROOT / "outputs" / "metrics" / "qwen_3b.csv")
    qwen_one = load_csv(ROOT / "outputs" / "metrics" / "qwen_3b_one.csv")
    tune = load_csv(ROOT / "outputs" / "metrics" / "qwen_shot_count_tune.csv")
    log_text = (ROOT / "outputs" / "logs" / "train_bio.log").read_text(encoding="utf-8")

    fig_split_scale(summary, profile)
    fig_type_counts(profile)
    fig_length_nested(profile)
    fig_span_mismatch(log_text)
    fig_main_prf(dictionary, bert, qwen, qwen_one)
    fig_tp_fp_fn(dictionary, bert, qwen, qwen_one)
    fig_bert_granularity(bert)
    fig_qwen_cost(qwen_cost, one_cost)
    fig_qwen_parse(qwen_cost, one_cost)
    fig_bio_dropped(dropped)
    fig_shot_tune(tune, shot_cost, selection)
    fig_train_curve(history)
    print("FIGURES_OK")


if __name__ == "__main__":
    main()
