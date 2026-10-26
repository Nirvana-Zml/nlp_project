# 中文医疗实体抽取

CMeEE-V2 上的词典基线、BERT-BIO，以及 Qwen2.5-3B 的零样本、单样本和三样本。提示条数在调参集上比较后再评主测试。下标左闭右开：`text[start_idx:end_idx] == entity`。种子 42。无标签的 `CMeEE-V2_test.json` 不参与评分。

作者环境是 conda `medical_ner`、Python 3.11。请先建隔离环境，再装依赖。虚拟环境不要打进提交 zip。

```text
# conda（与作者环境一致）
conda create -n medical_ner python=3.11 -y
conda activate medical_ner

# 或 venv
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
```

在已激活的环境中、仓库根目录安装依赖：

```text
pip install -r requirements.txt
```

`requirements.txt` 里的 `torch==2.6.0+cu124` 带 CUDA 12.4 构建标记。默认 PyPI 不一定能解析这一行。装不上时，先安装与本机 CUDA 匹配的 PyTorch 2.6.0，再安装其余依赖。完整流水线需要 NVIDIA GPU。Qwen 的 4bit 加载使用 `bitsandbytes`。

## 原始数据

把 CMeEE-V2 的三个 JSON 放到仓库内的下列路径：

```text
data/raw/CMeEE-V2/CMeEE-V2_train.json
data/raw/CMeEE-V2/CMeEE-V2_dev.json
data/raw/CMeEE-V2/CMeEE-V2_test.json
```

下载页：<https://tianchi.aliyun.com/dataset/dataDetail?dataId=95414>（CBLUE 数据集 95414）。说明见 <https://github.com/CBLUEbenchmark/CBLUE>。官方测试集没有标签，不参与评分，但校验会读取它。本机这份原始目录是 17,832,861 字节。

## 模型

流水线不会自动下载权重。配置里的路径相对仓库根目录：

| 模型 | 配置 | 放到 |
| --- | --- | --- |
| bert-base-chinese | `configs/bert_bio.yaml` 的 `model_path` | `models/bert-base-chinese` |
| Qwen2.5-3B-Instruct | `configs/qwen_3b.yaml` 的 `model_path` | `models/Qwen2.5-3B-Instruct` |

官方权重：

- BERT：<https://huggingface.co/google-bert/bert-base-chinese>
- Qwen：<https://huggingface.co/Qwen/Qwen2.5-3B-Instruct>
- Qwen 的 ModelScope 页面：<https://modelscope.cn/models/Qwen/Qwen2.5-3B-Instruct>

BERT 目录需要 `config.json` 和 `model.safetensors`，以及分词器文件。`model.safetensors` 为 411,553,788 字节。官方仓库还带 pytorch、flax、tf 权重；本机把这些格式都留下时，目录是 1,710,993,583 字节。训练只读 `model.safetensors`。

Qwen 目录需要 `config.json` 和 safetensors 权重。本机这份目录是 6,183,458,486 字节。

可在仓库根目录手动下载：

```text
huggingface-cli download google-bert/bert-base-chinese --local-dir models/bert-base-chinese
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir models/Qwen2.5-3B-Instruct
```

权重若已在仓库外，设置环境变量 `BERT_MODEL_PATH` 和 `QWEN_MODEL_PATH`，不要把本机路径写进配置文件。

## 一条命令

在仓库根目录执行：

```text
python scripts/run_pipeline.py
```

脚本先检查原始数据、两个模型目录和 CUDA。通过后依次运行：

1. `scripts/run_step1.py`：校验三份原始 JSON，按种子 42 划分，并在主测试上跑词典全匹配和最长匹配。
2. `scripts/train_bert_bio.py`：按 `configs/bert_bio.yaml` 训练 BERT-BIO，主测试只评一次。
3. `scripts/run_qwen_3b.py`：Qwen2.5-3B 的零样本和三样本主测试。
4. `scripts/run_qwen_shot_count.py`：调参集上比较 0、1、3 条示例，再评入选条数的主测试。
5. 四条 `scripts/recompute_*.py`：用预测文件重算指标，并与已写出的 CSV 对照。

成功时四条重算各打印 `RECOMPUTE_OK`，最后打印 `PIPELINE_OK`。`outputs/predictions/` 纳入仓库和提交 zip。检查点不放入提交 zip。

## 缺少模型或 GPU

缺数据、缺模型、目录里的权重对不上，或没有 CUDA 时，命令直接退出，不开始训练，也不自动下载、不改模型、不降低精度、不缩小测试集。

没有 BERT 权重时会写明需要 `models/bert-base-chinese` 里的 `config.json` 和 `model.safetensors`，并给出上面的 Hugging Face 地址。没有 Qwen 权重时会写明需要 `models/Qwen2.5-3B-Instruct` 里的 `config.json` 和 safetensors，并给出 Hugging Face 与 ModelScope 地址。

没有 CUDA 时打印：

```text
未检测到 CUDA。BERT-BIO 的 fp16 训练和 Qwen2.5-3B 的 4bit 推理都需要 NVIDIA GPU。
本程序不会改用 CPU，不会降低精度，也不会缩小测试集。
只重算已有预测时请运行：python scripts/run_pipeline.py --recompute
```

未安装 torch 时会提示先安装依赖，并说明 `torch==2.6.0+cu124` 在默认 PyPI 上不一定能装上。

## 预期文件和分数

主测试严格实体级 Micro-F1 来自已记录的 CSV。六位小数如下。

| 方法 | Micro-F1 | 文件 |
| --- | --- | --- |
| 词典全匹配 | 0.252369 | `outputs/metrics/dictionary_baseline.csv` |
| 词典最长匹配 | 0.392433 | `outputs/metrics/dictionary_baseline.csv` |
| BERT-BIO 实体 | 0.634777 | `outputs/metrics/bert_bio.csv` |
| BERT-BIO token | 0.776996 | `outputs/metrics/bert_bio.csv` |
| Qwen 0 示例 | 0.011517 | `outputs/metrics/qwen_3b.csv` |
| Qwen 1 示例 | 0.022367 | `outputs/metrics/qwen_3b_one.csv` |
| Qwen 3 示例 | 0.021382 | `outputs/metrics/qwen_3b.csv` |

主测试实体级 `TP+FN` 为 4877，调参集为 8147。同一原始文件和种子 42 下，词典基线应得到上表前两行。BERT 与 Qwen 重新训练、重新生成后，小数可能与这次记录不同。`--recompute` 核对的是当前预测和已保存 CSV，一致时打印 `RECOMPUTE_OK`。

流水线写出的主要文件：

| 内容 | 路径 |
| --- | --- |
| 划分 | `data/processed/train.json`、`tune.json`、`test.json`、`dev_full.json`、`split_manifest.json` |
| 校验 | `outputs/data_validation_summary.json`、`outputs/data_validation_errors.json` |
| 词典预测与指标 | `outputs/predictions/dictionary_all.json`、`dictionary_longest.json`；`outputs/metrics/dictionary_baseline.csv` |
| BERT 检查点、预测与指标 | `outputs/checkpoints/bert_bio/`；`outputs/predictions/bert_bio.json`；`outputs/metrics/bert_bio.csv` |
| Qwen 主测试 | `outputs/predictions/qwen_3b_zero.json`、`qwen_3b_three.json`、`qwen_3b_one.json` |
| Qwen 调参集 | `outputs/predictions/qwen_shot_0_tune.json`、`qwen_shot_1_tune.json`、`qwen_shot_3_tune.json` |
| Qwen 指标 | `outputs/metrics/qwen_3b.csv`、`qwen_3b_one.csv`、`qwen_shot_count_tune.csv`、`qwen_shot_count_selection.json` |
| 日志 | `outputs/logs/step1.log`、`train_bio.log`、`qwen_3b.log`、`qwen_shot_count.log` |

## 时间和磁盘

记录来自 RTX 4060 Laptop（`outputs/metrics/bert_inference_latency.json` 的 `device`）。

BERT 五轮的 `elapsed_sec` 在 `outputs/logs/train_bio.log` 中为 430.6、444.2、281.2、180.8、187.3，合计 1524.1 秒。这是训练墙钟，不含另外统计的模型加载。

Qwen 的 `mean_latency_sec` 只包住 `generate`。主测试 1000 篇：零样本 6.418967 秒、三样本 10.537989 秒、单样本 10.691977 秒（`outputs/metrics/qwen_3b_cost.json`、`qwen_3b_one_cost.json`）。调参集 1500 篇：0 条 8.695468 秒、1 条 12.472886 秒、3 条 12.660887 秒（`outputs/metrics/qwen_shot_count_cost.json`）。合计 78,392.8 秒，约 21.8 小时，不含模型加载。

已记录的 BERT 训练加上这段 Qwen 生成约 22.2 小时。数据校验和词典基线不加载模型，耗时远小于这两段。换机器后时间会变。

磁盘上已测到的体积：原始数据 17,832,861 字节，Qwen 目录 6,183,458,486 字节，BERT 的 `model.safetensors` 411,553,788 字节，BERT 检查点目录 407,231,325 字节，预测目录 17,668,762 字节。按 Qwen 全目录、BERT 的 `model.safetensors` 和检查点估算，模型和本次产物约 8 GB。PyTorch 安装另计。若 BERT 目录保留 pytorch、flax、tf 格式，再增加约 1.3 GB。

## 只重算指标

已有 `outputs/predictions/` 和 `outputs/metrics/` 时，在仓库根目录执行：

```text
python scripts/run_pipeline.py --recompute
```

这条命令不加载 BERT 或 Qwen，也不需要 GPU。它只运行四条重算脚本。预测或 CSV 缺失时会列出相对路径并退出。成功时打印四次 `RECOMPUTE_OK` 和一次 `PIPELINE_OK`。

训练配置在 `configs/bert_bio.yaml`。Qwen 配置在 `configs/qwen_3b.yaml`。提示条数配置在 `configs/qwen_shot_count.yaml`。Qwen 是本地推理，不记录 API 单价。提交包含 `outputs/predictions/`，不含权重、检查点、原始数据集、虚拟环境和 API Key。已有预测和 CSV 时，`--recompute` 不需要 GPU。
