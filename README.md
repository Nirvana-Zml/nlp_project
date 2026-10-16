# 中文医疗实体抽取

CMeEE-V2 上的词典基线、BERT-BIO，以及 Qwen2.5-3B 的零样本、单样本和三样本。提示条数在调参集上比较后再评主测试。下标左闭右开：`text[start_idx:end_idx] == entity`。种子 42。无标签的 `CMeEE-V2_test.json` 不参与评分。

环境：`medical_ner`，Python 3.11.17。解释器：

```text
D:/programdata/anaconda_envs/envs/medical_ner/python.exe
```

BERT 编码器：`D:/models/bert-base-chinese`。Qwen 模型：`D:/models/Qwen2.5-3B-Instruct`。4bit 加载使用已安装的 `bitsandbytes` 0.50.2。

## 复现

```text
D:/programdata/anaconda_envs/envs/medical_ner/python.exe scripts/run_step1.py
D:/programdata/anaconda_envs/envs/medical_ner/python.exe scripts/recompute_dictionary_metrics.py
D:/programdata/anaconda_envs/envs/medical_ner/python.exe scripts/train_bert_bio.py
D:/programdata/anaconda_envs/envs/medical_ner/python.exe scripts/recompute_bert_bio_metrics.py
D:/programdata/anaconda_envs/envs/medical_ner/python.exe scripts/run_qwen_3b.py
D:/programdata/anaconda_envs/envs/medical_ner/python.exe scripts/recompute_qwen_3b_metrics.py
D:/programdata/anaconda_envs/envs/medical_ner/python.exe scripts/run_qwen_shot_count.py
D:/programdata/anaconda_envs/envs/medical_ner/python.exe scripts/recompute_qwen_shot_count.py
```

重算命令只读预测文件和 CSV，成功时打印 `RECOMPUTE_OK`。训练配置在 `configs/bert_bio.yaml`。Qwen 配置在 `configs/qwen_3b.yaml`。提示条数配置在 `configs/qwen_shot_count.yaml`。主测试实体级 `TP+FN` 为 4877，调参集为 8147。Qwen 是本地推理，不记录 API 单价。

## 结果文件

| 内容 | 路径 |
| --- | --- |
| 词典基线 | `outputs/metrics/dictionary_baseline.csv` |
| BERT-BIO | `outputs/metrics/bert_bio.csv` |
| 训练曲线 | `outputs/metrics/bert_bio_history.json` |
| 训练日志 | `outputs/logs/train_bio.log` |
| 数据说明 | `过程文档/01_数据结果.md` |
| BERT-BIO 说明 | `过程文档/02_数据结果.md` |
| Qwen 零样本与三样本 | `outputs/metrics/qwen_3b.csv` |
| Qwen 延迟与 token | `outputs/metrics/qwen_3b_cost.json` |
| Qwen 日志 | `outputs/logs/qwen_3b.log` |
| Qwen 说明 | `过程文档/03_数据结果.md` |
| 提示条数调参集 | `outputs/metrics/qwen_shot_count_tune.csv` |
| 提示条数延迟与 token | `outputs/metrics/qwen_shot_count_cost.json` |
| 提示条数选择 | `outputs/metrics/qwen_shot_count_selection.json` |
| 单样本主测试 | `outputs/metrics/qwen_3b_one.csv` |
| 单样本主测试延迟与 token | `outputs/metrics/qwen_3b_one_cost.json` |
| 提示条数日志 | `outputs/logs/qwen_shot_count.log` |
| 提示条数说明 | `过程文档/04_数据结果.md` |

检查点在 `outputs/checkpoints/bert_bio/`，不放入提交 zip。Qwen 预测在 `outputs/predictions/`，同样不放入提交 zip。提交包不含权重、原始数据集、虚拟环境和 API Key。
