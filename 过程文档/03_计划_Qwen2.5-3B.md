# 03 计划：Qwen2.5-3B 零样本与三样本

**状态：** 已交付  
**日期：** 2026-10-14  
**对应评分项：** 实验与评估（同一主测试集上的大模型方案）、进度与工作量（推理日志、延迟和 token）、源码与可复现（外置配置、固定示例和种子）

## 本步做

在已经冻结的主测试集上，用本地 Qwen2.5-3B-Instruct 做零样本和三样本实体抽取，每种只评一次。

1. 模型目录是 `D:/models/Qwen2.5-3B-Instruct`。当前该目录不存在，只有 `bert-base-chinese`。确认执行前需要先放好模型。本步不自动下载，也不改用 1.5B 或 Qwen3。
2. 加载方式预设为 4bit NF4、double quant、计算精度 float16。加载失败就停止，不换精度，不缩小测试集，也不先跑完一部分再改。
3. 零样本不给示例。三样本只用下面 3 条，且每条测试文本都用这同一组、同一顺序。示例只来自 `data/processed/train.json`：有实体的样本按 `source_index` 排序后，用 `random.Random(42)` 打乱，取前 3 个。结果是 `3949`、`5340`、`1237`。看到分数后不更换。
4. 提示里的下标是左闭右开：`text[start_idx:end_idx] == entity`。不使用方案稿里的闭区间结束下标。
5. 两种方法都在 `data/processed/test.json` 的 1,000 条上各评一次。严格实体级 Micro P/R/F1 与词典、BERT-BIO 相同：`start_idx`、`end_idx`、`type` 都相同才算对。主测试 `TP+FN` 必须是 4877。
6. 程序记录每种方法的平均延迟、平均输入 token 数、平均输出 token 数。本地推理不写 API 单价。

## 本步不做

- 1.5B、Qwen3、GlobalPointer、bootstrap、单样本
- 在测试集或 tune 集上改提示、示例或 `max_new_tokens`
- 改 `split_manifest.json`、重抽测试集、重训 BERT
- 把解析失败的输出改写成正确答案
- 报告里的结论句和错误归因正文

## 预设配置

写入 `configs/qwen_3b.yaml` 后按文件推理。

| 项 | 值 |
| --- | --- |
| 模型 | `D:/models/Qwen2.5-3B-Instruct` |
| 量化 | 4bit NF4，double quant，compute float16 |
| temperature | 0 |
| do_sample | false |
| max_new_tokens | 1536 |
| repetition_penalty | 1.0 |
| batch_size | 1 |
| seed | 42 |
| 三样本 source_index | 3949, 5340, 1237 |

`max_new_tokens` 用 1536，是因为主测试单条最多 30 个实体，384 会截断长输出。这是预设，不是在测试集上搜出来的。

提示要求模型只输出 JSON，实体字段为 `entity`、`type`、`start_idx`、`end_idx`，类型限于九类。零样本和三样本使用同一模板，三样本在模板中插入上述 3 条的原文和金标。

计分规则：

- JSON 不能解析：该条预测为空，记一次解析失败，原始输出保存。
- 类型不在九类、下标不是整数或下标越界：该实体不进入计分，计入无效实体数，不改成正确下标。
- 下标合法但表面文本与切片不一致：仍按三元组计分，并另记次数。
- 同一条里的重复三元组只计一次。

## 输入与输出

| 路径 | 动作 |
| --- | --- |
| `data/processed/train.json` | 读，只取上面 3 条示例 |
| `data/processed/test.json` | 读，两种方法各评一次 |
| `configs/qwen_3b.yaml` | 写，上述预设 |
| `outputs/predictions/qwen_3b_zero.json` | 写，主测试原始金标、预测和原始输出 |
| `outputs/predictions/qwen_3b_three.json` | 写，同上 |
| `outputs/metrics/qwen_3b.csv` | 写，`qwen_3b_zero` 与 `qwen_3b_three` 两行 |
| `outputs/metrics/qwen_3b_cost.json` | 写，平均延迟、输入 token、输出 token、解析失败数、无效实体数 |
| `outputs/logs/qwen_3b.log` | 写，模型路径、量化、示例编号、生成参数、条数 |

CSV 列与词典基线相同：`method,precision,recall,micro_f1,tp,fp,fn`。

## 命令

```text
D:/programdata/anaconda_envs/envs/medical_ner/python.exe scripts/run_qwen_3b.py
D:/programdata/anaconda_envs/envs/medical_ner/python.exe scripts/recompute_qwen_3b_metrics.py
```

第二条只从两个预测文件重算 CSV。重算结果须与 CSV 一致。

1,000 条乘两种方法。8GB 显卡上 4bit 逐条生成可能要数小时。若要改成子集，先改本计划再确认。看到分数后不缩减测试集。

## 验收

- [x] 模型路径、4bit 和生成参数与本计划一致，并写在日志里
- [x] 三样本示例是 3949、5340、1237，且只来自 train
- [x] 两种方法各在同一 1,000 条上评一次，实体级 `TP+FN` 都是 4877
- [x] CSV 有两行 P/R/F1、TP、FP、FN
- [x] 成本文件有平均延迟、平均输入 token、平均输出 token
- [x] 重算命令输出与 CSV 一致
- [x] 解析失败和无效实体有计数，原始输出已保存

## 设计决策

等人写下「按计划执行」后才开始编码和推理。模型目录还没就位时不启动推理。本步不设目标 F1。
