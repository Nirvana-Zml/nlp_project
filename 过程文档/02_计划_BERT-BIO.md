# 02 计划：BERT-BIO

**状态：** 已交付  
**日期：** 2026-10-14  
**对应评分项：** 实验与评估（token 级与实体级口径）、进度与工作量（训练日志）、源码与可复现（外置配置和种子）

## 本步做

在已经冻结的划分上训练一个 BERT-BIO 序列标注模型，并在主测试集上只评一次。

1. 用本地 `D:/models/bert-base-chinese` 做编码器，上面接 token 分类层，解码成实体。
2. 把嵌套实体压成单层 BIO，压平只生成训练标签。冲突规则、舍弃计数和两套金标见下文「口径」。
3. 早停和选模只使用 `data/processed/tune.json` 的严格实体级 Micro-F1，金标用原始实体。并列时保留更早的 epoch。主测试用该检查点评一次。
4. 同一份主测试集上同时报告 token 级 Micro P/R/F1 和严格实体级 Micro P/R/F1。两种口径都写在下文「口径」里。
5. 保存真实 loss 和 tune 集实体级 F1 曲线。学习率使用预设值 `3e-5`，不宣称做过学习率搜索。

词典最长匹配的结果继续用 `outputs/metrics/dictionary_baseline.csv` 里的已有数字，本步不重抽测试集、不重建词典。

## 本步不做

- GlobalPointer、Qwen、1.5B 对照、bootstrap
- 在测试集上选 epoch、阈值或学习率
- 改 `split_manifest.json` 或两条已确认规则
- 报告里的结论句和错误归因正文

## 预设配置

写入 `configs/bert_bio.yaml` 后按文件训练。

| 项 | 值 |
| --- | --- |
| 模型 | `D:/models/bert-base-chinese` |
| 最大长度 | 256 |
| batch_size | 8 |
| gradient_accumulation_steps | 2 |
| 有效 batch | 16 |
| epochs | 5 |
| learning_rate | 3.0e-5 |
| weight_decay | 0.01 |
| warmup_ratio | 0.1 |
| fp16 | true |
| seed | 42 |
| early_stopping_patience | 2 |
| 早停指标 | tune 集原始金标的严格实体级 Micro-F1 |

8GB 显存装不下 batch 8 时，改为 `batch_size: 4`、`gradient_accumulation_steps: 4`，有效 batch 仍为 16，并把实际值写进日志。除此之外不改上表。

标签为 `O` 以及九类的 `B-` / `I-`。

## 口径

### 冲突与压平

压平只用于训练标签。按原始实体列表从 0 编号，排序键为长度 `end_idx - start_idx` 降序、编号升序。依次保留；与已经保留的实体共享至少一个字符下标则舍弃。左闭右开下首尾相接的实体都保留。

被舍弃的实体按 train、tune、test 分开，各类计数写入 `outputs/metrics/bio_dropped_entities.json`。实现结果须与下表一致。这张表是 2026-10-14 对现有划分套用上述规则得到的，本步不重抽数据。

| 划分 | 金标 | 舍弃 | bod | dis | sym | pro | ite | mic | dru | equ | dep |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 74502 | 13702 | 10706 | 393 | 10 | 583 | 1743 | 153 | 58 | 55 | 1 |
| tune | 8147 | 1458 | 1154 | 31 | 1 | 51 | 186 | 17 | 9 | 9 | 0 |
| 主测试 | 4877 | 839 | 657 | 25 | 0 | 26 | 123 | 4 | 2 | 2 | 0 |

### 两套金标

- 训练标签：压平后的 BIO。
- tune 选模和主测试的严格实体级：原始实体。一条预测算对，当且仅当 `start_idx`、`end_idx`、`type` 都相同。`outputs/predictions/bert_bio.json` 的 `gold` 保持原始实体。
- 主测试实体级 `TP+FN` 为 4877，与词典基线相同。tune 选模的实体级分母为 8147。

### Token 级

Token 金标是压平后的 BIO。实体覆盖的第一个 WordPiece 标 `B-类型`，同一实体的后续 WordPiece 标 `I-类型`。解码用 tokenizer 的 offset 映回原文字符下标。

计入 token 级 Micro 的位置，是去掉 `[CLS]`、`[SEP]` 和 padding 之后，金标或预测属于 `B-` / `I-` 的正文 token。

- TP：该位置金标与预测相同，且都是某个 `B-` 或 `I-`
- FP：预测是某个 `B-` 或 `I-`，且与金标不同
- FN：金标是某个 `B-` 或 `I-`，且与预测不同

`outputs/metrics/bert_bio.csv` 两行，列与词典基线相同：`method,precision,recall,micro_f1,tp,fp,fn`。`method` 分别为 `bert_bio_token` 和 `bert_bio_entity`。

### 切窗

最大长度 256 含 `[CLS]` 与 `[SEP]`，正文最多 254 个 token。正文 token 超过 254 时，按正文窗口 254、步长 64 切分，每个窗口再加 `[CLS]` 与 `[SEP]`。实体的起止字符都落在该窗口正文内才标 BIO；跨出窗口的实体，落在窗口内的字符标 `O`。预测合并后按 `(start_idx, end_idx, type)` 去重。各划分的窗口数写入日志。

字符长度超过 256 的样本是尾部：train 56、tune 9、主测试 3。窗口数以 tokenizer 的结果为准。

## 输入与输出

| 路径 | 动作 |
| --- | --- |
| `data/processed/train.json` | 读，训练 |
| `data/processed/tune.json` | 读，早停 |
| `data/processed/test.json` | 读，训练结束后评一次 |
| `configs/bert_bio.yaml` | 写，上述预设 |
| `outputs/logs/train_bio.log` | 写，loss、舍弃实体数、实际 batch、各划分窗口数 |
| `outputs/metrics/bert_bio_history.json` | 写，每个 epoch 的 train loss 和 tune 原始金标实体级 F1 |
| `outputs/metrics/bert_bio.csv` | 写，`bert_bio_token` 与 `bert_bio_entity` 两行 |
| `outputs/metrics/bio_dropped_entities.json` | 写，train / tune / test 的舍弃总数和九类计数 |
| `outputs/predictions/bert_bio.json` | 写，主测试集原始金标与预测 |
| `outputs/checkpoints/bert_bio/` | 写，tune 上实体级 Micro-F1 最高的检查点；并列取更早 epoch；不放入提交 zip |

## 命令

```text
D:/programdata/anaconda_envs/envs/medical_ner/python.exe scripts/train_bert_bio.py
D:/programdata/anaconda_envs/envs/medical_ner/python.exe scripts/recompute_bert_bio_metrics.py
```

第二条只从 `outputs/predictions/bert_bio.json` 重算 CSV。重算结果须与 CSV 一致。

## 验收

- [ ] 训练日志来自真实运行，早停依据是 tune 集原始金标的实体级 Micro-F1，主测试用的是该检查点
- [ ] 主测试集只评一次，且与词典基线是同一 1,000 条；实体级 `TP+FN` 为 4877
- [ ] CSV 有 `bert_bio_token` 和 `bert_bio_entity` 两行，各含 P/R/F1、TP、FP、FN
- [ ] `bio_dropped_entities.json` 的三个划分总数和九类计数与计划中的表一致
- [ ] 重算命令输出与 CSV 一致
- [ ] 学习率仍是预设的 3e-5，日志里的 batch 与实际运行一致

## 设计决策

等人写下「按计划执行」后才开始编码和训练。本步不设目标 F1。
