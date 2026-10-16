# 02 评审：BERT-BIO

**结论：** 已确认  
**日期：** 2026-10-14  
**确认日期：** 2026-10-14  
**计划：** [02_计划_BERT-BIO.md](02_计划_BERT-BIO.md)  
**数据结果：** [02_数据结果.md](02_数据结果.md)

## 实际跑过的命令

```text
D:/programdata/anaconda_envs/envs/medical_ner/python.exe scripts/train_bert_bio.py
D:/programdata/anaconda_envs/envs/medical_ner/python.exe scripts/recompute_bert_bio_metrics.py
```

第二条在新进程中只读取 `outputs/predictions/bert_bio.json` 和 `outputs/metrics/bert_bio.csv`，输出 `RECOMPUTE_OK`。主测试实体级 `TP+FN` 为 4877。

## 数字出处

| 数字 | 值 | 文件 | 字段或行 |
| --- | --- | --- | --- |
| 学习率 / batch / 累积步数 / 种子 | 3e-5 / 8 / 2 / 42 | `outputs/logs/train_bio.log` | `learning_rate`、`batch_size`、`gradient_accumulation_steps`、`seed` |
| 显存回退 | 无 | `outputs/logs/train_bio.log` | `oom_fallback=none` |
| 早停 | 5 个 epoch 都刷新了检查点，未触发早停 | `outputs/metrics/bert_bio_history.json` | `best_epoch=5`，`stopped_early=false` |
| 主测试前向次数 | 1，使用第 5 个 epoch | `outputs/logs/train_bio.log` | `test_eval_count=1`，`test_checkpoint_epoch=5` |
| 实体级 P / R / F1 | 0.688144 / 0.589092 / 0.634777 | `outputs/metrics/bert_bio.csv` | `bert_bio_entity` |
| 实体级 TP / FP / FN | 2873 / 1302 / 2004 | `outputs/metrics/bert_bio.csv` | `bert_bio_entity` |
| Token 级 P / R / F1 | 0.795151 / 0.759651 / 0.776996 | `outputs/metrics/bert_bio.csv` | `bert_bio_token` |
| Token 级 TP / FP / FN | 16726 / 4309 / 5292 | `outputs/metrics/bert_bio.csv` | `bert_bio_token` |
| 各 epoch 的 train loss 与 tune 实体级 F1 | 见日志第 29–33 行 | `outputs/logs/train_bio.log` | `train_loss`、`tune_micro_f1`；完整浮点在 `bert_bio_history.json` 的 `epochs` |
| tune 第 5 个 epoch 的 TP+FN | 4980+3167=8147 | `outputs/metrics/bert_bio_history.json` | `epochs[4].tune_tp`、`tune_fn` |
| 压平舍弃数 | train 13702、tune 1458、主测试 839 | `outputs/metrics/bio_dropped_entities.json` | 各划分的 `total`；九类计数与计划表一致 |
| 主测试窗口 | 1000 条都是单窗 | `outputs/logs/train_bio.log` | `test_windows=1000`，`test_multi_window_documents=0` |
| WordPiece 跨度与金标字符边界不一致 | train 156、tune 34、主测试 12 | `outputs/logs/train_bio.log` | `token_span_mismatch` |
| 窗口内没有可标 token 的保留实体 | train 17、tune 7、主测试 1 | `outputs/logs/train_bio.log` | `no_token` |

词典最长匹配的数字没有重算，仍是 `outputs/metrics/dictionary_baseline.csv` 里 `dictionary_longest` 的 Micro-F1 0.392433。

## 清单

- [x] 复现命令已在新进程跑通
- [x] 关键数字都能在 outputs 中找到
- [x] 划分与种子未被事后改动
- [x] 本步没有读取禁止使用的数据

主测试索引在训练前与 `data/processed/split_manifest.json` 的 `test.indices` 核对过。官方无标签 `CMeEE-V2_test.json` 和 `dev_full` 没有参与评分。权重在 `outputs/checkpoints/bert_bio/`，不放入提交 zip。

## 退回项

无。

## 留给人的结论

这里只列需要人写成文字的判断，不代写结论。

- 主测试实体级 Micro-F1 是否与词典最长匹配放在同一张表里比较。
- Token 级 Micro-F1 和严格实体级 Micro-F1 的差异如何解释。
- 主测试压平舍弃 839 个实体（其中 `bod` 657 个），以及 12 个实体的 WordPiece 跨度与金标字符边界不一致，是否写入 BIO 限制的说明。
- 本步没有目标 F1。优劣、错误归因和报告措辞由人定稿。
