# 03 评审：Qwen2.5-3B 零样本与三样本

**结论：** 已确认  
**日期：** 2026-10-15  
**确认日期：** 2026-10-15  
**计划：** [03_计划_Qwen2.5-3B.md](03_计划_Qwen2.5-3B.md)  
**数据结果：** [03_数据结果.md](03_数据结果.md)

## 实际跑过的命令

```text
D:/programdata/anaconda_envs/envs/medical_ner/python.exe scripts/run_qwen_3b.py
D:/programdata/anaconda_envs/envs/medical_ner/python.exe scripts/recompute_qwen_3b_metrics.py
```

推理进程退出码为 0，日志有 `RUN_OK` 之前写入的两行 `RESULT`。第二条在新进程中只读取两个预测文件、`outputs/metrics/qwen_3b.csv` 和 `outputs/metrics/qwen_3b_cost.json`，输出 `RECOMPUTE_OK`。两种方法的实体级 `TP+FN` 都是 4877。

执行前环境里没有 `bitsandbytes`。为按计划做 4bit NF4 加载，安装了 `bitsandbytes` 0.50.2。加载成功，没有改精度，也没有缩小测试集。

## 数字出处

| 数字 | 值 | 文件 | 字段或行 |
| --- | --- | --- | --- |
| 模型路径 | `D:/models/Qwen2.5-3B-Instruct` | `outputs/logs/qwen_3b.log` | `model_path` |
| 量化 | 4bit NF4，double quant，float16 | `outputs/logs/qwen_3b.log` | `quantization=4bit_nf4 double_quant=true compute_dtype=float16` |
| 生成参数 | temperature 0，do_sample false，max_new_tokens 1536，repetition_penalty 1.0，batch 1，seed 42 | `outputs/logs/qwen_3b.log` | 同名各行 |
| 三样本编号 | 3949, 5340, 1237，只来自 train | `outputs/logs/qwen_3b.log` | `example_source_index`，`example_split=train` |
| 每种方法生成次数 | 各从 0 条已完成开始，主测试 1000 条 | `outputs/logs/qwen_3b.log` | `already_done=0`，`test_documents=1000` |
| 零样本 P / R / F1 | 0.022067 / 0.007792 / 0.011517 | `outputs/metrics/qwen_3b.csv` | `qwen_3b_zero` |
| 零样本 TP / FP / FN | 38 / 1684 / 4839 | `outputs/metrics/qwen_3b.csv` | `qwen_3b_zero` |
| 三样本 P / R / F1 | 0.027155 / 0.017634 / 0.021382 | `outputs/metrics/qwen_3b.csv` | `qwen_3b_three` |
| 三样本 TP / FP / FN | 86 / 3081 / 4791 | `outputs/metrics/qwen_3b.csv` | `qwen_3b_three` |
| 零样本平均延迟、输入 token、输出 token | 6.418967 / 228.386 / 71.522 | `outputs/metrics/qwen_3b_cost.json` | `qwen_3b_zero` |
| 三样本平均延迟、输入 token、输出 token | 10.537989 / 629.386 / 124.49 | `outputs/metrics/qwen_3b_cost.json` | `qwen_3b_three` |
| 解析失败 | 零样本 32，三样本 25 | `outputs/metrics/qwen_3b_cost.json` | `parse_failures` |
| 无效实体 | 零样本 164，三样本 473 | `outputs/metrics/qwen_3b_cost.json` | `invalid_entities` |
| 表面不一致 | 零样本 1668，三样本 3058 | `outputs/metrics/qwen_3b_cost.json` | `surface_mismatches` |
| API 单价 | 不记录 | `outputs/logs/qwen_3b.log` | `api_unit_price=none` |

## 清单

- [x] 复现命令已在新进程跑通
- [x] 关键数字都能在 outputs 中找到
- [x] 划分与种子未被事后改动
- [x] 本步没有读取禁止使用的数据

主测试索引在推理前与 `data/processed/split_manifest.json` 的 `test.indices` 核对过。示例不在 tune 或主测试中。官方无标签 `CMeEE-V2_test.json` 和 `dev_full` 没有参与评分。预测在 `outputs/predictions/`，不放入提交 zip。

## 退回项

无。

## 留给人的结论

这里只列需要人写成文字的判断，不代写结论。

- 零样本和三样本是否与词典最长匹配、BERT-BIO 放在同一张主测试表里。
- 表面不一致（下标合法，但模型给出的表面与切片不同，仍按三元组计分）和解析失败是否写入说明。
- 本步没有目标 F1，也没有在看到分数后改提示、示例或 `max_new_tokens`。优劣、错误归因和报告措辞由人定稿。
