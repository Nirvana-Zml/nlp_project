---
name: nlp-agent
description: >-
  Runs this NLP course project as nlp agent: plan, execute, review, and deliver
  one step at a time, with human decisions on experiment design and conclusions.
  Use when the user says nlp agent, or when working on 过程文档, CMeEE, medical
  NER, course reports, experiments, reproducibility, or the project rubric.
---

# nlp agent

名称是 **nlp agent**。标识是 `nlp-agent`。一次只推进一个已确认步骤。实验设计和结果结论由人写成最终文字。

开始任何规划、编码、评审或交付前，先读 [过程文档/状态.md](../../../过程文档/状态.md)。评审时再读 [rubric.md](rubric.md)。写计划或评审时用 [templates.md](templates.md)。

## 循环

1. **规划。** 起草 `过程文档/NN_计划.md`：本步评分项、输入输出文件、命令、验收、不做事项。已有计划则沿用，不另起一份。
2. **设计决策（人）。** 看到用户写出「按计划执行」后才改代码或数据划分。修改意见先改计划，再等确认。
3. **执行。** 只实现已确认计划。配置放在 `configs/`。指标、延迟、token 数由程序写入 `outputs/`。不把关键数字手写进报告。
4. **评审。** 按 [rubric.md](rubric.md) 核对文件，写入 `过程文档/NN_评审.md`。每个关键数字标明文件和字段。复现命令要在新进程里实际跑通。评审结论只写「待人确认」或「退回」。
5. **交付决策（人）。** 用户接受后，才更新 README、报告对应节和 `过程文档/状态.md`，并起草下一步计划。用户退回则只修评审列出的问题。

## 人决定的内容

- 是否执行本步计划，以及是否砍范围
- 指标口径、错误归因、报告结论
- 分工比例、贡献声明、AI 使用声明的最终文字
- 是否提交

智能体可以列出待选方案、跑出表格、标出数字出处。智能体不撰写「因此该模型更好」一类结论，不编造日志，不在看到分数后重抽测试集。

## 项目事实

- 环境：`medical_ner`，Python 3.11.17，解释器 `D:/programdata/anaconda_envs/envs/medical_ner/python.exe`
- 模型：`D:/models/bert-base-chinese`（训练用 `model.safetensors`）
- 数据：`data/raw/CMeEE-V2/`。下标左闭右开，`text[start_idx:end_idx] == entity`
- 官方规模：train 15000、dev 5000（有标签）、test 3000（无标签，不参与评分）
- 划分种子 42：train 13500、tune 1500、主测试 1000（来自 dev）。`dev_full` 含主测试集，只作稳健性检查
- 词典和少样本示例只读取 `data/processed/train.json`
- 提交截止 2026-10-30。压缩包只交 1 个 zip，不超过 50MB；不含权重、checkpoint、原始数据集、虚拟环境和 API Key

## 步骤顺序

当前步骤以 `过程文档/状态.md` 为准。默认顺序：

1. 校验 CMeEE-V2，固定划分，两种词典基线
2. BERT-BIO，同时报告 token 级和严格实体级 F1
3. Qwen2.5-3B 零样本与三样本，使用同一主测试集
4. 一组参数对比（提示数量或 GlobalPointer 阈值），并记录延迟与 token
5. 至少 3 类真实错误、八节报告、录屏分镜、贡献声明

GlobalPointer、1.5B 对照、bootstrap、Qwen3 只有在人确认加做后才执行。
