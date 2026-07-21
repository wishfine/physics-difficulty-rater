# 物理题难度模型：数据构建与评估说明

本文说明 `physics-difficulty-rater` 当前版本如何从原始初中物理题构造训练数据、教师标签数据、审计数据和外部 gold 评估集。本文的目标是让后续维护者能够回答四个问题：

1. 一条训练样本来自哪里，文本是怎样拼接的？
2. 模型实际学习的是哪一个难度标签？
3. 哪些数据可用于调参，哪些数据只能用于最终报告？
4. 出现标签冲突时，应该如何处理而不是误改数据？

## 1. 数据流总览

```text
原始物理题 25,000 条
  ├─ 题干、选项、解析、小题、图片 URL
  ├─ 历史题库难度/采样桶（仅供审计）
  │
  └─ 冻结教师链路
       API Prompt -> 18 维特征 -> V7 后处理 -> 五档教师标签
          │
          ├─ 合同校验、去重、隔离异常样本
          ├─ 按题目组分层切分 train / validation / teacher-test
          └─ OOF 文本审计（只标记，不自动改标签）
                    │
                    └─ Qwen3.5-4B + LoRA 训练

独立 GPT-5.6 逐题复核集 1,066 条
  ├─ 修订后主标签
  ├─ 可接受等级（允许的相邻难度档）
  └─ 外部 gold 评估集（先做训练集 ID 防泄漏检查）
```

## 2. 难度档位与标签原则

统一五档从低到高编码为：

```yaml
0: 送分题
1: 基础题
2: 中等题
3: 拔高题
4: 压轴题
```

### 2.1 训练主标签

训练主标签是冻结教师链路的最终结果，不是原始题库难度字段。

```yaml
primary_label:
  source: difficulty_rating.difficulty_level
  generation:
    - 冻结的初中物理难度打标 Prompt 调用 API
    - 提取冻结的 18 维教师特征
    - 执行 V7 后处理规则
  prepared_fields:
    - teacher_difficulty_level
    - teacher_difficulty_id
  usage:
    - 训练 Qwen 难度主分类头
    - teacher-validation 与 teacher-test 评估
```

它应被准确称为“教师标签”或“API+V7 标签”。它是当前大规模监督来源，不等同于人工真值。

### 2.2 原始题库标签

原始数据中的历史难度字段、采样桶或 `difficulty_distribution` 不保证正确，也可能缺失。处理后如存在，会保留为 `raw_difficulty`；部分样本的该字段为 `null`。

```yaml
raw_difficulty:
  allowed_usage:
    - 原始分层抽样
    - 教师标签冲突发现
    - 人工复核排序
  forbidden_usage:
    - 直接作为训练主标签
    - 自动覆盖 API+V7 教师标签
```

## 3. 原始题目输入与文本规范

原始 25,000 条文件：

```yaml
file: data/raw/physics_sampled_5000_per_difficulty_v2.jsonl
records: 25000
content:
  - 题干
  - 选项
  - 官方解析
  - 小题及其题干、选项、解析
  - 题目图片 URL、解析图片 URL
```

训练和推理使用相同的标准化文本拼接。字段缺失时省略该段；小题按 `question_id` 排序。

```text
【题干】
{题干}

【选项】
{选项}

【解析】
{解析}

【小题】

  小题1:

    题干: {小题题干}

    选项: {小题选项}

    解析: {小题解析}
```

实现位置：`src/physics_difficulty/data/formatting.py`。

模型输入长度默认上限为 1,024 token。超长题由 `truncation.py` 做分段截断：优先保留题干、选项、小题标题和必需题目内容，再分配解析文本预算。模型是纯文本模型，不下载、不识别图片内容；图片 URL 和“如图”提示只作为质量诊断信号。

## 4. 冻结教师特征

API Prompt 与 V7 后处理已经冻结，因此每条教师数据必须完整保留原始 18 维，而不是改写教师体系。

```yaml
teacher_features_legacy18:
  - step_count
  - formula_count
  - calculation_complexity
  - reasoning_chain
  - problem_structure
  - additional_structure
  - information_carrier
  - reality_question
  - subquestion_dependency
  - knowledge_count
  - knowledge_diff
  - cross_module
  - state_count
  - constraint_count
  - variable_relation
  - experiment_requirement
  - graph_table_requirement
  - error_risk
```

本地 Qwen 模型不直接训练全部 18 维，而是从中稳定派生 10 个单标签辅助任务。这样既不破坏冻结教师链路，又避免将稀疏或语义不稳定字段强行作为多任务目标。

```yaml
teacher_features:
  problem_structure: "9 分类：概念判断、直接计算、实验探究、图像表格分析、电路综合、力学综合、热学综合、光学声学综合、跨模块综合"
  step_count: 4 分类
  calculation_complexity: 4 分类
  reasoning_chain: 4 分类
  knowledge_count: 3 分类
  subquestion_dependency: 3 分类
  state_count: 4 分类
  constraint_count: 3 分类
  variable_relation: 4 分类
  information_processing: "由 graph_table_requirement 与 experiment_requirement 合并"
```

`knowledge_domains` 仅作为元数据，不参与损失。完整 18 维保存于 `teacher_features_legacy18`，派生 10 维保存于 `teacher_features`，版本标记为 `v2_frozen18`。

## 5. 教师打标、校验与清洗

### 5.1 教师导出

教师打标程序在 `prompt_test` 项目中运行，输出到本项目：

```yaml
teacher_export:
  file: data/teacher_labeled/physics_teacher_frozen18_25000.jsonl
  records: 25000
  prompt_version: frozen_physics_prompt
  postprocess_version: v7
  teacher_model: current_physics_api
```

先执行：

```bash
python scripts/validate_teacher_labels.py \
  --input data/teacher_labeled/physics_teacher_frozen18_25000.jsonl
```

该步骤检查冻结 18 维与主标签合同；本批次结果为 `PASS`。

### 5.2 规范化、去重与隔离

执行：

```bash
python scripts/prepare_teacher_data.py \
  --input data/teacher_labeled/physics_teacher_frozen18_25000.jsonl \
  --output data/curated/physics_teacher_v2_frozen18.jsonl \
  --manifest data/curated/physics_teacher_v2_frozen18.manifest.json \
  --prompt-version frozen_physics_prompt \
  --postprocess-version v7 \
  --teacher-model current_physics_api \
  --seed 42
```

本次产物统计：

```yaml
input_records: 25000
exact_duplicate_removed: 4
quarantined_records: 0
curated_records: 24996

teacher_label_distribution:
  送分题: 3642
  基础题: 8338
  中等题: 8041
  拔高题: 3007
  压轴题: 1968
```

重复样本的冲突信息会写入 `.conflicts.jsonl`；它是审计资料，不会静默覆盖任一教师标签。

## 6. 训练、验证和 teacher-test 切分

通过 `scripts/split_teacher_data.py`，以题目组为单位进行固定随机种子（42）的分层切分。相同 `parent_id` 的小题不能跨 split，以避免同题泄漏。

```yaml
split_directory: data/curated/split_v2_frozen18
train:
  records: 19997
  labels:
    送分题: 2914
    基础题: 6670
    中等题: 6433
    拔高题: 2406
    压轴题: 1574
validation:
  records: 2500
  labels:
    送分题: 364
    基础题: 834
    中等题: 804
    拔高题: 301
    压轴题: 197
teacher_test:
  file: test.jsonl
  records: 2499
  labels:
    送分题: 364
    基础题: 834
    中等题: 804
    拔高题: 300
    压轴题: 197
```

使用原则：

```yaml
train.jsonl: 训练
validation.jsonl: "比较 checkpoint、选择模型、温度校准"
test.jsonl: "API+V7 教师链路上的最终留出集；不用于调参"
```

`test.jsonl` 不等于人工真值，只能衡量模型对教师链路的复现能力。

## 7. OOF 标签审计

OOF 审计文件：

```yaml
input: data/curated/split_v2_frozen18/train.jsonl
output: data/curated/train_with_oof.jsonl
method:
  splitter: StratifiedGroupKFold
  folds: 5
  grouping: "source_dataset_id + parent_id"
  text_feature: "字符 TF-IDF，2-4 gram，最多 120000 特征"
  classifier: "LogisticRegression(class_weight=balanced)"
```

每道题的 OOF 概率均由“没有见过该题”的折模型产生。审计结果附在 `oof_audit`：

```yaml
oof_audit:
  predicted_level_id: "轻量审计模型的预测"
  probabilities: "五档概率"
  top1_probability: "最高概率"
  top1_top2_margin: "第一、二候选间隔"
  entropy: "不确定性"
  distance_from_teacher_label: "与教师标签相差档数"
  rejudge_recommended: "高置信度且相差至少两档时为 true"
```

本批 19,997 条训练数据中，严格规则选出 13 条人工复核候选。OOF 不会自动改标签，也不作为训练监督；它仅用于寻找可能存在教师标签异常、Prompt/V7 边界问题或文本缺失的问题样本。

## 8. GPT-5.6 逐题复核 gold 集

外部复核 CSV：

```yaml
file: physics_adjudicated_labels_gpt56_rereview_1066.csv
source_label_column: 修订后主标签
acceptable_level_column: 可接受等级
confidence_column: 修订后置信度
records_in_csv: 1066
```

该集用于衡量业务难度判断质量，优先级高于 teacher-test。构建时：

1. `修订后主标签` 作为严格 gold 标签；
2. `可接受等级` 转为允许档位集合，用于补充的 `acceptable_level_accuracy`；
3. `修订后置信度` 保留为高/中质量切片；
4. 绝不读取其他 JSON 文件中不可信的难度字段；
5. 构建前必须与 `train.jsonl` 按题目 ID 求交集，发现重叠即失败；
6. 没有题干、选项、解析的纯图片题不能由当前文本模型公平评估，显式跳过并记录 ID。

构建命令：

```bash
python scripts/prepare_adjudicated_gold.py \
  --labels_csv data/gold/physics_adjudicated_labels_gpt56_rereview_1066.csv \
  --reference_train_file data/curated/split_v2_frozen18/train.jsonl \
  --output data/gold/physics_adjudicated_gold_1065.jsonl \
  --skip_unrenderable
```

当前 CSV 有 1 条纯图片且没有文本输入（ID `3659087300292263936`），因此当前文本模型的可评估 gold 集为 1,065 条。若后续接入视觉输入或从原始 JSON 补回可读文本，可重新纳入该题。

## 9. 模型选择与最终报告流程

```yaml
phase_1_training:
  train_file: train.jsonl
  checkpoint_frequency: "每 0.25 epoch"

phase_2_selection:
  eval_file: validation.jsonl
  objective: "优先 macro_f1、balanced_accuracy，结合 MAE 与混淆矩阵"
  forbidden: "使用 teacher-test 或 gold 选择 checkpoint"

phase_3_calibration:
  input: "选出的 checkpoint + validation.jsonl"
  output: calibration.json

phase_4_final_reporting:
  teacher_fidelity: test.jsonl
  business_gold: physics_adjudicated_gold_1065.jsonl
  required_metrics:
    - strict accuracy
    - acceptable_level_accuracy（仅 gold）
    - macro_f1
    - balanced_accuracy
    - mean_absolute_error
    - adjacent_accuracy
    - quadratic_weighted_kappa
    - 置信度/图片风险切片
```

对外解释时应区分两类结论：

```yaml
teacher_test_result: "模型对冻结 API+V7 教师链路的模仿程度"
gold_result: "模型对逐题复核难度真值的业务表现"
```

## 10. 版本、可追溯性与禁止事项

每次训练输出目录应保存 `training_config.json`、checkpoint、优化器状态、调度器状态、训练日志和评估 JSON。教师准备产物通过 manifest 保存 Prompt、后处理、教师模型和随机种子来源。

禁止事项：

```yaml
- 不要把 raw_difficulty 当作主训练标签。
- 不要让 OOF 预测自动覆盖教师或 gold 标签。
- 不要用 teacher-test 或 gold 选择 checkpoint、学习率或其他超参数。
- 不要将 gold 与 train 重叠后仍作为最终测试结果汇报。
- 不要把图片依赖高的文本模型结果与真正多模态能力混为一谈。
```
