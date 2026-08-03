# V3 单题推理、双版本对照与五档阈值实验

## 1. 实验目的

本实验回答三个不同问题：

1. V1（BT-only）和 V3（BT + Aux10，辅助损失权重 0.03）对同一批新题给出的全局排序是否一致；
2. V3 的十个辅助头在单题推理时具体输出什么，其置信度和标签一致性如何；
3. 连续 BT 标量如何冻结为可部署的百分位分数和五档难度，以及 1,000 题是否足以稳定估计四个阈值。

V3 validation 由 500 道题组成 1,891 个 pair，适合评价 pairwise log loss、AUC 和 Aux10 同源标签表现，但不适合估计业务五档阈值。它既太小，又是实验验证集；把它用于阈值拟合会同时造成估计方差和模型选择泄漏。

## 2. Reference Pool

阈值 reference 应来自与未来线上推理一致、按业务自然流入或库存分布抽取的独立题目集合。若只能使用当前源题库，应明确写作“源题库分布”，不能声称是自然业务分布。

reference 的冻结规则：

- 不使用 `difficulty`、`raw_difficulty`、V3/V4 教师五档或 Aux 特征进行抽样；
- 使用现有稳定 `question_id`，按 `sha256(seed, question_group_id)` 固定顺序；
- 排除 V3 train/validation/test 中出现过的 ID 和规范化文本；
- 预先冻结 10,000 道题，其中前 1,000 道为 smoke 子集；
- 两个 checkpoint 必须读取同一文件，且保持相同题目顺序。

1,000 题只用于检查运行链路、分数是否塌缩、两模型排序趋势和 Aux 输出。正式阈值建议使用 10,000 题；至少使用 5,000 题，并报告 bootstrap 置信区间与样本量敏感性。

## 3. 两版模型对照

固定对照 checkpoint：

- V1 BT-only：`v1_bt_only/checkpoint-epoch-3-step-1176`；
- V3 BT + Aux10：`v3_bt_aux10_w003/checkpoint-epoch-3`。

两版各自在一张 GPU 上同时推理。比较指标包括：

- 原始标量 Pearson：比较数值线性关系，但原始 BT 标量存在平移和尺度自由度，不能单独解释；
- Spearman：主排序一致性指标；
- top/bottom 10% overlap：检查最难、最易题集合是否稳定；
- 各自按相同目标分布拟合阈值后的五档一致率与迁移矩阵；
- 分数最小值、最大值、均值和标准差，用于发现标量塌缩。

阈值必须绑定 checkpoint fingerprint。V1 的 raw score 阈值不能直接用于 V3，反之亦然。

## 4. V3 Aux10 单题输出

V3 推理一次 backbone 后，同时输出一个连续难度标量和十个辅助分类头：

- `problem_structure`
- `step_count`
- `calculation_complexity`
- `reasoning_chain`
- `knowledge_count`
- `subquestion_dependency`
- `state_count`
- `constraint_count`
- `variable_relation`
- `information_processing`

每个头保存预测标签、类别 ID、最大概率和归一化熵；需要排错时可额外保存完整类别概率。V1 没有辅助头，不能输出这些字段。

若在原 V3 frozen18 数据上评估，Aux10 指标属于同源 held-out accuracy。若用当前 V4 Prompt + 后处理生成的特征进行对照，则 `information_processing` 需要由 Frozen18 中的图表要求和实验要求重新合并，`step_count` 也必须映射到 checkpoint 自带的旧类别表；此时指标只能叫“跨 pipeline 一致率”，不能当成人工真值准确率。

## 5. 五档阈值与稳定性

默认目标分布暂定：

```yaml
送分题: 0.20
基础题: 0.20
中等题: 0.30
拔高题: 0.20
压轴题: 0.10
```

四个 raw score 阈值分别是 reference 分数的 `q20 / q40 / q70 / q90`。同时保存完整排序分数，推理时得到：

- `raw_difficulty_score`：checkpoint 原始 BT 标量，无固定 0–1 范围；
- `difficulty_percentile`：相对冻结 reference 的经验 CDF，范围 0–1；
- `difficulty_score`：百分位乘 100，范围 0–100；
- `difficulty_level`：由四个冻结阈值映射出的五档。

阈值稳定性报告应包含：

- 四个阈值的 bootstrap 95% 区间；
- 1k、2k、5k、10k 子样本阈值偏差；
- 子样本阈值相对全量阈值造成的五档迁移率；
- reference 数据版本、哈希、checkpoint fingerprint 和目标分布。

## 6. GPU 暂停原则

Qwen3-32B 教师打标的每个 worker 使用两张卡，并把每一轮结果持续追加到 shard 的 `raw_votes.jsonl`。单独终止一个 worker 后，其已有有效 votes 不会丢失；后续以完全相同的模型、配置、候选文件和输出文件重启时，只补缺失票。

不要直接终止外层 `server_run_cascade_production.sh`，因为它的退出清理会同时终止两个 shard。只终止准备释放的那个 `run_local_pairwise_teacher.py` worker，并在 GPU 显存释放后运行 V1/V3 单卡推理。恢复打标前先停止两项推理，再以原参数恢复对应 shard；不得更换配置或输出路径，否则配置哈希保护会拒绝续跑。

## 7. 判定标准

本轮不只看“是否能输出分数”。满足以下条件后才进入正式阈值冻结：

- 两版均完成同一 10k reference 推理，无缺失 ID、重复 ID或文本哈希不一致；
- score 标准差非零，四个阈值严格递增；
- 5k 到 10k 的阈值迁移率和 bootstrap 区间已达到可接受水平；
- V3 Aux10 未出现单一类别异常塌缩，且已区分同源 accuracy 与跨 pipeline agreement；
- 最终选择 V1 或 V3 checkpoint 后，只为被选版本发布 calibration 文件。
