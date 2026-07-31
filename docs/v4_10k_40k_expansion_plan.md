# QuRating V4 数据扩量方案：10k Questions / 40k Pairs

## 1. 目标

本轮目标是在保持 QuRating 成对比较范式不变的前提下，将训练数据从 V3 的 2,000 道题、
7,509 条有效 pair，扩展到 10,000 道题、40,000 条候选 pair。扩量重点不是单纯增加边数，
而是提高题目覆盖、比较图质量以及 teacher soft label 的全局一致性。

```yaml
data_scope:
  source_teacher_records: 58977
  normalized_distinct_questions: 58962
  selected_training_questions: 10000
  candidate_training_pairs: 40000
  expected_mean_degree: 8

primary_objectives:
  - 覆盖送分题到压轴题的完整难度范围
  - 覆盖11个辅助特征的主要类别和稀有类别
  - 构建单连通、度数受控的全局比较图
  - 通过成对比较获得稳定的soft preference监督
  - 在正式训练前验证pair数据是否支持一维全局难度排序
```

### 1.1 数据使用边界

原始题库字段 `difficulty` 及其派生字段 `raw_difficulty` 已确认不可靠，不得用于选题、
构图、pair 标注或训练。允许用于私有选题的难度字段仅为 Prompt 与 V7 后处理生成的最终
教师档位：

```yaml
sampling_difficulty:
  field: teacher_difficulty_level
  source: difficulty_rating.difficulty_level
  meaning: Prompt + V7后处理后的最终教师档位
  usage: 仅用于10k题目的粗粒度分层抽样

forbidden_fields:
  - difficulty
  - raw_difficulty
  - teacher_difficulty_id
```

最终教师档位和 Aux11 特征只存在于私有选题数据中，不写入交给 Qwen3-32B 的候选 pair。
Qwen3-32B 只能看到题干、选项、解析和小题结构，不能看到已有难度结论或辅助标签。

图片不作为模型输入。题目统一使用文本字段和解析字段；图片 URL、图片标记仅用于数据切片
和风险诊断。

### 1.2 范围与非目标

本方案负责回答三个问题：从扩充后的教师数据中选择哪些题、这些题之间建立哪些比较关系、
以及如何证明最终 pair 数据足以支撑稳定的相对难度学习。以下事项不属于本方案范围：

- 不重新定义 V3 已确定的学生模型主体结构和 Bradley–Terry 训练目标；
- 不在本轮用教师五档直接训练难度分类头；
- 不把 10,000 题的抽样分布解释为真实线上题库分布；
- 不在本文规定训练、评测或部署命令，执行命令由独立运行手册维护；
- 不在数据构建阶段确定最终“送分题—压轴题”的线上分档阈值。

### 1.3 设计原则

- **标签与采样解耦**：教师五档用于控制选题范围，pairwise soft target 才是难度主监督。
- **局部可辨、全局可比**：既保留相近题之间的细粒度边，也保留跨区域边和桥接边。
- **覆盖优先于机械均衡**：保证难度两端与稀有特征可见，同时尽量保留源池主体分布。
- **公开数据最小化**：教师档位和具体辅助标签只进入私有审计文件，不进入 pair teacher 输入。
- **先验证、后投入**：图结构未通过 Pre-label Gate 时不启动大模型标注。
- **全过程可复现**：所有输入快照、随机种子、配置、模型版本、提示词版本和输出哈希均需登记。

## 2. 总体流程

```mermaid
flowchart TD
    A["58,977条教师结果"] --> B["字段规范化、文本去重与ID校验"]
    B --> C["按题目组划分Train / Validation / Test"]
    C --> D["排除旧V3训练题和所有验证测试题"]
    D --> E["按最终教师五档分配10k选题配额"]
    E --> F["档内Aux11覆盖与分布平衡"]
    F --> G["10,000个训练节点"]
    G --> H["构建40,000条feature-aware候选边"]
    H --> I["打标前数据验证"]
    I --> J{"Pre-label Gate"}
    J -- "不通过" --> E
    J -- "通过" --> K["Qwen3-32B级联成对标注"]
    K --> L["正反序聚合、soft target与可靠性加权"]
    L --> M["隔离异常pair并生成训练数据"]
    M --> N["打标后离线Bradley-Terry验证"]
    N --> O{"Post-label Gate"}
    O -- "不通过" --> P["复判、补边或重新标注"]
    O -- "通过" --> Q["冻结V4训练数据版本"]
```

流程包含两次性质不同的数据验证：

- **打标前验证**：检查选出的题目和候选图是否覆盖合理、结构健康，判断是否值得投入
  Qwen3-32B 标注成本。
- **打标后验证**：检查 teacher soft labels 是否自洽，能否在同一尺度上恢复稳定的一维
  难度排序。

## 3. 数据构建策略

### 3.1 选择 10,000 道题

#### 五档配额

从去重、隔离后的 Train Pool 中统计最终教师五档分布。每个档位先分配
`min(1,000, 该档可用题数)`，剩余名额按照各档在源 Train Pool 中的题目数量比例分配。
若某档容量不足，未使用名额自动分配给仍有容量的档位。

```yaml
teacher_level_quota:
  levels:
    - 送分题
    - 基础题
    - 中等题
    - 拔高题
    - 压轴题
  minimum_per_level_when_available: 1000
  remaining_quota: proportional_to_source_distribution
  selected_total: 10000
```

该策略同时保留难度两端和源数据的主体分布。它不要求每档固定 2,000 题，也不把最终
10,000 题解释为线上自然难度分布。业务五档校准仍需使用独立、冻结且代表真实业务分布的
参考池。

旧 V1 BT-only checkpoint 不再用于全池选题。它只可在 10,000 题选定后作为可选诊断，
用于检查旧 pairwise 排序与教师五档是否存在明显冲突，不得参与选题、生成 pair 标签或
作为数据门禁。

#### Aux11 覆盖

在每个教师档位内部，按 11 个辅助特征控制边际覆盖：

```yaml
auxiliary_category_floor:
  global_per_category: 20_when_available
  per_teacher_level_per_category: 3_when_available

remaining_slots_after_floors:
  marginal_distribution_matching: 0.80
  rare_category_protection: 0.10
  deterministic_random_exploration: 0.10
```

- **类别下限**：防止稀有题型或过程特征在抽样后消失。
- **边际分布匹配**：使各档入选题的单特征分布尽量接近对应源池。
- **稀有类别保护**：提高低频类别的有效样本量。
- **随机探索**：避免选题完全受教师档位和既有特征体系约束。

不对“五档 × 11 个特征”构造联合笛卡尔积分层。联合组合高度稀疏，会产生大量空单元和
极小样本单元。选题采用确定性 set-cover 和固定随机种子，同一数据快照可重复得到相同结果。

#### 数据隔离

选题前必须排除：

- V3 已使用的 2,000 个训练节点；
- V3 与 V4 validation/test 中的全部题目；
- 与上述集合规范化文本重复的题目；
- 缺失最终教师档位或任一 Aux11 字段的题目；
- 语义为空、存在明确标签泄漏或 ID 冲突的题目。

选题产物分为两层：

- **公开候选题文件**：仅包含稳定 ID、规范化文本、split 和必要诊断字段；用于后续构图。
- **私有审计文件**：额外保存最终教师档位、Aux11、选中原因和来源信息；不得传给 pair
  teacher。

#### Aux11 数据契约

V4 选题与构图使用当前代码冻结的 11 个单标签辅助特征。它们不是难度真值，也不参与
teacher 的 pair 判断，仅用于保证题目类型和物理过程覆盖。

| 特征 | 当前候选类别 |
|---|---|
| `problem_structure` | 概念判断、直接计算、实验探究、图像表格分析、电路综合、力学综合、热学综合、光学声学综合、跨模块综合 |
| `step_count` | 1-2步、3-5步、6-8步、9步以上 |
| `calculation_complexity` | 口算或直接判断、简单笔算、多公式联立、复杂方程或范围计算 |
| `reasoning_chain` | 直接套用、简单因果推理、多层因果推理、逆向推理或临界分析 |
| `knowledge_count` | 1个、2-3个、4个及以上 |
| `subquestion_dependency` | 无多问、多问但相互独立、多问且层层递进 |
| `state_count` | 单状态、双状态、多状态、连续变化或临界状态 |
| `constraint_count` | 无约束、单一约束、多约束 |
| `variable_relation` | 无变量关系、简单正反比、图像函数关系、多变量耦合关系 |
| `graph_table_requirement` | 无、直接读数、多组比较归纳、图像反推或外推 |
| `experiment_requirement` | 无、基础操作或读数、控制变量或故障分析、方案设计或误差评价 |

V3 的 `information_processing` 是图表要求与实验要求的合并字段；V4 已将两者拆回独立特征，
因此由 10 个辅助头变为 11 个辅助头。当前 `step_count` 仍是四档，旧数据中的“9-12步”
和“12步以上”统一映射为“9步以上”。如果后续决定恢复五档，必须新建特征 schema 版本、
完成高步数样本独立复核，并重新生成对应的选题审计和辅助训练数据，不能在同一版本内静默修改。

### 3.2 构建 40,000 条候选 pair

10,000 道题作为图节点，40,000 条 pair 作为无向比较边。目标图约束为：

```yaml
pair_graph:
  nodes: 10000
  edges: 40000
  node_coverage: 1.0
  connected_components: 1
  expected_mean_degree: 8
  minimum_degree: 4
  maximum_degree: 16
  self_loops: 0
  duplicate_edges: 0
```

候选边由七类策略混合生成：

```yaml
pair_source_weights:
  feature_near: 0.30
  feature_contrast: 0.25
  random_global: 0.15
  lexical_near: 0.10
  structure_matched: 0.10
  graph_bridge: 0.05
  low_degree_repair: 0.05
```

- `feature_near`、`lexical_near` 和 `structure_matched` 提供局部、细粒度比较。
- `feature_contrast` 和 `random_global` 提供跨题型、跨难度区域的全局尺度约束。
- `graph_bridge` 合并局部连通分量。
- `low_degree_repair` 保证所有节点获得最低比较次数。

构边时可读取 Aux11，但只允许在候选 metadata 中保留特征距离、匹配数量等聚合统计，不能
保留具体特征标签或教师难度档位。A/B 方向使用稳定随机规则平衡，避免候选顺序形成位置先验。

### 3.3 Teacher 成对标注

Teacher 继续使用 Qwen3-32B，并沿用已验证的级联策略：

1. 对 `(A, B)` 和 `(B, A)` 分别进行多次 nonthinking 采样；
2. 位置稳定且判断明确的 pair 直接接受；
3. 位置敏感或接近 0.5 的 pair 升级到 thinking_1024；
4. 使用正反序有效票数聚合得到偏好概率；
5. 通过 Jeffreys 平滑生成 `soft_target = P(A>B)`；
6. 根据位置偏差和投票稳定性生成 `sample_weight`；
7. 无法稳定解析、严重位置冲突或证据不足的 pair 进入 quarantine。

最终训练数据的难度监督来自成对偏好概率，而不是选题阶段使用的五档标签：

$$
P(A>B)=\sigma(s_A-s_B)
$$

因此，使用最终教师五档进行抽样不会把学生模型重新变成绝对五分类模型；五档只决定哪些
题目进入比较图，Bradley–Terry soft preference 才是训练主监督。

## 4. 数据验证

### 4.1 输入数据验证

在选题前验证数据快照和字段契约：

```yaml
input_gate:
  source_manifest_and_sha256_registered: true
  stable_question_id_unique: true
  normalized_text_duplicate_removed: true
  train_validation_test_group_isolation: true
  missing_teacher_level: 0
  missing_aux11: 0
  invalid_aux11_value: 0
  raw_difficulty_used: false
  image_uploaded: false
```

同时记录五档分布、Aux11 各类别分布、文本长度、是否含解析、多小题比例和图片依赖风险，
作为选题前基线。

### 4.2 10k 选题验证

```yaml
question_selection_gate:
  selected_questions: 10000
  duplicate_question_ids: 0
  question_or_normalized_text_overlap_with_exclusions: 0
  selected_counts_equal_frozen_level_quotas: true
  minimum_per_level: 1000_when_available
  zero_covered_aux11_source_categories: 0
  minimum_category_count_global: 20_when_available
  minimum_category_count_per_level: 3_when_available
  maximum_marginal_jensen_shannon_divergence: <=_0.05
  forbidden_fields_in_public_output: 0
```

报告必须同时给出源池与入选集的五档分布、Aux11 边际分布以及选中原因分布。普通 Accuracy
式的“覆盖率”不足以描述抽样质量，必须报告每个类别的绝对数量和边际 JSD。

### 4.3 候选图打标前验证

Pre-label Gate 不使用新 teacher 结果，只检查候选数据本身：

```yaml
prelabel_pair_gate:
  unique_pair_ids: 40000
  unique_undirected_edges: 40000
  self_loops: 0
  unknown_endpoints: 0
  node_coverage: 1.0
  connected_components: 1
  minimum_degree: >=_4
  maximum_degree: <=_16
  mean_degree: approximately_8
  A_B_orientation_balanced: true
  all_pair_source_types_present: true
  auxiliary_endpoint_coverage_pass: true
```

除总体度数外，还需按 pair source、教师档位和 Aux11 类别报告：

- 节点数量与端点出现次数；
- 度数的最小值、中位数、均值、P90 和最大值；
- 五档 pair 组合矩阵；
- 特征 Hamming 距离分布；
- 局部边、全局边和桥接边占比。

这里可以使用私有教师档位做结构审计，但档位不能写入正式候选文件。旧 BT 分差分析仅为
可选诊断，不属于本轮硬门禁。

### 4.4 Teacher 标注过程验证

标注过程中持续记录：

```yaml
teacher_monitoring:
  completed_pairs: REQUIRED
  valid_vote_parse_rate: REQUIRED
  direct_acceptance_rate: REQUIRED
  thinking_escalation_rate: REQUIRED
  position_sensitive_rate: REQUIRED
  quarantine_rate_and_reasons: REQUIRED
  label_source_counts: REQUIRED
```

不能使用 V3 的接受率推算 V4 最终训练 pair 数。新题池与新构图策略会改变比较难度，最终
有效记录数以冻结的 final manifest 为准。

### 4.5 打标后离线 Bradley–Terry 验证

对最终接受的 pair 拟合一个不读取文本、不读取 Aux11 的经典标量 Bradley–Terry 模型。
该模型只为每个 `question_id` 学习一个自由标量，用于检验 pair 数据本身能否支持一致的
全局排序。

五折交叉验证必须保留连通骨架，再划分冗余边，保证每折训练图包含全部节点且保持单连通。
bootstrap 同样保留生成树，只对冗余边做有放回采样；断连的 bootstrap 结果无效。

```yaml
postlabel_bt_gate:
  graph_connected: true
  heldout_weighted_log_loss_beats_constant_baseline: true
  heldout_unweighted_log_loss_beats_constant_baseline: true
  heldout_weighted_brier_beats_constant_baseline: true
  heldout_unweighted_brier_beats_constant_baseline: true
  severe_residual:
    absolute_error_threshold: 0.5
    maximum_rate: 0.05
  connectivity_preserving_bootstrap:
    runs: 20
    minimum_mean_spearman: 0.90
```

必须报告：

- 加权与非加权 Soft Pairwise Log Loss、Brier Score；
- Pairwise Accuracy、AUC 和 Decisive Accuracy；
- 每道题 BT score 的近似标准误与 95% 区间；
- bootstrap 排名 Spearman，以及最难/最易 10% 集合重叠率；
- teacher soft target 与离线 BT 概率残差最大的 pair；
- 按 pair source、教师档位组合和可靠性类型划分的错误切片。

如果 Post-label Gate 不通过，优先检查图桥接和低度节点，再对高残差 pair 进行
thinking_1024 复判或人工审核；必要时补充针对性边并重新运行审计。不得通过直接删除困难
pair 或降低门槛来凑够训练数量。

离线 BT 审计只能证明数据支持稳定的一维相对排序，不能证明 teacher 符合真实教研标准。
最终业务效果仍需在题目不重叠的独立 validation 和冻结 test/OOT 上验证。

## 5. 验收与决策规则

V4 数据只有在以下四层结果同时成立时才可冻结：

1. **输入合格**：字段契约、去重、ID 和数据隔离全部通过；
2. **选题合格**：五档配额、Aux11 类别下限和边际分布偏差达到预设标准；
3. **候选图合格**：节点全覆盖、单连通、度数受控且各类边达到合理占比；
4. **标注数据合格**：teacher 解析与稳定性指标合格，离线 BT 的泛化误差和排序稳定性通过门禁。

门禁失败时按失败阶段处理：

| 失败阶段 | 处理原则 |
|---|---|
| 输入或选题失败 | 修正字段、排除集或配额后重新选题，不进入构图和标注 |
| Pre-label Gate 失败 | 调整构边权重、桥接边或低度修复边，然后重新审计 |
| Teacher 过程失败 | 检查提示词、解析器、采样配置与推理服务，不直接接受不完整结果 |
| Post-label Gate 失败 | 优先复判高残差边、补充低度节点和薄弱区域边，再重新拟合 BT |
| 独立验证失败 | 数据版本不得作为正式升级版本；分析 teacher 偏差、领域切片与模型容量 |

不允许为了通过门禁而事后删除大量模糊 pair、只保留容易比较的题目，或把验证集反馈用于
反复修改测试集。模糊 pair 本身包含难度接近信息，应通过 soft target 和可靠性权重表达。

## 6. 交付物与版本治理

每次正式数据构建至少冻结以下产物：

- 原始教师数据快照的 manifest、记录数和 SHA-256；
- 规范化 Train Pool 及其去重、隔离报告；
- 10k 公开题目文件、私有选题审计文件和选题 manifest；
- 40k 候选 pair、构图 manifest 和 Pre-label 验证报告；
- nonthinking 与 thinking_1024 原始投票、解析统计和路由报告；
- 最终训练 pair、quarantine、final manifest 与 Post-label BT 审计报告；
- 独立 validation/test 的数据版本与重叠检查报告；
- 数据卡：记录 schema、教师模型、提示词、后处理、随机种子及所有输入输出哈希。

冻结后的数据版本不可原地覆盖。任何会改变题目集合、pair 边、soft target、sample weight、
Aux11 schema 或隔离集合的变更，都必须产生新的版本号，并重新执行受影响阶段之后的全部门禁。

## 7. 主要风险与控制措施

| 风险 | 可能影响 | 控制措施 |
|---|---|---|
| 最终教师五档存在系统性偏差 | 10k 选题覆盖偏移 | 仅将其用于粗分层；报告源池与入选集分布；最终效果使用独立集验证 |
| 五档与 Aux11 来自同一教师链路 | 抽样误差具有相关性 | 保留随机探索份额；对稀有类别和冲突样本做独立抽查 |
| 源数据存在历史分层采样偏差 | 10k 分布不代表线上分布 | 明确其为训练覆盖集；分档校准使用独立业务参考池 |
| 近邻边过多 | 图局部充分但全局尺度漂移 | 配置跨区域边、随机边和桥接边，并检查单连通和谱/度数诊断 |
| 随机远距离边过多 | 比较过易、有效信息量低 | 控制全局随机边比例，保留 feature-near 和结构近邻边 |
| 高步数类别被合并 | 难以区分复杂题内部层级 | 当前版本如实记录四档；通过专项复核决定是否在新 schema 恢复五档 |
| 不上传图片 | 强图依赖题信息不足 | 保留图片依赖风险切片；高风险题进入专项评估或隔离，而非默认等同普通文本题 |
| pair teacher 位置偏差 | soft target 方向性失真 | 强制正反序采样、位置偏差监控和 thinking 升级 |
| 图中少量异常边 | BT 排序局部扭曲 | 使用可靠性权重、残差审计、复判与针对性补边，不盲目全量删除 |

## 8. 最终成功标准

本轮扩量成功不以“生成了 40,000 条 pair”作为唯一标准，而以以下结果共同判断：

- 10,000 道题在五档和 Aux11 上达到预定覆盖，且与验证测试集合严格隔离；
- 40,000 条候选边形成全覆盖、单连通、度数健康且兼顾局部与全局比较的图；
- teacher 标注具有较高解析成功率、可控的位置敏感率和可解释的 quarantine；
- 离线 BT 在留出边上优于常数基线，残差率与 bootstrap 排名稳定性通过门禁；
- 新数据训练得到的模型在独立 validation/test 上稳定优于 V3 基线，而不是只降低训练损失；
- 全流程可由冻结的数据快照、配置、manifest 和哈希完整复现。
