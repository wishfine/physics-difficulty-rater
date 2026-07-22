# Physics Difficulty Rater 实验记录

```yaml
document_type: append_only_experiment_log
project: physics-difficulty-rater
current_route: QuRating_V3_pairwise
owner: zhangyonglin
timezone: Asia/Shanghai
created_at: 2026-07-22
last_updated_at: 2026-07-22
repository: git@github.com:wishfine/physics-difficulty-rater.git
status: active
```

## 1. 记录规则

本文件是项目实验事实的主索引，记录数据版本、代码提交、参数、运行环境、日志、指标、
异常、结论和下一步。大体积原始数据、模型权重、checkpoint、逐票输出和完整日志不进入
Git，只记录其服务器绝对路径、SHA256（如有）、关键摘要和获取方式。

更新时遵守以下约束：

- 不覆盖历史实验；修复后新增 attempt 或 run。
- 未从日志、manifest 或指标 JSON 中确认的值写 `UNKNOWN` 或 `PENDING`。
- `difficulty`、`raw_difficulty` 以及旧 API+V7 难度不能进入 V3 数据、标签或指标。
- 每个结论必须能追溯到代码 commit、配置文件和输出文件。
- validation 用于选模型和调参；test/OOT 只在方案冻结后使用，不能反复据此调参。
- 不把跨 teacher 模式一致性当作人工真值准确率。

## 2. 固定实验边界

```yaml
task:
  input:
    - 题干
    - 选项
    - 解析
    - 小题
  images_uploaded: false
  output:
    continuous_score: s(q)
    pair_probability: sigmoid(s_a - s_b)
    final_levels:
      - 送分题
      - 基础题
      - 中等题
      - 拔高题
      - 压轴题

teacher:
  model: Qwen3-32B
  model_path: /home/share_ssd_data/nfs-env/llm_models/Qwen/Qwen3-32B
  inference: vLLM_offline
  physical_gpus: [6, 7]
  tensor_parallel_size: 2

student:
  planned_model: Qwen3.5-4B
  tuning: LoRA
  head: shared_scalar_head
  primary_loss: soft_Bradley_Terry
  auxiliary_feature_heads: false

forbidden_v3_supervision:
  - difficulty
  - raw_difficulty
  - teacher_difficulty_id
  - teacher_difficulty_level
  - teacher_features
  - teacher_features_legacy18
```

## 3. 实验索引

### V3-DATA-001：原始 25k 题目准备

```yaml
date: 2026-07-22
status: PASS
source_file: data/physics_sampled_5000_per_difficulty_v2.jsonl
server_source_resolved: /data/zhangyonglin/physics-difficulty-runtime/rater-data/physics_sampled_5000_per_difficulty_v2.jsonl
source_sha256: 2a85c25f43408b5b3d38ad8322b57457e6559e1db005566e4a5a284078624dfb
script: scripts/prepare_raw_v3_questions.py
schema_version: v3_raw25k_preparation_v1
seed: 42
split_method: sha256(seed, question_group_id)
split_ratio: [0.8, 0.1, 0.1]
deduplication: sha256(NFKC(normalized_rendered_text))
raw_difficulty_used: false
images_uploaded: false

counts:
  source: 25000
  accepted: 24983
  train: 19988
  validation: 2468
  test: 2527
  quarantine: 17
  exact_or_normalized_duplicates: 6
  label_leakage: 10
  semantically_empty: 1

diagnostics:
  has_analysis: 24924
  has_subquestions: 2855
  has_image_metadata: 24983
  image_dependency_medium: 8131
  image_dependency_high: 16852
  length_short: 17991
  length_medium: 6579
  length_long: 413

manifest: /data/zhangyonglin/physics-difficulty-runtime/pairwise_v3/questions.manifest.json
```

结论：数据准备通过；历史 `difficulty` 未被使用。源文件曾按错误难度字段各抽 5,000 条，
因此只能作为 pairwise 题目池，不能代表自然业务难度分布。

### V3-GRAPH-SMOKE-001：无标签比较图 smoke

```yaml
date: 2026-07-22
status: PASS
questions: 100
pairs: 400
mean_degree: 8.0
min_degree: 6
max_degree: 10
connected_components: 1
node_coverage: 1.0

pair_sources:
  random_global: 111
  lexical_near: 131
  graph_bridge: 10
  structure_matched: 113
  low_degree_repair: 35

lexical_check:
  lexical_near_mean_jaccard: 0.01808
  random_global_mean_jaccard: 0.00995

candidates: /data/zhangyonglin/physics-difficulty-runtime/pairwise_v3/smoke/candidates.jsonl
```

结论：图连通、覆盖完整、度数受控；词面近邻平均 Jaccard 高于随机边。这里仅证明词面
召回有效，不宣称它代表深层语义相似。

### V3-TEACHER-ABLATION-001：reasoning 模式对照（20 pair）

```yaml
date: 2026-07-22
status: COMPLETED_THINKING_1024_SELECTED_FOR_NEXT_PILOT
pair_count: 20
output_root: /data/zhangyonglin/physics-difficulty-runtime/pairwise_v3/smoke/reasoning_ablation_20
code_commit_initial: 73345a1
code_commit_current: 8e0ea27

modes:
  nonthinking:
    enable_thinking: false
    max_new_tokens: 4
    temperature: 0.7
    top_p: 0.8
    top_k: 20
    min_p: 0.0
    prompt_batch_size: 8
  thinking_512:
    enable_thinking: true
    max_new_tokens: 512
    temperature: 0.6
    top_p: 0.95
    top_k: 20
    min_p: 0.0
    prompt_batch_size: 4
  thinking_1024:
    enable_thinking: true
    max_new_tokens: 1024
    temperature: 0.6
    top_p: 0.95
    top_k: 20
    min_p: 0.0
    prompt_batch_size: 4

common_engine_config:
  dtype: bfloat16
  tensor_parallel_size: 2
  gpu_memory_utilization: 0.82
  max_num_batched_tokens: 4096
  max_num_seqs: 32
  flashinfer_sampler: disabled
  attention_backend: FlashAttention_2

adaptive_votes_per_direction: [3, 5, 10]
```

#### Attempt 1：FlashInfer sampler JIT 失败

```yaml
started_at: 2026-07-22T14:39:33+08:00
status: FAILED_BEFORE_GENERATION
failure_stage: vLLM_warmup
model_weight_size_gib: 61.02
model_memory_per_gpu_gib: 30.59
error: FlashInfer_sampling_ninja_build_failed
root_cause: system_/usr/bin/nvcc_is_below_CUDA_12_but_FlashInfer_0.6.12_requires_CUDA_12_plus
fix_commit: aaa406c
fix: VLLM_USE_FLASHINFER_SAMPLER=0
votes_written: 0
```

#### Attempt 2：进入生成后显存不足，同时发现配置覆盖错误

```yaml
started_at: 2026-07-22T14:46:03+08:00
status: FAILED_DURING_FIRST_GENERATE
flashinfer_fallback_verified: true
model_memory_per_gpu_gib: 30.59
cuda_graph_memory_per_gpu_gib: 6.60
kv_cache_tokens: 318992
max_model_length: 40960
processed_prompts_before_failure: 15
expanded_requests_in_batch: 120
oom_requested_mib: 800
free_memory_at_oom_mib: 287.69
observed_wrong_top_p: 0.9
expected_top_p: 0.8
root_causes:
  - gpu_memory_utilization_0.9_left_insufficient_activation_headroom
  - default_max_num_batched_tokens_16384_created_large_prefill_activation
  - outer_batch_64_with_n_3_expanded_to_120_requests
  - argparse_argument_defaults_overrode_JSON_config
fix_commit: 8e0ea27
fixes:
  - gpu_memory_utilization_0.82
  - max_num_batched_tokens_4096
  - max_num_seqs_32
  - smaller_mode_specific_prompt_batches
  - apply_JSON_defaults_after_argument_registration
votes_written: 0
```

#### 最终产物状态

```yaml
nonthinking:
  status: PASS_WITH_HIGH_POSITION_BIAS
  manifest: .../nonthinking/teacher.manifest.json
  log: .../logs/nonthinking.log
thinking_512:
  status: FAIL_INCOMPLETE
  manifest: .../thinking_512/teacher.manifest.json
  log: .../logs/thinking_512.log
thinking_1024:
  status: PASS
  manifest: .../thinking_1024/teacher.manifest.json
  log: .../logs/thinking_1024.log
comparison:
  status: PASS
  file: .../comparison.json
```

#### thinking_512 运行中快照（333 vote rows）

```yaml
snapshot_status: INTERIM_NOT_FINAL
snapshot_received_at: 2026-07-22
source: user_uploaded_thinking_512/raw_votes.jsonl
rows: 333
json_parse_errors: 0
schema_version: qwen_pair_vote_v2
teacher_mode: thinking_512
effective_sampling_config:
  enable_thinking: true
  temperature: 0.6
  top_p: 0.95
  top_k: 20
  max_new_tokens: 512

vote_quality:
  valid_votes: 112
  invalid_votes: 221
  valid_rate: 0.336336
  truncation_rate: 0.663664
  finish_reason_length: 221
  finish_reason_stop: 112
  total_output_tokens: 158469
  mean_output_tokens_all: 475.88
  mean_output_tokens_valid: 404.62
  invalid_output_tokens:
    min: 512
    median: 512
    max: 512

coverage_at_snapshot:
  pairs: 20
  pair_directions: 40
  pair_directions_below_3_valid_votes: 13
  pairs_with_both_directions_at_least_3_valid_votes: 10
  pairs_aggregatable_with_at_least_1_valid_vote_each_direction: 18

provisional_position_bias_on_18_aggregatable_pairs:
  mean_gap: 0.0913
  median_gap: 0.0417
  max_gap: 0.7083
  gap_above_0.15_count: 3
  gap_above_0.30_count: 2
  warning: uneven_and_incomplete_vote_counts_make_these_non_final
```

阶段性结论：全部 221 个无效输出都正好达到 512 Token，说明失败由思考内容截断主导，
而不是 A/B 解析器随机漏判。示例 pair 在正序输出 A、反序输出 B，均还原到同一真实
`question_id`，证明方向映射正确。`thinking_512` 不能作为正式 teacher 配置或训练标签来源；
若要判断 thinking 本身是否有收益，需要继续运行 `thinking_1024`。不得从被截断的自然语言
末尾猜测 A/B 并补票。

#### thinking_1024 最终结果

```yaml
status: PASS
started_at: 2026-07-22T15:33:12+08:00
completed_at: 2026-07-22T15:46:53+08:00
schema_version: local_pairwise_teacher_run_v2
teacher_mode: thinking_1024
config_hash: 53ece673ef4e7a2967cb49b6ac4336ab389cfe2bd821d447fcdcbfe464e7197a
prompt_version: physics_pair_vote_v1
model: /home/share_ssd_data/nfs-env/llm_models/Qwen/Qwen3-32B
pair_file: /data/zhangyonglin/physics-difficulty-runtime/pairwise_v3/smoke/candidates.jsonl
raw_votes: /data/zhangyonglin/physics-difficulty-runtime/pairwise_v3/smoke/reasoning_ablation_20/thinking_1024/raw_votes.jsonl
manifest: /data/zhangyonglin/physics-difficulty-runtime/pairwise_v3/smoke/reasoning_ablation_20/thinking_1024/teacher.manifest.json

completion:
  pairs_requested: 20
  pairs_completed_minimum: 20
  total_vote_rows: 157
  valid_votes: 156
  invalid_or_truncated_votes: 1
  parse_success_rate: 0.993631
  truncation_or_parse_failure_rate: 0.006369
  sampling_round_cumulative_votes: [120, 156, 157]
  warnings: []

cost_and_speed:
  generation_wall_seconds: 787.283
  generation_wall_time: 00:13:07
  valid_votes_per_second: 0.19815
  output_tokens: 84089
  valid_output_tokens: 83065
  mean_output_tokens_per_valid_vote: 532.468

effective_sampling_config:
  enable_thinking: true
  temperature: 0.6
  top_p: 0.95
  top_k: 20
  min_p: 0.0
  max_new_tokens: 1024
  prompt_batch_size: 4
  adaptive_votes_per_direction: [3, 5, 10]

effective_engine_config:
  dtype: bfloat16
  tensor_parallel_size: 2
  gpu_memory_utilization: 0.82
  max_num_batched_tokens: 4096
  max_num_seqs: 32
```

结论：`thinking_1024` 已完整覆盖 20/20 pair，只有 1/157 个输出无效，说明 1024 Token
基本解决了 `thinking_512` 的系统性截断问题。它生成 156 个有效票仅消耗 84,089 个输出
Token；相比之下，`thinking_512` 快照已经消耗 158,469 Token，却只有 112 个有效票且实验
尚未完成。因此在两个 thinking 配置之间，`thinking_1024` 明显更可靠，也更节省有效标签
成本。退出阶段的 TCPStore/NCCL 警告发生在结果生成完成之后，且日志明确记录全部 worker
正常退出，不判定为实验失败。最终是否选择 thinking，仍须结合 `nonthinking` 结果、
`comparison.json` 和人工审计，不能仅凭解析成功率决定。

#### 三种模式最终对比

```yaml
evidence_received_at: 2026-07-22
local_evidence_dir: log/
raw_vote_sha256:
  nonthinking: f56b95e7421e8543c4201466f45ebc3bba8e291ff623fbecbccfcb69669b527a
  thinking_512: 3547fb2d00f20256ff2e43ddb88b38f5c4d336de469a601874e45b13fbb83694
  thinking_1024: 8b4285f7c4212b46ef35a88f8efd891736bd6928f906a61cf47cf385e8c5b111

nonthinking:
  completed_pairs: 20
  vote_rows: 254
  valid_votes: 254
  parse_success_rate: 1.0
  output_tokens: 508
  generation_wall_seconds: 34.440
  valid_votes_per_second: 7.3751
  mean_position_bias_gap: 0.275758
  high_position_bias_rate: 0.40
  uncertain_pair_rate: 0.35

thinking_512:
  completed_pairs_minimum: 16
  aggregated_pairs: 19
  unaggregated_pairs: 1
  vote_rows: 429
  valid_votes: 131
  invalid_truncated_votes: 298
  parse_success_rate: 0.305361
  output_tokens: 206173
  generation_wall_seconds: 2284.292
  valid_votes_per_second: 0.05735
  mean_position_bias_gap: 0.087333
  high_position_bias_rate: 0.105263
  uncertain_pair_rate: 0.157895
  warning: stopped_after_8_adaptive_rounds_with_insufficient_valid_votes

thinking_1024:
  completed_pairs: 20
  aggregated_pairs: 20
  vote_rows: 157
  valid_votes: 156
  invalid_truncated_votes: 1
  parse_success_rate: 0.993631
  output_tokens: 84089
  generation_wall_seconds: 787.283
  valid_votes_per_second: 0.19815
  mean_position_bias_gap: 0.076515
  high_position_bias_rate: 0.10
  uncertain_pair_rate: 0.10

cross_mode:
  nonthinking_vs_thinking_1024:
    common_pairs: 20
    hard_label_agreement: 0.80
    mean_absolute_soft_target_difference: 0.117992
  thinking_512_vs_thinking_1024:
    common_pairs: 19
    hard_label_agreement: 1.0
    mean_absolute_soft_target_difference: 0.020115
```

结果解释：`nonthinking` 的解析与速度很好，但顺序敏感性明显。20 对题中有 40% 的
`position_bias_gap > 0.30`，而 `thinking_1024` 为 10%；平均位置偏差从 0.2758 降到
0.0765。逐对检查发现，三组跨模式硬标签不一致来自 nonthinking 在交换顺序后跟随位置
改变答案，最终被聚合成假 `tie`；thinking 两个长度配置在这些题上的真实题目方向一致。
第四组分歧本身高度模糊，`thinking_1024` 也存在较大顺序差异，不能当作任一模式的正确性
证据。

模式选择结论：排除 `thinking_512`；`thinking_1024` 作为下一阶段 pilot 的首选 teacher，
但尚未批准直接用于全量标注。20 pair 只是技术 smoke，不足以证明人工真值准确率。同时，
按本次速度线性外推，8,000 pair 约需 87.5 个 GPU6/7 双卡墙钟小时，因此进入大规模生产前
必须先做带人工判断的代表性小样本审计，并评估分层或级联标注以控制成本。

#### 路由探索：文本长度与 nonthinking 可靠性

使用相同 seed 和源文件在本机确定性还原 400 个 smoke candidates，20 个消融 pair 全部
匹配。以渲染后题目文本字符数和数据自带长度桶分析，结果不支持“短题直接走
nonthinking”的假设：

```yaml
route_hypothesis: short_pairs_use_nonthinking
status: REJECTED_ON_CURRENT_SMOKE

short_short:
  pairs: 12
  hard_agreement_with_thinking_1024: 0.6667
  nonthinking_mean_position_bias_gap: 0.3396
  nonthinking_high_position_bias_rate: 0.4167
  mean_absolute_soft_target_difference: 0.1496

short_medium:
  pairs: 5
  hard_agreement_with_thinking_1024: 1.0
  nonthinking_mean_position_bias_gap: 0.1424
  nonthinking_high_position_bias_rate: 0.20
  mean_absolute_soft_target_difference: 0.0470

medium_medium:
  pairs: 3
  hard_agreement_with_thinking_1024: 1.0
  nonthinking_mean_position_bias_gap: 0.2424
  nonthinking_high_position_bias_rate: 0.6667
  mean_absolute_soft_target_difference: 0.1098
```

最大单题字符数阈值也未呈现单调收益：例如 `max_chars <= 403` 的 5 对题，方向一致率仅
0.80，平均位置偏差为 0.542；`max_chars <= 842` 的 10 对题，方向一致率仍为 0.80，平均
位置偏差为 0.308。文本短不等于比较简单；两个基础短题接近时，反而更容易受位置影响。

#### 推荐探索：nonthinking 一阶段筛选 + thinking_1024 升级

只使用 nonthinking 第一轮每方向 3 票进行离线回放，不追加 nonthinking 自适应票：

```yaml
stage_1:
  mode: nonthinking
  votes_per_direction: 3
  accept_rule:
    position_bias_gap: "<= 0.25"
    soft_target: "< 0.30 or > 0.70"

smoke_replay:
  accepted_without_thinking: 14
  escalated_to_thinking_1024: 6
  accepted_coverage: 0.70
  accepted_hard_agreement_with_thinking_1024: 1.0
  accepted_mean_absolute_soft_target_difference: 0.05
  escalated_share_of_observed_thinking_tokens: 0.3579

strict_variant:
  accept_rule:
    position_bias_gap: 0.0
    soft_target: "< 0.30 or > 0.70"
  accepted_without_thinking: 9
  accepted_coverage: 0.45
  accepted_hard_agreement_with_thinking_1024: 1.0
  accepted_mean_absolute_soft_target_difference: 0.0
```

这只是同一批 20 pair 上的回放，agreement 不是人工准确率，阈值也不能据此直接冻结。
但它比长度路由更有依据，因为路由信号直接测量同一 pair 的方向稳定性和决策强度。按宽松
规则和本次 Token 占比粗略外推，8,000 pair 使用 4 卡（两个 TP=2 实例）约需 17--20 小时，
显著低于全量 `thinking_1024` 的约 44--50 小时。正式采用前应在新的 100--200 对代表性
pair 上预注册阈值并复验，避免在这 20 对题上过拟合路由规则。

## 4. 必须记录的关键指标

### 4.1 Teacher 标注阶段

每个模式至少记录以下内容：

```yaml
identity:
  run_id: REQUIRED
  code_commit: REQUIRED
  config_file: REQUIRED
  model_path: REQUIRED
  pair_file: REQUIRED
  pair_file_sha256: RECOMMENDED
  seed: REQUIRED

completion:
  pairs_requested: REQUIRED
  pairs_completed_minimum: REQUIRED
  total_vote_rows: REQUIRED
  valid_votes: REQUIRED
  parse_success_rate: REQUIRED
  truncated_vote_count: REQUIRED

cost_and_speed:
  generation_wall_seconds: REQUIRED
  valid_votes_per_second: REQUIRED
  output_tokens: REQUIRED
  mean_output_tokens_per_valid_vote: REQUIRED
  peak_gpu_memory_mib: RECOMMENDED

label_quality_without_gold:
  mean_position_bias_gap: REQUIRED
  high_position_bias_rate: REQUIRED
  uncertain_pair_rate: REQUIRED
  forward_backward_hard_consistency: REQUIRED
  cycle_violation_rate: REQUIRED_FOR_PILOT
  cross_mode_hard_label_agreement: REQUIRED_FOR_ABLATION
  mean_absolute_soft_target_difference: REQUIRED_FOR_ABLATION

human_audit:
  audited_pairs: REQUIRED_BEFORE_MODE_SELECTION
  decisive_pair_accuracy: REQUIRED
  tie_handling_policy: REQUIRED
  severe_error_count: REQUIRED
```

解释：

- `parse_success_rate`：能从生成结果得到合法最终 A/B 的比例；低值说明 Prompt、思考截断
  或解析规则有问题。
- `position_bias_gap`：同一真实 pair 正序和反序得到的 A 更难概率之差；越小越好。
- `uncertain_pair_rate`：soft target 落在模糊区间的比例；不是越低越好，它描述题目接近程度。
- `cycle_violation_rate`：若 A>B、B>C，却得到 C>A 的比例；反映全局排序自洽性。
- `valid_votes_per_second` 和平均输出 Token：共同衡量 teacher 成本，不能只比较准确率。
- `cross_mode_hard_label_agreement`：只表示模式之间是否一致，不表示哪个模式正确。
- `decisive_pair_accuracy`：排除人工标为平局的 pair 后，与人工比较结果一致的比例；这是
  reasoning 模式选择的主质量指标。

### 4.2 Bradley–Terry student 训练阶段

每次训练启动必须先记录完整参数：

```yaml
run_identity:
  run_id: REQUIRED
  code_commit: REQUIRED
  train_pairs_manifest: REQUIRED
  validation_pairs_manifest: REQUIRED
  base_model_path: REQUIRED
  output_dir: REQUIRED
  resume_checkpoint: null_or_path
  seed: REQUIRED

model:
  backbone: Qwen3.5-4B
  tuning: LoRA
  lora_rank: REQUIRED
  lora_alpha: REQUIRED
  lora_dropout: REQUIRED
  lora_target_modules: REQUIRED
  pooling: REQUIRED
  scalar_head_init: REQUIRED

optimization:
  epochs: REQUIRED
  per_gpu_batch_size: REQUIRED
  world_size: REQUIRED
  gradient_accumulation_steps: REQUIRED
  effective_global_pair_batch: REQUIRED
  learning_rate_lora: REQUIRED
  learning_rate_head: REQUIRED
  scheduler: REQUIRED
  warmup_ratio_or_steps: REQUIRED
  weight_decay: REQUIRED
  max_grad_norm: REQUIRED
  precision: REQUIRED
  gradient_checkpointing: REQUIRED
  max_length: REQUIRED
  checkpoint_interval_epochs: REQUIRED
```

训练过程中每个日志窗口和 checkpoint 记录：

```yaml
training_progress:
  epoch: REQUIRED
  optimizer_step: REQUIRED
  seen_pairs: REQUIRED
  learning_rate: REQUIRED
  train_soft_bt_loss_window_mean: REQUIRED
  train_hard_pair_accuracy_window: RECOMMENDED
  gradient_norm: REQUIRED
  optimizer_updates_per_second: REQUIRED
  pairs_per_second: REQUIRED
  tokens_per_second: RECOMMENDED
  peak_gpu_memory_mib: REQUIRED
  nan_or_inf_count: REQUIRED
```

不能用单个 `last_loss` 判断训练趋势，主观察值必须是固定窗口平均 loss。训练 loss 下降只
证明对训练 pair 拟合增强，不代表排序泛化提升。

### 4.3 Pairwise validation/test 指标

```yaml
primary:
  soft_bt_log_loss: lower_is_better
  decisive_pair_accuracy: higher_is_better

ranking_and_calibration:
  pairwise_auc: higher_is_better
  brier_score: lower_is_better
  expected_calibration_error: lower_is_better
  spearman_rank_correlation: higher_is_better_if_reference_ranking_exists
  kendall_tau: higher_is_better_if_reference_ranking_exists

robustness:
  source_slice_metrics: REQUIRED
  length_slice_metrics: REQUIRED
  image_dependency_risk_slice_metrics: REQUIRED
  close_pair_metrics: REQUIRED
  graph_cycle_violation_rate: REQUIRED
```

- `soft_bt_log_loss`：预测概率与 teacher 软概率的交叉熵；能惩罚过度自信，适合作为主要
  checkpoint 选择指标。
- `decisive_pair_accuracy`：只看目标明显偏离 0.5 的 pair，判断难易方向是否正确。
- `pairwise_auc`：模型把较难题排在前面的整体能力，对阈值不敏感。
- `Brier score`：预测概率与目标概率的平方误差，兼顾方向和概率校准。
- `ECE`：置信度与实际正确率的分桶偏差；低 ECE 表示概率更可信。
- `Spearman/Kendall`：比较整体排序相关性；只有存在独立参考排序时才有意义。
- `cycle_violation_rate`：模型输出是否出现大量非传递关系；标量 `s(q)` 理论上应显著降低
  此类矛盾。

### 4.4 固定阈值五档指标

只有在锚点和四个固定阈值冻结后才记录五档指标：

```yaml
threshold_calibration:
  anchor_set_version: REQUIRED
  anchor_label_source: REQUIRED
  reference_population: REQUIRED
  thresholds: [t1, t2, t3, t4]
  fitting_method: REQUIRED

five_level_metrics:
  accuracy: REQUIRED
  macro_f1: REQUIRED
  balanced_accuracy: REQUIRED
  mean_absolute_error: REQUIRED
  adjacent_accuracy: REQUIRED
  quadratic_weighted_kappa: REQUIRED
  confusion_matrix: REQUIRED
  class_support: REQUIRED
  negative_log_likelihood: RECOMMENDED
  expected_calibration_error: RECOMMENDED
```

- `macro_f1`：五档分别计算 F1 后等权平均，能防止大类掩盖送分题、压轴题等小类表现。
- `balanced_accuracy`：五档召回率的平均值，衡量各档是否都能被识别。
- `MAE`：预测档位编号与真值编号的平均距离；错两档比错一档处罚更重。
- `adjacent_accuracy`：预测在真值相邻一档以内的比例，反映业务可容忍误差。
- `QWK`：考虑有序距离和偶然一致性的加权一致性，越接近 1 越好。
- `confusion_matrix`：定位具体混淆方向，必须同时报告每档样本数。

### 4.5 过拟合判断

同一批 checkpoint 画或记录以下序列：

```yaml
overfitting_signals:
  - train_soft_bt_loss_continues_down_but_validation_soft_bt_log_loss_rises
  - train_pair_accuracy_rises_but_validation_pair_accuracy_falls
  - validation_probability_becomes_more_extreme_while_Brier_or_ECE_worsens
  - validation_improves_but_frozen_test_or_OOT_degrades
  - rare_slices_and_close_pairs_degrade_before_overall_average
```

模型选择以 validation 主指标为准，不默认取最后一个 epoch。test/OOT 不能参与 checkpoint
选择，否则它不再是独立泛化评估。

## 5. 如何把服务器信息交给 Codex 回填

你不需要自己编辑本文。以后在对话中说“更新实验记录”，并任选一种方式提供信息。

### 方式 A：直接粘贴

适合短日志和命令输出：

```bash
tail -n 200 /path/to/run.log
cat /path/to/teacher.manifest.json
cat /path/to/comparison.json
```

把输出直接粘贴到对话即可。请同时给出 run 名称、服务器输出目录和对应代码 commit；
如果不知道 commit，可执行：

```bash
git -C ~/physics-difficulty-rater rev-parse HEAD
```

### 方式 B：从服务器下载到 Mac 后上传附件

在能连接服务器的终端执行：

```bash
scp zhangyonglin@172.22.0.45:/服务器/绝对路径/comparison.json ~/Downloads/
scp zhangyonglin@172.22.0.45:/服务器/绝对路径/teacher.manifest.json ~/Downloads/
scp zhangyonglin@172.22.0.45:/服务器/绝对路径/run.log ~/Downloads/
```

然后把下载的文件作为对话附件发给 Codex。

### 方式 C：打包一个实验的轻量证据

不要打包 checkpoint 和模型权重，只打包配置、manifest、metrics 和日志：

```bash
RUN=/data/zhangyonglin/physics-difficulty-runtime/某个实验目录
tar -czf /tmp/physics_experiment_evidence.tgz \
  -C "$RUN" \
  logs evaluations comparison.json 2>/dev/null || true

scp zhangyonglin@172.22.0.45:/tmp/physics_experiment_evidence.tgz ~/Downloads/
```

若目录结构不同，先提供：

```bash
find "$RUN" -maxdepth 3 -type f -printf '%p\n' | sort
```

Codex 会告诉你应下载哪些小文件。原始 25k 数据、模型权重和 checkpoint 无需传回本机。

## 6. 每次更新所需的最小信息

```yaml
required_from_user:
  run_name: "实验名称"
  purpose: "本次验证什么"
  command: "实际启动命令"
  output_dir: "服务器绝对路径"
  code_commit: "git rev-parse HEAD"
  artifacts:
    - config
    - manifest
    - metrics_json
    - relevant_log_tail
  interruption: "无，或停止/崩溃时间与原因"
```

收到这些信息后，Codex 负责：核对参数是否真正生效、解析关键指标、判断运行是否完整、
把事实追加到本文件、标注结论置信度，并提交推送到 GitHub。

## 7. 当前下一步

```yaml
next_actions:
  - completed: V3-TEACHER-ABLATION-001
    provisional_mode: thinking_1024
  - perform: representative_human_pair_audit
    purpose: measure_decisive_pair_accuracy_and_tie_handling
  - evaluate: staged_or_cascade_teacher_strategy
    purpose: reduce_thinking_1024_labeling_cost_without_accepting_nonthinking_position_bias
    candidate_rule: nonthinking_3_votes_each_direction_then_escalate_unstable_or_uncertain_pairs
  - expand_if_passed: bounded_pairwise_pilot_before_8000_pairs
  - then: train_soft_Bradley_Terry_student
```
