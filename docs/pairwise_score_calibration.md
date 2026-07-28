# Pairwise scalar score calibration

The pairwise student already learns one shared scalar `s(q)` per question:

```text
P(A harder than B) = sigmoid(s(A) - s(B))
```

Deployment therefore scores each question once. Pair construction is training
supervision, not an inference requirement.

## Artifacts and isolation

Use a selected V1/V2/V3 checkpoint and a frozen reference pool. The first
experimental reference pool is the raw-V3 `train` split after excluding the
2,000 questions used in the production comparison graph:

```yaml
reference_source: pairwise_v3/questions/train.jsonl
excluded_questions: pairwise_v3/pilot/questions.jsonl
expected_reference_records: 17988
forbidden_reference_splits:
  - validation
  - test
```

This prevents validation/test leakage and reduces direct reuse of questions
that trained the scalar head. The original 25k file was sampled into five
equal buckets using an unusable historical `difficulty` field, so this first
reference is an experimental fixed population, not a claim about the natural
online business distribution. Replace it with a representative production
sample before freezing business thresholds.

## 1. Export reference raw scores

Set `CHECKPOINT` only after V1/V2/V3 model selection:

```bash
cd ~/physics-difficulty-rater
conda activate QuRater

MODEL=/home/zhangyonglin/models/models/Qwen--Qwen3.5-4B/snapshots/master
PAIR_ROOT=/data/$USER/physics-difficulty-runtime/pairwise_v3
CHECKPOINT=/data/$USER/physics-difficulty-runtime/outputs/pairwise_v3_production_8000/v1_bt_only/checkpoint-epoch-3-step-1176
CAL_ROOT=/data/$USER/physics-difficulty-runtime/pairwise_v3/calibration/physics_reference_v1

mkdir -p "$CAL_ROOT"

nohup env CUDA_VISIBLE_DEVICES=0 \
  python score_pairwise_questions.py \
  --model-path "$MODEL" \
  --checkpoint-dir "$CHECKPOINT" \
  --questions "$PAIR_ROOT/questions/train.jsonl" \
  --exclude-question-ids "$PAIR_ROOT/pilot/questions.jsonl" \
  --output "$CAL_ROOT/reference_scores.jsonl" \
  --manifest "$CAL_ROOT/reference_scores.manifest.json" \
  --max-length 1024 \
  --batch-size 4 \
  > "$CAL_ROOT/reference_scoring.log" 2>&1 &
```

The score manifest binds the output to the input question hash and a
fingerprint covering the LoRA adapter, scalar head, and pairwise config.

## 2. Freeze the empirical CDF and four thresholds

```bash
python scripts/fit_pairwise_difficulty_calibration.py \
  --scores "$CAL_ROOT/reference_scores.jsonl" \
  --scores-manifest "$CAL_ROOT/reference_scores.manifest.json" \
  --output "$CAL_ROOT/calibration.json" \
  --calibration-version physics_reference_v1 \
  --distribution 0.20,0.20,0.30,0.20,0.10 \
  --minimum-records 1000 \
  --reference-note "Experimental raw-V3 train reference excluding 2000 pair-training questions"
```

The calibration stores:

- the exact sorted reference scores used as the empirical CDF;
- raw-score quantiles at 20%, 40%, 70%, and 90%;
- the reference and checkpoint hashes;
- a content-derived `calibration_id`.

The thresholds are frozen. They must never be recomputed from an inference
batch.

## 3. Score and classify a new prepared question file

```bash
PRED_ROOT=/data/$USER/physics-difficulty-runtime/pairwise_v3/predictions/batch_v1
mkdir -p "$PRED_ROOT"

env CUDA_VISIBLE_DEVICES=0 python score_pairwise_questions.py \
  --model-path "$MODEL" \
  --checkpoint-dir "$CHECKPOINT" \
  --questions /path/to/prepared_questions.jsonl \
  --output "$PRED_ROOT/raw_scores.jsonl" \
  --manifest "$PRED_ROOT/raw_scores.manifest.json" \
  --max-length 1024 \
  --batch-size 4

python predict_pairwise_difficulty.py \
  --scores "$PRED_ROOT/raw_scores.jsonl" \
  --scores-manifest "$PRED_ROOT/raw_scores.manifest.json" \
  --calibration "$CAL_ROOT/calibration.json" \
  --output "$PRED_ROOT/predictions.jsonl" \
  --manifest "$PRED_ROOT/predictions.manifest.json"
```

Each prediction contains:

```json
{
  "question_id": "example",
  "raw_difficulty_score": 1.2847,
  "difficulty_percentile": 0.823,
  "difficulty_score": 82.3,
  "difficulty_level_id": 3,
  "difficulty_level": "拔高题",
  "calibration_version": "physics_reference_v1",
  "calibration_id": "..."
}
```

`difficulty_score` means percentile within the frozen reference population.
It is not a probability that the question is difficult. A later production
batch uses the same CDF and thresholds, so its level proportions are allowed
to differ from 20/20/30/20/10.
