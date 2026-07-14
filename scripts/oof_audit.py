#!/usr/bin/env python3
"""Create independent 5-fold OOF label-audit evidence without rewriting labels."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="curated V2 teacher JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = [row["teacher_difficulty_id"] for row in rows]
    counts = Counter(labels)
    if min(counts.values()) < args.folds:
        raise ValueError(f"Every label needs at least {args.folds} records for OOF; got {dict(counts)}")
    groups = [f"{row.get('source_dataset_id', 'unknown')}::{row.get('parent_id', row['id'])}" for row in rows]
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    probabilities = [[0.0] * 5 for _ in rows]
    for train_index, test_index in splitter.split([row["text"] for row in rows], labels, groups):
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2, max_features=120000, sublinear_tf=True)
        train_text = [rows[index]["text"] for index in train_index]
        test_text = [rows[index]["text"] for index in test_index]
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed)
        model.fit(vectorizer.fit_transform(train_text), [labels[index] for index in train_index])
        fold_probabilities = model.predict_proba(vectorizer.transform(test_text))
        for index, probability in zip(test_index, fold_probabilities):
            for class_id, value in zip(model.classes_, probability):
                probabilities[index][int(class_id)] = float(value)
    output_rows = []
    for row, probability, label in zip(rows, probabilities, labels):
        prediction = max(range(5), key=lambda class_id: probability[class_id])
        ordered = sorted(probability, reverse=True)
        confidence, margin = ordered[0], ordered[0] - ordered[1]
        entropy = -sum(value * math.log(max(value, 1e-12)) for value in probability)
        distance = abs(prediction - label)
        audit = {"predicted_level_id": prediction, "probabilities": probability, "top1_probability": confidence, "top1_top2_margin": margin, "entropy": entropy, "distance_from_teacher_label": distance, "rejudge_recommended": bool(confidence >= 0.7 and distance >= 2)}
        output_rows.append({**row, "oof_audit": audit})
    Path(args.output).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows), encoding="utf-8")
    print(json.dumps({"records": len(output_rows), "folds": args.folds, "rejudge_recommended": sum(row["oof_audit"]["rejudge_recommended"] for row in output_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
