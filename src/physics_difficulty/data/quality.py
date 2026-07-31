"""Non-destructive quality scoring. Rules flag risk; they never rewrite labels."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from physics_difficulty.schema import DIFFICULTY_TO_ID


def feature_conflicts(level: str, features: Dict[str, Any]) -> List[Tuple[str, str]]:
    conflicts: List[Tuple[str, str]] = []
    hard_steps = {"6-8步", "9步以上"}
    if level == "送分题" and (features["step_count"] in hard_steps or features["constraint_count"] == "多约束" or features["variable_relation"] == "多变量耦合关系"):
        conflicts.append(("severe", "送分题与高复杂度特征冲突"))
    if level == "压轴题" and all((
        features["step_count"] == "1-2步", features["calculation_complexity"] == "口算或直接判断",
        features["reasoning_chain"] == "直接套用", features["knowledge_count"] == "1个", features["constraint_count"] == "无约束",
    )):
        conflicts.append(("severe", "压轴题缺少高阶特征"))
    if level in {"送分题", "基础题"} and features["reasoning_chain"] == "逆向推理或临界分析":
        conflicts.append(("mild", "低难度标签与逆向或临界推理不一致"))
    return conflicts


def score_label_quality(level: str, features: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    if level not in DIFFICULTY_TO_ID:
        return {"label_quality": "invalid", "sample_weight": 0.0, "conflicts": ["难度标签非法"], "conflict_severity": "severe", "feature_level_consistent": False, "review_action": "exclude"}
    if not str(record.get("stem") or "").strip() and not record.get("sub_questions"):
        return {"label_quality": "invalid", "sample_weight": 0.0, "conflicts": ["题干和小题均为空"], "conflict_severity": "severe", "feature_level_consistent": False, "review_action": "exclude"}
    conflicts = feature_conflicts(level, features)
    severity = "severe" if any(item[0] == "severe" for item in conflicts) else "mild" if conflicts else "none"
    review = record.get("independent_review") or {}
    if severity == "severe":
        return {"label_quality": "low", "sample_weight": 0.0, "conflicts": [item[1] for item in conflicts], "conflict_severity": severity, "feature_level_consistent": False, "review_action": "rejudge"}
    if review.get("verdict") == "accept" and review.get("confidence") == "high":
        return {"label_quality": "high", "sample_weight": 1.0, "conflicts": [item[1] for item in conflicts], "conflict_severity": severity, "feature_level_consistent": not conflicts, "review_action": "keep"}
    return {"label_quality": "medium", "sample_weight": 0.7, "conflicts": [item[1] for item in conflicts], "conflict_severity": severity, "feature_level_consistent": not conflicts, "review_action": "keep"}
