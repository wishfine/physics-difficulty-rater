"""Non-destructive label-quality scoring for API teacher labels."""
from __future__ import annotations
from typing import Any, Dict, List

from physics_difficulty.schema import DIFFICULTY_TO_ID

def feature_conflicts(level: str, features: Dict[str, str]) -> List[str]:
    conflicts: List[str] = []
    hard = {"6-8步", "9步以上"}
    if level == "送分题" and (features["step_count"] in hard or features["constraint_count"] == "多约束" or features["variable_relation"] == "多变量耦合关系"):
        conflicts.append("送分题与高复杂度特征冲突")
    if level == "压轴题" and all([
        features["step_count"] == "1-2步", features["calculation_complexity"] == "口算或直接判断",
        features["reasoning_chain"] == "直接套用", features["knowledge_count"] == "1个",
        features["constraint_count"] == "无约束",
    ]):
        conflicts.append("压轴题缺少高阶特征")
    return conflicts

def score_label_quality(level: str, features: Dict[str, str], record: Dict[str, Any]) -> Dict[str, Any]:
    if level not in DIFFICULTY_TO_ID:
        return {"label_quality": "invalid", "sample_weight": 0.0, "conflicts": ["难度标签非法"], "review_action": "exclude"}
    if not str(record.get("stem") or "").strip() and not record.get("sub_questions"):
        return {"label_quality": "invalid", "sample_weight": 0.0, "conflicts": ["题干和小题均为空"], "review_action": "exclude"}
    conflicts = feature_conflicts(level, features)
    if conflicts:
        return {"label_quality": "low", "sample_weight": 0.0, "conflicts": conflicts, "review_action": "rejudge"}
    return {"label_quality": "medium", "sample_weight": 0.7, "conflicts": [], "review_action": "keep"}
