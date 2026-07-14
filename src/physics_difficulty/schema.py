"""Stable V2 label schema and conversion from the legacy 18-feature output."""
from __future__ import annotations

from typing import Any, Dict, List

DIFFICULTY_LEVELS = ["送分题", "基础题", "中等题", "拔高题", "压轴题"]
DIFFICULTY_TO_ID = {name: index for index, name in enumerate(DIFFICULTY_LEVELS)}

PROBLEM_STRUCTURE_TAGS = [
    "概念判断", "直接计算", "多条件建模", "实验探究", "图像表格分析", "单模块综合", "跨模块综合",
]
KNOWLEDGE_DOMAINS = ["力学", "电路", "热学", "光学声学"]

FEATURE_VALUES = {
    "problem_structure": PROBLEM_STRUCTURE_TAGS,
    "step_count": ["1-2步", "3-5步", "6-8步", "9步以上"],
    "calculation_complexity": ["口算或直接判断", "简单笔算", "多公式联立", "复杂方程或范围计算"],
    "reasoning_chain": ["直接套用", "简单因果推理", "多层因果推理", "逆向推理或临界分析"],
    "knowledge_count": ["1个", "2-3个", "4个及以上"],
    "subquestion_dependency": ["无多问", "多问但相互独立", "多问且层层递进"],
    "state_count": ["单状态", "双状态", "多状态", "连续变化或临界状态"],
    "constraint_count": ["无约束", "单一约束", "多约束"],
    "variable_relation": ["无变量关系", "简单正反比", "图像函数关系", "多变量耦合关系"],
    "information_processing": ["无", "图表直接读数", "图表多组比较归纳", "图像反推或外推", "实验基础操作或读数", "实验控制变量或故障分析", "实验方案设计或误差评价", "图表与实验混合处理"],
}
FEATURE_TO_ID = {name: {value: index for index, value in enumerate(values)} for name, values in FEATURE_VALUES.items()}
MULTI_LABEL_FEATURES = {"problem_structure"}
SINGLE_LABEL_FEATURES = tuple(name for name in FEATURE_VALUES if name not in MULTI_LABEL_FEATURES)

LEGACY_DEFAULTS = {
    "problem_structure": "概念判断", "step_count": "1-2步", "calculation_complexity": "口算或直接判断",
    "reasoning_chain": "直接套用", "knowledge_count": "1个", "subquestion_dependency": "无多问",
    "state_count": "单状态", "constraint_count": "无约束", "variable_relation": "无变量关系",
    "graph_table_requirement": "无", "experiment_requirement": "无",
}

def _valid(field: str, value: Any, default: str) -> str:
    value = str(value or "").strip()
    return value if value in FEATURE_VALUES[field] else default

def merge_information_processing(graph: Any, experiment: Any) -> str:
    graph = str(graph or "无").strip()
    experiment = str(experiment or "无").strip()
    if graph != "无" and experiment != "无":
        return "图表与实验混合处理"
    graph_map = {"直接读数": "图表直接读数", "多组比较归纳": "图表多组比较归纳", "图像反推或外推": "图像反推或外推"}
    experiment_map = {"基础操作或读数": "实验基础操作或读数", "控制变量或故障分析": "实验控制变量或故障分析", "方案设计或误差评价": "实验方案设计或误差评价"}
    return graph_map.get(graph, experiment_map.get(experiment, "无"))

def normalize_problem_structure(value: Any) -> List[str]:
    """Convert the legacy one-of-nine field to V2 structural tags.

    The old feature mixed physical domains (such as 电路综合) and task forms.
    V2 keeps only task/knowledge organisation in the trainable tag set; domain
    details are deliberately not inferred from this lossy legacy field.
    """
    if isinstance(value, list):
        tags = [str(tag).strip() for tag in value if str(tag).strip() in PROBLEM_STRUCTURE_TAGS]
        return list(dict.fromkeys(tags)) or ["概念判断"]
    mapping = {
        "概念判断": ["概念判断"],
        "直接计算": ["直接计算"],
        "实验探究": ["实验探究"],
        "图像表格分析": ["图像表格分析"],
        "电路综合": ["单模块综合"],
        "力学综合": ["单模块综合"],
        "热学综合": ["单模块综合"],
        "光学声学综合": ["单模块综合"],
        "跨模块综合": ["跨模块综合"],
        "多条件建模": ["多条件建模"],
        "单模块综合": ["单模块综合"],
    }
    return mapping.get(str(value or "").strip(), ["概念判断"])

def normalize_knowledge_domains(features: Dict[str, Any] | None) -> List[str]:
    """Keep physical-domain tags as metadata, not as an auxiliary loss."""
    features = features or {}
    value = features.get("knowledge_domains", features.get("knowledge_domain"))
    if isinstance(value, list):
        return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip() in KNOWLEDGE_DOMAINS))
    legacy_domain = {
        "力学综合": "力学", "电路综合": "电路", "热学综合": "热学", "光学声学综合": "光学声学",
    }.get(str(features.get("problem_structure") or "").strip())
    return [legacy_domain] if legacy_domain else []

def normalize_v2_features(legacy: Dict[str, Any] | None) -> Dict[str, Any]:
    legacy = legacy or {}
    step = str(legacy.get("step_count", LEGACY_DEFAULTS["step_count"]))
    if step in {"9-12步", "12步以上"}:
        step = "9步以上"
    result = {
        "problem_structure": normalize_problem_structure(legacy.get("problem_structure")),
        "step_count": _valid("step_count", step, "1-2步"),
        "calculation_complexity": _valid("calculation_complexity", legacy.get("calculation_complexity"), "口算或直接判断"),
        "reasoning_chain": _valid("reasoning_chain", legacy.get("reasoning_chain"), "直接套用"),
        "knowledge_count": _valid("knowledge_count", legacy.get("knowledge_count"), "1个"),
        "subquestion_dependency": _valid("subquestion_dependency", legacy.get("subquestion_dependency"), "无多问"),
        "state_count": _valid("state_count", legacy.get("state_count"), "单状态"),
        "constraint_count": _valid("constraint_count", legacy.get("constraint_count"), "无约束"),
        "variable_relation": _valid("variable_relation", legacy.get("variable_relation"), "无变量关系"),
        "information_processing": merge_information_processing(legacy.get("graph_table_requirement"), legacy.get("experiment_requirement")),
    }
    return result

def difficulty_id(level: str) -> int:
    if level not in DIFFICULTY_TO_ID:
        raise ValueError(f"Unknown difficulty level: {level}")
    return DIFFICULTY_TO_ID[level]
