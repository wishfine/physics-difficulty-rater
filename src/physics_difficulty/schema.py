"""Frozen-18-compatible V2 schema for the physics local model."""
from __future__ import annotations

from typing import Any, Dict, List

DIFFICULTY_LEVELS = ["送分题", "基础题", "中等题", "拔高题", "压轴题"]
DIFFICULTY_TO_ID = {name: index for index, name in enumerate(DIFFICULTY_LEVELS)}

PROBLEM_STRUCTURE_VALUES = ["概念判断", "直接计算", "实验探究", "图像表格分析", "电路综合", "力学综合", "热学综合", "光学声学综合", "跨模块综合"]
KNOWLEDGE_DOMAINS = ["力学", "电路", "热学", "光学声学"]
FROZEN_18_FEATURE_NAMES = (
    "step_count", "formula_count", "calculation_complexity", "reasoning_chain", "problem_structure", "additional_structure",
    "information_carrier", "reality_question", "subquestion_dependency", "knowledge_count", "knowledge_diff", "cross_module",
    "state_count", "constraint_count", "variable_relation", "experiment_requirement", "graph_table_requirement", "error_risk",
)

FEATURE_VALUES = {
    "problem_structure": PROBLEM_STRUCTURE_VALUES,
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

def normalize_problem_structure(value: Any) -> str:
    """Keep the frozen teacher's nine-way structural label without inference."""
    return _valid("problem_structure", value, "概念判断")

def normalize_knowledge_domains(features: Dict[str, Any] | None) -> List[str]:
    """Keep physical-domain tags as metadata, not as an auxiliary loss."""
    features = features or {}
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
