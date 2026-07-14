"""Canonical text formatting shared by labeling, training, and inference."""
from __future__ import annotations
from typing import Any, Dict

def _part(title: str, value: Any) -> str:
    text = str(value or "").strip()
    return f"【{title}】\n{text}" if text else ""

def format_question(record: Dict[str, Any]) -> str:
    parts = [_part("题干", record.get("stem")), _part("选项", record.get("options")), _part("解析", record.get("analysis"))]
    children = record.get("sub_questions") or []
    if children:
        child_parts = ["【小题】"]
        for index, child in enumerate(children, 1):
            if isinstance(child, dict):
                child_parts.extend([f"小题{index}:", _part("题干", child.get("stem")), _part("选项", child.get("options")), _part("解析", child.get("analysis"))])
            else:
                child_parts.append(f"小题{index}:\n{child}")
        parts.append("\n".join(item for item in child_parts if item))
    return "\n\n".join(item for item in parts if item)

def diagnostics(record: Dict[str, Any], text: str) -> Dict[str, Any]:
    return {
        "has_analysis": bool(str(record.get("analysis") or "").strip()),
        "has_subquestions": bool(record.get("sub_questions")),
        "has_image_marker": "<image>" in text or "如图" in text,
        "char_length": len(text),
    }
