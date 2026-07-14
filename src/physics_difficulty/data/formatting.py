"""Canonical input rendering shared by teacher labeling, training and serving."""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

FORMATTER_VERSION = "v1"
IMAGE_REFERENCE_PATTERN = re.compile(r"(?:如图|见图|图中|下图|上图|图示|如右图|如左图)")


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _child_sort_key(child: Any) -> tuple[int, str]:
    if not isinstance(child, dict):
        return (1, "")
    value = child.get("question_id", "")
    try:
        return (0, f"{int(value):020d}")
    except (TypeError, ValueError):
        return (1, str(value))


def sorted_subquestions(record: Dict[str, Any]) -> List[Any]:
    """Return a sorted copy; never mutate the source record in preprocessing."""
    children = record.get("sub_questions") or []
    return sorted(list(children), key=_child_sort_key)


def canonical_sections(record: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build ordered renderable chunks in the exact API Prompt text convention."""
    sections: List[Dict[str, str]] = []
    for name, value in (("题干", record.get("stem")), ("选项", record.get("options")), ("解析", record.get("analysis"))):
        text = _clean(value)
        if text:
            sections.append({"kind": f"parent_{name}", "title": f"【{name}】", "text": text, "required": name != "解析"})

    children = sorted_subquestions(record)
    if children:
        sections.append({"kind": "subquestions_header", "title": "【小题】", "text": "", "required": True})
    for index, child in enumerate(children, 1):
        sections.append({"kind": "subquestion_header", "title": f"  小题{index}:", "text": "", "required": True})
        if isinstance(child, dict):
            for name, value in (("题干", child.get("stem")), ("选项", child.get("options")), ("解析", child.get("analysis"))):
                text = _clean(value)
                if text:
                    sections.append({"kind": f"child_{name}", "title": f"    {name}:", "text": text, "required": name != "解析", "inline": True})
        else:
            text = _clean(child)
            if text:
                sections.append({"kind": "child_题干", "title": "    题干:", "text": text, "required": True, "inline": True})
    return sections


def render_sections(sections: Iterable[Dict[str, str]]) -> str:
    rendered = []
    for section in sections:
        title, text = section["title"], section["text"]
        rendered.append(title if not text else f"{title} {text}" if section.get("inline") else f"{title}\n{text}")
    return "\n\n".join(rendered)


def format_question(record: Dict[str, Any]) -> str:
    return render_sections(canonical_sections(record))


def image_urls(record: Dict[str, Any]) -> List[str]:
    urls = [_clean(record.get("stem_pic_url")), _clean(record.get("analysis_pic_url"))]
    for child in sorted_subquestions(record):
        if isinstance(child, dict):
            urls.extend((_clean(child.get("stem_pic_url")), _clean(child.get("analysis_pic_url"))))
    return list(dict.fromkeys(url for url in urls if url))


def diagnostics(record: Dict[str, Any], text: str) -> Dict[str, Any]:
    urls = image_urls(record)
    marker = bool(IMAGE_REFERENCE_PATTERN.search(text))
    char_length = len(text)
    return {
        "formatter_version": FORMATTER_VERSION,
        "has_analysis": bool(_clean(record.get("analysis"))) or any(bool(_clean(child.get("analysis"))) for child in sorted_subquestions(record) if isinstance(child, dict)),
        "has_subquestions": bool(record.get("sub_questions")),
        "has_image_url": bool(urls),
        "has_image_reference_marker": marker,
        "image_dependency_risk": "high" if urls and marker else "medium" if urls or marker else "none",
        "char_length": char_length,
        "input_length_bucket": "short" if char_length < 1000 else "medium" if char_length < 3000 else "long",
    }
