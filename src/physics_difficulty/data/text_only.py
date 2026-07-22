"""Text-only normalization and leakage diagnostics for pairwise rating."""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Iterable, List

HTTP_URL = re.compile(r"https?://\S+", re.IGNORECASE)
MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
HTML_IMAGE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
EXPLICIT_IMAGE_PLACEHOLDER = re.compile(r"<(?:image|img)>|\[图片\]|【图片】", re.IGNORECASE)
EXCESS_WHITESPACE = re.compile(r"[ \t]+")
EXCESS_NEWLINES = re.compile(r"\n{3,}")

LEAKAGE_PATTERNS = {
    "explicit_level": re.compile(r"送分题|基础题|中等题|拔高题|压轴题"),
    "difficulty_coefficient": re.compile(r"难度系数|难度等级"),
    # Generic "难易程度" is common physics wording (for example,
    # conductivity).  Only flag it when it explicitly assesses a question.
    "difficulty_assessment": re.compile(r"(?:本题|该题|题目|试题|该试题).{0,12}难易程度|难易程度.{0,12}(?:本题|该题|题目|试题|该试题)"),
    "difficulty_adjective": re.compile(r"本题(?:较为|比较|非常|相对)?(?:简单|容易|较难|困难)"),
}
FORBIDDEN_SOURCE_LABEL_KEYS = {"difficulty", "raw_difficulty"}


def normalize_text_only(text: Any) -> str:
    """Remove image payload references while preserving physics wording."""
    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    value = MARKDOWN_IMAGE.sub("", value)
    value = HTML_IMAGE.sub("", value)
    value = EXPLICIT_IMAGE_PLACEHOLDER.sub("", value)
    # Curated model input should not contain remote image URLs.  Other URLs are
    # also irrelevant for the local text-only rater and can encode source IDs.
    value = HTTP_URL.sub("", value)
    value = "\n".join(EXCESS_WHITESPACE.sub(" ", line).strip() for line in value.splitlines())
    return EXCESS_NEWLINES.sub("\n\n", value).strip()


def normalize_for_dedup(text: Any) -> str:
    """Create a conservative comparison key without changing model input."""
    value = unicodedata.normalize("NFKC", normalize_text_only(text))
    return re.sub(r"\s+", " ", value).strip()


def has_semantic_content(text: Any) -> bool:
    """Reject title-only/image-only records while retaining formulas."""
    value = normalize_for_dedup(text)
    value = re.sub(r"【(?:题干|选项|解析|小题)】", "", value)
    value = re.sub(r"(?:^|\s)小题\d+\s*[:：]", " ", value)
    value = re.sub(r"(?:^|\s)(?:题干|选项|解析)\s*[:：]", " ", value)
    value = re.sub(r"[\s:：,，.。；;!！?？【】\[\]()（）]+", "", value)
    return bool(value)


def leakage_findings(text: str) -> List[Dict[str, str]]:
    findings = []
    for name, pattern in LEAKAGE_PATTERNS.items():
        for match in pattern.finditer(text):
            left, right = max(0, match.start() - 30), min(len(text), match.end() + 30)
            findings.append({"type": name, "match": match.group(0), "context": text[left:right]})
    return findings


def forbidden_source_label_paths(value: Any, prefix: str = "") -> List[str]:
    """Find the unusable historical label even when nested in metadata."""
    findings: List[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key) in FORBIDDEN_SOURCE_LABEL_KEYS:
                findings.append(path)
            findings.extend(forbidden_source_label_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(forbidden_source_label_paths(child, f"{prefix}[{index}]"))
    return findings


def question_identifier(record: Dict[str, Any]) -> str:
    """Return the existing stable ID; V3 must never invent a replacement."""
    value = record.get("id") or record.get("question_id")
    if value is None or not str(value).strip():
        raise ValueError("question is missing its existing stable id/question_id")
    return str(value)


def question_group_identifier(record: Dict[str, Any], question_id: str) -> str:
    """Reuse an existing grouping key; a standalone item groups with itself."""
    value = record.get("question_group_id") or record.get("parent_id") or question_id
    return str(value)


def source_text(record: Dict[str, Any], formatter: Any) -> str:
    if str(record.get("text") or "").strip():
        return str(record["text"])
    return formatter(record)


def strip_metadata_fields(record: Dict[str, Any], fields: Iterable[str]) -> Dict[str, Any]:
    result = dict(record)
    for field in fields:
        result.pop(field, None)
    return result
