"""Deterministic section-aware truncation for canonical question input."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from physics_difficulty.data.formatting import render_sections

TRUNCATION_STRATEGY_VERSION = "v1"


def _token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _truncate_tokens(tokenizer: Any, text: str, budget: int) -> str:
    if budget <= 0:
        return ""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= budget:
        return text
    return tokenizer.decode(tokens[:budget], skip_special_tokens=True).rstrip()


def _section_text(section: Dict[str, str]) -> str:
    return section["title"] if not section["text"] else f"{section['title']}\n{section['text']}"


def render_with_token_budget(sections: List[Dict[str, str]], tokenizer: Any, max_length: int) -> Tuple[str, Dict[str, Any]]:
    """Preserve every title and required field before allocating analysis budget.

    If required text alone exceeds the limit, each required content field gets a
    deterministic equal share after all titles are reserved.  This avoids the
    unsafe default behaviour of retaining only the beginning of a long parent.
    """
    full = render_sections(sections)
    original_tokens = _token_count(tokenizer, full)
    if original_tokens <= max_length:
        return full, {"truncated": False, "original_token_count": original_tokens, "retained_token_count": original_tokens, "truncation_strategy_version": TRUNCATION_STRATEGY_VERSION}

    title_tokens = sum(_token_count(tokenizer, section["title"]) for section in sections)
    separators = max(0, len(sections) - 1) * _token_count(tokenizer, "\n\n")
    content_budget = max(0, max_length - title_tokens - separators)
    required = [section for section in sections if section.get("required") and section["text"]]
    optional = [section for section in sections if not section.get("required") and section["text"]]
    rendered_sections = [dict(section) for section in sections]

    required_tokens = sum(_token_count(tokenizer, section["text"]) for section in required)
    if required_tokens > content_budget:
        # All required fields survive, but their contents share the limited budget.
        base, remainder = divmod(content_budget, max(1, len(required)))
        allocation = {id(section): base + (1 if index < remainder else 0) for index, section in enumerate(required)}
    else:
        allocation = {id(section): _token_count(tokenizer, section["text"]) for section in required}
        remaining = content_budget - required_tokens
        # Divide available analysis capacity across all analyses, so late small
        # questions are not erased merely because an early analysis is verbose.
        base, remainder = divmod(remaining, max(1, len(optional)))
        for index, section in enumerate(optional):
            allocation[id(section)] = min(_token_count(tokenizer, section["text"]), base + (1 if index < remainder else 0))

    for original, rendered in zip(sections, rendered_sections):
        if original["text"]:
            rendered["text"] = _truncate_tokens(tokenizer, original["text"], allocation.get(id(original), 0))

    text = render_sections(rendered_sections)
    # Tokenizers sometimes add/merge boundary tokens differently.  Apply a final
    # bounded trim while retaining the section-aware allocation in normal cases.
    if _token_count(tokenizer, text) > max_length:
        text = _truncate_tokens(tokenizer, text, max_length)
    retained = _token_count(tokenizer, text)
    return text, {"truncated": True, "original_token_count": original_tokens, "retained_token_count": retained, "truncation_strategy_version": TRUNCATION_STRATEGY_VERSION}
