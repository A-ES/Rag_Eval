"""Post-extraction text cleaning for ingested PDF pages.

Provides a pure function `clean_text` that:
1. Strips repeated headers/footers (strings appearing on >80% of pages).
2. Removes page-number-only lines.
3. Normalizes whitespace (collapses runs of spaces/tabs, trims blank lines).
4. De-hyphenates words broken across line wraps.
"""

from __future__ import annotations

import re
from collections import Counter


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def clean_text(text: str, *, repeated_lines: set[str] | None = None) -> str:
    """Apply all cleaning steps to *text* and return the cleaned result.

    Parameters
    ----------
    text:
        The raw extracted text (may span multiple lines).
    repeated_lines:
        Optional pre-computed set of header/footer lines to strip.  When
        processing a multi-page document, call `detect_repeated_lines` once
        on all pages and pass the result here so each page is cleaned
        consistently.
    """
    if not text:
        return ""

    lines = text.splitlines()

    # 1. Strip repeated headers/footers
    if repeated_lines:
        lines = _strip_repeated_lines(lines, repeated_lines)

    # 2. Remove page-number-only lines
    lines = _remove_page_number_lines(lines)

    # 3. De-hyphenate words broken across line wraps
    lines = _dehyphenate(lines)

    # 4. Normalize whitespace
    result = _normalize_whitespace(lines)

    return result


def detect_repeated_lines(pages: list[str], *, threshold: float = 0.8) -> set[str]:
    """Detect lines that appear on more than *threshold* fraction of pages.

    These are typically headers or footers inserted by the PDF renderer.
    """
    if not pages:
        return set()

    num_pages = len(pages)
    line_page_counts: Counter[str] = Counter()

    for page_text in pages:
        # Use a set so each line is counted at most once per page
        unique_lines = {_normalize_for_comparison(line) for line in page_text.splitlines() if line.strip()}
        for line in unique_lines:
            line_page_counts[line] += 1

    min_occurrences = num_pages * threshold
    return {line for line, count in line_page_counts.items() if count > min_occurrences}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_for_comparison(line: str) -> str:
    """Collapse whitespace for fuzzy header/footer matching."""
    return " ".join(line.split())


def _strip_repeated_lines(lines: list[str], repeated: set[str]) -> list[str]:
    return [line for line in lines if _normalize_for_comparison(line) not in repeated]


_PAGE_NUMBER_RE = re.compile(
    r"^\s*"
    r"(?:"
    r"(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?"  # "12", "page 12", "3 of 10", "3/10"
    r"|"
    r"[-–—]\s*\d+\s*[-–—]"                     # "- 5 -", "— 5 —"
    r")"
    r"\s*$",
    re.IGNORECASE,
)


def _remove_page_number_lines(lines: list[str]) -> list[str]:
    return [line for line in lines if not _PAGE_NUMBER_RE.match(line)]


_HYPHEN_RE = re.compile(r"(\w+)-\s*$")


def _dehyphenate(lines: list[str]) -> list[str]:
    """Rejoin words split across lines with a trailing hyphen."""
    if not lines:
        return lines

    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        match = _HYPHEN_RE.search(line)
        if match and i + 1 < len(lines):
            next_line = lines[i + 1]
            # Extract the first word of the next line to join
            next_words = next_line.split(None, 1)
            if next_words:
                # Remove the hyphen and join with the next word
                joined_line = line[: match.start()] + match.group(1) + next_words[0]
                remainder = next_words[1] if len(next_words) > 1 else ""
                result.append(joined_line)
                # Replace the next line with the remainder (may be empty)
                if remainder:
                    lines[i + 1] = remainder
                else:
                    i += 1  # skip the consumed next line
            else:
                result.append(line)
        else:
            result.append(line)
        i += 1

    return result


def _normalize_whitespace(lines: list[str]) -> str:
    """Collapse inline whitespace and reduce multiple blank lines to one."""
    cleaned: list[str] = []
    prev_blank = False
    for line in lines:
        # Collapse horizontal whitespace
        normalized = " ".join(line.split())
        if not normalized:
            if not prev_blank:
                cleaned.append("")
            prev_blank = True
        else:
            cleaned.append(normalized)
            prev_blank = False

    # Strip leading/trailing blank lines from result
    text = "\n".join(cleaned).strip()
    return text
