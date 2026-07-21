"""Unit tests for src/ingestion/clean.py covering each cleaning step."""

from __future__ import annotations

import pytest

from ingestion.clean import clean_text, detect_repeated_lines


# ---------------------------------------------------------------------------
# detect_repeated_lines
# ---------------------------------------------------------------------------


class TestDetectRepeatedLines:
    def test_detects_header_on_all_pages(self):
        pages = [
            "ACME Corp Confidential\nFirst page content.",
            "ACME Corp Confidential\nSecond page content.",
            "ACME Corp Confidential\nThird page content.",
            "ACME Corp Confidential\nFourth page content.",
            "ACME Corp Confidential\nFifth page content.",
        ]
        repeated = detect_repeated_lines(pages)
        assert "ACME Corp Confidential" in repeated

    def test_ignores_line_below_threshold(self):
        pages = [
            "Header\nContent A",
            "Header\nContent B",
            "Different header\nContent C",
            "Different header\nContent D",
            "Different header\nContent E",
        ]
        repeated = detect_repeated_lines(pages)
        # "Header" appears on 2/5 = 40%, below 80%
        assert "Header" not in repeated

    def test_normalizes_whitespace_for_matching(self):
        pages = [
            "  ACME   Corp  \nContent",
            "ACME Corp\nContent",
            "ACME  Corp\nContent",
            "ACME Corp\nContent",
            "ACME Corp\nContent",
        ]
        repeated = detect_repeated_lines(pages)
        assert "ACME Corp" in repeated

    def test_empty_pages_returns_empty_set(self):
        assert detect_repeated_lines([]) == set()

    def test_single_page_no_repeats(self):
        # A line on a single page is 1/1 = 100% so it counts
        repeated = detect_repeated_lines(["Only line"])
        assert "Only line" in repeated


# ---------------------------------------------------------------------------
# Repeated header/footer stripping
# ---------------------------------------------------------------------------


class TestStripRepeatedHeaders:
    def test_removes_repeated_header_and_footer(self):
        repeated = {"Company Header", "Page Footer"}
        text = "Company Header\nActual content here.\nPage Footer"
        result = clean_text(text, repeated_lines=repeated)
        assert "Company Header" not in result
        assert "Page Footer" not in result
        assert "Actual content here." in result

    def test_keeps_non_repeated_lines(self):
        repeated = {"Remove Me"}
        text = "Remove Me\nKeep this\nAnd this"
        result = clean_text(text, repeated_lines=repeated)
        assert "Keep this" in result
        assert "And this" in result


# ---------------------------------------------------------------------------
# Page-number-only line removal
# ---------------------------------------------------------------------------


class TestRemovePageNumberLines:
    @pytest.mark.parametrize(
        "page_num_line",
        [
            "1",
            "  42  ",
            "page 7",
            "Page 12",
            "3 of 10",
            "3/10",
            "- 5 -",
            "— 12 —",
            "– 99 –",
        ],
    )
    def test_removes_page_number_variants(self, page_num_line: str):
        text = f"Real content\n{page_num_line}\nMore content"
        result = clean_text(text)
        # The page number line should be gone; content remains
        assert "Real content" in result
        assert "More content" in result
        lines = result.splitlines()
        for line in lines:
            stripped = line.strip()
            if stripped:
                assert stripped not in (page_num_line.strip(),)

    def test_keeps_numbers_in_context(self):
        text = "There are 42 items in the list."
        result = clean_text(text)
        assert "42" in result


# ---------------------------------------------------------------------------
# Whitespace normalization
# ---------------------------------------------------------------------------


class TestNormalizeWhitespace:
    def test_collapses_multiple_spaces(self):
        text = "Hello    world   foo"
        result = clean_text(text)
        assert result == "Hello world foo"

    def test_collapses_tabs(self):
        text = "col1\t\tcol2\t\tcol3"
        result = clean_text(text)
        assert result == "col1 col2 col3"

    def test_reduces_multiple_blank_lines(self):
        text = "Paragraph one.\n\n\n\n\nParagraph two."
        result = clean_text(text)
        assert result == "Paragraph one.\n\nParagraph two."

    def test_strips_leading_trailing_blanks(self):
        text = "\n\n  Content here  \n\n"
        result = clean_text(text)
        assert result == "Content here"

    def test_empty_input(self):
        assert clean_text("") == ""
        assert clean_text("   \n\n  ") == ""


# ---------------------------------------------------------------------------
# De-hyphenation
# ---------------------------------------------------------------------------


class TestDehyphenation:
    def test_rejoins_hyphenated_word(self):
        text = "The quick brown fox jumped over the laz-\ny dog."
        result = clean_text(text)
        assert "lazy" in result
        assert "laz-" not in result

    def test_preserves_intentional_hyphens(self):
        # Hyphen mid-line (not at end) should stay
        text = "well-known fact"
        result = clean_text(text)
        assert "well-known" in result

    def test_handles_multiple_hyphenations(self):
        text = "This is a com-\nplex sen-\ntence structure."
        result = clean_text(text)
        assert "complex" in result
        assert "sentence" in result

    def test_no_dehyphenation_at_last_line(self):
        # Trailing hyphen on last line has nothing to join with
        text = "some word-"
        result = clean_text(text)
        assert "word-" in result


# ---------------------------------------------------------------------------
# Integration: all steps together
# ---------------------------------------------------------------------------


class TestCleanTextIntegration:
    def test_full_pipeline(self):
        repeated = {"CONFIDENTIAL DRAFT"}
        text = (
            "CONFIDENTIAL DRAFT\n"
            "  This   is  the intro-\n"
            "duction to our   paper.\n"
            "\n"
            "\n"
            "\n"
            "Page 3\n"
            "CONFIDENTIAL DRAFT\n"
            "Second paragraph   here."
        )
        result = clean_text(text, repeated_lines=repeated)

        assert "CONFIDENTIAL DRAFT" not in result
        assert "introduction" in result
        assert "Page 3" not in result
        # Multiple blanks collapsed
        assert "\n\n\n" not in result
        # Whitespace normalized
        assert "  " not in result
        assert "Second paragraph here." in result
