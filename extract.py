#!/usr/bin/env python3
"""
Extract displayed Lean declarations from a mathematics article PDF and pair
each declaration with the nearest preceding mathematical prose.

This version is designed for articles where Lean code is syntax-highlighted in
the PDF. It avoids most inline-code false positives by extracting only visually
displayed Lean blocks, then extending each block until the displayed code ends.

Dependencies:
    python -m pip install pymupdf requests beautifulsoup4

Examples:
    # Article page containing a PDF link
    python extract.py https://afm.episciences.org/15978 \
        --out lean_items.json --csv lean_items.csv --markdown lean_items.md

    # Direct PDF or local PDF
    python extract.py article.pdf --out lean_items.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup


# Top-level Lean declaration starts. Attributes are normally extracted as
# separate lines immediately above the declaration, but the regex also accepts
# one-line attributes for robustness.
LEAN_DECL_START = re.compile(
    r"""
    ^\s*
    (?:@\[[^\]]+\]\s*)*
    (?:(?:noncomputable|private|protected|unsafe|partial|mutual)\s+)*
    (?P<keyword>
        def|lemma|theorem|structure|class|abbrev|instance|inductive|
        axiom|constant|example
    )
    (?:\s+|$)
    (?P<name>
        «[^»]+»
        |
        [A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)*
    )?
    """,
    re.VERBOSE,
)

DEFINITION_KEYWORDS = {
    "def", "structure", "class", "abbrev", "instance", "inductive", "axiom", "constant"
}

# Strings that often indicate a Lean signature/body rather than prose.
LEAN_TOKENS = (
    ":=", "=>", "↔", "∀", "∃", "→", "⟶", "←", "↦", "≃", "≅", "⊢",
    "⟦", "⟧", "⟨", "⟩", "·", "by", "where", "fun", "match", "with",
    " := ", " : ", " = ", " // ", " | ", "⟪", "⟫", "≫", "⤳"
)

LEAN_CONTINUATION_START = re.compile(
    r"""
    ^\s*
    (
        where|by|exact|have|let|show|intro|intros|refine|apply|simp|rw|rfl|
        constructor|aesop|omega|linarith|ring|norm_num|cases|induction|
        ·|\||\{|\}|\(|\)|\[|\]|,|\.\.\.|…
    )
    (\s|$)
    """,
    re.VERBOSE,
)

SECTION_HEADING = re.compile(r"^\d+(?:\.\d+)*\s+.{1,120}$")
PARAGRAPH_LABEL = re.compile(r"^\d+(?:\.\d+)+\s+")
FOOTNOTE_START = re.compile(r"^\d+[A-Z]?\w")


@dataclass
class PdfLine:
    idx: int
    page: int                  # 1-based page number
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    height: float
    page_width: float
    page_height: float
    mono_frac: float           # fraction of characters in monospaced font
    avg_size: float
    max_size: float
    fonts: list[str]


@dataclass
class LeanRecord:
    kind: str                  # definition | lemma | theorem | example
    lean_keyword: str          # def | lemma | theorem | structure | ...
    name: str
    page_start: int
    page_end: int
    math_text: str             # nearest preceding mathematical prose
    lean_context_before: str   # nearby Lean setup lines such as variable/universe, if any
    lean_code: str             # whole displayed code piece: setup lines + declaration body
    declaration_code: str      # declaration body only
    abbreviated: bool          # True if code contains "..." or "…"


def resolve_pdf_url(url: str) -> str:
    """Accept either a direct PDF URL or an article page containing a PDF link."""
    headers = {"User-Agent": "Mozilla/5.0 lean-pdf-extractor/2.0"}
    r = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
    r.raise_for_status()

    ctype = r.headers.get("content-type", "").lower()
    if "application/pdf" in ctype or r.url.lower().endswith((".pdf", "/pdf")):
        return r.url

    soup = BeautifulSoup(r.text, "html.parser")
    candidates: list[str] = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        label = a.get_text(" ", strip=True).lower()
        href_l = href.lower()
        if (
            href_l.endswith(".pdf")
            or href_l.endswith("/pdf")
            or "/pdf" in href_l
            or "download article" in label
            or label == "pdf"
        ):
            candidates.append(urljoin(r.url, href))

    if not candidates:
        raise RuntimeError(f"Could not find a PDF link in {url!r}")

    # Prefer candidates that really return a PDF.
    for candidate in candidates:
        try:
            rr = requests.get(
                candidate,
                headers=headers,
                timeout=30,
                stream=True,
                allow_redirects=True,
            )
            rr.raise_for_status()
            ctype = rr.headers.get("content-type", "").lower()
            if "application/pdf" in ctype or rr.url.lower().endswith((".pdf", "/pdf")):
                return rr.url
        except requests.RequestException:
            continue

    return candidates[0]


def download_pdf(pdf_url: str, out_path: Path) -> Path:
    headers = {"User-Agent": "Mozilla/5.0 lean-pdf-extractor/2.0"}
    with requests.get(pdf_url, headers=headers, timeout=60, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)
    return out_path


def font_is_mono(font_name: str) -> bool:
    f = font_name.lower()
    return any(
        token in f
        for token in (
            "mono", "typewriter", "courier", "cmtt", "lmmono",
            "berasansmono", "inconsolata", "dejavusansmono"
        )
    )


def clean_text(s: str) -> str:
    """Clean common PDF extraction artifacts while preserving Lean Unicode."""
    return (
        s.replace("\u00ad", "")
        .replace("\ufb00", "ff")
        .replace("\ufb01", "fi")
        .replace("\ufb02", "fl")
        .replace("\ufb03", "ffi")
        .replace("\ufb04", "ffl")
        .replace("\u2212", "−")
        .rstrip()
    )


def line_from_pymupdf(
    idx: int,
    page_number: int,
    line: dict,
    page_width: float,
    page_height: float,
) -> PdfLine | None:
    spans = line.get("spans", [])
    text = clean_text("".join(span.get("text", "") for span in spans))
    if not text.strip():
        return None

    total_chars = sum(len(span.get("text", "")) for span in spans) or 1
    mono_chars = 0
    weighted_size = 0.0
    max_size = 0.0
    fonts = []

    for span in spans:
        span_text = span.get("text", "")
        n = len(span_text)
        font = span.get("font", "")
        size = float(span.get("size", 0.0))
        fonts.append(font)
        weighted_size += n * size
        max_size = max(max_size, size)
        if font_is_mono(font):
            mono_chars += n

    x0, y0, x1, y1 = line.get("bbox", (0, 0, 0, 0))

    return PdfLine(
        idx=idx,
        page=page_number,
        text=text,
        x0=float(x0),
        y0=float(y0),
        x1=float(x1),
        y1=float(y1),
        height=float(y1 - y0),
        page_width=float(page_width),
        page_height=float(page_height),
        mono_frac=mono_chars / total_chars,
        avg_size=weighted_size / total_chars,
        max_size=max_size,
        fonts=sorted(set(f for f in fonts if f)),
    )


def extract_pdf_lines(pdf_path: Path) -> list[PdfLine]:
    """Extract all visible text lines with geometry and font information."""
    doc = fitz.open(str(pdf_path))
    lines: list[PdfLine] = []
    idx = 0

    for pno, page in enumerate(doc, start=1):
        page_width, page_height = page.rect.width, page.rect.height
        text_dict = page.get_text("dict", sort=True)

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue

            for line in block.get("lines", []):
                pdf_line = line_from_pymupdf(idx, pno, line, page_width, page_height)
                if pdf_line is not None:
                    lines.append(pdf_line)
                    idx += 1

    return lines


def is_header_footer(line: PdfLine) -> bool:
    s = line.text.strip()

    if re.fullmatch(r"\d+", s):
        return True

    # Page headers and page numbers.
    if line.y0 < 0.045 * line.page_height:
        return True
    if line.y1 > 0.965 * line.page_height:
        return True

    return False


def is_probable_footnote(line: PdfLine) -> bool:
    """Footnotes are prose, but should not be used as mathematical context."""
    s = line.text.strip()
    if line.y0 > 0.82 * line.page_height and line.avg_size < 10.4:
        return True
    if line.y0 > 0.78 * line.page_height and line.max_size < 10.8 and FOOTNOTE_START.match(s):
        return True
    return False


def lean_decl_match(text: str) -> re.Match[str] | None:
    return LEAN_DECL_START.match(text.strip())


def declaration_start_looks_structural(text: str, match: re.Match[str]) -> bool:
    """Reject prose such as 'structure ShortComplex C which bundles ...'."""
    s = text.strip()
    keyword = match.group("keyword")
    name = match.group("name")

    if keyword == "example":
        return True

    if not name:
        return False

    # Look at the first non-space character after the matched prefix.
    rest = s[match.end():].lstrip()
    if not rest:
        return True

    # Very common after a declaration name in Lean.
    if rest.startswith(("[", "(", "{", ":", ":=", "where", "extends")):
        return True

    # If the line is strongly monospaced we allow the common Lean form
    # 'def f x y := ...'. The caller checks the monospaced condition.
    return False


def has_lean_token(s: str) -> bool:
    return any(tok in s for tok in LEAN_TOKENS)


def is_displayed_declaration_start(line: PdfLine) -> bool:
    """True only for displayed Lean declaration starts, not inline/prose mentions."""
    if is_header_footer(line) or is_probable_footnote(line):
        return False

    s = line.text.strip()
    m = lean_decl_match(s)
    if not m:
        return False

    structural = declaration_start_looks_structural(s, m)

    # Strong monospaced evidence: accept most Lean declarations.
    if line.mono_frac >= 0.50 and line.avg_size <= 11.2:
        return True

    # Weaker monospaced evidence requires structural Lean syntax.
    if structural and line.mono_frac >= 0.35 and line.avg_size <= 11.2:
        return True

    # Some syntax-highlighted code has math symbols in non-mono fonts.
    if structural and line.avg_size <= 10.8 and has_lean_token(s):
        return True

    return False


def is_attribute_line(line: PdfLine) -> bool:
    s = line.text.strip()
    return s.startswith("@[") and line.avg_size <= 11.2 and line.mono_frac >= 0.35


def looks_like_prose(line: PdfLine) -> bool:
    s = line.text.strip()
    words = re.findall(r"[A-Za-z]{3,}", s)
    leanish = has_lean_token(s) or LEAN_CONTINUATION_START.match(s) is not None

    if line.avg_size >= 11.0 and line.mono_frac < 0.20 and len(words) >= 4 and not leanish:
        return True

    # A long full-width line with little monospaced material is almost always prose.
    if (
        len(s) > 60
        and line.x1 > 0.82 * line.page_width
        and line.mono_frac < 0.25
        and line.avg_size >= 10.8
        and len(words) >= 5
    ):
        return True

    return False


def is_code_continuation(line: PdfLine, first_line: PdfLine, previous: PdfLine) -> bool:
    """Decide whether a line after a declaration start is still part of the displayed code."""
    if is_header_footer(line) or is_probable_footnote(line):
        return False

    s = line.text.strip()
    if not s:
        return False

    # New declaration starts are handled separately by the caller.
    if is_displayed_declaration_start(line):
        return False

    # Page-continuation geometry is handled by caller; here we only inspect content.
    near_code_indent = (
        first_line.x0 - 8 <= line.x0 <= first_line.x0 + 90
        or previous.x0 - 8 <= line.x0 <= previous.x0 + 90
    )

    if s in {"...", "....", "…", ". . ."}:
        return near_code_indent

    # Lean doc-comments/comments in printed code often use a non-monospace
    # font after PDF extraction, but they are still part of a declaration body.
    if near_code_indent and line.avg_size <= 10.8 and (
        s.startswith("/--")
        or s.startswith("/-")
        or s.startswith("/−−")
        or s.startswith("/−")
        or s.startswith("--")
        or s.startswith("−−")
    ):
        return True

    if looks_like_prose(line):
        return False

    # Displayed code in the sample PDFs is noticeably smaller than prose. A
    # prose line can still contain many monospaced inline identifiers, so do
    # not rely on mono_frac alone unless the average font size is code-sized.
    if line.mono_frac >= 0.30 and line.avg_size <= 10.6 and near_code_indent:
        return True

    if line.avg_size <= 10.6 and near_code_indent and has_lean_token(s):
        return True

    if LEAN_CONTINUATION_START.match(s) and near_code_indent and line.avg_size <= 10.8:
        return True

    # Lines consisting mostly of mathematical/Lean punctuation, variables, and symbols.
    non_letters = sum(1 for ch in s if not ch.isalpha())
    if near_code_indent and line.avg_size <= 10.6 and non_letters / max(len(s), 1) > 0.35:
        return True

    return False


def same_visual_code_block(next_line: PdfLine, previous: PdfLine) -> bool:
    """Geometric adjacency test for lines in one displayed code block."""
    if next_line.page == previous.page:
        gap = next_line.y0 - previous.y1
        return gap <= max(22.0, 2.8 * max(previous.height, 1.0))

    # Allow code to continue at the top of the next page.
    if next_line.page == previous.page + 1:
        return previous.y1 > 0.80 * previous.page_height and next_line.y0 < 0.18 * next_line.page_height

    return False


def code_from_lines(lines: Iterable[PdfLine]) -> str:
    text = "\n".join(line.text.rstrip() for line in lines).strip()
    text = re.sub(r"\.\s*\.\s*\.", "...", text)
    return text


def normalize_prose(lines: Iterable[PdfLine]) -> str:
    text = " ".join(line.text.strip() for line in lines if line.text.strip())
    text = re.sub(r"\s+", " ", text)
    # Fix some PDF line-break artifacts without altering mathematical content too much.
    text = text.replace(" - ", "-")
    text = text.replace(" ,", ",")
    text = text.replace(" .", ".")
    return text.strip()


def line_starts_paragraph(line: PdfLine) -> bool:
    """Heuristic for the first line of a paragraph in the article body."""
    s = line.text.strip()

    if PARAGRAPH_LABEL.match(s) or re.fullmatch(r"\d+(?:\.\d+)+", s):
        return True

    # Many TeX articles indent the first line of a paragraph by about 15-25 pt.
    return 88 <= line.x0 <= 110 and line.avg_size >= 10.8 and line.mono_frac < 0.30


def is_section_heading(line: PdfLine) -> bool:
    s = line.text.strip()

    if is_header_footer(line) or is_probable_footnote(line):
        return False

    # Examples: "2 Homology ...", "4.3.4 | The octahedron axiom".
    if SECTION_HEADING.match(s) and line.max_size >= 10.0 and line.mono_frac < 0.20:
        return True

    return False


def is_any_display_code_line(line: PdfLine) -> bool:
    """Coarse test used while collecting prose context backwards."""
    if is_displayed_declaration_start(line) or is_attribute_line(line):
        return True

    s = line.text.strip()
    if s in {"...", "....", "…", ". . ."} and line.avg_size <= 11.2:
        return True

    if line.mono_frac >= 0.35 and line.avg_size <= 10.6:
        return True

    if line.avg_size <= 10.6 and line.mono_frac >= 0.20 and has_lean_token(s):
        return True

    return False


def extract_math_text_before(
    lines: list[PdfLine],
    start_idx: int,
    max_chars: int = 1200,
) -> str:
    """
    Extract the closest preceding mathematical prose.

    It skips headers, footers, footnotes, and previous displayed Lean code. For
    consecutive declarations with no prose between them, this means each
    declaration receives the same preceding prose context.
    """
    collected: list[PdfLine] = []
    chars = 0
    j = start_idx - 1

    # Skip code immediately above the current declaration. This handles groups
    # like several one-line defs after one explanatory sentence.
    while j >= 0:
        line = lines[j]
        if is_header_footer(line) or is_probable_footnote(line) or is_any_display_code_line(line):
            j -= 1
            continue
        break

    while j >= 0:
        line = lines[j]
        s = line.text.strip()

        if not s:
            j -= 1
            continue

        if is_header_footer(line) or is_probable_footnote(line):
            j -= 1
            continue

        if is_any_display_code_line(line):
            # Stop once we have prose; otherwise skip older code.
            if collected:
                break
            j -= 1
            continue

        if is_section_heading(line):
            break

        collected.append(line)
        chars += len(s) + 1

        # Stop after the beginning of the nearest paragraph, but only after
        # collecting a meaningful amount of context.
        if line_starts_paragraph(line) and chars >= 40:
            break

        if chars >= max_chars:
            break

        j -= 1

    return normalize_prose(reversed(collected))



def is_lean_setup_line(line: PdfLine) -> bool:
    """Lines such as `universe`, `variable`, `open`, or parameter lines before a declaration."""
    if is_header_footer(line) or is_probable_footnote(line):
        return False

    s = line.text.strip()

    if not s:
        return False

    if is_displayed_declaration_start(line):
        return False

    if line.avg_size > 10.8:
        return False

    if s.startswith(("universe ", "variable ", "variables ", "open ", "namespace ", "section ", "end ")):
        return True

    if s.startswith(("/--", "/-", "/−−", "/−", "--", "−−")):
        return True

    if line.mono_frac >= 0.35:
        return True

    if has_lean_token(s) and line.mono_frac >= 0.20:
        return True

    return False


def extract_lean_setup_before(lines: list[PdfLine], decl_start_idx: int) -> list[PdfLine]:
    """
    Extract immediately preceding displayed Lean setup lines.

    This captures code such as:
        universe u
        variable {C : Type u} [Category C]

    These lines are not part of the declaration body, but they are often needed
    to understand the declaration as printed in the paper.
    """
    setup: list[PdfLine] = []
    j = decl_start_idx - 1
    current = lines[decl_start_idx]

    while j >= 0:
        candidate = lines[j]

        if is_header_footer(candidate) or is_probable_footnote(candidate):
            j -= 1
            continue

        if is_displayed_declaration_start(candidate):
            break

        if not is_lean_setup_line(candidate):
            break

        same_code_area = same_visual_code_block(current, candidate) or (
            candidate.page == current.page
            and 0 <= current.y0 - candidate.y1 <= 32.0
            and abs(candidate.x0 - current.x0) <= 35.0
        )

        if not same_code_area:
            break

        setup.append(candidate)
        current = candidate
        j -= 1

    return list(reversed(setup))


def make_record(
    body: list[PdfLine],
    lines: list[PdfLine],
    context_chars: int,
) -> LeanRecord | None:
    decl_line = next((l for l in body if is_displayed_declaration_start(l)), None)
    if decl_line is None:
        return None

    m = lean_decl_match(decl_line.text)
    if m is None:
        return None

    keyword = m.group("keyword")
    name = m.group("name") or "<anonymous>"
    kind = "definition" if keyword in DEFINITION_KEYWORDS else keyword

    setup_lines = extract_lean_setup_before(lines, decl_line.idx)
    lean_context_before = code_from_lines(setup_lines) if setup_lines else ""
    declaration_code = code_from_lines(body)
    all_code_lines = setup_lines + body
    lean_code = code_from_lines(all_code_lines)

    return LeanRecord(
        kind=kind,
        lean_keyword=keyword,
        name=name,
        page_start=all_code_lines[0].page,
        page_end=all_code_lines[-1].page,
        math_text=extract_math_text_before(lines, decl_line.idx, max_chars=context_chars),
        lean_context_before=lean_context_before,
        lean_code=lean_code,
        declaration_code=declaration_code,
        abbreviated=bool(re.search(r"…|\.\.\.", lean_code)),
    )


def extract_lean_records(
    pdf_path: Path,
    context_chars: int = 1200,
) -> list[LeanRecord]:
    lines = extract_pdf_lines(pdf_path)
    records: list[LeanRecord] = []

    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        if not is_displayed_declaration_start(line):
            i += 1
            continue

        # Include attribute lines immediately above the declaration.
        start_i = i
        while start_i > 0 and is_attribute_line(lines[start_i - 1]) and same_visual_code_block(line, lines[start_i - 1]):
            start_i -= 1

        body = lines[start_i:i + 1]
        previous = line
        j = i + 1

        while j < n:
            candidate = lines[j]

            # The next top-level declaration starts a new piece.
            if is_displayed_declaration_start(candidate):
                break

            if not same_visual_code_block(candidate, previous):
                break

            if not is_code_continuation(candidate, line, previous):
                break

            body.append(candidate)
            previous = candidate
            j += 1

        record = make_record(body, lines, context_chars)
        if record is not None:
            records.append(record)

        # Continue at the next unconsumed line. If we stopped because of a new
        # declaration, j points to it and it will be processed next.
        i = max(j, i + 1)

    # Deduplicate repeated extractions.
    seen: set[tuple[str, str, int, str]] = set()
    unique: list[LeanRecord] = []
    for r in records:
        key = (r.lean_keyword, r.name, r.page_start, r.lean_code)
        if key not in seen:
            unique.append(r)
            seen.add(key)

    return unique


def grouped_json(records: list[LeanRecord]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {
        "definitions": [],
        "lemmas": [],
        "theorems": [],
        "examples": [],
        "other_declarations": [],
    }

    for r in records:
        item = asdict(r)
        if r.kind == "definition":
            groups["definitions"].append(item)
        elif r.kind == "lemma":
            groups["lemmas"].append(item)
        elif r.kind == "theorem":
            groups["theorems"].append(item)
        elif r.kind == "example":
            groups["examples"].append(item)
        else:
            groups["other_declarations"].append(item)

    return groups


def write_json(records: list[LeanRecord], path: Path) -> None:
    path.write_text(
        json.dumps(grouped_json(records), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(records: list[LeanRecord], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "kind",
                "lean_keyword",
                "name",
                "page_start",
                "page_end",
                "math_text",
                "lean_context_before",
                "lean_code",
                "declaration_code",
                "abbreviated",
            ],
        )
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))


def write_markdown(records: list[LeanRecord], path: Path) -> None:
    groups = grouped_json(records)
    lines: list[str] = ["# Lean declarations extracted from PDF", ""]

    sections = (
        ("Definitions", groups["definitions"]),
        ("Lemmas", groups["lemmas"]),
        ("Theorems", groups["theorems"]),
        ("Examples", groups["examples"]),
        ("Other declarations", groups["other_declarations"]),
    )

    for title, items in sections:
        lines.append(f"## {title}")
        lines.append("")

        if not items:
            lines.append("_None found._")
            lines.append("")
            continue

        for item in items:
            page = (
                str(item["page_start"])
                if item["page_start"] == item["page_end"]
                else f"{item['page_start']}–{item['page_end']}"
            )
            lines.append(f"### `{item['name']}` — page {page}")

            if item.get("math_text"):
                lines.append("")
                lines.append("**Mathematical text.**")
                lines.append("")
                lines.append(item["math_text"])

            if item.get("abbreviated"):
                lines.append("")
                lines.append(
                    "**Note.** The PDF appears to abbreviate this Lean code with an ellipsis; "
                    "the omitted body cannot be recovered from the PDF alone."
                )

            lines.append("")
            lines.append("```lean")
            lines.append(item["lean_code"])
            lines.append("```")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def get_pdf_path(input_arg: str, save_pdf: str | None) -> tuple[Path, str]:
    """
    Return (local_pdf_path, source_description).

    input_arg may be a local path, direct PDF URL, or article page URL.
    """
    candidate = Path(input_arg).expanduser()
    if candidate.exists():
        return candidate, str(candidate)

    pdf_url = resolve_pdf_url(input_arg)

    if save_pdf:
        pdf_path = download_pdf(pdf_url, Path(save_pdf))
        return pdf_path, pdf_url

    # Use a real temporary directory path. Do not attach custom attributes to
    # pathlib.Path objects; PosixPath/WindowsPath do not support that.
    tmpdir = Path(tempfile.mkdtemp(prefix="lean_pdf_extract_"))
    pdf_path = tmpdir / "article.pdf"
    download_pdf(pdf_url, pdf_path)
    return pdf_path, pdf_url


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract displayed Lean code blocks and corresponding mathematics text from a PDF."
    )
    parser.add_argument("input", help="Local PDF path, direct PDF URL, or article page URL containing a PDF link")
    parser.add_argument("--out", default="lean_items.json", help="JSON output path")
    parser.add_argument("--csv", default=None, help="Optional CSV output path")
    parser.add_argument("--markdown", default=None, help="Optional Markdown output path")
    parser.add_argument("--save-pdf", default=None, help="Optional path to save the downloaded PDF")
    parser.add_argument(
        "--context-chars",
        type=int,
        default=1200,
        help="Maximum number of characters of preceding mathematical text per declaration",
    )
    args = parser.parse_args()

    pdf_path, source = get_pdf_path(args.input, args.save_pdf)
    records = extract_lean_records(pdf_path, context_chars=args.context_chars)

    write_json(records, Path(args.out))
    if args.csv:
        write_csv(records, Path(args.csv))
    if args.markdown:
        write_markdown(records, Path(args.markdown))

    counts = grouped_json(records)

    print(f"Source: {source}")
    print(f"Wrote {args.out}")
    if args.csv:
        print(f"Wrote {args.csv}")
    if args.markdown:
        print(f"Wrote {args.markdown}")

    print(
        "Found "
        f"{len(counts['definitions'])} definitions, "
        f"{len(counts['lemmas'])} lemmas, "
        f"{len(counts['theorems'])} theorems, "
        f"{len(counts['examples'])} examples, "
        f"{len(counts['other_declarations'])} other declarations."
    )
    print("Items marked abbreviated=true contain an ellipsis in the displayed PDF code.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
