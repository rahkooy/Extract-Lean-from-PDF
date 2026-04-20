#!/usr/bin/env python3
"""
Extract Lean declarations from a mathematics article PDF.

Dependencies:
    python -m pip install pymupdf requests beautifulsoup4

Example:
    python extract_lean_items_from_pdf.py https://afm.episciences.org/15978 \
        --out lean_items.json --csv lean_items.csv --markdown lean_items.md
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


LEAN_DECL_START = re.compile(
    r"""
    ^\s*
    (?:@\[[^\]]+\]\s*)*
    (?:(?:noncomputable|private|protected|unsafe|partial)\s+)*
    (?P<keyword>def|lemma|theorem|structure|class|abbrev|instance|inductive)\s+
    (?P<name>
        «[^»]+»
        |
        [A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)*
    )
    """,
    re.VERBOSE,
)

DEFINITION_KEYWORDS = {"def", "structure", "class", "abbrev", "instance", "inductive"}
LEAN_OPERATORS = (
    ":=", "=>", "↔", "∀", "∃", "→", "⟶", "←", "↦", "≃", "≅", "⊢",
    "⟦", "⟧", "⟨", "⟩", "·", "by", "where", "fun", "match", "with",
)
SECTION_HEADING = re.compile(r"^\d+(?:\.\d+)*\s+[A-Z].{0,120}$")


@dataclass
class PdfLine:
    idx: int
    page: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    height: float
    page_width: float
    page_height: float
    mono_frac: float
    fonts: list[str]


@dataclass
class LeanRecord:
    kind: str
    lean_keyword: str
    name: str
    page: int
    context: str
    lean_code: str
    abbreviated: bool


def resolve_pdf_url(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 lean-pdf-extractor/1.0"}
    r = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
    r.raise_for_status()

    ctype = r.headers.get("content-type", "").lower()
    if "application/pdf" in ctype or r.url.lower().endswith((".pdf", "/pdf")):
        return r.url

    soup = BeautifulSoup(r.text, "html.parser")
    candidates = []

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

    return candidates[0]


def download_pdf(pdf_url: str, out_path: Path) -> Path:
    headers = {"User-Agent": "Mozilla/5.0 lean-pdf-extractor/1.0"}
    with requests.get(pdf_url, headers=headers, timeout=60, stream=True) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)
    return out_path


def font_is_mono(font_name: str) -> bool:
    f = font_name.lower()
    return any(token in f for token in ("mono", "typewriter", "courier", "cmtt", "lmmono"))


def clean_text(s: str) -> str:
    return (
        s.replace("\u00ad", "")
        .replace("\ufb00", "ff")
        .replace("\ufb01", "fi")
        .replace("\ufb02", "fl")
        .replace("\ufb03", "ffi")
        .replace("\ufb04", "ffl")
        .rstrip()
    )


def extract_pdf_lines(pdf_path: Path) -> list[PdfLine]:
    doc = fitz.open(str(pdf_path))
    lines = []
    idx = 0

    for pno, page in enumerate(doc, start=1):
        page_width, page_height = page.rect.width, page.rect.height
        text_dict = page.get_text("dict", sort=True)

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue

            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = clean_text("".join(span.get("text", "") for span in spans))
                if not text.strip():
                    continue

                total_chars = sum(len(span.get("text", "")) for span in spans) or 1
                mono_chars = sum(
                    len(span.get("text", ""))
                    for span in spans
                    if font_is_mono(span.get("font", ""))
                )

                fonts = sorted({span.get("font", "") for span in spans if span.get("font", "")})
                x0, y0, x1, y1 = line.get("bbox", block.get("bbox", (0, 0, 0, 0)))

                lines.append(
                    PdfLine(
                        idx=idx,
                        page=pno,
                        text=text,
                        x0=float(x0),
                        y0=float(y0),
                        x1=float(x1),
                        y1=float(y1),
                        height=float(y1 - y0),
                        page_width=float(page_width),
                        page_height=float(page_height),
                        mono_frac=mono_chars / total_chars,
                        fonts=fonts,
                    )
                )
                idx += 1

    return lines


def is_header_footer(line: PdfLine) -> bool:
    s = line.text.strip()
    if re.fullmatch(r"\d+", s):
        return True
    if line.y0 < 0.045 * line.page_height:
        return True
    if line.y1 > 0.955 * line.page_height:
        return True
    return False


def lean_decl_match(text: str):
    return LEAN_DECL_START.match(text.strip())


def is_codeish(line: PdfLine) -> bool:
    s = line.text.strip()

    if not s or is_header_footer(line):
        return False

    if lean_decl_match(s):
        return True

    if s.startswith("@["):
        return True

    if line.mono_frac >= 0.45:
        return True

    if re.match(r"^(where|by|exact|have|let|intro|rw|simp|rfl|constructor)\b", s):
        return True

    token_hits = sum(1 for t in LEAN_OPERATORS if t in s)
    return token_hits >= 2 and len(s) < 180


def group_code_regions(lines: list[PdfLine]) -> list[list[PdfLine]]:
    regions = []
    current = []

    def close_current():
        nonlocal current
        if current:
            regions.append(current)
            current = []

    for line in lines:
        if not is_codeish(line):
            close_current()
            continue

        if not current:
            current = [line]
            continue

        prev = current[-1]
        same_page_close = (
            line.page == prev.page
            and (line.y0 - prev.y1) <= max(18.0, 2.2 * max(prev.height, 1.0))
        )

        if same_page_close:
            current.append(line)
        else:
            close_current()
            current = [line]

    close_current()
    return regions


def context_before(lines: list[PdfLine], start_idx: int, max_chars: int = 700) -> str:
    collected = []
    chars = 0
    j = start_idx - 1

    while j >= 0 and (is_codeish(lines[j]) or is_header_footer(lines[j])):
        j -= 1

    while j >= 0:
        line = lines[j]
        s = line.text.strip()

        if not s or is_header_footer(line):
            j -= 1
            continue

        if is_codeish(line) and collected:
            break

        if SECTION_HEADING.match(s) and collected:
            break

        collected.append(s)
        chars += len(s) + 1

        if chars >= max_chars:
            break

        j -= 1

    text = " ".join(reversed(collected))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def code_from_lines(region: Iterable[PdfLine]) -> str:
    text = "\n".join(line.text.rstrip() for line in region).strip()
    text = re.sub(r"\.\s*\.\s*\.", "...", text)
    return text


def split_region_into_declarations(region: list[PdfLine], all_lines: list[PdfLine]) -> list[LeanRecord]:
    starts = []

    for i, line in enumerate(region):
        if lean_decl_match(line.text):
            k = i
            while k > 0 and region[k - 1].text.strip().startswith("@["):
                k -= 1
            if not starts or starts[-1] != k:
                starts.append(k)

    records = []

    for pos, start in enumerate(starts):
        end = starts[pos + 1] if pos + 1 < len(starts) else len(region)
        segment = region[start:end]

        decl_line = next((l for l in segment if lean_decl_match(l.text)), None)
        if decl_line is None:
            continue

        m = lean_decl_match(decl_line.text)
        keyword = m.group("keyword")
        name = m.group("name")

        kind = "definition" if keyword in DEFINITION_KEYWORDS else keyword
        code = code_from_lines(segment)

        records.append(
            LeanRecord(
                kind=kind,
                lean_keyword=keyword,
                name=name,
                page=decl_line.page,
                context=context_before(all_lines, decl_line.idx),
                lean_code=code,
                abbreviated=bool(re.search(r"…|\.\.\.", code)),
            )
        )

    return records


def extract_lean_records(pdf_path: Path) -> list[LeanRecord]:
    lines = extract_pdf_lines(pdf_path)
    regions = group_code_regions(lines)

    records = []
    for region in regions:
        records.extend(split_region_into_declarations(region, lines))

    seen = set()
    unique = []

    for r in records:
        key = (r.kind, r.name, r.page, r.lean_code)
        if key not in seen:
            unique.append(r)
            seen.add(key)

    return unique


def grouped_json(records: list[LeanRecord]) -> dict[str, list[dict]]:
    groups = {"definitions": [], "lemmas": [], "theorems": []}

    for r in records:
        item = asdict(r)
        if r.kind == "definition":
            groups["definitions"].append(item)
        elif r.kind == "lemma":
            groups["lemmas"].append(item)
        elif r.kind == "theorem":
            groups["theorems"].append(item)

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
            fieldnames=["kind", "lean_keyword", "name", "page", "context", "lean_code", "abbreviated"],
        )
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))


def write_markdown(records: list[LeanRecord], path: Path) -> None:
    groups = grouped_json(records)
    lines = ["# Lean declarations extracted from PDF", ""]

    for title, items in (
        ("Definitions", groups["definitions"]),
        ("Lemmas", groups["lemmas"]),
        ("Theorems", groups["theorems"]),
    ):
        lines.append(f"## {title}")
        lines.append("")

        if not items:
            lines.append("_None found._")
            lines.append("")
            continue

        for item in items:
            lines.append(f"### `{item['name']}` — page {item['page']}")

            if item.get("context"):
                lines.append("")
                lines.append(f"**Context.** {item['context']}")

            if item.get("abbreviated"):
                lines.append("")
                lines.append("**Note.** The PDF appears to abbreviate this Lean code with an ellipsis.")

            lines.append("")
            lines.append("```lean")
            lines.append(item["lean_code"])
            lines.append("```")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract Lean definitions, lemmas, and theorems from an article PDF."
    )
    parser.add_argument("url", help="Direct PDF URL, or article page URL containing a PDF link")
    parser.add_argument("--out", default="lean_items.json", help="JSON output path")
    parser.add_argument("--csv", default=None, help="Optional CSV output path")
    parser.add_argument("--markdown", default=None, help="Optional Markdown output path")
    parser.add_argument("--save-pdf", default=None, help="Optional path to save the downloaded PDF")
    args = parser.parse_args()

    pdf_url = resolve_pdf_url(args.url)

    if args.save_pdf:
        pdf_path = download_pdf(pdf_url, Path(args.save_pdf))
    else:
        tmpdir = tempfile.TemporaryDirectory()
        pdf_path = download_pdf(pdf_url, Path(tmpdir.name) / "article.pdf")

    records = extract_lean_records(pdf_path)

    write_json(records, Path(args.out))

    if args.csv:
        write_csv(records, Path(args.csv))

    if args.markdown:
        write_markdown(records, Path(args.markdown))

    counts = grouped_json(records)

    print(f"PDF: {pdf_url}")
    print(f"Wrote {args.out}")

    if args.csv:
        print(f"Wrote {args.csv}")

    if args.markdown:
        print(f"Wrote {args.markdown}")

    print(
        "Found "
        f"{len(counts['definitions'])} definitions, "
        f"{len(counts['lemmas'])} lemmas, "
        f"{len(counts['theorems'])} theorems."
    )
    print("Items marked abbreviated=true contain an ellipsis in the PDF snippet.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
