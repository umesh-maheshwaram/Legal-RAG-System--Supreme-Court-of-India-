"""
Supreme Court Judgments — Data Preprocessing & Text Chunking Pipeline
=====================================================================

Strategies Applied:
───────────────────
1. PDF TEXT EXTRACTION (PyMuPDF / fitz)
   - Chosen over pdfplumber/PyPDF2 for speed and accuracy on scanned+digital PDFs.
   - Extracts text page-by-page, preserving reading order.

2. TEXT CLEANING
   - Removes page numbers, repeated headers/footers, and legal boilerplate lines.
   - Normalises whitespace (collapses multiple spaces/newlines).
   - Strips non-printable / control characters.
   - Preserves paragraph boundaries (double newline → single separator).

3. METADATA EXTRACTION (from filename)
   - Parses case title, parties (petitioner vs respondent), judgment date, and year
     directly from the structured filename convention:
       <Petitioner>_vs_<Respondent>_on_<Date>_<Seq>.PDF

4. TEXT CHUNKING — Recursive Character Splitting with Overlap
   - Primary split on paragraph boundaries ("\n\n").
   - Secondary split on sentence boundaries (". ", "? ", "! ").
   - Tertiary split on word boundaries (" ").
   - chunk_size  = 1000 characters  (fits ~250 tokens for most embedding models).
   - chunk_overlap = 200 characters  (preserves cross-boundary context).
   - Each chunk is enriched with source metadata (case name, year, filename, page range).

5. OUTPUT
   - Writes a single JSONL file (one JSON object per chunk) for easy downstream
     ingestion into vector stores (Qdrant).

Usage:
    python preprocess_and_chunk.py
"""

import os
import re
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from tqdm import tqdm

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent          # supreme_court_judgments/
OUTPUT_FILE = ROOT_DIR / "chunks.jsonl"             # output
LOG_FILE = ROOT_DIR / "preprocess.log"

CHUNK_SIZE = 1200       # characters per chunk
CHUNK_OVERLAP = 200     # overlap between consecutive chunks
MIN_CHUNK_LENGTH = 50   # discard very small trailing chunks

YEAR_RANGE = range(1950, 2026)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────
@dataclass
class CaseMetadata:
    filename: str
    year: int
    case_title: str = ""
    petitioner: str = ""
    respondent: str = ""
    judgment_date: str = ""


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)


# ──────────────────────────────────────────────
# 1. Metadata extraction from filename
# ──────────────────────────────────────────────
def parse_filename(filename: str, year: int) -> CaseMetadata:
    """Extract structured metadata from the PDF filename."""
    meta = CaseMetadata(filename=filename, year=year)
    name = Path(filename).stem  # strip .PDF

    # Remove trailing sequence number (e.g. "_1")
    name = re.sub(r"_\d+$", "", name)

    # Split on "_vs_" to get petitioner / respondent + date
    vs_match = re.split(r"_vs_", name, maxsplit=1, flags=re.IGNORECASE)
    if len(vs_match) == 2:
        meta.petitioner = vs_match[0].replace("_", " ").strip()
        remainder = vs_match[1]

        # Extract date portion: "_on_<day>_<Month>_<year>"
        date_match = re.search(
            r"_on_(\d{1,2}_\w+_\d{4})$", remainder, re.IGNORECASE
        )
        if date_match:
            meta.judgment_date = date_match.group(1).replace("_", " ")
            meta.respondent = remainder[: date_match.start()].replace("_", " ").strip()
        else:
            meta.respondent = remainder.replace("_", " ").strip()

        meta.case_title = f"{meta.petitioner} vs {meta.respondent}"
    else:
        meta.case_title = name.replace("_", " ").strip()

    return meta


# ──────────────────────────────────────────────
# 2. PDF text extraction
# ──────────────────────────────────────────────
def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """Extract all text from a PDF using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        pages = []
        for page in doc:
            text = page.get_text("text")
            if text:
                pages.append(text)
        doc.close()
        return "\n\n".join(pages) if pages else None
    except Exception as e:
        logger.error("Failed to extract text from %s: %s", pdf_path, e)
        return None


# ──────────────────────────────────────────────
# 3. Text cleaning
# ──────────────────────────────────────────────

# Patterns for common Supreme Court PDF noise
_PAGE_NUMBER_RE = re.compile(
    r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$", re.MULTILINE
)
_HEADER_FOOTER_RE = re.compile(
    r"(?:SUPREME COURT OF INDIA|REPORTABLE|NON[- ]?REPORTABLE|"
    r"Page \d+ of \d+|www\.judis\.nic\.in|"
    r"Digitally signed by|ITEM NO\.\s*\d+)",
    re.IGNORECASE,
)
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_MULTI_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINES_RE = re.compile(r"\n{3,}")


def clean_text(raw: str) -> str:
    """Clean extracted PDF text for downstream NLP."""
    text = raw

    # Remove control characters
    text = _CONTROL_CHARS_RE.sub("", text)

    # Remove standalone page numbers
    text = _PAGE_NUMBER_RE.sub("", text)

    # Remove repeated headers / footers
    text = _HEADER_FOOTER_RE.sub("", text)

    # Collapse horizontal whitespace (preserve newlines)
    text = _MULTI_WHITESPACE_RE.sub(" ", text)

    # Normalise paragraph breaks
    text = _MULTI_NEWLINES_RE.sub("\n\n", text)

    # Strip leading/trailing whitespace on each line
    text = "\n".join(line.strip() for line in text.split("\n"))

    return text.strip()


# ──────────────────────────────────────────────
# 4. Recursive character text splitting
# ──────────────────────────────────────────────
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


def recursive_split(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    separators: list[str] | None = None,
) -> list[str]:
    """
    Split *text* into chunks of at most *chunk_size* characters,
    with *chunk_overlap* characters of overlap between consecutive chunks.

    Tries each separator in order; when a separator produces pieces
    that are still too long, it recurses with the next separator.
    """
    if separators is None:
        separators = SEPARATORS

    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    sep = separators[0]
    remaining_seps = separators[1:]

    if sep == "":
        # Character-level fallback
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - chunk_overlap
        return chunks

    pieces = text.split(sep)
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        candidate = (current + sep + piece) if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # If single piece is still too large, recurse with finer separator
            if len(piece) > chunk_size and remaining_seps:
                sub_chunks = recursive_split(
                    piece, chunk_size, chunk_overlap, remaining_seps
                )
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = piece

    if current:
        chunks.append(current)

    # Apply overlap: merge trailing context of previous chunk into next
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap_text = prev[-chunk_overlap:]
            overlapped.append(overlap_text + sep + chunks[i])
        chunks = overlapped

    return [c for c in chunks if len(c.strip()) >= MIN_CHUNK_LENGTH]


# ──────────────────────────────────────────────
# 5. Main pipeline
# ──────────────────────────────────────────────
def collect_pdf_paths(root: Path) -> list[tuple[int, Path]]:
    """Return list of (year, pdf_path) sorted by year then name."""
    results = []
    for year in YEAR_RANGE:
        year_dir = root / str(year)
        if not year_dir.is_dir():
            continue
        for pdf_file in sorted(year_dir.glob("*.PDF")):
            results.append((year, pdf_file))
        # Also match lowercase .pdf
        for pdf_file in sorted(year_dir.glob("*.pdf")):
            if pdf_file not in [r[1] for r in results]:
                results.append((year, pdf_file))
    return results


def process_all(root: Path = ROOT_DIR, output: Path = OUTPUT_FILE):
    """Run the full preprocessing + chunking pipeline."""
    pdf_paths = collect_pdf_paths(root)
    logger.info("Found %d PDF files across %d–%d", len(pdf_paths), YEAR_RANGE.start, YEAR_RANGE.stop - 1)

    total_chunks = 0
    failed_files = 0

    with open(output, "w", encoding="utf-8") as out_f:
        for year, pdf_path in tqdm(pdf_paths, desc="Processing PDFs"):
            # --- metadata ---
            meta = parse_filename(pdf_path.name, year)

            # --- extract ---
            raw_text = extract_text_from_pdf(str(pdf_path))
            if not raw_text:
                logger.warning("No text extracted: %s", pdf_path)
                failed_files += 1
                continue

            # --- clean ---
            cleaned = clean_text(raw_text)
            if not cleaned:
                logger.warning("Empty after cleaning: %s", pdf_path)
                failed_files += 1
                continue

            # --- chunk ---
            chunks = recursive_split(cleaned)

            # --- write chunks as JSONL ---
            for idx, chunk_text in enumerate(chunks):
                chunk = Chunk(
                    chunk_id=f"{year}/{pdf_path.stem}__chunk_{idx:04d}",
                    text=chunk_text,
                    metadata={
                        **asdict(meta),
                        "chunk_index": idx,
                        "total_chunks": len(chunks),
                        "char_count": len(chunk_text),
                    },
                )
                out_f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

            total_chunks += len(chunks)

    logger.info(
        "Done — %d chunks written to %s  |  %d files failed",
        total_chunks, output, failed_files,
    )


if __name__ == "__main__":
    process_all()
