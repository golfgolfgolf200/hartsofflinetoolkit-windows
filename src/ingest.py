import hashlib
import re
import sys
from pathlib import Path
import pdfplumber
from src.config import settings

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def compute_file_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()

def _split_blocks_preserving_tables(text: str) -> list[tuple[bool, str]]:
    """
    Splits text into ordered blocks of (is_table, content), keeping contiguous
    Markdown table lines together so rows/columns are never split mid-table.
    """
    lines = text.splitlines()
    blocks: list[tuple[bool, str]] = []
    current_lines: list[str] = []
    current_is_table: bool | None = None

    for line in lines:
        stripped = line.strip()
        is_table_line = stripped.startswith("|") and stripped.endswith("|") and len(stripped) >= 2

        if current_is_table is None:
            current_is_table = is_table_line
            current_lines.append(line)
        elif is_table_line == current_is_table:
            current_lines.append(line)
        else:
            block_text = "\n".join(current_lines).strip()
            if block_text:
                blocks.append((current_is_table, block_text))
            current_lines = [line]
            current_is_table = is_table_line

    if current_lines and current_is_table is not None:
        block_text = "\n".join(current_lines).strip()
        if block_text:
            blocks.append((current_is_table, block_text))

    return blocks

def chunk_text_preserving_tables(
    text: str,
    chunk_words: int | None = None,
    overlap_words: int | None = None,
    max_chunks: int | None = None,
) -> list[str]:
    """
    Splits prose into overlapping word windows while treating Markdown tables
    as indivisible atomic blocks.
    """
    target_words = chunk_words or settings.chunk_words
    overlap = min(overlap_words or settings.chunk_overlap_words, max(0, target_words - 1))
    cap = max_chunks or settings.max_chunks_per_doc

    blocks = _split_blocks_preserving_tables(text)
    chunks: list[str] = []
    prose_buffer: list[str] = []

    def flush_prose():
        nonlocal prose_buffer
        if not prose_buffer:
            return
        step = max(1, target_words - overlap)
        for start in range(0, len(prose_buffer), step):
            window = prose_buffer[start : start + target_words]
            if window:
                chunks.append(" ".join(window))
            if start + target_words >= len(prose_buffer):
                break
        prose_buffer = []

    for is_table, block_text in blocks:
        if is_table:
            flush_prose()
            chunks.append(block_text)
        else:
            words = block_text.split()
            prose_buffer.extend(words)

    flush_prose()
    return chunks[:cap]

def parse_accounting_pdf(pdf_path: Path, file_sha256: str) -> list[dict]:
    documents = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_text_blocks = []
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue
                headers = [clean_text(str(c)) if c else "" for c in table[0]]
                sep = ["---"] * len(headers)
                md_table = [f"| {' | '.join(headers)} |", f"| {' | '.join(sep)} |"]
                for row in table[1:]:
                    cleaned_row = [clean_text(str(c)) if c else "" for c in row]
                    md_table.append(f"| {' | '.join(cleaned_row)} |")
                page_text_blocks.append("\n".join(md_table))

            raw_text = page.extract_text()
            if raw_text:
                cleaned = clean_text(raw_text)
                if cleaned:
                    page_text_blocks.append(cleaned)

            combined_content = "\n\n".join(page_text_blocks).strip()
            if combined_content:
                page_chunks = chunk_text_preserving_tables(combined_content)
                for chunk_idx, chunk_content in enumerate(page_chunks):
                    documents.append({
                        "doc_id": f"{pdf_path.name}_p{page_idx + 1}_c{chunk_idx}",
                        "source": str(pdf_path),
                        "file_sha256": file_sha256,
                        "page": page_idx + 1,
                        "chunk_index": chunk_idx,
                        "content": chunk_content,
                    })
    return documents

def scan_directory_files(target_dir: str) -> list[tuple[Path, str]]:
    """
    Safely enumerates supported files within target_dir with symlink,
    path-containment, and file-size guardrails. Returns [(resolved_path, sha256)].
    """
    root = Path(target_dir).resolve()
    if not root.exists() or not root.is_dir():
        return []

    valid_files: list[tuple[Path, str]] = []
    for f in sorted(root.rglob("*")):
        # Security: reject symlinks to prevent directory traversal outside root
        if f.is_symlink() or not f.is_file():
            continue

        resolved = f.resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            sys.stderr.write(f"[SECURITY WARN] Skipped path escaping root: {f}\n")
            continue

        if resolved.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            size = resolved.stat().st_size
        except OSError:
            continue

        if size == 0 or size > settings.max_file_bytes:
            sys.stderr.write(f"[WARN] Skipped file outside size bounds ({size} bytes): {resolved.name}\n")
            continue

        valid_files.append((resolved, compute_file_sha256(resolved)))

    return valid_files

def parse_file(file_path: Path, file_sha256: str) -> list[dict]:
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return parse_accounting_pdf(file_path, file_sha256)
    elif ext in {".txt", ".md"}:
        text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            return []
        chunks = chunk_text_preserving_tables(text)
        return [
            {
                "doc_id": f"{file_path.name}_c{chunk_idx}",
                "source": str(file_path),
                "file_sha256": file_sha256,
                "page": 1,
                "chunk_index": chunk_idx,
                "content": chunk_content,
            }
            for chunk_idx, chunk_content in enumerate(chunks)
        ]
    return []

def parse_directory(target_dir: str) -> list[dict]:
    extracted = []
    for resolved_path, file_sha256 in scan_directory_files(target_dir):
        extracted.extend(parse_file(resolved_path, file_sha256))
    return extracted
