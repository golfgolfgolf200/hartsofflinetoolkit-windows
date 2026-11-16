import hashlib
import pytest
from src.config import settings, apply_hardware_governor
from src.engine import SearchEngine
from src.ingest import chunk_text_preserving_tables

def test_hardware_governor():
    apply_hardware_governor()
    import torch
    assert torch.get_num_threads() == settings.worker_threads

def test_chunking_preserves_markdown_tables():
    prose_before = " ".join([f"word{i}" for i in range(450)])
    md_table = "| Account | Balance |\n| --- | --- |\n| Operating | $125,000.00 |\n| Reserve | $50,000.00 |"
    prose_after = " ".join([f"tail{i}" for i in range(120)])
    combined = f"{prose_before}\n\n{md_table}\n\n{prose_after}"

    chunks = chunk_text_preserving_tables(combined, chunk_words=200, overlap_words=30)
    assert len(chunks) >= 4
    # Ensure the Markdown table stayed intact in a single chunk
    assert any(md_table in c for c in chunks)

def test_incremental_indexing_and_search(tmp_path):
    raw_dir = tmp_path / "raw"
    db_dir = tmp_path / "db"
    raw_dir.mkdir()
    db_dir.mkdir()

    f1 = raw_dir / "invoice_1099.txt"
    f1.write_text("Invoice #1099: Total $4,500. Tax EIN: 88-1928371. Payable within 30 days.", encoding="utf-8")

    f2 = raw_dir / "sla_policy.txt"
    f2.write_text("Service level agreement requires 99.99% uptime. System crashes result in 10% penalty fee.", encoding="utf-8")

    engine = SearchEngine(db_path=str(db_dir))

    # First pass indexes both files
    s1 = engine.index_directory_incremental(str(raw_dir))
    assert s1["added_files"] == 2
    assert s1["skipped_files"] == 0

    # Second pass with no changes skips both files (zero duplicates)
    s2 = engine.index_directory_incremental(str(raw_dir))
    assert s2["added_files"] == 0
    assert s2["skipped_files"] == 2
    assert len(engine.corpus_ids) == s1["chunks_indexed"]

    # Modifying f1 updates only f1
    f1.write_text("Invoice #1099-REV: Total $5,200. Tax EIN: 88-1928371. Updated terms.", encoding="utf-8")
    s3 = engine.index_directory_incremental(str(raw_dir))
    assert s3["updated_files"] == 1
    assert s3["skipped_files"] == 1

    # Exact Keyword Retrieval
    kw_results = engine.hybrid_search("88-1928371", top_n=1)
    assert len(kw_results) > 0
    assert "invoice_1099.txt" in kw_results[0]["doc_id"]
    assert "$5,200" in kw_results[0]["content"]

    # Semantic Retrieval
    sem_results = engine.hybrid_search("What financial penalty happens if servers crash?", top_n=1)
    assert len(sem_results) > 0
    assert "sla_policy.txt" in sem_results[0]["doc_id"]
