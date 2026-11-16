import re
import shutil
from pathlib import Path
import numpy as np
import pyarrow as pa
import torch
import lancedb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from src.config import settings, apply_hardware_governor
from src.ingest import scan_directory_files, parse_file

SHA256_RE = re.compile(r"^[a-f0-9]{64}$")

def _escape_sql_literal(value: str) -> str:
    """Escapes single quotes and rejects null bytes for LanceDB filter expressions."""
    if "\x00" in value:
        raise ValueError("Null byte not permitted in filter predicate.")
    return value.replace("'", "''")

class SearchEngine:
    def __init__(self, db_path: str | None = None, rebuild: bool = False):
        apply_hardware_governor(allow_hub_download=False)
        if not settings.model_path:
            raise RuntimeError(
                "MODEL_PATH is not configured. Run 'python3 cli.py init' or set MODEL_PATH in .env."
            )

        self.db_path = db_path or settings.db_path
        if rebuild:
            p = Path(self.db_path)
            if p.exists() and p.is_dir():
                shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)

        print(f"Loading local weights: {settings.model_path} on CPU (float32)...")
        self.model = SentenceTransformer(
            settings.model_path,
            device="cpu",
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float32},
        )
        self.dim = settings.matryoshka_dim

        self.db = lancedb.connect(self.db_path)
        self.schema = pa.schema([
            pa.field("vector", pa.list_(pa.float32(), self.dim)),
            pa.field("doc_id", pa.string()),
            pa.field("source", pa.string()),
            pa.field("file_sha256", pa.string()),
            pa.field("page", pa.int32()),
            pa.field("chunk_index", pa.int32()),
            pa.field("content", pa.string()),
        ])
        self.table = self._open_or_migrate_table()

        self.corpus_ids: list[str] = []
        self.corpus_docs: list[dict] = []
        self.bm25: BM25Okapi | None = None
        self._rebuild_bm25_index()

    def _open_or_migrate_table(self):
        """Opens existing LanceDB table or recreates it if schema predates incremental tracking."""
        if "documents" in self.db.list_tables():
            tbl = self.db.open_table("documents")
            existing_cols = set(tbl.schema.names)
            required_cols = {"vector", "doc_id", "source", "file_sha256", "page", "chunk_index", "content"}
            if required_cols.issubset(existing_cols):
                return tbl
            self.db.drop_table("documents")
        return self.db.create_table("documents", schema=self.schema, exist_ok=True)

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _rebuild_bm25_index(self):
        records = self.table.to_pandas()
        if not records.empty:
            self.corpus_ids = records["doc_id"].tolist()
            self.corpus_docs = records.to_dict("records")
            tokenized = [self._tokenize(doc["content"]) for doc in self.corpus_docs]
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.corpus_ids = []
            self.corpus_docs = []
            self.bm25 = None

    def get_indexed_file_hashes(self) -> dict[str, set[str]]:
        """Returns mapping of {source_path: {sha256, ...}} currently stored in LanceDB."""
        records = self.table.to_pandas()
        mapping: dict[str, set[str]] = {}
        if records.empty:
            return mapping
        for row in records[["source", "file_sha256"]].drop_duplicates().to_dict("records"):
            mapping.setdefault(row["source"], set()).add(row["file_sha256"])
        return mapping

    def delete_by_source(self, source_path: str):
        safe_source = _escape_sql_literal(source_path)
        self.table.delete(f"source = '{safe_source}'")

    def embed_batch(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        if not texts:
            return []
        prefix = "task: search result | query: " if is_query else "title: none | text: "
        formatted = [f"{prefix}{t}" for t in texts]

        with torch.no_grad():
            full_vectors = self.model.encode(
                formatted,
                batch_size=settings.embed_batch_size,
                normalize_embeddings=False,
                show_progress_bar=False,
            )

        truncated = np.asarray(full_vectors)[:, : self.dim]
        norms = np.linalg.norm(truncated, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        normalized = truncated / norms
        return normalized.tolist()

    def embed_text(self, text: str, is_query: bool = False) -> list[float]:
        return self.embed_batch([text], is_query=is_query)[0]

    def index_documents(self, docs: list[dict]) -> int:
        if not docs:
            return 0

        vectors = self.embed_batch([d["content"] for d in docs], is_query=False)
        records = []
        for d, vec in zip(docs, vectors):
            sha = d.get("file_sha256", "0" * 64)
            if not SHA256_RE.match(sha):
                raise ValueError(f"Invalid SHA-256 hash format: {sha}")
            records.append({
                "vector": vec,
                "doc_id": d["doc_id"],
                "source": d["source"],
                "file_sha256": sha,
                "page": int(d.get("page", 1)),
                "chunk_index": int(d.get("chunk_index", 0)),
                "content": d["content"],
            })
        self.table.add(records)
        self._rebuild_bm25_index()
        return len(records)

    def index_directory_incremental(self, target_dir: str) -> dict[str, int]:
        """
        Incrementally indexes target_dir:
        - Skips files whose source path and SHA-256 hash already match LanceDB.
        - Replaces chunks for modified files.
        - Removes orphaned records for deleted files.
        """
        disk_files = scan_directory_files(target_dir)
        disk_map = {str(path): sha for path, sha in disk_files}
        indexed_map = self.get_indexed_file_hashes()

        stats = {"added_files": 0, "updated_files": 0, "skipped_files": 0, "removed_files": 0, "chunks_indexed": 0}
        docs_to_insert: list[dict] = []

        # Remove chunks for files deleted from disk under target_dir
        resolved_root = str(Path(target_dir).resolve())
        for existing_source in list(indexed_map.keys()):
            if existing_source.startswith(resolved_root) and existing_source not in disk_map:
                self.delete_by_source(existing_source)
                stats["removed_files"] += 1

        for path_obj, sha in disk_files:
            source_str = str(path_obj)
            existing_hashes = indexed_map.get(source_str)

            if existing_hashes == {sha}:
                stats["skipped_files"] += 1
                continue

            if existing_hashes is not None:
                self.delete_by_source(source_str)
                stats["updated_files"] += 1
            else:
                stats["added_files"] += 1

            docs_to_insert.extend(parse_file(path_obj, sha))

        if docs_to_insert:
            stats["chunks_indexed"] = self.index_documents(docs_to_insert)
        elif stats["removed_files"] > 0:
            self._rebuild_bm25_index()

        return stats

    def hybrid_search(self, query: str, top_n: int = 5) -> list[dict]:
        if not self.bm25 or len(self.corpus_ids) == 0:
            return []

        tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_ranked = np.argsort(bm25_scores)[::-1]

        bm25_ranks = {}
        for rank, idx in enumerate(bm25_ranked):
            if bm25_scores[idx] > 0:
                bm25_ranks[self.corpus_ids[idx]] = rank + 1

        q_vec = self.embed_text(query, is_query=True)
        dense_results = self.table.search(q_vec).limit(len(self.corpus_ids)).to_list()

        dense_ranks = {}
        dense_doc_map = {}
        for rank, res in enumerate(dense_results):
            doc_id = res["doc_id"]
            dense_ranks[doc_id] = rank + 1
            dense_doc_map[doc_id] = res

        all_candidate_ids = set(bm25_ranks.keys()).union(set(dense_ranks.keys()))
        fused = []
        k = settings.rrf_k
        for doc_id in all_candidate_ids:
            score = 0.0
            if doc_id in bm25_ranks:
                score += 1.0 / (k + bm25_ranks[doc_id])
            if doc_id in dense_ranks:
                score += 1.0 / (k + dense_ranks[doc_id])

            doc_data = dense_doc_map.get(doc_id) or next(d for d in self.corpus_docs if d["doc_id"] == doc_id)
            fused.append({
                "doc_id": doc_id,
                "score": score,
                "source": doc_data["source"],
                "page": doc_data["page"],
                "chunk_index": doc_data.get("chunk_index", 0),
                "content": doc_data["content"],
            })

        fused.sort(key=lambda x: x["score"], reverse=True)
        return fused[:top_n]
