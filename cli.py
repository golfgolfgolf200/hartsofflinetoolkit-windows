import json
import os
import re
import stat
import time
from pathlib import Path
import psutil
import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from src.config import settings, apply_hardware_governor

app = typer.Typer(help="HartsOfflineToolkit: Local Hybrid Knowledge Search")
console = Console()
engine = None

MAX_QUERY_CHARS = 2000

def _sanitize_query(raw: str) -> str:
    # Strip non-printable ASCII/Unicode control characters except spaces/newlines
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw).strip()
    return cleaned[:MAX_QUERY_CHARS]

def get_engine(rebuild: bool = False):
    global engine
    if engine is None:
        from src.engine import SearchEngine
        with console.status("[bold cyan]Initializing Engine & Loading Local Weights..."):
            engine = SearchEngine(rebuild=rebuild)
    return engine

def _render_results_table(query: str, results: list[dict], elapsed_ms: float | None = None):
    title = f"Results for: '{escape(query)}'"
    if elapsed_ms is not None:
        title += f" ({elapsed_ms:.1f} ms)"

    table = Table(title=title, show_lines=True)
    table.add_column("Rank", justify="center", style="cyan", width=6)
    table.add_column("Score (RRF)", justify="center", style="magenta", width=12)
    table.add_column("Document ID", style="green", width=28)
    table.add_column("Content Snippet", style="white")

    for i, res in enumerate(results, 1):
        raw_snippet = res["content"][:280] + "..." if len(res["content"]) > 280 else res["content"]
        # Escape Rich markup tags so document text cannot inject terminal formatting
        table.add_row(str(i), f"{res['score']:.5f}", escape(res["doc_id"]), escape(raw_snippet))

    console.print(table)

@app.command()
def init(
    model_id: str = typer.Option("google/embeddinggemma-2", help="Hugging Face model repository ID"),
    lock_weights: bool = typer.Option(True, help="Set cached model files read-only after download"),
):
    """One-time setup: creates runtime dirs, downloads model weights, locks permissions, and writes .env."""
    if not re.match(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$", model_id):
        console.print("[bold red]Invalid Hugging Face model ID format.[/]")
        raise typer.Exit(code=1)

    project_root = Path(__file__).resolve().parent
    raw_dir = project_root / "data" / "raw"
    db_dir = project_root / "data" / "db"
    env_path = project_root / ".env"
    env_example = project_root / ".env.example"

    raw_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)

    if not env_path.exists() and env_example.exists():
        env_path.write_text(env_example.read_text(encoding="utf-8"), encoding="utf-8")

    console.print(f"[bold cyan]Downloading snapshot for {escape(model_id)} (network enabled for init)...[/]")
    apply_hardware_governor(allow_hub_download=True)

    from huggingface_hub import snapshot_download
    snapshot_path = Path(snapshot_download(model_id)).resolve()

    if not snapshot_path.exists() or not any(snapshot_path.iterdir()):
        console.print("[bold red]Snapshot download failed or directory is empty.[/]")
        raise typer.Exit(code=1)

    if lock_weights:
        for root, dirs, files in os.walk(snapshot_path):
            for fname in files:
                fpath = Path(root) / fname
                if not fpath.is_symlink():
                    os.chmod(fpath, stat.S_IREAD)

    # Update or append MODEL_PATH in project-local .env
    existing_lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    updated = False
    new_lines = []
    for line in existing_lines:
        if line.startswith("MODEL_PATH="):
            new_lines.append(f"MODEL_PATH={snapshot_path}")
            updated = True
        else:
            new_lines.append(line)
    if not updated:
        new_lines.append(f"MODEL_PATH={snapshot_path}")
    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    console.print(f"[bold green]Setup complete.[/] Weights locked at: {escape(str(snapshot_path))}")
    console.print(f"[bold green]Updated:[/] {escape(str(env_path))}")

@app.command()
def index(
    path: str = typer.Option("./data/raw", help="Directory path to parse and index"),
    rebuild: bool = typer.Option(False, "--rebuild", help="Wipe existing LanceDB index and re-index from scratch"),
):
    """Incrementally indexes new and modified documents into LanceDB."""
    eng = get_engine(rebuild=rebuild)
    console.print(f"[bold blue]Scanning directory:[/] {escape(path)} (rebuild={rebuild})")
    t0 = time.perf_counter()
    stats = eng.index_directory_incremental(path)
    elapsed = time.perf_counter() - t0

    console.print(
        f"[bold green]Done in {elapsed:.2f}s[/] — "
        f"Added: {stats['added_files']} | Updated: {stats['updated_files']} | "
        f"Skipped (unchanged): {stats['skipped_files']} | Removed: {stats['removed_files']} | "
        f"Chunks embedded: {stats['chunks_indexed']}"
    )

@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(3, "--limit", "-n", min=1, max=50, help="Number of results to return"),
    json_output: bool = typer.Option(False, "--json", help="Emit raw JSON output for scripting/jq"),
):
    """Runs a single hybrid BM25 + EmbeddingGemma 2 search query."""
    clean_q = _sanitize_query(query)
    if not clean_q:
        if json_output:
            print(json.dumps([]))
        else:
            console.print("[yellow]Empty query provided.[/]")
        return

    eng = get_engine()
    t0 = time.perf_counter()
    results = eng.hybrid_search(clean_q, top_n=limit)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    if json_output:
        print(json.dumps(results, indent=2))
        return

    if not results:
        console.print("[yellow]No relevant matches found.[/]")
        return

    _render_results_table(clean_q, results, elapsed_ms=elapsed_ms)

@app.command()
def interactive(
    limit: int = typer.Option(5, "--limit", "-n", min=1, max=50, help="Number of results per query"),
):
    """Loads weights once into RAM and starts a low-latency interactive search REPL."""
    eng = get_engine()
    console.print("[bold cyan]Interactive Search Mode[/] — Type a query and press Enter (or 'exit' / Ctrl+C to quit).")

    while True:
        try:
            raw_q = console.input("\n[bold green]search>[/] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Exiting interactive session.[/]")
            break

        clean_q = _sanitize_query(raw_q)
        if not clean_q:
            continue
        if clean_q.lower() in {"exit", "quit", ":q"}:
            console.print("[dim]Exiting interactive session.[/]")
            break

        t0 = time.perf_counter()
        results = eng.hybrid_search(clean_q, top_n=limit)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if not results:
            console.print(f"[yellow]No matches found ({elapsed_ms:.1f} ms).[/]")
            continue

        _render_results_table(clean_q, results, elapsed_ms=elapsed_ms)

@app.command()
def status():
    """Displays vault index counts, disk usage, CPU affinity, and current RAM footprint."""
    apply_hardware_governor(allow_hub_download=False)
    proc = psutil.Process()
    rss_mb = proc.memory_info().rss / (1024 * 1024)
    affinity = proc.cpu_affinity() if hasattr(proc, "cpu_affinity") else settings.affinity_cores

    db_dir = Path(settings.db_path)
    db_bytes = sum(f.stat().st_size for f in db_dir.rglob("*") if f.is_file()) if db_dir.exists() else 0
    db_mb = db_bytes / (1024 * 1024)

    chunk_count = 0
    file_count = 0
    if db_dir.exists():
        import lancedb
        db = lancedb.connect(str(db_dir))
        if "documents" in db.list_tables():
            df = db.open_table("documents").to_pandas()
            chunk_count = len(df)
            if "source" in df.columns and not df.empty:
                file_count = df["source"].nunique()

    table = Table(title="HartsOfflineToolkit System & Vault Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Model Path", escape(settings.model_path or "NOT CONFIGURED"))
    table.add_row("Offline Enforcement", f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE', '0')}")
    table.add_row("CPU Core Affinity", str(affinity))
    table.add_row("Worker Threads / Batch Size", f"{settings.worker_threads} threads / batch={settings.embed_batch_size}")
    table.add_row("Process RAM (RSS)", f"{rss_mb:.1f} MB")
    table.add_row("LanceDB Path & Size", f"{escape(str(db_dir))} ({db_mb:.2f} MB)")
    table.add_row("Indexed Files / Chunks", f"{file_count} files / {chunk_count} chunks")
    console.print(table)

if __name__ == "__main__":
    app()
