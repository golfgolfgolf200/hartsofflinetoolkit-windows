# HartsOfflineToolkit (Windows 10 / 11 Edition)

**Official Website:** [https://hartsofflinetoolkit.com](https://hartsofflinetoolkit.com)  
**Linux Edition:** [https://github.com/golfgolfgolf200/hartsofflinetoolkit-linux](https://github.com/golfgolfgolf200/hartsofflinetoolkit-linux)

**HartsOfflineToolkit for Windows** is an air-gapped, CPU-only hybrid semantic search engine and Model Context Protocol (MCP) server built natively for Windows 10 and Windows 11 (x86_64) — no WSL, Docker, or GPU required.

It combines **Google EmbeddingGemma 2** (`google/embeddinggemma-2`, truncated via Matryoshka Representation Learning to 256 dimensions) with **BM25 lexical scoring** and **Reciprocal Rank Fusion (RRF)** inside a local **LanceDB** vector store, governed by Windows CPU core affinity (`cpu_affinity`), `BELOW_NORMAL_PRIORITY_CLASS` / `IDLE_PRIORITY_CLASS` scheduling, and `IOPRIO_VERYLOW` background disk I/O.

---

## Key Capabilities

- **100% Air-Gapped Runtime (`HF_HUB_OFFLINE=1`):** Zero outbound telemetry or network calls after one-time `cli.py init` setup.
- **Native Windows Hardware Governor:** Pins process execution to designated CPU cores (`CPU_AFFINITY_CORES=0,1,2`), caps PyTorch/OMP/MKL worker threads, and sets `BELOW_NORMAL_PRIORITY_CLASS` + `IOPRIO_VERYLOW` so foreground apps, video calls, and games never stutter.
- **Incremental SHA-256 Delta Indexing:** Skips unchanged files in milliseconds, atomically replaces modified files, and purges deleted documents.
- **Table-Preserving Overlapping Chunking:** Extracts PDF tables via `pdfplumber` as intact Markdown grids and splits long `.pdf`, `.md`, and `.txt` files into overlapping word windows (`CHUNK_WORDS=400`, `CHUNK_OVERLAP_WORDS=50`).
- **Hardened `stdio` MCP Server (`mcp_server.py`):** Plugs directly into VS Code Copilot, Claude Desktop, Cursor, or local MCP hosts with root-jailed indexing (`./data/raw`), optional `MCP_READ_ONLY=1` lock, and warm in-RAM sub-50ms search.

---

## Quick Start (PowerShell)

```powershell
# 1. Clone the Windows repository
git clone https://github.com/golfgolfgolf200/hartsofflinetoolkit-linux-windows.git
cd hartsofflinetoolkit-windows

# 2. Verify repository file hashes against SHA256SUMS
powershell -ExecutionPolicy Bypass -File .\verify.ps1

# 3. Create .venv, install CPU-only PyTorch, download EmbeddingGemma 2, and lock weights
powershell -ExecutionPolicy Bypass -File .\setup.ps1

# 4. Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

## Daily CLI Commands

```powershell
# Check CPU core affinity, RAM usage, and LanceDB status
python cli.py status

# Incrementally index PDFs, Markdown, and text files in .\data\raw
python cli.py index

# Hybrid semantic + exact keyword search
python cli.py search "30-day cancellation terms" --limit 5

# Machine-readable JSON output for PowerShell pipelines
python cli.py search "REF-99482-TX" --json

# Zero-cold-start interactive REPL mode (keeps model warm in RAM)
python cli.py interactive
```

## Windows MCP Client Configuration (mcp.json / claude_desktop_config.json)

```json
{
  "mcpServers": {
    "hartsofflinetoolkit": {
      "command": "C:\\Users\\<username>\\projects\\hartsofflinetoolkit-windows\\.venv\\Scripts\\python.exe",
      "args": [
        "C:\\Users\\<username>\\projects\\hartsofflinetoolkit-windows\\mcp_server.py"
      ]
    }
  }
}
```

## Model Attribution & Licensing

Powered by Google EmbeddingGemma 2 (google/embeddinggemma-2), governed by the Gemma Terms of Use. Toolkit source code is licensed under LICENSE.md