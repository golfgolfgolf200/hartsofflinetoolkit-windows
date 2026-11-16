import json
import pytest
from mcp.server.mcpserver.exceptions import ToolError
import mcp_server

def test_mcp_jail_blocks_traversal():
    with pytest.raises(ToolError):
        mcp_server._resolve_jailed_path("/etc/passwd")

    with pytest.raises(ToolError):
        mcp_server._resolve_jailed_path("../../outside_vault")

    allowed = mcp_server._resolve_jailed_path("./data/raw")
    assert allowed == mcp_server.ALLOWED_INGEST_ROOT

def test_mcp_tools_end_to_end(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    db_dir = tmp_path / "db"
    raw_dir.mkdir()
    db_dir.mkdir()

    sample_file = raw_dir / "contract_audit.md"
    sample_file.write_text(
        "# Master Service Contract\n\n"
        "| Clause | Reference | Penalty |\n"
        "| --- | --- | --- |\n"
        "| SLA Breach | SLA-404-TX | 15% monthly credit |\n\n"
        "All termination notices require 60 days written notice.",
        encoding="utf-8",
    )

    monkeypatch.setattr(mcp_server, "ALLOWED_INGEST_ROOT", raw_dir.resolve())
    monkeypatch.setattr(mcp_server.settings, "ingest_path", str(raw_dir))
    monkeypatch.setattr(mcp_server.settings, "db_path", str(db_dir))
    monkeypatch.setattr(mcp_server, "_engine", None)

    # 1. Index via MCP tool
    idx_resp = json.loads(mcp_server.index_vault(path=str(raw_dir)))
    assert idx_resp["status"] == "ok"
    assert idx_resp["stats"]["added_files"] == 1

    # 2. Search via MCP tool
    search_resp = json.loads(mcp_server.search_vault(query="SLA-404-TX", limit=2))
    assert search_resp["total_returned"] >= 1
    assert search_resp["results"][0]["file_name"] == "contract_audit.md"
    assert "15% monthly credit" in search_resp["results"][0]["content"]

    # 3. Fetch document chunks via MCP tool
    chunks_resp = json.loads(mcp_server.get_document_chunks(file_name="contract_audit.md"))
    assert chunks_resp["total_returned"] >= 2

    # 4. Vault status via MCP tool
    status_resp = json.loads(mcp_server.vault_status())
    assert status_resp["offline_mode"] is True
    assert status_resp["indexed_file_count"] == 1
