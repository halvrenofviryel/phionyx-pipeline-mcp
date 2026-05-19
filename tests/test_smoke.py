"""Smoke tests for phionyx-pipeline-mcp.

These tests verify the package imports cleanly and the shared-trace helpers
behave correctly. Full behavioural tests (gate decisions, claim verification)
require phionyx-core fixtures and live in the developer's private monorepo;
this public package keeps the test surface minimal so CI is fast and the
package boundary stays clean.
"""
from __future__ import annotations

import sys

import pytest


def test_package_imports() -> None:
    """The package must be importable without optional companions installed."""
    import phionyx_pipeline_mcp

    assert phionyx_pipeline_mcp.__version__


def test_active_trace_env_var_precedence(monkeypatch, tmp_path) -> None:
    """PHIONYX_TRACE_ID overrides any active_trace file content."""
    from phionyx_pipeline_mcp.server import _active_trace_id

    monkeypatch.setenv("PHIONYX_TRACE_ID", "trace-env-001")
    monkeypatch.setenv(
        "PHIONYX_ACTIVE_TRACE_FILE", str(tmp_path / "active_trace")
    )

    (tmp_path / "active_trace").write_text("trace-from-file", encoding="utf-8")

    assert _active_trace_id() == "trace-env-001"


def test_active_trace_file_fallback(monkeypatch, tmp_path) -> None:
    """When env var is unset, the active_trace file is the source of truth."""
    from phionyx_pipeline_mcp.server import _active_trace_id

    monkeypatch.delenv("PHIONYX_TRACE_ID", raising=False)
    monkeypatch.setenv(
        "PHIONYX_ACTIVE_TRACE_FILE", str(tmp_path / "active_trace")
    )

    (tmp_path / "active_trace").write_text("trace-from-file-abc", encoding="utf-8")

    assert _active_trace_id() == "trace-from-file-abc"


def test_active_trace_generates_and_persists(monkeypatch, tmp_path) -> None:
    """First caller generates a UUID-derived id and writes it to the file."""
    from phionyx_pipeline_mcp.server import _active_trace_id

    trace_file = tmp_path / "active_trace"
    monkeypatch.delenv("PHIONYX_TRACE_ID", raising=False)
    monkeypatch.setenv("PHIONYX_ACTIVE_TRACE_FILE", str(trace_file))

    assert not trace_file.exists()

    first = _active_trace_id()
    assert first.startswith("trace-")
    assert trace_file.read_text(encoding="utf-8").strip() == first

    second = _active_trace_id()
    assert second == first


def test_envelope_chain_graceful_without_server_mcp(
    monkeypatch, tmp_path
) -> None:
    """When phionyx-mcp-server is not importable, chain reports the install
    gap instead of raising."""
    from phionyx_pipeline_mcp.server import _read_envelope_chain

    monkeypatch.setenv("PHIONYX_TRACE_ID", "trace-no-server-001")
    monkeypatch.setenv(
        "PHIONYX_ACTIVE_TRACE_FILE", str(tmp_path / "active_trace")
    )

    # Hide phionyx_mcp_server from the import system.
    monkeypatch.setitem(sys.modules, "phionyx_mcp_server", None)
    monkeypatch.setitem(sys.modules, "phionyx_mcp_server.audit_chain", None)

    chain = _read_envelope_chain()

    assert chain["trace_id"] == "trace-no-server-001"
    assert chain["count"] == 0
    assert chain["valid"] is None
    assert chain["reason"] == "phionyx-mcp-server not installed"
