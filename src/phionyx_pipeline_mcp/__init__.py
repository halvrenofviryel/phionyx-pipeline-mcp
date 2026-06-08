"""Phionyx Pipeline MCP — agent self-claim governance gate for Claude Code.

Three-layer claim verification for AI coding agents:

    Layer 1 — LLM declaration: the agent says what it changed and tested.
    Layer 2 — Repo truth:      git diff + function extraction verify declarations.
    Layer 3 — Gate decision:   physics state + revision thresholds produce
                                a directive (pass | regenerate | reject).

The gate is deterministic; the input generation (Layer 1) is stochastic.
The input verification layer (Layer 2) closes the gap.

Companion package: ``phionyx-mcp-server`` (outward-facing MCP trust boundary).
When both are installed and registered with a Claude Code host, they share a
single ``trace_id`` per session via the ``PHIONYX_TRACE_ID`` env var (with
``~/.phionyx/active_trace`` file fallback) for end-to-end auditability.

See README.md for the .claude/mcp.json registration snippet and tool surface.
"""
from __future__ import annotations

__version__ = "0.3.0"
__all__ = ["__version__"]
