#!/usr/bin/env python3
"""
MCP Gate Check — reads telemetry files to verify governance state.

Usage:
    python3 tools/claude_code_mcp/check_mcp_gate.py [--mode pre-commit|pre-tool]

Exit codes:
    0 — gate passed (recent MCP call, no reject/regenerate)
    1 — gate failed (no recent call, or last directive was reject/regenerate)
    2 — no telemetry data (MCP not active, pass with warning)
"""

import json
import sys
import time
from pathlib import Path


def find_latest_session(telemetry_dir: Path) -> dict | None:
    if not telemetry_dir.exists():
        return None

    sessions = sorted(
        telemetry_dir.glob("session_*.json"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not sessions:
        return None

    try:
        return json.loads(sessions[0].read_text())
    except (json.JSONDecodeError, OSError):
        return None


def check_gate(mode: str = "pre-commit") -> int:
    root = Path(__file__).resolve().parent.parent.parent
    telemetry_dir = root / "data" / "mcp_telemetry"

    session = find_latest_session(telemetry_dir)

    if session is None:
        print("MCP_GATE: no telemetry data (MCP server may not be active)")
        return 2

    age = time.time() - session.get("last_update", 0)
    session_id = session.get("session_id", "unknown")
    call_count = session.get("call_count", 0)
    drift = session.get("drift_metrics", {})
    last_directive = drift.get("last_directive", "unknown")
    current_phi = session.get("current_phi", 0)

    # Check if session is stale (>30 min old)
    if age > 1800:
        print(f"MCP_GATE: session {session_id} is stale ({age/60:.0f}min old)")
        return 2

    # Check last directive
    if last_directive in ("reject", "regenerate"):
        print(
            f"MCP_GATE: BLOCKED — last directive is {last_directive.upper()}"
            f" (session {session_id}, {call_count} calls, phi={current_phi:.4f})"
        )
        print("  Fix the gaps identified by the MCP gate before committing.")
        return 1

    # Check if any verification was done (not just checkpoints)
    timeline = session.get("timeline", [])
    verification_calls = [
        e for e in timeline
        if e.get("tool") in ("phionyx_verify_claim", "phionyx_response_gate")
    ]

    if mode == "pre-commit" and not verification_calls:
        print(
            f"MCP_GATE: WARNING — no verify_claim or response_gate calls in session"
            f" (session {session_id}, {call_count} calls)"
        )
        print("  Consider calling phionyx_response_gate before committing.")
        # Warning only, not blocking — some commits don't need full verification
        return 0

    reject_count = drift.get("reject_count", 0)
    print(
        f"MCP_GATE: PASS — directive={last_directive}, phi={current_phi:.4f},"
        f" calls={call_count}, rejects={reject_count}"
    )
    return 0


if __name__ == "__main__":
    mode = "pre-commit"
    if len(sys.argv) > 1 and sys.argv[1] == "--mode":
        mode = sys.argv[2] if len(sys.argv) > 2 else "pre-commit"
    sys.exit(check_gate(mode))
