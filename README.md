# phionyx-pipeline-mcp

> A self-governance MCP server for Claude Code — gates the agent's own *"I fixed this / I tested that / this code path changed"* declarations against `git diff` truth and a deterministic physics gate.

`phionyx-pipeline-mcp` solves a less-discussed agent-trust problem: when an AI coding agent reports back on its own work, that report is itself a trust object. Without verification, you accept it on faith.

This package gives any MCP-capable host (Claude Desktop, Cursor, Zed, VS Code, JetBrains) a six-tool surface that turns those self-reports into reviewable evidence — coverage of paths claimed vs. paths actually touched, severity-weighted evidence taxonomy, drift tracking across a session, and an explicit `pass | regenerate | reject` directive before the agent claims "done."

## How it works — three-layer verification

```
Layer 1: LLM declaration   →   Layer 2: Repo truth          →   Layer 3: Physics gate
  "I fixed X, tested Y,         git diff parsed; functions       phi + entropy + revision
   affected paths a,b,c"        extracted; cross-checked         thresholds → directive
```

The gate is deterministic. Layer 1 (the agent's input) is stochastic. Layer 2 — verifying the agent's path declarations against the actual `git diff` — is what narrows the gap.

## Companion package: phionyx-mcp-server

This package is the **inward-facing** layer: it gates what the agent says about its own work.

A companion package, [`phionyx-mcp-server`](https://github.com/halvrenofviryel/phionyx-mcp-server), is the **outward-facing** layer: it sees the host calling a third-party MCP server and signs evidence of that call (descriptor hash, drift detection, audit chain).

When both packages are installed and registered with the same Claude Code host, they agree on a single `trace_id` per session via `PHIONYX_TRACE_ID` (with `~/.phionyx/active_trace` file fallback). One conversation = one trace = end-to-end view of every third-party tool call AND every agent self-claim gate decision.

`phionyx_session_report` (this package) surfaces the server-MCP envelope chain head + validity inline, so a reviewer can see both layers in one JSON.

## Install

```bash
# This package only:
pip install phionyx-pipeline-mcp

# With the outward-facing companion (recommended for full session evidence):
pip install "phionyx-pipeline-mcp[mcp-server-integration]"
```

## Register with Claude Code

Add to `.claude/mcp.json` in your project:

```json
{
  "mcpServers": {
    "phionyx-pipeline": {
      "command": "phionyx-pipeline-mcp"
    },
    "phionyx-mcp-server": {
      "command": "phionyx-mcp-server"
    }
  }
}
```

Both servers will share `~/.phionyx/active_trace` by default.

## Tool surface

| Tool | When to call |
|---|---|
| `phionyx_verify_claim` | **Before** claiming "fixed" or "done." Takes the claim, the evidence, the evidence type, paths tested, paths affected. Returns a directive (`pass / regenerate / reject`) plus diagnosis. |
| `phionyx_causal_trace` | **While debugging.** Validates a causal chain from symptom to root cause. Chains shorter than 3 links or weaker than 40% code-specificity get a `deepen` directive. |
| `phionyx_response_gate` | **Before committing or deploying.** Action-type-specific thresholds: `claim_fixed` is strictest, `deploy` is very strict, `default` is standard. |
| `phionyx_verify_paths` | Cross-check claimed-affected and claimed-tested paths against `git diff`. Flags underreporting. |
| `phionyx_checkpoint` | Lightweight physics snapshot — call frequently to keep the session telemetry dense. |
| `phionyx_session_report` | End-of-session summary: claims, directives, drift metrics, evidence taxonomy, and (if the server MCP is installed) the audit chain head + validity. |

## Evidence taxonomy

Not all evidence is equal. The gate weights confidence by the type of test that backs a claim:

| Evidence type | Weight |
|---|---|
| `browser_test` | 0.9 |
| `manual_repro` | 0.8 |
| `integration_test` | 0.7 |
| `endpoint_test` | 0.6 |
| `log_inspection` | 0.5 |
| `unit_test` | 0.4 |
| `code_review` | 0.3 |
| `none` | 0.0 |

A `claim_fixed` action with `evidence_type=none` will be rejected outright; even high confidence values cannot compensate for missing test evidence.

## Pre-commit gate helper

A standalone `phionyx-pipeline-check-gate` CLI reads the recent telemetry and exits non-zero if the most recent gate directive was `reject` or `regenerate`. Drop it into your pre-commit hook to enforce the discipline:

```bash
phionyx-pipeline-check-gate --mode pre-commit
```

Exit codes: `0` pass · `1` failed (recent reject/regenerate) · `2` no telemetry (warn-only).

## Shared-trace contract

When `phionyx-mcp-server` is also installed, this package extends `phionyx_session_report`'s output with:

```json
{
  "trace_id": "<active trace>",
  "mcp_envelope_chain": {
    "trace_id": "<same trace>",
    "count": 7,
    "head_hash": "sha256:...",
    "valid": true,
    "broken_at": null
  }
}
```

The integration is read-only — this package imports `FilesystemEnvelopeStore` + `verify_chain` from `phionyx-mcp-server` and reads its chain. No cross-package write coupling. When the server package isn't installed, the field reports `"phionyx-mcp-server not installed"` and the rest of the report continues to work.

## Tests

```bash
pip install -e ".[test]"
pytest tests/ -q
```

## License

AGPL-3.0-or-later. See [`LICENSE`](LICENSE).

## See also

- Project hub: [github.com/halvrenofviryel/phionyx-research](https://github.com/halvrenofviryel/phionyx-research)
- Outward layer (MCP trust boundary): [halvrenofviryel/phionyx-mcp-server](https://github.com/halvrenofviryel/phionyx-mcp-server)
- Phionyx Core SDK (PyPI): [`phionyx-core`](https://pypi.org/project/phionyx-core/)
