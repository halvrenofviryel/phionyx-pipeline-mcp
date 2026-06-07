#!/usr/bin/env python3
"""Constraint ledger — İz manufacture for the read→applied guarantee (B1).

At session start the binding constraints that govern this repo are extracted
from their canonical sources and written to a per-session ledger
(`~/.phionyx/active_constraints.jsonl`). This is the "manufactured İz": a
persistent, re-soundable trace of *what binds the assistant this session*, so
that a later proposed action can be checked against it (B2, the gate's
constraint-recognition step) instead of relying on the assistant to remember to
apply a note it merely read.

Sources (canonical binding corpus):
  1. CLAUDE.md  — "## Forbidden Actions" items + "## Escalation" bullets
  2. .claude/rules/*.md — header "> ... Rule" lines + "**Rule:**" lines
  3. memory feedback_*.md — "**How to apply:**" lines (the binding application rules)

Each record:
  {id, text, source, machine_checkable: bool, check_hint: str|None, registered_ts}

`machine_checkable` is a conservative heuristic flag — the ACTUAL checks live in
B2 (the gate). This module only catalogs; it does not decide groundedness.

Every function fails safe (returns what it has; never raises) — a ledger is a
best-effort trace, never a session-breaker.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

DEFAULT_LEDGER = "~/.phionyx/active_constraints.jsonl"

# Heuristic: a constraint is machine-checkable if it names a concrete, inspectable
# artifact class (imports, paths, file types, gate calls, source reads).
_CHECKABLE_HINTS: list[tuple[str, re.Pattern[str]]] = [
    ("import_boundary", re.compile(r"\bimport\b|\bfastapi\b|\buvicorn\b|\bsqlalchemy\b", re.I)),
    ("path_boundary", re.compile(r"phionyx_core/|phionyx_bridge/|\bcore\b.*\bbridge\b", re.I)),
    ("call_core_not_reimplement", re.compile(r"call the core|re-?implement|bespoke|regex hook", re.I)),
    ("read_source_first", re.compile(r"\bREAD\b.*\b(first|before)\b|read the (real )?(module|source|artifact)", re.I)),
    ("gate_before_commit", re.compile(r"response_gate|verify_claim|before commit", re.I)),
    ("no_secrets", re.compile(r"\.env\b|API[_ ]?key|secret|token|credential", re.I)),
]


def _classify(text: str) -> tuple[bool, str | None]:
    for hint, pat in _CHECKABLE_HINTS:
        if pat.search(text):
            return True, hint
    return False, None


def _rec(idx: str, text: str, source: str, ts: str) -> dict[str, Any]:
    text = " ".join(text.split())
    checkable, hint = _classify(text)
    return {
        "id": idx,
        "text": text[:400],
        "source": source,
        "machine_checkable": checkable,
        "check_hint": hint,
        "registered_ts": ts,
    }


def _section(lines: list[str], header_pat: re.Pattern[str]) -> list[str]:
    """Return the lines of the first markdown section whose header matches."""
    out: list[str] = []
    in_sec = False
    for ln in lines:
        if ln.startswith("## "):
            if in_sec:
                break
            in_sec = bool(header_pat.search(ln))
            continue
        if in_sec:
            out.append(ln)
    return out


def _from_claude_md(path: Path, ts: str) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return recs
    # Forbidden Actions — numbered "N. Do NOT ..."
    for ln in _section(lines, re.compile(r"Forbidden Actions", re.I)):
        m = re.match(r"\s*(\d+)\.\s+(.*)", ln)
        if m and m.group(2).strip():
            recs.append(_rec(f"forbidden-{m.group(1)}", m.group(2), ".claude/CLAUDE.md#forbidden", ts))
    # Escalation — bullet lines
    n = 0
    for ln in _section(lines, re.compile(r"Escalation", re.I)):
        m = re.match(r"\s*[-*]\s+(.*)", ln)
        if m and m.group(1).strip():
            n += 1
            recs.append(_rec(f"escalation-{n}", m.group(1), ".claude/CLAUDE.md#escalation", ts))
    return recs


def _from_rules(rules_dir: Path, ts: str) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    if not rules_dir.is_dir():
        return recs
    for p in sorted(rules_dir.glob("*.md")):
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        n = 0
        for ln in text.splitlines():
            m = re.search(r"\*\*Rule:\*\*\s*(.*)", ln)
            if m and m.group(1).strip():
                n += 1
                recs.append(_rec(f"rule-{p.stem}-{n}", m.group(1),
                                 f".claude/rules/{p.name}", ts))
    return recs


def _from_memory_feedback(memory_dir: Path, ts: str) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    if not memory_dir.is_dir():
        return recs
    for p in sorted(memory_dir.glob("feedback_*.md")):
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        for m in re.finditer(r"\*\*How to apply:?\*\*\s*(.+)", text):
            body = m.group(1).strip()
            if body:
                recs.append(_rec(f"howto-{p.stem}", body, f"memory/{p.name}", ts))
    return recs


def extract_constraints(
    repo_root: Path,
    memory_dir: Path | None = None,
    ts: str | None = None,
) -> list[dict[str, Any]]:
    """Extract the binding-constraint corpus from canonical sources. Never raises."""
    ts = ts or time.strftime("%Y-%m-%dT%H:%M:%S")
    recs: list[dict[str, Any]] = []
    try:
        recs += _from_claude_md(repo_root / ".claude" / "CLAUDE.md", ts)
        recs += _from_rules(repo_root / ".claude" / "rules", ts)
        if memory_dir is not None:
            recs += _from_memory_feedback(memory_dir, ts)
    except Exception:  # pragma: no cover — best-effort
        pass
    return recs


def write_ledger(constraints: list[dict[str, Any]], path: str | Path = DEFAULT_LEDGER) -> int:
    """Write the constraint ledger as JSONL. Returns count written (0 on failure)."""
    try:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for rec in constraints:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return len(constraints)
    except Exception:  # pragma: no cover — best-effort
        return 0


def load_ledger(path: str | Path = DEFAULT_LEDGER) -> list[dict[str, Any]]:
    """Load the constraint ledger. Returns [] if absent/unreadable."""
    out: list[dict[str, Any]] = []
    try:
        p = Path(path).expanduser()
        if not p.exists():
            return out
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return out


def _resolve_source(source: str, repo_root: Path, memory_dir: Path | None) -> Path | None:
    """Map a ledger `source` string back to the file it was extracted from."""
    src = source.split("#", 1)[0].strip()
    if not src:
        return None
    if src.startswith("memory/"):
        if memory_dir is None:
            return None
        return memory_dir / src.split("/", 1)[1]
    return repo_root / src


# ── P2b — per-constraint satisfaction (read_but_not_bound caught) ──
#
# ledger_freshness() (P2 v0) answers only "is the ledger bound + fresh" — a COARSE
# signal that can't see whether THIS action satisfies the constraints it touches.
# assess_constraint_satisfaction() is the per-constraint upgrade: given the action
# context (what the claim references, what was actually read/bound this turn, what
# files were edited), it evaluates each machine-checkable constraint and returns a
# real per-action external_eval for compute_dds. A genuine violation drives
# external_eval low → DDS flags "confident while a read constraint was not applied"
# = read_but_not_bound caught. Conservative: only CLEAR violations penalise; anything
# not evaluable in the given context is left out (returns None → ignored), so the
# default behaviour stays non-regressive.

_HOOK_OR_TOOL = re.compile(r"hook|tools/claude_code_mcp|\.claude/|regex", re.I)
_REIMPL_CLAIM = re.compile(r"\b(regex hook|new hook|re-?implement|bespoke)\b", re.I)
_CALLS_CORE_CLAIM = re.compile(r"call(s|ed)? (the )?core|import\w*\s+phionyx_core|via (the )?core", re.I)
_FORBIDDEN_IMPORT = re.compile(r"\b(fastapi|uvicorn|flask|django|sqlalchemy|litellm|anthropic|openai)\b", re.I)


def _norm_paths(xs: Any) -> set[str]:
    return {str(x).lower() for x in (xs or []) if str(x).strip()}


def _ref_matches(ref: str, path: str) -> bool:
    """A referenced source counts as bound by a read path ONLY on a path-SEGMENT or basename
    match — never a raw bidirectional substring. The naive `ref in path or path in ref` lets
    short/common tokens spuriously satisfy binding (e.g. ref 'x' "matches" 'proxy.md'), which
    would mask the very read_but_not_bound violations this check exists to catch."""
    ref = ref.strip().lower()
    if len(ref) < 3:
        return False  # too short to match without spurious hits
    segs = [s for s in re.split(r"[/\\.\s_-]+", path.lower()) if s]
    if ref in segs:                       # exact path-segment / basename-stem match
        return True
    base = path.lower().rsplit("/", 1)[-1]
    return len(ref) >= 4 and ref in base  # substring only within the basename, min length 4


def _eval_read_binding(ctx: dict[str, Any]) -> bool | None:
    """Structural read_but_not_bound check — LEDGER-INDEPENDENT (so the benchmark is
    deterministic). Violated when the action references a source but that source was not
    bound (Read) this turn. Evaluable only when BOTH referenced_sources and read_paths are
    explicitly provided (a real turn without a wired read-set → not evaluable → non-regressive)."""
    if "read_paths" not in ctx:
        return None  # no read-set → binding can't be judged
    refs = _norm_paths(ctx.get("referenced_sources"))
    if not refs:
        return None  # nothing referenced → not evaluable
    read = _norm_paths(ctx.get("read_paths"))
    if not read:
        return False  # referenced sources but bound nothing → read-but-not-bound
    for r in refs:
        if not any(_ref_matches(r, p) for p in read):
            return False  # a referenced source was not bound this turn
    return True


def _eval_call_core_not_reimplement(text: str, ctx: dict[str, Any]) -> bool | None:
    """Violated when an edited hook/tool adds decision logic (e.g. regex) without
    importing phionyx_core, or the claim describes a re-implementation while no core call."""
    edits = [e for e in (ctx.get("edited_files") or []) if _HOOK_OR_TOOL.search(str(e.get("path", "")))]
    if edits:
        for e in edits:
            if e.get("adds_decision_regex") and not e.get("imports_core"):
                return False
        return True
    ct = ctx.get("claim_text", "") or text
    if _REIMPL_CLAIM.search(ct):
        return bool(_CALLS_CORE_CLAIM.search(ct))  # claims reimpl → ok only if also calls core
    return None


def _eval_import_boundary(text: str, ctx: dict[str, Any]) -> bool | None:
    """Violated when an edited phionyx_core file imports a delivery framework."""
    for e in ctx.get("edited_files") or []:
        path = str(e.get("path", ""))
        if "phionyx_core/" in path and _FORBIDDEN_IMPORT.search(str(e.get("added_text", ""))):
            return False
    return None  # not evaluable without core-file edits


# Ledger-constraint evaluators (read_but_not_bound is handled structurally, not here).
_EVALUATORS: dict[str, Any] = {
    "call_core_not_reimplement": _eval_call_core_not_reimplement,
    "import_boundary": _eval_import_boundary,
}


def assess_constraint_satisfaction(
    constraints: list[dict[str, Any]],
    ctx: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """P2b — evaluate machine-checkable constraints against the action context.

    Returns {external_eval, n_relevant, n_satisfied, n_violated, violations:[{id,check_hint,text}],
    evaluated}. `external_eval ∈ [0,1]` feeds compute_dds: 1.0 when nothing relevant is
    evaluable or all satisfied; ≤0.15 when any constraint is clearly violated (so a
    high-confidence claim that violated a read constraint is flagged). Never raises.
    """
    ctx = ctx or {}
    out: dict[str, Any] = {
        "external_eval": 1.0, "n_relevant": 0, "n_satisfied": 0, "n_violated": 0,
        "violations": [], "evaluated": False,
        "source": "constraint_ledger.assess_constraint_satisfaction",
    }
    try:
        # (a) structural read_but_not_bound check — ledger-independent, deterministic.
        rb = _eval_read_binding(ctx)
        if rb is not None:
            out["n_relevant"] += 1
            out["evaluated"] = True
            if rb:
                out["n_satisfied"] += 1
            else:
                out["n_violated"] += 1
                out["violations"].append({
                    "id": "read_but_not_bound", "check_hint": "read_source_first",
                    "text": "referenced source(s) not bound (read) this turn"})
        # (b) per-ledger-constraint checks for the other machine-checkable hints
        for c in constraints or []:
            hint = c.get("check_hint")
            ev = _EVALUATORS.get(hint or "")
            if ev is None:
                continue
            verdict = ev(c.get("text", ""), ctx)
            if verdict is None:
                continue  # not evaluable in this context → ignore (non-regressive)
            out["n_relevant"] += 1
            out["evaluated"] = True
            if verdict:
                out["n_satisfied"] += 1
            else:
                out["n_violated"] += 1
                out["violations"].append({"id": c.get("id"), "check_hint": hint,
                                          "text": (c.get("text") or "")[:160]})
    except Exception:  # pragma: no cover — best-effort, never break the gate
        pass
    # external_eval is derived OUTSIDE the try so it stays consistent with n_violated even if
    # the loop raised mid-way (avoids a desync where n_violated>0 but external_eval reads 1.0).
    if out["n_violated"] > 0:
        out["external_eval"] = 0.15  # clear violation → drive DDS to flag
    elif out["n_relevant"] > 0:
        out["external_eval"] = 1.0
    return out


def ledger_freshness(
    repo_root: Path,
    memory_dir: Path | None = None,
    path: str | Path = DEFAULT_LEDGER,
    now: float | None = None,
) -> dict[str, Any]:
    """P2 — is the session's constraint binding still fresh?

    `bound`  = the ledger exists and holds constraints (B1 ran this session).
    `stale`  = at least one binding SOURCE file was modified AFTER the ledger was
               registered → the binding no longer reflects the source (re-bind needed).
    Returns a structured continuity-binding status. Never raises (best-effort).
    """
    import time as _time
    status: dict[str, Any] = {
        "bound": False, "constraint_count": 0, "machine_checkable_count": 0,
        "stale": False, "stale_sources": [], "registered_ts": None,
        "source": "constraint_ledger.ledger_freshness",
    }
    try:
        recs = load_ledger(path)
        status["constraint_count"] = len(recs)
        status["machine_checkable_count"] = sum(1 for r in recs if r.get("machine_checkable"))
        status["bound"] = len(recs) > 0
        if not recs:
            return status
        ts_str = recs[0].get("registered_ts")
        status["registered_ts"] = ts_str
        try:
            registered_epoch = _time.mktime(_time.strptime(ts_str, "%Y-%m-%dT%H:%M:%S"))
        except (ValueError, TypeError):
            return status  # can't compare → report bound, not stale
        seen: set[str] = set()
        for r in recs:
            src = r.get("source", "")
            if src in seen:
                continue
            seen.add(src)
            fp = _resolve_source(src, repo_root, memory_dir)
            try:
                if fp is not None and fp.exists() and fp.stat().st_mtime > registered_epoch + 1:
                    status["stale_sources"].append(src)
            except OSError:
                continue
        status["stale"] = bool(status["stale_sources"])
    except Exception:  # pragma: no cover — best-effort
        pass
    return status
