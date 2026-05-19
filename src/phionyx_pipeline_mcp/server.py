"""
Phionyx Claude Code MCP Server — v4 (Extended Pipeline Integration)
====================================================================

Three-layer claim verification:
    Layer 1 — LLM declaration: Claude says what it changed and tested
    Layer 2 — Repo truth: git diff + function extraction verifies declarations
    Layer 3 — Gate decision: physics state + revision thresholds produce directive

The gate is deterministic; the input generation (Layer 1) is stochastic.
The input verification layer (Layer 2) closes the gap.

Pipeline Blocks Used (9/46):
    Block  3 — input_safety_gate (input validation + quality check)
    Block 15 — knowledge_boundary_check (KnowledgeBoundaryDetector)
    Block 16 — trust_evaluation (inline heuristic: t_meta, risk, entropy)
    Block 23 — behavioral_drift_detection (SelfModelDrift)
    Block 37 — phi_computation (calculate_phi_v2_1, Hybrid Resonance v2.2)
    Block 38 — entropy_computation (state-derived)
    Block 39 — confidence_fusion (w_final = 0.4φ + 0.35conf + 0.25safety)
    Block 41 — response_revision_gate (ResponseRevisionGateBlock._decide)
    Block 44 — audit_layer (integrity assessment)

Evidence taxonomy:
    browser_test (0.9) > manual_repro (0.8) > integration_test (0.7)
    > endpoint_test (0.6) > log_inspection (0.5) > unit_test (0.4)
    > code_review (0.3) > none (0.0)

Usage:
    python -m phionyx_pipeline_mcp
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import hashlib
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

from phionyx_core.physics.formulas import calculate_phi_v2_1
from phionyx_core.meta.knowledge_boundary import (
    KnowledgeBoundaryDetector,
    BoundaryAssessment,
)
from phionyx_core.meta.self_model_drift import SelfModelDrift, DriftReport
from phionyx_core.pipeline.blocks.response_revision_gate import (
    ResponseRevisionGateBlock,
    RevisionDirective,
    RevisionThresholds,
)


# ── Evidence Taxonomy ─────────────────────────────────────────────
# Different test types carry different confidence weights.
# A browser test that proves the user-facing flow > a unit test
# that proves an internal function.

EVIDENCE_WEIGHTS: Dict[str, float] = {
    "browser_test": 0.9,
    "manual_repro": 0.8,
    "integration_test": 0.7,
    "endpoint_test": 0.6,
    "log_inspection": 0.5,
    "unit_test": 0.4,
    "code_review": 0.3,
    "none": 0.0,
}


# ── Action-Type Threshold Profiles ───────────────────────────────
# claim_fixed with user-facing impact needs stricter thresholds
# than an investigation or refactor.

_THRESHOLD_PROFILES: Dict[str, RevisionThresholds] = {
    "claim_fixed": RevisionThresholds(
        entropy_damp=0.60,
        entropy_rewrite=0.75,
        entropy_reject=0.90,
        phi_min=0.08,
        confidence_regenerate=0.40,
        confidence_rewrite=0.55,
        drift_rewrite=0.50,
    ),
    "deploy": RevisionThresholds(
        entropy_damp=0.55,
        entropy_rewrite=0.70,
        entropy_reject=0.85,
        phi_min=0.10,
        confidence_regenerate=0.45,
        confidence_rewrite=0.60,
        drift_rewrite=0.45,
    ),
    "default": RevisionThresholds(),  # Standard thresholds from block
}


# ── Context Weights ───────────────────────────────────────────────
# Code review is analytical (high cognitive weight, like SCHOOL context)
_W_C = 0.75
_W_P = 0.25
_GAMMA = 0.15


# ── Session Cognitive State ───────────────────────────────────────

@dataclass
class _CognitiveState:
    """Session cognitive state — maps to EchoState2 (A, V, H) vector."""
    A: float = 0.5
    V: float = 0.0
    H: float = 0.5

    @property
    def stability(self) -> float:
        return max(0.0, 1.0 - self.H)

    @property
    def amplitude(self) -> float:
        return 5.0 + (self.A - 0.5) * 2.0


@dataclass
class _DriftMetrics:
    """Richer drift tracking beyond EMA."""
    false_claim_count: int = 0
    reject_count: int = 0
    reject_then_pass: int = 0
    user_contradictions: int = 0
    last_directive: str = "pass"


# Session singletons
_state = _CognitiveState()
_kb_detector = KnowledgeBoundaryDetector(boundary_threshold=0.4, hedge_threshold=0.6)
_drift_tracker = SelfModelDrift()
_drift_metrics = _DriftMetrics()
_claim_history: list[dict[str, Any]] = []
_previous_phi: float = 0.5
_last_call_time: float = time.time()
_session_id: str = uuid.uuid4().hex[:12]
_session_start: float = time.time()
_call_count: int = 0


# ── Shared-Trace Coordination (ADR-0006) ─────────────────────────
# Both phionyx-pipeline and phionyx-mcp-server agree on a single
# trace_id per Claude Code session so their telemetry can be joined
# without merging the packages. Resolution order:
#   1. PHIONYX_TRACE_ID env var
#   2. PHIONYX_ACTIVE_TRACE_FILE (default ~/.phionyx/active_trace)
#   3. Generate UUID and persist to the file.

_DEFAULT_ACTIVE_TRACE_FILE = "~/.phionyx/active_trace"


def _active_trace_file() -> Path:
    return Path(
        os.environ.get("PHIONYX_ACTIVE_TRACE_FILE", _DEFAULT_ACTIVE_TRACE_FILE)
    ).expanduser()


def _active_trace_id(persist_if_missing: bool = True) -> str:
    """Return the active trace id, creating one if necessary (ADR-0006)."""
    env_value = os.environ.get("PHIONYX_TRACE_ID")
    if env_value:
        return env_value
    path = _active_trace_file()
    if path.exists():
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    new_id = "trace-" + uuid.uuid4().hex[:16]
    if persist_if_missing:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_id, encoding="utf-8")
    return new_id


def _read_envelope_chain() -> dict[str, Any]:
    """Surface the server-MCP envelope chain for the active trace.

    Returns a dict suitable for embedding in session_report. Imports the
    server-MCP package lazily so the pipeline MCP works even when
    phionyx-mcp-server is not installed.
    """
    trace_id = _active_trace_id(persist_if_missing=False)
    try:
        from phionyx_mcp_server.audit_chain import (  # type: ignore[import-not-found]
            FilesystemEnvelopeStore,
            verify_chain,
        )
    except ImportError:
        return {
            "trace_id": trace_id,
            "count": 0,
            "head_hash": None,
            "valid": None,
            "broken_at": None,
            "reason": "phionyx-mcp-server not installed",
        }
    store = FilesystemEnvelopeStore()
    envelopes = list(store.iter_chain(trace_id))
    verdict = verify_chain(envelopes)
    head = envelopes[-1]["integrity"]["current"] if envelopes else None
    return {
        "trace_id": trace_id,
        "count": len(envelopes),
        "head_hash": head,
        "valid": verdict["valid"],
        "broken_at": verdict["broken_at"],
        "reason": verdict["reason"],
    }


# ── Persistence Layer ────────────────────────────────────────────
# Writes telemetry to disk after every tool call so Founder Console
# can poll and display live session state.

def _telemetry_dir() -> Path:
    root = os.environ.get("PHIONYX_PROJECT_ROOT", "")
    if not root:
        root = os.environ.get("PYTHONPATH", ".")
    p = Path(root) / "data" / "mcp_telemetry"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _persist_state(tool_name: str, result: dict) -> None:
    """Write current session state to disk after every tool call."""
    global _call_count
    _call_count += 1

    telemetry_entry = {
        "call_number": _call_count,
        "tool": tool_name,
        "timestamp": time.time(),
        "iso_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "state": {
            "A": round(_state.A, 4),
            "V": round(_state.V, 4),
            "H": round(_state.H, 4),
            "stability": round(_state.stability, 4),
            "amplitude": round(_state.amplitude, 2),
        },
        "phi": result.get("physics", {}).get("phi", _previous_phi),
        "directive": result.get("directive", "n/a"),
        "drift_severity": result.get("drift_severity", "none"),
        "w_final": result.get("confidence_fusion", {}).get("w_final", None),
        "trust": result.get("trust_evaluation", {}).get("direct_trust", None),
        "integrity": result.get("audit", {}).get("integrity_score", None),
    }

    session_file = _telemetry_dir() / f"session_{_session_id}.json"

    # Read existing or create new
    if session_file.exists():
        try:
            session_data = json.loads(session_file.read_text())
        except (json.JSONDecodeError, OSError):
            session_data = _new_session_data()
    else:
        session_data = _new_session_data()

    # Update session-level aggregates
    session_data["last_update"] = time.time()
    session_data["last_update_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    session_data["call_count"] = _call_count
    session_data["current_state"] = telemetry_entry["state"]
    session_data["current_phi"] = telemetry_entry["phi"]
    session_data["drift_metrics"] = asdict(_drift_metrics)
    session_data["claims_total"] = len(_claim_history)
    session_data["timeline"].append(telemetry_entry)

    # Keep timeline bounded (last 200 entries)
    if len(session_data["timeline"]) > 200:
        session_data["timeline"] = session_data["timeline"][-200:]

    try:
        session_file.write_text(json.dumps(session_data, indent=2))
    except OSError:
        pass  # Non-fatal: telemetry loss doesn't block the gate


def _new_session_data() -> dict:
    return {
        "session_id": _session_id,
        "session_start": _session_start,
        "session_start_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(_session_start)),
        "call_count": 0,
        "last_update": _session_start,
        "last_update_iso": "",
        "current_state": {"A": 0.5, "V": 0.0, "H": 0.5, "stability": 0.5, "amplitude": 5.0},
        "current_phi": 0.5,
        "drift_metrics": asdict(_drift_metrics),
        "claims_total": 0,
        "timeline": [],
    }


# ── Helpers ───────────────────────────────────────────────────────

def _compute_dt() -> float:
    global _last_call_time
    now = time.time()
    dt = max(0.001, now - _last_call_time)
    _last_call_time = now
    return dt


def _compute_phi(dt: float) -> Dict[str, float]:
    return calculate_phi_v2_1(
        valence=_state.V,
        arousal=_state.A,
        amplitude=_state.amplitude,
        time_delta=dt,
        gamma=_GAMMA,
        stability=_state.stability,
        entropy=_state.H,
        w_c=_W_C,
        w_p=_W_P,
    )


def _get_revision_gate(action_type: str) -> ResponseRevisionGateBlock:
    """Get a revision gate with action-type-specific thresholds."""
    thresholds = _THRESHOLD_PROFILES.get(action_type, _THRESHOLD_PROFILES["default"])
    return ResponseRevisionGateBlock(thresholds=thresholds)


def _run_revision_gate(
    phi: float,
    confidence: float,
    drift_score: float,
    action_type: str = "default",
) -> RevisionDirective:
    gate = _get_revision_gate(action_type)
    return gate._decide({
        "phi": phi,
        "entropy": _state.H,
        "coherence": 1.0,
        "coherence_leak": False,
        "ethics_enforced": False,
        "ethics_risk": 0.0,
        "conflict_score": 0.0,
        "arbitration_strategy": "none",
        "confidence": confidence,
        "drift_score": drift_score,
        "cep_flagged": False,
    })


def _track_directive(directive: str) -> None:
    """Track directive patterns for richer drift signals."""
    if directive in ("reject", "regenerate"):
        _drift_metrics.reject_count += 1
    if _drift_metrics.last_directive in ("reject", "regenerate") and directive == "pass":
        _drift_metrics.reject_then_pass += 1
    _drift_metrics.last_directive = directive


def _physics_snapshot(phi_result: Dict[str, float]) -> Dict[str, Any]:
    return {
        "phi": round(phi_result["phi"], 4),
        "phi_cognitive": round(phi_result["phi_cognitive"], 4),
        "phi_physical": round(phi_result["phi_physical"], 4),
        "entropy": round(_state.H, 4),
        "valence": round(_state.V, 4),
        "arousal": round(_state.A, 4),
        "stability": round(_state.stability, 4),
        "amplitude": round(_state.amplitude, 2),
        "w_c": _W_C,
        "w_p": _W_P,
        "gamma": _GAMMA,
    }


# ── Block #3: Input Safety Gate ──────────────────────────────────
# Validates MCP tool inputs before processing. Rejects empty/trivial
# claims and ensures minimum quality for verification to be meaningful.

_MIN_CLAIM_LENGTH = 10
_MIN_CHAIN_LENGTH = 3

def _input_safety_check(text: str, input_type: str = "claim") -> Dict[str, Any]:
    """Block #3 adaptation: validate tool input quality."""
    text = text.strip() if text else ""
    length = len(text)
    issues = []

    if length < _MIN_CLAIM_LENGTH:
        issues.append(f"{input_type} too short ({length} chars, min {_MIN_CLAIM_LENGTH})")

    if input_type == "claim" and not any(
        kw in text.lower()
        for kw in ("fix", "add", "implement", "update", "refactor", "remove", "change", "create", "resolve")
    ):
        issues.append(f"{input_type} lacks action verb — may be vague")

    if input_type == "causal_chain":
        links = [part.strip() for part in text.split("→") if part.strip()]
        if len(links) < _MIN_CHAIN_LENGTH:
            issues.append(f"Causal chain has {len(links)} links (min {_MIN_CHAIN_LENGTH})")

    return {
        "gate_passed": len(issues) == 0,
        "input_length": length,
        "input_type": input_type,
        "issues": issues,
    }


# ── Block #39: Confidence Fusion ─────────────────────────────────
# Fuses phi, declared confidence, and evidence weight into w_final.
# Inline adaptation of the v2.5 deterministic path.

def _fuse_confidence(
    phi: float,
    declared_confidence: float,
    evidence_weight: float,
) -> Dict[str, float]:
    """Block #39 adaptation: w_final = 0.4φ + 0.35conf + 0.25safety."""
    phi_signal = max(0.0, min(1.0, phi))
    conf_signal = max(0.0, min(1.0, declared_confidence))
    safety_signal = max(0.0, min(1.0, evidence_weight))

    w_final = 0.4 * phi_signal + 0.35 * conf_signal + 0.25 * safety_signal
    w_final = max(0.0, min(1.0, w_final))

    is_uncertain = w_final < 0.5
    if w_final >= 0.6:
        recommendation = "proceed"
    elif w_final >= 0.4:
        recommendation = "hedge"
    else:
        recommendation = "block"

    return {
        "w_final": round(w_final, 4),
        "is_uncertain": is_uncertain,
        "recommendation": recommendation,
        "phi_signal": round(phi_signal, 4),
        "conf_signal": round(conf_signal, 4),
        "safety_signal": round(safety_signal, 4),
    }


# ── Block #16: Trust Evaluation ──────────────────────────────────
# Inline heuristic: trust = 0.6*t_meta + 0.3*(1-risk) + 0.1*(1-entropy)
# t_meta = declaration_coverage, risk = drift_severity_numeric, entropy = H

def _evaluate_trust(
    declaration_coverage: float,
    drift_severity: str,
    entropy: float,
) -> Dict[str, Any]:
    """Block #16 adaptation: trust from declaration quality, drift risk, entropy."""
    t_meta = max(0.0, min(1.0, declaration_coverage))

    severity_to_risk = {
        "none": 0.0, "low": 0.2, "medium": 0.4, "high": 0.7, "critical": 0.9,
    }
    risk = severity_to_risk.get(drift_severity, 0.3)

    direct_trust = t_meta * 0.6 + (1.0 - risk) * 0.3 + (1.0 - entropy) * 0.1
    direct_trust = max(0.0, min(1.0, direct_trust))
    is_trusted = direct_trust >= 0.5

    return {
        "direct_trust": round(direct_trust, 4),
        "is_trusted": is_trusted,
        "t_meta": round(t_meta, 4),
        "risk": round(risk, 4),
        "entropy": round(entropy, 4),
        "reasoning": f"trust={direct_trust:.2f} (t_meta={t_meta:.2f}, risk={risk:.2f}, H={entropy:.2f})",
    }


# ── Block #44: Audit Layer ───────────────────────────────────────
# Inline integrity assessment: checks verification completeness.

def _audit_integrity(
    has_evidence: bool,
    evidence_weight: float,
    phi: float,
    declaration_trust: str,
    gaps: List[str],
) -> Dict[str, Any]:
    """Block #44 adaptation: integrity score for the verification itself."""
    integrity = 1.0
    issues = []

    if not has_evidence:
        integrity -= 0.25
        issues.append("no_evidence_provided")
    elif evidence_weight < 0.4:
        integrity -= 0.1
        issues.append("low_evidence_weight")

    if phi < 0.05:
        integrity -= 0.2
        issues.append("phi_collapsed")

    if declaration_trust in ("low", "untrusted"):
        integrity -= 0.2
        issues.append(f"declaration_trust_{declaration_trust}")

    if len(gaps) >= 3:
        integrity -= 0.15
        issues.append(f"multiple_gaps ({len(gaps)})")

    integrity = max(0.0, min(1.0, integrity))

    return {
        "integrity_score": round(integrity, 4),
        "status": "ok" if integrity > 0.7 else "degraded" if integrity > 0.4 else "critical",
        "issues": issues,
    }


# ── Input Verification Layer (git-derived) ────────────────────────

def _git_changed_files(ref: str = "HEAD") -> List[str]:
    """Get files changed in working tree + staged vs ref."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", ref],
            capture_output=True, text=True, timeout=10,
            cwd=os.environ.get("PYTHONPATH", "."),
        )
        staged = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True, text=True, timeout=10,
            cwd=os.environ.get("PYTHONPATH", "."),
        )
        files = set()
        for line in (result.stdout + "\n" + staged.stdout).strip().split("\n"):
            if line.strip():
                files.add(line.strip())
        return sorted(files)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def _git_changed_functions(ref: str = "HEAD") -> List[str]:
    """Extract function/method names from git diff."""
    try:
        result = subprocess.run(
            ["git", "diff", "-U0", ref],
            capture_output=True, text=True, timeout=10,
            cwd=os.environ.get("PYTHONPATH", "."),
        )
        functions = set()
        for line in result.stdout.split("\n"):
            if line.startswith("+") and not line.startswith("+++"):
                for match in re.finditer(r"def\s+(\w+)\s*\(", line):
                    functions.add(match.group(1))
                for match in re.finditer(r"(?:async\s+)?function\s+(\w+)\s*\(", line):
                    functions.add(match.group(1))
                for match in re.finditer(r"(?:export\s+)?(?:const|let)\s+(\w+)\s*=\s*(?:async\s*)?\(", line):
                    functions.add(match.group(1))
            elif line.startswith("@@"):
                context = line.split("@@")[-1].strip()
                for match in re.finditer(r"(?:def|function|class)\s+(\w+)", context):
                    functions.add(match.group(1))
        return sorted(functions)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def _verify_paths_impl(
    claimed_affected: str,
    claimed_tested: str,
) -> dict[str, Any]:
    """
    Input verification layer: cross-check LLM declarations against repo truth.

    Runs git diff to discover actually-changed files and functions,
    then compares against what Claude claims to have affected/tested.

    This is the critical anti-gaming layer. Claude can say "I affected
    3 paths" but git diff might show 12 files touching 20 functions.
    """
    claimed_aff = set(p.strip() for p in claimed_affected.split(",") if p.strip())
    claimed_tst = set(p.strip() for p in claimed_tested.split(",") if p.strip())

    # Repo truth
    changed_files = _git_changed_files()
    changed_functions = _git_changed_functions()

    file_basenames = set()
    for f in changed_files:
        base = f.split("/")[-1]
        file_basenames.add(base.replace(".py", "").replace(".ts", "").replace(".tsx", ""))

    repo_surface = set(changed_functions) | file_basenames

    # Discrepancy analysis
    missed_by_llm = sorted(repo_surface - claimed_aff)
    phantom_claims = sorted(claimed_aff - repo_surface)
    untested_repo = sorted(repo_surface - claimed_tst)

    # Trust score: how much of repo truth did the LLM correctly declare?
    if repo_surface:
        declaration_coverage = len(claimed_aff & repo_surface) / len(repo_surface)
    else:
        declaration_coverage = 1.0 if not claimed_aff else 0.5

    # Warnings
    warnings = []
    if missed_by_llm:
        warnings.append(
            f"Git shows {len(missed_by_llm)} affected paths not in your declaration: "
            f"{', '.join(missed_by_llm[:10])}"
        )
    if phantom_claims:
        warnings.append(
            f"{len(phantom_claims)} claimed paths not found in git diff: "
            f"{', '.join(phantom_claims[:10])}"
        )
    if len(repo_surface) > len(claimed_aff) * 2 and claimed_aff:
        warnings.append(
            f"Possible underreporting: repo shows {len(repo_surface)} affected surfaces "
            f"but only {len(claimed_aff)} declared"
        )

    # Trust assessment
    if declaration_coverage >= 0.8:
        trust = "high"
    elif declaration_coverage >= 0.5:
        trust = "medium"
    elif declaration_coverage >= 0.2:
        trust = "low"
    else:
        trust = "untrusted"

    return {
        "declaration_trust": trust,
        "declaration_coverage": round(declaration_coverage, 2),
        "repo_changed_files": changed_files[:20],
        "repo_changed_functions": changed_functions[:30],
        "repo_surface_size": len(repo_surface),
        "claimed_affected_size": len(claimed_aff),
        "missed_by_llm": missed_by_llm[:15],
        "phantom_claims": phantom_claims[:10],
        "untested_repo_paths": untested_repo[:15],
        "warnings": warnings,
    }


# ── Tool Implementations ─────────────────────────────────────────

def _verify_claim_impl(
    claim: str,
    evidence: str,
    evidence_type: str,
    code_paths_tested: str,
    code_paths_affected: str,
) -> dict[str, Any]:
    """
    Three-layer claim verification with 9-block pipeline:
        Block  3: Input safety gate (validate claim quality)
        Block 15: Knowledge boundary check
        Block 16: Trust evaluation
        Block 23: Behavioral drift detection
        Block 37: Phi computation
        Block 38: Entropy computation
        Block 39: Confidence fusion (w_final)
        Block 41: Response revision gate
        Block 44: Audit integrity
    """
    global _previous_phi

    # Block 3: Input safety gate
    safety = _input_safety_check(claim, "claim")
    if not safety["gate_passed"]:
        _state.H = 0.95
        _state.V = -0.5
        dt = _compute_dt()
        phi_result = _compute_phi(dt)
        return {
            "directive": "reject",
            "revision_reasons": safety["issues"],
            "damp_factor": 0.3,
            "coverage": 0.0,
            "declared_coverage": 0.0,
            "evidence_type": evidence_type,
            "evidence_weight": 0.0,
            "boundary_score": 1.0,
            "drift_severity": "none",
            "gaps": safety["issues"],
            "untested_paths": [],
            "input_verification": {"declaration_trust": "untrusted", "declaration_coverage": 0.0, "repo_surface_size": 0, "missed_by_llm": [], "warnings": safety["issues"]},
            "confidence_fusion": {"w_final": 0.0, "recommendation": "block"},
            "trust_evaluation": {"direct_trust": 0.0, "is_trusted": False},
            "audit": {"integrity_score": 0.0, "status": "critical", "issues": ["input_rejected"]},
            "physics": _physics_snapshot(phi_result),
        }

    tested = [p.strip() for p in code_paths_tested.split(",") if p.strip()]
    affected = [p.strip() for p in code_paths_affected.split(",") if p.strip()]

    # Evidence weight from taxonomy
    ev_weight = EVIDENCE_WEIGHTS.get(evidence_type, EVIDENCE_WEIGHTS.get("none", 0.0))
    has_evidence = bool(evidence.strip()) and ev_weight > 0

    # Layer 2: Input verification against git
    path_verification = _verify_paths_impl(code_paths_affected, code_paths_tested)

    # Adjust coverage using both declared and repo truth
    if affected:
        declared_coverage = len(set(tested) & set(affected)) / len(set(affected))
    else:
        declared_coverage = 1.0 if tested else 0.0

    # Blend declared coverage with declaration trust
    trust_factor = path_verification["declaration_coverage"]
    effective_coverage = declared_coverage * (0.5 + 0.5 * trust_factor)

    # Block 38: Entropy computation (state-derived)
    _state.H = max(0.01, min(0.99, 1.0 - effective_coverage))
    _state.V = max(-0.8, min(0.8, (effective_coverage - 0.5) * ev_weight * 2))
    if not has_evidence:
        _state.V = min(_state.V, -0.3)
    drift_magnitude = _drift_tracker.get_drift()
    _state.A = max(0.25, min(0.8, 0.5 + drift_magnitude * 0.5))

    # Block 37: Phi computation
    dt = _compute_dt()
    phi_result = _compute_phi(dt)

    # Block 15: Knowledge boundary check
    boundary: BoundaryAssessment = _kb_detector.assess(
        ood_score=1.0 - effective_coverage,
        graph_relevance=effective_coverage,
        novelty_score=0.5 if not has_evidence else 0.0,
    )

    # Block 23: Behavioral drift detection
    _drift_tracker.observe(effective_coverage)
    drift: DriftReport = _drift_tracker.get_report()

    # Block 39: Confidence fusion (replaces simple confidence calc)
    fusion = _fuse_confidence(
        phi=phi_result["phi"],
        declared_confidence=effective_coverage,
        evidence_weight=ev_weight,
    )
    confidence = fusion["w_final"]

    # Block 16: Trust evaluation
    trust = _evaluate_trust(
        declaration_coverage=path_verification["declaration_coverage"],
        drift_severity=drift.severity.value,
        entropy=_state.H,
    )

    # Block 41: Response revision gate — action-specific thresholds
    revision: RevisionDirective = _run_revision_gate(
        phi=phi_result["phi"],
        confidence=confidence,
        drift_score=drift.current_drift,
        action_type="claim_fixed",
    )
    _track_directive(revision.directive)

    # Gap analysis
    untested = sorted(set(affected) - set(tested))
    gaps = []
    if untested:
        gaps.append(f"Untested code paths: {', '.join(untested)}")
    if not has_evidence:
        gaps.append("No test evidence provided")
    elif ev_weight < 0.5:
        gaps.append(f"Evidence type '{evidence_type}' has low weight ({ev_weight}). Consider browser_test or integration_test.")
    if path_verification["warnings"]:
        gaps.extend(path_verification["warnings"])
    if drift.severity.value in ("high", "critical"):
        gaps.append(f"Drift alert: {drift.severity.value} — {drift.reasoning}")
    if phi_result["phi"] < 0.05:
        gaps.append(f"Phi collapsed below floor ({phi_result['phi']:.4f} < 0.05)")
    if not trust["is_trusted"]:
        gaps.append(f"Trust below threshold: {trust['reasoning']}")

    # Block 44: Audit integrity
    audit = _audit_integrity(
        has_evidence=has_evidence,
        evidence_weight=ev_weight,
        phi=phi_result["phi"],
        declaration_trust=path_verification["declaration_trust"],
        gaps=gaps,
    )

    _previous_phi = phi_result["phi"]
    claim_hash = hashlib.sha256(claim.encode()).hexdigest()[:12]
    _claim_history.append({
        "claim": claim,
        "claim_hash": claim_hash,
        "directive": revision.directive,
        "coverage": round(effective_coverage, 2),
        "phi": phi_result["phi"],
        "entropy": _state.H,
        "evidence_type": evidence_type,
        "evidence_weight": ev_weight,
        "declaration_trust": path_verification["declaration_trust"],
        "w_final": fusion["w_final"],
        "trust": trust["direct_trust"],
        "integrity": audit["integrity_score"],
        "timestamp": time.time(),
    })

    return {
        "directive": revision.directive,
        "revision_reasons": revision.reasons,
        "damp_factor": revision.damp_factor,
        "coverage": round(effective_coverage, 2),
        "declared_coverage": round(declared_coverage, 2),
        "evidence_type": evidence_type,
        "evidence_weight": ev_weight,
        "boundary_score": round(boundary.boundary_score, 2),
        "drift_severity": drift.severity.value,
        "gaps": gaps,
        "untested_paths": untested,
        "input_verification": {
            "declaration_trust": path_verification["declaration_trust"],
            "declaration_coverage": path_verification["declaration_coverage"],
            "repo_surface_size": path_verification["repo_surface_size"],
            "missed_by_llm": path_verification["missed_by_llm"][:5],
            "warnings": path_verification["warnings"],
        },
        "confidence_fusion": fusion,
        "trust_evaluation": trust,
        "audit": audit,
        "physics": _physics_snapshot(phi_result),
    }


def _causal_trace_impl(
    symptom: str,
    causal_chain: str,
) -> dict[str, Any]:
    """Validate a causal debugging chain. Blocks 3, 37, 38, 39, 44."""
    # Block 3: Input safety gate
    safety = _input_safety_check(causal_chain, "causal_chain")

    links = [link.strip() for link in causal_chain.split("→") if link.strip()]

    issues = []
    suggestions = []

    if safety["issues"]:
        issues.extend(safety["issues"])

    if len(links) < 3:
        issues.append(f"Chain has only {len(links)} links — likely missing intermediate causes")
        suggestions.append("Add intermediate steps: what connects each link to the next?")

    code_markers = ["()", ".", "get(", "return", "line", "def ", "class ", "import"]
    code_refs = sum(1 for link in links if any(c in link for c in code_markers))
    specificity = code_refs / max(len(links), 1)
    if specificity < 0.4:
        issues.append(f"Only {code_refs}/{len(links)} links reference specific code — chain may be too abstract")
        suggestions.append("Ground each link in a specific function, line, or variable")

    seen = set()
    for link in links:
        normalized = link.lower().strip()
        if normalized in seen:
            issues.append(f"Circular reference detected: '{link}' appears twice")
        seen.add(normalized)

    if links:
        if symptom.lower() not in links[0].lower() and len(links[0]) < 10:
            suggestions.append(f"First link should clearly state the observable symptom: '{symptom}'")

    has_counterfactual = any(
        "if" in link.lower() or "because" in link.lower() or "when" in link.lower()
        for link in links
    )
    if not has_counterfactual:
        suggestions.append("Add a counterfactual: 'X happens because Y returns Z instead of W'")

    chain_quality = 1.0
    chain_quality -= 0.25 * len(issues)
    if not has_counterfactual:
        chain_quality -= 0.15
    chain_quality = max(0.0, min(1.0, chain_quality))

    # Block 38: Entropy
    _state.H = max(0.01, min(0.99, 1.0 - chain_quality))
    _state.V = max(-0.8, min(0.8, specificity - 0.3))
    _state.A = max(0.25, min(0.8, 0.4 + len(links) * 0.05))

    # Block 37: Phi computation
    dt = _compute_dt()
    phi_result = _compute_phi(dt)

    if len(issues) >= 2:
        directive = "incomplete"
    elif issues:
        directive = "weak"
    else:
        directive = "solid"

    # Block 39: Confidence fusion
    fusion = _fuse_confidence(
        phi=phi_result["phi"],
        declared_confidence=chain_quality,
        evidence_weight=specificity,
    )

    # Block 44: Audit integrity
    audit = _audit_integrity(
        has_evidence=len(links) >= 3,
        evidence_weight=specificity,
        phi=phi_result["phi"],
        declaration_trust="high" if specificity >= 0.6 else "medium" if specificity >= 0.3 else "low",
        gaps=issues,
    )

    return {
        "directive": directive,
        "chain_length": len(links),
        "code_specificity": round(specificity, 2),
        "issues": issues,
        "suggestions": suggestions,
        "links_parsed": links,
        "confidence_fusion": fusion,
        "audit": audit,
        "physics": _physics_snapshot(phi_result),
    }


def _response_gate_impl(
    action_type: str,
    confidence: float,
    evidence_count: int,
    evidence_type: str,
    affects_user_facing: bool,
) -> dict[str, Any]:
    """Response revision gate with 9-block pipeline. Blocks 3, 16, 23, 37, 38, 39, 41, 44."""
    ev_weight = EVIDENCE_WEIGHTS.get(evidence_type, 0.3)
    evidence_factor = min(1.0, evidence_count * ev_weight)

    # Block 38: Entropy computation
    if action_type == "claim_fixed" and evidence_count == 0:
        entropy = 0.96
    elif action_type == "claim_fixed":
        entropy = max(0.01, 1.0 - (confidence * evidence_factor))
    elif affects_user_facing and evidence_count < 1:
        entropy = 0.88
    else:
        entropy = max(0.01, 1.0 - confidence)

    _state.H = entropy
    _state.V = max(-0.8, min(0.8, confidence - 0.5))
    _state.A = max(0.25, min(0.8, 0.5 + (0.2 if affects_user_facing else 0.0)))

    # Block 37: Phi computation
    dt = _compute_dt()
    phi_result = _compute_phi(dt)

    # Block 23: Drift detection
    drift = _drift_tracker.get_report()

    # Block 39: Confidence fusion
    fusion = _fuse_confidence(
        phi=phi_result["phi"],
        declared_confidence=confidence,
        evidence_weight=ev_weight,
    )

    # Block 16: Trust evaluation
    trust = _evaluate_trust(
        declaration_coverage=confidence,
        drift_severity=drift.severity.value,
        entropy=_state.H,
    )

    # Block 41: Response revision gate
    revision: RevisionDirective = _run_revision_gate(
        phi=phi_result["phi"],
        confidence=confidence,
        drift_score=drift.current_drift,
        action_type=action_type,
    )
    _track_directive(revision.directive)

    # Block 44: Audit integrity
    gaps = revision.reasons if revision.directive != "pass" else []
    audit = _audit_integrity(
        has_evidence=evidence_count > 0,
        evidence_weight=ev_weight,
        phi=phi_result["phi"],
        declaration_trust="high" if trust["is_trusted"] else "low",
        gaps=gaps,
    )

    chain = _read_envelope_chain()
    return {
        "directive": revision.directive,
        "revision_reasons": revision.reasons,
        "damp_factor": revision.damp_factor,
        "entropy_floor": revision.entropy_floor,
        "evidence_type": evidence_type,
        "evidence_weight": ev_weight,
        "drift_severity": drift.severity.value,
        "confidence_fusion": fusion,
        "trust_evaluation": trust,
        "audit": audit,
        "physics": _physics_snapshot(phi_result),
        "trace_id": chain["trace_id"],
        "mcp_envelope_chain_head": chain["head_hash"],
    }


def _session_report_impl() -> dict[str, Any]:
    """Session summary with physics, drift metrics, and evidence taxonomy."""
    total = len(_claim_history)
    by_directive: dict[str, int] = {}
    by_evidence: dict[str, int] = {}
    for c in _claim_history:
        by_directive[c["directive"]] = by_directive.get(c["directive"], 0) + 1
        et = c.get("evidence_type", "unknown")
        by_evidence[et] = by_evidence.get(et, 0) + 1

    drift = _drift_tracker.get_report()
    phi_result = _compute_phi(max(0.001, time.time() - _last_call_time))

    return {
        "total_claims": total,
        "directives": by_directive,
        "evidence_types_used": by_evidence,
        "drift_severity": drift.severity.value,
        "drift_magnitude": round(drift.current_drift, 4),
        "drift_metrics": {
            "reject_count": _drift_metrics.reject_count,
            "reject_then_pass": _drift_metrics.reject_then_pass,
            "false_claim_count": _drift_metrics.false_claim_count,
        },
        "auto_corrections": drift.auto_corrections_applied,
        "mean_coverage": round(
            sum(c["coverage"] for c in _claim_history) / max(total, 1), 2
        ),
        "mean_phi": round(
            sum(c.get("phi", 0) for c in _claim_history) / max(total, 1), 4
        ),
        "mean_w_final": round(
            sum(c.get("w_final", 0) for c in _claim_history) / max(total, 1), 4
        ),
        "mean_trust": round(
            sum(c.get("trust", 0) for c in _claim_history) / max(total, 1), 4
        ),
        "mean_integrity": round(
            sum(c.get("integrity", 0) for c in _claim_history) / max(total, 1), 4
        ),
        "current_physics": _physics_snapshot(phi_result),
        "claims": [
            {
                "claim": c["claim"][:80],
                "directive": c["directive"],
                "coverage": c["coverage"],
                "phi": round(c.get("phi", 0), 4),
                "w_final": round(c.get("w_final", 0), 4),
                "trust": round(c.get("trust", 0), 4),
                "integrity": round(c.get("integrity", 0), 4),
                "evidence_type": c.get("evidence_type", "unknown"),
                "declaration_trust": c.get("declaration_trust", "unknown"),
            }
            for c in _claim_history[-10:]
        ],
        "blocks_active": [
            "03:input_safety_gate",
            "15:knowledge_boundary_check",
            "16:trust_evaluation",
            "23:behavioral_drift_detection",
            "37:phi_computation",
            "38:entropy_computation",
            "39:confidence_fusion",
            "41:response_revision_gate",
            "44:audit_layer",
        ],
        "trace_id": _active_trace_id(persist_if_missing=False),
        "mcp_envelope_chain": _read_envelope_chain(),
    }


def _checkpoint_impl(context: str = "") -> dict[str, Any]:
    """Lightweight snapshot — no git diff, no verification overhead."""
    dt = _compute_dt()
    phi_result = _compute_phi(dt)
    drift = _drift_tracker.get_report()

    return {
        "directive": "checkpoint",
        "context": context,
        "session_id": _session_id,
        "trace_id": _active_trace_id(),
        "call_count": _call_count + 1,
        "session_duration_s": round(time.time() - _session_start, 1),
        "drift_severity": drift.severity.value,
        "drift_magnitude": round(drift.current_drift, 4),
        "claims_total": len(_claim_history),
        "drift_metrics": {
            "reject_count": _drift_metrics.reject_count,
            "reject_then_pass": _drift_metrics.reject_then_pass,
            "false_claim_count": _drift_metrics.false_claim_count,
        },
        "physics": _physics_snapshot(phi_result),
    }


# ── MCP Server ────────────────────────────────────────────────────

def main() -> None:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        sys.stderr.write("Install MCP SDK: pip install mcp\n")
        sys.exit(1)

    mcp = FastMCP("phionyx-pipeline", json_response=True)

    @mcp.tool()
    def phionyx_verify_claim(
        claim: str,
        evidence: str,
        evidence_type: str,
        code_paths_tested: str,
        code_paths_affected: str,
    ) -> dict[str, Any]:
        """Verify a completion claim using three-layer verification. Call BEFORE saying 'fixed' or 'done'.

        Layer 1: Parse your declarations (claim, evidence, paths)
        Layer 2: Cross-check paths against git diff (input verification)
        Layer 3: Physics gate (phi + entropy + revision thresholds)

        Args:
            claim: What you're claiming (e.g. "scenario continuation bug is fixed")
            evidence: What test output proves it (e.g. "12 scenes played, quest_complete=True")
            evidence_type: Type of evidence — determines confidence weight.
                          One of: browser_test, manual_repro, integration_test,
                          endpoint_test, log_inspection, unit_test, code_review, none
            code_paths_tested: Comma-separated functions/endpoints you actually tested
            code_paths_affected: Comma-separated functions/endpoints affected by the change
        """
        result = _verify_claim_impl(claim, evidence, evidence_type, code_paths_tested, code_paths_affected)
        _persist_state("phionyx_verify_claim", result)
        return result

    @mcp.tool()
    def phionyx_verify_paths(
        claimed_affected: str,
        claimed_tested: str,
    ) -> dict[str, Any]:
        """Cross-check your claimed code paths against git diff reality. Call to verify your own declarations.

        Compares what you say you affected/tested against what git diff actually shows.
        Returns discrepancies, trust score, and warnings about underreporting.

        Args:
            claimed_affected: Comma-separated paths you claim are affected
            claimed_tested: Comma-separated paths you claim to have tested
        """
        result = _verify_paths_impl(claimed_affected, claimed_tested)
        _persist_state("phionyx_verify_paths", result)
        return result

    @mcp.tool()
    def phionyx_causal_trace(
        symptom: str,
        causal_chain: str,
    ) -> dict[str, Any]:
        """Validate a causal debugging chain. Call when investigating a bug.

        Args:
            symptom: What the user observes (e.g. "scenarios end at scene 2")
            causal_chain: Arrow-separated chain from symptom to root cause
                         (e.g. "0 choices shown → play page reads res.choices → play_card returns empty → make_choice uses wrong key")
        """
        result = _causal_trace_impl(symptom, causal_chain)
        _persist_state("phionyx_causal_trace", result)
        return result

    @mcp.tool()
    def phionyx_response_gate(
        action_type: str,
        confidence: float,
        evidence_count: int,
        evidence_type: str = "code_review",
        affects_user_facing: bool = False,
    ) -> dict[str, Any]:
        """Response revision gate with action-type-specific thresholds. Call before committing.

        Different action types trigger different threshold profiles:
        - claim_fixed: strictest (entropy_reject=0.90, phi_min=0.08)
        - deploy: very strict (entropy_reject=0.85, phi_min=0.10)
        - default: standard pipeline thresholds

        Args:
            action_type: claim_fixed | claim_working | deploy | refactor | investigate
            confidence: Your confidence 0.0-1.0
            evidence_count: Number of independent test/verification points
            evidence_type: Type of evidence (see phionyx_verify_claim for options)
            affects_user_facing: Whether this change is visible to end users
        """
        result = _response_gate_impl(action_type, confidence, evidence_count, evidence_type, affects_user_facing)
        _persist_state("phionyx_response_gate", result)
        return result

    @mcp.tool()
    def phionyx_checkpoint(
        context: str = "",
    ) -> dict[str, Any]:
        """Lightweight physics state snapshot. Call frequently — after completing any subtask,
        before switching context, or when reporting progress.

        This is cheap (no git diff, no verification). Use it to keep the telemetry
        timeline dense so the founder can track session physics in real time.

        Args:
            context: Brief note of what you're doing (e.g. "finished implementing selector")
        """
        result = _checkpoint_impl(context)
        _persist_state("phionyx_checkpoint", result)
        return result

    @mcp.tool()
    def phionyx_session_report() -> dict[str, Any]:
        """Session summary: claims, directives, drift metrics, evidence taxonomy, physics state."""
        result = _session_report_impl()
        _persist_state("phionyx_session_report", result)
        return result

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
