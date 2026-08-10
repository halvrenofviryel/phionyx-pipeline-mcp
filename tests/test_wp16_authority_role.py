"""WP-16 — authority-role tagging on the self-governance gate (MG-5 authority separation).

The gate tags the DECLARED authority of each claim (implementer_self_claim vs
independent_verification) and NEVER emits acceptance evidence: it is a self-governance check, not
the authorized acceptance stage. "A statement by the implementing agent is never acceptance
evidence" (constitution §4 / MG-5). A self-claim `pass` therefore can never be read as ACCEPTED.
"""
from __future__ import annotations

import phionyx_pipeline_mcp.server as server
from phionyx_pipeline_mcp.server import (
    _authority_block,
    _response_gate_impl,
    _session_report_impl,
)


def test_authority_block_normalizes_and_fails_safe():
    # only the exact literal is independent; everything else -> implementer_self_claim
    assert _authority_block("independent_verification")["claim_authority"] == "independent_verification"
    for bad in ("implementer_self_claim", None, "typo", "INDEPENDENT_VERIFICATION", "", "reviewer"):
        b = _authority_block(bad)
        assert b["claim_authority"] == "implementer_self_claim", bad
        assert b["is_self_claim"] is True


def test_authority_block_never_acceptance_evidence():
    # the gate is not the acceptance stage — acceptance evidence is False for BOTH authorities
    assert _authority_block("implementer_self_claim")["is_acceptance_evidence"] is False
    assert _authority_block("independent_verification")["is_acceptance_evidence"] is False


def test_authority_block_note_names_the_discipline():
    self_note = _authority_block("implementer_self_claim")["note"].lower()
    assert "never accepted" in self_note and "mg-5" in self_note
    assert "acceptance evidence" in self_note


def test_gate_result_carries_authority_and_is_never_acceptance():
    r_self = _response_gate_impl("claim_fixed", 0.9, 3, "unit_test", False, claim_authority="implementer_self_claim")
    r_ind = _response_gate_impl("claim_fixed", 0.9, 3, "unit_test", False, claim_authority="independent_verification")
    assert r_self["authority"]["claim_authority"] == "implementer_self_claim"
    assert r_ind["authority"]["claim_authority"] == "independent_verification"
    # the invariant: neither is acceptance evidence (a pass is developer evidence, never ACCEPTED)
    assert r_self["authority"]["is_acceptance_evidence"] is False
    assert r_ind["authority"]["is_acceptance_evidence"] is False


def test_gate_default_authority_is_self_claim():
    # an unspecified authority defaults to the safe implementer_self_claim
    r = _response_gate_impl("refactor", 0.8, 2, "code_review", False)
    assert r["authority"]["claim_authority"] == "implementer_self_claim"
    assert r["authority"]["is_acceptance_evidence"] is False


def test_session_report_counts_authorities_from_telemetry(tmp_path, monkeypatch):
    """Non-vacuous rollup: persisting gate results makes session_report count the ACTUAL
    authorities from the recorded timeline (would fail if the rollup loop were hardcoded to {})."""
    monkeypatch.setattr(server, "_telemetry_dir", lambda: tmp_path)
    monkeypatch.setattr(server, "_session_id", "wp16test")
    r_self = _response_gate_impl("claim_fixed", 0.9, 3, "unit_test", False, claim_authority="implementer_self_claim")
    r_ind = _response_gate_impl("claim_fixed", 0.9, 3, "unit_test", False, claim_authority="independent_verification")
    server._persist_state("phionyx_response_gate", r_self)
    server._persist_state("phionyx_response_gate", r_ind)
    server._persist_state("phionyx_response_gate", r_self)
    rep = _session_report_impl()
    counts = rep["authority"]["counts"]
    assert counts.get("implementer_self_claim") == 2 and counts.get("independent_verification") == 1
    assert rep["authority"]["acceptance_evidence_emitted"] is False
    assert "mg-5" in rep["authority"]["note"].lower() and "accepted" in rep["authority"]["note"].lower()


def test_session_report_rollup_fail_safe_on_corrupt_session(tmp_path, monkeypatch):
    """A missing/corrupt session file must not crash the report — empty counts, no raise."""
    monkeypatch.setattr(server, "_telemetry_dir", lambda: tmp_path)
    monkeypatch.setattr(server, "_session_id", "wp16corrupt")
    (tmp_path / "session_wp16corrupt.json").write_text("{ not valid json", encoding="utf-8")
    rep = _session_report_impl()  # must not raise
    assert rep["authority"]["counts"] == {}
    assert rep["authority"]["acceptance_evidence_emitted"] is False
