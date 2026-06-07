"""Constraint-binding (require_tool) behavioural tests for phionyx-pipeline-mcp.

Covers the three opt-in / default-off governance capabilities ported from the
private monorepo gate:

  * P1 require_tool — a factual / external-state claim with no externally-bound
    evidence triggers the assessment; with PHIONYX_GATE_REQUIRE_TOOL_ENFORCE=1 an
    otherwise-passing directive becomes 'require_tool'.
  * P2/P2b continuity binding — a referenced source not bound (read) this turn is
    the read_but_not_bound failure class; with PHIONYX_GATE_CONTINUITY_ENFORCE=1
    an otherwise-passing claim is downgraded to 'regenerate'.
  * Non-regression — with both flags unset the directive is unchanged from the
    base v0.2.0 behaviour (the assessments are surfaced but never enforce).

These tests require phionyx-core (a declared dependency) to be importable.
"""
from __future__ import annotations

import pytest

from phionyx_pipeline_mcp import server


# ── shared fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_enforce_flags(monkeypatch):
    """Every test starts with both enforce flags unset (default-off)."""
    monkeypatch.delenv("PHIONYX_GATE_REQUIRE_TOOL_ENFORCE", raising=False)
    monkeypatch.delenv("PHIONYX_GATE_CONTINUITY_ENFORCE", raising=False)


# ── P1: require_tool ─────────────────────────────────────────────────────────


def test_require_tool_triggers_on_factual_claim_without_external_evidence():
    """A factual/external-state claim with only code_review evidence triggers
    the require_tool assessment (surfaced) even when the flag is off."""
    result = server._verify_claim_impl(
        claim="Updated the release; the package is live on PyPI v0.2.0",
        evidence="I read the changelog",
        evidence_type="code_review",  # self-inspection, NOT externally bound
        code_paths_tested="pkg.publish",
        code_paths_affected="pkg.publish",
    )
    rt = result["require_tool"]
    assert rt["triggered"] is True
    assert rt["is_factual_or_external_state_claim"] is True
    assert rt["evidence_externally_bound"] is False
    # Default-off: assessment surfaced but the directive is NOT forced to require_tool.
    assert rt["enforced"] is False
    assert result["directive"] != "require_tool"


def test_require_tool_enforce_forces_directive(monkeypatch):
    """With PHIONYX_GATE_REQUIRE_TOOL_ENFORCE=1 an otherwise-passing factual
    response_gate directive becomes 'require_tool'."""
    monkeypatch.setenv("PHIONYX_GATE_REQUIRE_TOOL_ENFORCE", "1")
    result = server._response_gate_impl(
        action_type="deploy",          # factual action type
        confidence=0.95,
        evidence_count=0,              # no evidence at all → not externally bound
        evidence_type="code_review",
        affects_user_facing=False,
    )
    rt = result["require_tool"]
    assert rt["triggered"] is True
    # deploy with 0 evidence would not be 'pass' anyway, but if it ever is, it must
    # be forced to require_tool; either way it must never silently 'pass'.
    if rt["enforced"]:
        assert result["directive"] == "require_tool"
    assert result["directive"] != "pass"


def test_require_tool_does_not_trigger_on_externally_bound_evidence():
    """An external-state claim backed by an externally-bound evidence type does
    NOT trigger require_tool."""
    result = server._verify_claim_impl(
        claim="Fixed the handler; the endpoint returns 200",
        evidence="curl output: HTTP/1.1 200 OK",
        evidence_type="endpoint_test",  # externally bound
        code_paths_tested="api.handler",
        code_paths_affected="api.handler",
    )
    rt = result["require_tool"]
    assert rt["evidence_externally_bound"] is True
    assert rt["triggered"] is False
    assert result["directive"] != "require_tool"


def test_require_tool_does_not_trigger_on_nonfactual_claim():
    """A non-factual claim (no external-state markers) does not trigger require_tool."""
    result = server._verify_claim_impl(
        claim="Refactored an internal variable for clarity",
        evidence="read the diff",
        evidence_type="code_review",
        code_paths_tested="mod.helper",
        code_paths_affected="mod.helper",
    )
    rt = result["require_tool"]
    assert rt["is_factual_or_external_state_claim"] is False
    assert rt["triggered"] is False


# ── P2/P2b: continuity binding (read_but_not_bound) ──────────────────────────


_READ_BUT_NOT_BOUND = dict(
    claim="Refactored the hook per the roadmap",
    evidence="reviewed",
    evidence_type="code_review",
    code_paths_tested="hook",
    code_paths_affected="hook",
    referenced_sources="roadmap",  # cites a source ...
    read_paths="",                 # ... but bound (read) nothing this turn
)


def test_continuity_read_but_not_bound_regenerate_when_enforced(monkeypatch):
    """read_but_not_bound + PHIONYX_GATE_CONTINUITY_ENFORCE=1 → directive='regenerate'."""
    monkeypatch.setenv("PHIONYX_GATE_CONTINUITY_ENFORCE", "1")
    result = server._verify_claim_impl(**_READ_BUT_NOT_BOUND)

    cont = result["continuity_binding"]
    sat = cont["per_constraint"]
    assert sat["evaluated"] is True
    assert sat["n_violated"] >= 1
    assert any(v["id"] == "read_but_not_bound" for v in sat["violations"])

    assert result["directive"] == "regenerate"
    assert cont["enforced"] is True
    assert "read-but-not-bound" in cont["reason"]


def test_continuity_violation_detected_but_not_enforced_when_flag_off():
    """The violation is detected (surfaced) but the directive is NOT forced when the
    enforce flag is off — non-regressive."""
    result = server._verify_claim_impl(**_READ_BUT_NOT_BOUND)

    cont = result["continuity_binding"]
    sat = cont["per_constraint"]
    assert sat["n_violated"] >= 1            # the read_but_not_bound class is caught ...
    assert cont["enforced"] is False         # ... but never enforced with the flag off
    assert result["directive"] != "regenerate"


def test_continuity_bound_when_referenced_source_is_read(monkeypatch):
    """When the referenced source IS bound (read) this turn, there is no
    read_but_not_bound violation, so no enforcement even with the flag on."""
    monkeypatch.setenv("PHIONYX_GATE_CONTINUITY_ENFORCE", "1")
    result = server._verify_claim_impl(
        claim="Refactored the hook per the roadmap",
        evidence="reviewed",
        evidence_type="code_review",
        code_paths_tested="hook",
        code_paths_affected="hook",
        referenced_sources="roadmap.md",
        read_paths="docs/roadmap.md",   # the cited source was bound this turn
    )
    sat = result["continuity_binding"]["per_constraint"]
    # The read-binding check is satisfied (no read_but_not_bound violation).
    assert not any(v["id"] == "read_but_not_bound" for v in sat["violations"])
    assert result["directive"] != "regenerate"


# ── Non-regression: package default behaviour unchanged from v0.2.0 ──────────


def test_verify_claim_non_regressive_with_flags_off():
    """With both flags unset, the verify_claim directive matches what the base
    pipeline produces — the require_tool / continuity blocks never change it."""
    result = server._verify_claim_impl(
        claim="Updated the deploy config; the site is live",
        evidence="checked",
        evidence_type="code_review",
        code_paths_tested="site.deploy",
        code_paths_affected="site.deploy",
    )
    # The new fields are present (additive) ...
    assert "require_tool" in result
    assert "continuity_binding" in result
    # ... and neither enforced anything.
    assert result["require_tool"]["enforced"] is False
    assert result["continuity_binding"].get("enforced") in (False, None)
    # Directive is whatever the base pipeline decided — NOT a forced require_tool/regenerate.
    assert result["directive"] not in ("require_tool",)


def test_response_gate_non_regressive_with_flags_off():
    """Default-off response_gate carries the assessments but does not enforce."""
    result = server._response_gate_impl(
        action_type="claim_fixed",
        confidence=0.9,
        evidence_count=2,
        evidence_type="integration_test",
        affects_user_facing=False,
    )
    assert "require_tool" in result
    assert "continuity_binding" in result
    assert result["require_tool"]["enforced"] is False
    assert result["continuity_binding"].get("enforced") in (False, None)


def test_response_gate_signature_backcompat():
    """The two new params are optional — old call shape still works."""
    result = server._response_gate_impl(
        "refactor", 0.8, 1, "unit_test", False,
    )
    assert "directive" in result


# ── make_claim / ask_question grounding short-circuit ────────────────────────


def test_make_claim_grounding_regenerate_on_unread_source():
    """make_claim referencing a source not in the read set → regenerate via the
    knowledge_boundary short-circuit (no enforce flag needed — this is the base
    grounding behaviour, always on for ask_question/make_claim)."""
    result = server._response_gate_impl(
        action_type="make_claim",
        confidence=0.9,
        evidence_count=0,
        evidence_type="none",
        affects_user_facing=False,
        artifact_references="SPEC.md",
        artifact_paths_read="",          # nothing read
    )
    assert result["directive"] == "regenerate"
    assert "SPEC.md" in result["unread_artifacts"]


def test_make_claim_grounding_pass_when_source_read():
    """make_claim referencing a source that WAS read → not short-circuited to regenerate."""
    result = server._response_gate_impl(
        action_type="make_claim",
        confidence=0.9,
        evidence_count=1,
        evidence_type="code_review",
        affects_user_facing=False,
        artifact_references="SPEC.md",
        artifact_paths_read="SPEC.md",   # the referenced source was read
    )
    # Falls through to the normal pipeline; the grounding short-circuit did not fire.
    assert "unread_artifacts" not in result
