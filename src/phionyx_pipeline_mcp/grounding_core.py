#!/usr/bin/env python3
"""Claim-grounding core path — the single place detection + decision live.

This module is the *core-calling adapter* for claim/question grounding. ALL
"is this grounded?" logic lives here and is realized THROUGH phionyx-core's
`KnowledgeBoundaryDetector` — never re-implemented as a hand-tuned threshold.

Two consumers share this one decision path:
  * the Stop hook `check_claim_grounding.py` — a pure trigger that pipes the
    draft text + read-set in and obeys the verdict (it contains no detection
    logic of its own), and
  * the gate `phionyx_claude_mcp._response_gate_impl` (action_type in
    {ask_question, make_claim}) — which receives explicit references.

Design split (per founder directive 2026-05-28, "sıfır mantık tetik"):
  - The HOOK decides only *when* to consult the core (something is finalising
    and the grounding gate was not called this turn).
  - THIS module decides *what to look at* (extraction) and *the verdict*
    (KnowledgeBoundaryDetector). Text→source extraction is irreducible adapter
    glue; it is co-located here with the core call, not scattered in a hook.

Robustness: every public function fails OPEN (returns "no finding") if
phionyx-core is unavailable or anything unexpected happens — a grounding
adapter must never break a session.
"""

from __future__ import annotations

import re
from typing import Any

# NOTE (package port): In the monorepo this module ran from a hook subprocess and
# injected the repo root onto sys.path so phionyx_core could be found. In the
# packaged distribution phionyx-core is a declared dependency (pyproject.toml),
# so no sys.path manipulation is needed — the import below resolves normally.


# ─── Extraction patterns (the "what to look at" glue) ────────────────────────
# Named-artifact references. Each match anchors a "was this read?" check.
ARTIFACT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b[A-Za-z][\w\-]*\.md\b"),
    re.compile(r"\b[A-Za-z][\w\-/]*\.py\b"),
    re.compile(r"\b[A-Za-z][\w\-/]*\.tsx?\b"),
    re.compile(r"\b[A-Za-z][\w\-/]*\.json\b"),
    re.compile(r"https?://\S+"),
    re.compile(r"#\d+\b"),
    re.compile(r"\bpost[_ ]?\d+\b", re.IGNORECASE),
    re.compile(r"\b[Pp]aper\s+\d+\b"),
    re.compile(r"\b[Bb]ook\s+\d+\b"),
    re.compile(r"(?<![\w/])/[a-z][a-z0-9\-]*(?:/[a-z][a-z0-9\-]*)+\b"),
]

_ARTIFACT_BLACKLIST = {"README.md", "node_modules", "package.json", "/path/to", "/etc/passwd"}

# Question detection — conservative (ends with "?" OR explicit interrogative).
QUESTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\?\s*$", re.MULTILINE),
    re.compile(r"\b(should we|should I|do we|do you|does it|is it|is the|"
               r"are these|are those|are we|would you|can you|hangisi|"
               r"yapayım mı|silelim mi|değiştirelim mi|edelim mi)\b", re.IGNORECASE),
]

# Attribution cue — a verb/preposition binding a proposition to a source.
# A plain mention ("I edited foo.py") carries no cue and must NOT match.
ATTRIBUTION_CUES: re.Pattern[str] = re.compile(
    r"\b(says?|stated?|states|argues?|claims?|shows?|defines?|describes?|"
    r"contains?|covers?|establishes?|specif(?:y|ies)|proves?|demonstrates?|"
    r"mentions?|notes?|requires?|lists?|documents?|according to|per the|as the)\b"
    r"|"  # Turkish attribution cues
    r"\b(diyor|der|belirt(?:iyor|ir)|söyl(?:üyor|er)|gösteriyor|tanıml(?:ıyor|ar)|"
    r"içeriyor|kapsıyor|göre|açıkl(?:ıyor|ar)|iddia ediyor)\b",
    re.IGNORECASE,
)

_SENTENCE_SPLIT: re.Pattern[str] = re.compile(r"(?<=[.!?\n])\s+")


def _artifacts_in(text: str) -> set[str]:
    refs: set[str] = set()
    for pattern in ARTIFACT_PATTERNS:
        for m in pattern.finditer(text):
            ref = m.group(0).strip(".,;:)(\"'`")
            if ref and len(ref) > 1:
                refs.add(ref)
    return {r for r in refs if r not in _ARTIFACT_BLACKLIST}


def extract_grounding_targets(text: str) -> set[str]:
    """Named sources the draft is accountable for this turn.

    Union of (a) artifacts named inside a QUESTION, and (b) sources the draft
    ATTRIBUTES content to (attribution cue + artifact in the same sentence).
    A plain mention with neither a question nor an attribution cue contributes
    nothing.
    """
    targets: set[str] = set()
    # (a) questions about a named artifact → all artifacts in the draft
    if any(p.search(text) for p in QUESTION_PATTERNS):
        targets |= _artifacts_in(text)
    # (b) attribution claims → only artifacts in attributing sentences
    for sentence in _SENTENCE_SPLIT.split(text):
        if ATTRIBUTION_CUES.search(sentence):
            targets |= _artifacts_in(sentence)
    return targets


def reference_covered(ref: str, read_set: set[str]) -> bool:
    """Lenient path-vs-basename match between a reference and the read set."""
    if ref in read_set:
        return True
    base = ref.rsplit("/", 1)[-1]
    if base and base in read_set:
        return True
    for r in read_set:
        if r.endswith(ref) or ref.endswith(r):
            return True
    return False


def assess_sources(refs: set[str], read_set: set[str]) -> dict[str, Any]:
    """The verdict — realized THROUGH phionyx_core.KnowledgeBoundaryDetector.

    For each referenced source: a read source maps to in-distribution / high
    graph-relevance / low novelty (→ within boundary); an unread source maps to
    OOD / low relevance / high novelty (→ outside). The core module decides
    within/without; this adapter does not hand-tune the threshold.

    Returns {"ungrounded": [...sorted...], "reasons": [...],
             "source": "<module>", "available": bool}. Fails OPEN
    (empty ungrounded) if phionyx-core cannot be imported.
    """
    if not refs:
        return {"ungrounded": [], "reasons": [], "source": "", "available": True}
    try:
        from phionyx_core.meta.knowledge_boundary import KnowledgeBoundaryDetector
    except Exception as exc:  # pragma: no cover — fail open if core missing
        return {
            "ungrounded": [],
            "reasons": [f"knowledge_boundary unavailable: {exc!r}"],
            "source": "",
            "available": False,
        }

    detector = KnowledgeBoundaryDetector(boundary_threshold=0.4, hedge_threshold=0.6)
    ungrounded: list[str] = []
    reasons: list[str] = []
    for ref in sorted(refs):
        is_read = reference_covered(ref, read_set)
        b = detector.assess(
            ood_score=0.1 if is_read else 0.9,
            graph_relevance=0.9 if is_read else 0.2,
            novelty_score=0.0 if is_read else 0.8,
        )
        if not b.within_boundary:
            ungrounded.append(ref)
            reasons.append(f"{ref}: {b.reasoning}")
    return {
        "ungrounded": ungrounded,
        "reasons": reasons,
        "source": "phionyx_core.meta.knowledge_boundary.KnowledgeBoundaryDetector",
        "available": True,
    }


def evaluate_draft(text: str, read_set: set[str]) -> dict[str, Any]:
    """Hook entry point: extract targets from the draft, then assess them.

    The hook calls only this — it contains no detection logic itself.
    """
    return assess_sources(extract_grounding_targets(text), read_set)
