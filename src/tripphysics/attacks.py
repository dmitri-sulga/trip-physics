"""Adversarial attack registry: what this detector was tried against, and what beat it.

An evaluation harness that only reports the score it achieves on its own
corpus is grading its own homework. The registry loaded here is the other
half of the deliverable: a named, versioned list of the manipulations an
account owner could apply to their own telemetry, each carrying a verdict
about whether the current scoring survives it.

Many of these attacks degrade the detector, and several defeat it outright.
That is the point. Every
detector has an evasion surface; the only question is whether the people
operating it know where it is before an adversary maps it for them. A
registry that quietly omitted its own failures would be worth less than no
registry at all, because it would create false confidence about exactly the
accounts that matter.

Registry format
---------------
JSON, not YAML. The package's only runtime dependency is numpy (see
``pyproject.toml``, where that constraint is deliberate and documented), so
PyYAML is unavailable, and a hand-rolled YAML subset parser would be a
silent-misparse risk sitting in the middle of a fraud artifact. JSON parses
with the standard library, has one unambiguous grammar, and the registry
gains nothing from YAML's comments and anchors that a ``notes`` field does
not already provide.

Every entry carries:

``id``          stable identifier, referenced from write-ups and tests
``name``        human name, as an analyst would say it out loud
``mechanism``   the physical or behavioural reason the attack works
``params``      the concrete realization: ``synth.py`` parameters where the
                attack is already parameterized there, otherwise the exact
                manipulation applied to the trip log
``targets``     which of the eleven features it is meant to defeat or exploit
``detected``    ``caught`` / ``partial`` / ``passes`` / ``unmeasured``
``evidence``    HOW that verdict was reached: a measured number with the
                population and seed that produced it, or the literal string
                ``unmeasured -- reasoning only``. Reasoning is never dressed
                up as measurement; an entry whose verdict came from reading
                the code says so.

The verdict scale
-----------------
``caught``      the attack does not meaningfully help: the attacked
                population is still flagged at close to the un-attacked rate
``partial``     detection degrades materially but survives above chance
``passes``      the attack defeats the current scoring
``unmeasured``  no measurement was run; the verdict field is absent and the
                entry states what would have to be measured

``ATTACKS.md`` is GENERATED from this file by :func:`render_markdown` and is
asserted byte-identical in ``tests/test_attacks.py``. A document that
contradicts its own data is the failure mode this repo cares most about
avoiding, so the document is never hand-edited.
"""

from __future__ import annotations

import json
import os
from typing import Any

__all__ = [
    "REGISTRY_PATH",
    "DETECTED_VALUES",
    "REQUIRED_FIELDS",
    "load_registry",
    "verdict_counts",
    "render_markdown",
]

# Registry lives next to the generated document it feeds, one level up from
# the package: it is a reviewable artifact in its own right, not package data
# that happens to be readable.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
REGISTRY_PATH = os.path.join(_REPO_ROOT, "attacks", "registry.json")
MARKDOWN_PATH = os.path.join(_REPO_ROOT, "attacks", "ATTACKS.md")

# Verdicts are a closed set on purpose. A free-text verdict field drifts into
# "mostly caught?" within two revisions, and nobody can then count the
# failures -- which is the one number a reader of this registry actually
# wants.
DETECTED_VALUES = ("caught", "partial", "passes", "unmeasured")

REQUIRED_FIELDS = ("id", "name", "mechanism", "params", "targets",
                   "detected", "evidence")

# The exact feature names the `targets` field may reference. Kept as a literal
# rather than imported from features.FEATURE_NAMES so that a typo in the
# registry fails against the *contract* the registry was written against,
# even if someone renames a feature; the test asserts the two agree.
_FEATURE_NAMES = (
    "max_jump_m", "max_speed_kmh", "accel_p99_ms2", "gps_jitter_m",
    "mock_rate", "cell_mismatch_rate", "handover_gap_score",
    "style_bimodality", "overlap_rate", "night_share", "nonfinite_rate",
)

# Targets that name something other than a feature: the scoring machinery
# itself, or the absence of any feature at all.
_NON_FEATURE_TARGETS = ("score_ordering", "evidence_free_path", "all")

_UNMEASURED_EVIDENCE = "unmeasured -- reasoning only"


def load_registry(path: str | None = None) -> dict[str, Any]:
    """Parse the registry and enforce every structural invariant it claims.

    Validation happens at load time rather than in the tests alone, because
    the renderer below trusts the shape completely and a malformed entry
    would otherwise reach ``ATTACKS.md`` as a plausible-looking row. Raises
    ValueError with the offending id on the first violation.
    """
    with open(path or REGISTRY_PATH, encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, dict) or "attacks" not in data:
        raise ValueError("registry must be an object with an 'attacks' list")
    attacks = data["attacks"]
    if not isinstance(attacks, list) or not attacks:
        raise ValueError("registry 'attacks' must be a non-empty list")

    seen: set[str] = set()
    for entry in attacks:
        if not isinstance(entry, dict):
            raise ValueError("each attack entry must be an object")
        missing = [f for f in REQUIRED_FIELDS if f not in entry]
        if missing:
            raise ValueError(
                f"attack {entry.get('id', '<no id>')!r} missing fields: {missing}")

        aid = entry["id"]
        if not isinstance(aid, str) or not aid:
            raise ValueError(f"attack id must be a non-empty string, got {aid!r}")
        if aid in seen:
            raise ValueError(f"duplicate attack id: {aid!r}")
        seen.add(aid)

        for field in ("name", "mechanism", "params", "evidence"):
            value = entry[field]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"attack {aid!r}: {field} must be a non-empty string")

        # `notes` is optional but, when present, must be a string: the
        # renderer interpolates it directly, so a stray list or number would
        # reach the document as a Python repr.
        if "notes" in entry and not isinstance(entry["notes"], str):
            raise ValueError(f"attack {aid!r}: notes must be a string")

        targets = entry["targets"]
        if not isinstance(targets, list) or not targets:
            raise ValueError(f"attack {aid!r}: targets must be a non-empty list")
        allowed = set(_FEATURE_NAMES) | set(_NON_FEATURE_TARGETS)
        unknown = [t for t in targets if t not in allowed]
        if unknown:
            raise ValueError(f"attack {aid!r}: unknown targets {unknown}")

        verdict = entry["detected"]
        if verdict not in DETECTED_VALUES:
            raise ValueError(
                f"attack {aid!r}: detected must be one of {DETECTED_VALUES}, "
                f"got {verdict!r}")

        # The whole credibility of the registry rests on this rule: a verdict
        # that claims a measurement must carry one, and an entry that did not
        # measure must say so in the exact agreed words rather than in prose
        # that reads like a result.
        if verdict == "unmeasured":
            if entry["evidence"] != _UNMEASURED_EVIDENCE:
                raise ValueError(
                    f"attack {aid!r}: an unmeasured verdict must carry the "
                    f"evidence string {_UNMEASURED_EVIDENCE!r}")
        elif entry["evidence"] == _UNMEASURED_EVIDENCE:
            raise ValueError(
                f"attack {aid!r}: verdict {verdict!r} claims a measurement but "
                f"the evidence says none was taken")

    return data


def verdict_counts(registry: dict[str, Any]) -> dict[str, int]:
    """Count entries per verdict, in the fixed ``DETECTED_VALUES`` order.

    Counting is a function rather than a hand-maintained line in the document
    for the obvious reason: an eyeballed count in a fraud artifact is a
    number that will eventually be wrong, and it will be wrong in the
    flattering direction.
    """
    counts = {v: 0 for v in DETECTED_VALUES}
    for entry in registry["attacks"]:
        counts[entry["detected"]] += 1
    return counts


def _md_escape(text: str) -> str:
    """Escape the one character that would break a Markdown table cell."""
    return text.replace("|", "\\|")


def render_markdown(registry: dict[str, Any]) -> str:
    """Render the registry as the human-readable matrix in ``ATTACKS.md``.

    Grouped by verdict, worst news first: an operator reading this should hit
    the attacks that defeat the detector before the ones it survives. Within
    a group, entries keep registry order, which is thematic (documented
    evasions, then handover geometry, then client-controlled fields, then
    the aggregate-shape attacks).
    """
    attacks = registry["attacks"]
    counts = verdict_counts(registry)
    total = len(attacks)

    lines: list[str] = []
    lines.append("# Attack registry")
    lines.append("")
    lines.append(
        "GENERATED FILE -- do not edit. Rendered from `attacks/registry.json` by")
    lines.append(
        "`tripphysics.attacks.render_markdown`; `tests/test_attacks.py` asserts this")
    lines.append("file is byte-identical to a fresh render.")
    lines.append("")
    for para in registry.get("preamble", []):
        lines.append(para)
        lines.append("")

    lines.append("## Verdict summary")
    lines.append("")
    lines.append(f"{total} attacks recorded.")
    lines.append("")
    lines.append("| verdict | count | meaning |")
    lines.append("| --- | --- | --- |")
    for verdict in DETECTED_VALUES:
        lines.append(f"| `{verdict}` | {counts[verdict]} | "
                     f"{_md_escape(registry['verdict_meanings'][verdict])} |")
    lines.append("")

    for verdict in DETECTED_VALUES:
        group = [a for a in attacks if a["detected"] == verdict]
        if not group:
            continue
        lines.append(f"## {verdict} ({len(group)})")
        lines.append("")
        for entry in group:
            lines.append(f"### `{entry['id']}` -- {entry['name']}")
            lines.append("")
            lines.append(f"**Mechanism.** {entry['mechanism']}")
            lines.append("")
            lines.append(f"**Realization.** {entry['params']}")
            lines.append("")
            lines.append("**Targets.** " + ", ".join(f"`{t}`" for t in entry["targets"]))
            lines.append("")
            lines.append(f"**Evidence.** {entry['evidence']}")
            lines.append("")
            # `notes` is where the interpretation lives -- why a result is
            # weaker or stronger than it looks, what it would mean on real
            # data. Dropping it from the render would leave the document a
            # table of numbers with the reasoning stripped out.
            if entry.get("notes", "").strip():
                lines.append(f"**Notes.** {entry['notes']}")
                lines.append("")

    return "\n".join(lines).rstrip("\n") + "\n"


def main() -> None:
    """Regenerate ``attacks/ATTACKS.md`` from the registry."""
    registry = load_registry()
    with open(MARKDOWN_PATH, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(registry))
    counts = verdict_counts(registry)
    total = len(registry["attacks"])
    print(f"wrote {MARKDOWN_PATH}: {total} attacks "
          + ", ".join(f"{k}={v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
