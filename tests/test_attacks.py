"""Structural guarantees for the adversarial attack registry.

The registry's value is entirely in being trustworthy: it is read as a
statement about what this detector has and has not survived, and it is the
one artifact in the repo that is *supposed* to contain bad news. Three
classes of rot would destroy that, and each is pinned here.

* **A verdict without a measurement.** Any entry claiming the detector caught
  or partially caught an attack has to carry evidence saying how that was
  established. The failure mode is not malice but drift -- a reasoned guess
  written in the past tense reads exactly like a result six months later.
* **A document that disagrees with its data.** ``ATTACKS.md`` is generated,
  so the only way it can contradict ``registry.json`` is if someone edited
  the document by hand. Asserted byte-for-byte rather than by parsing the
  markdown back, because a lenient comparison would let formatting drift
  accumulate until the assertion means nothing.
* **Targets naming features that do not exist.** A registry entry pointing at
  a renamed or deleted feature is silently describing an attack against
  nothing.

The counts themselves are deliberately NOT pinned to specific numbers here.
Adding a measured attack is the intended direction of travel for this file,
and a test that has to be edited every time an attack is added trains the
next person to edit tests instead of reading them.
"""

import json

import pytest

from tripphysics.attacks import (
    DETECTED_VALUES,
    MARKDOWN_PATH,
    REQUIRED_FIELDS,
    _UNMEASURED_EVIDENCE,
    load_registry,
    render_markdown,
    verdict_counts,
)
from tripphysics.features import FEATURE_NAMES


@pytest.fixture(scope="module")
def registry():
    return load_registry()


def test_registry_parses_and_is_non_empty(registry):
    """The shipped registry loads through the validating loader."""
    assert registry["attacks"], "registry contains no attacks"


def test_every_entry_has_every_required_field(registry):
    for entry in registry["attacks"]:
        missing = [f for f in REQUIRED_FIELDS if f not in entry]
        assert not missing, f"{entry.get('id')!r} is missing {missing}"


def test_ids_are_unique(registry):
    ids = [a["id"] for a in registry["attacks"]]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    assert not duplicates, f"duplicate attack ids: {duplicates}"


def test_verdicts_are_from_the_allowed_set(registry):
    for entry in registry["attacks"]:
        assert entry["detected"] in DETECTED_VALUES, (
            f"{entry['id']!r} has verdict {entry['detected']!r}")


def test_decided_verdicts_carry_evidence(registry):
    """caught / partial / passes must each say how the verdict was reached.

    This is the assertion the whole file exists for. A verdict is a claim
    about the detector's behaviour; without evidence attached it is an
    opinion formatted as a result.
    """
    for entry in registry["attacks"]:
        if entry["detected"] == "unmeasured":
            continue
        evidence = entry["evidence"].strip()
        assert evidence, f"{entry['id']!r} has an empty evidence field"
        assert evidence != _UNMEASURED_EVIDENCE, (
            f"{entry['id']!r} claims verdict {entry['detected']!r} while its "
            "evidence says nothing was measured")


def test_unmeasured_entries_admit_it_in_the_agreed_words(registry):
    """An unmeasured entry must be unmistakable, not merely vague.

    Pinning the exact string keeps 'unmeasured' greppable and stops the
    category from being softened into prose that reads like a finding.
    """
    for entry in registry["attacks"]:
        if entry["detected"] == "unmeasured":
            assert entry["evidence"] == _UNMEASURED_EVIDENCE, (
                f"{entry['id']!r} is unmeasured but its evidence field reads "
                f"{entry['evidence']!r}")


def test_measured_evidence_names_a_population_size(registry):
    """Measured evidence has to state what it was measured on.

    A bare "31/300 caught" is unreproducible: at these population sizes the
    repo's own documented spread (U sd ~8pp at 2000 accounts) is large enough
    that the reader needs the n and the seed to know how much of a difference
    is real. Requiring the word 'seed' is a cheap proxy for that discipline.
    """
    for entry in registry["attacks"]:
        if entry["detected"] == "unmeasured":
            continue
        assert "seed" in entry["evidence"], (
            f"{entry['id']!r} reports a measurement without naming the seed "
            "that produced it")


def test_targets_name_real_features_or_declared_non_features(registry):
    """Every feature named in `targets` still exists in features.py."""
    declared = set(FEATURE_NAMES) | {"score_ordering", "evidence_free_path", "all"}
    for entry in registry["attacks"]:
        unknown = [t for t in entry["targets"] if t not in declared]
        assert not unknown, f"{entry['id']!r} targets unknown names {unknown}"


def test_registry_documents_every_verdict_value(registry):
    """The rendered summary table needs a meaning for each verdict."""
    for verdict in DETECTED_VALUES:
        assert verdict in registry["verdict_meanings"], (
            f"no documented meaning for verdict {verdict!r}")


def test_verdict_counts_sum_to_the_entry_count(registry):
    counts = verdict_counts(registry)
    assert sum(counts.values()) == len(registry["attacks"])


def test_committed_markdown_matches_a_fresh_render(registry):
    """ATTACKS.md is generated; a hand edit must fail the suite.

    Byte-identical on purpose. The document is the thing an outside reader
    sees, so any divergence between it and the data means the reader is being
    shown something the repo cannot back up.
    """
    with open(MARKDOWN_PATH, encoding="utf-8") as fh:
        committed = fh.read()
    assert committed == render_markdown(registry), (
        "attacks/ATTACKS.md is stale or hand-edited; regenerate with "
        "`python -m tripphysics.attacks`")


def test_loader_rejects_a_duplicate_id(tmp_path, registry):
    """The loader's own invariants fail loudly rather than rendering junk."""
    broken = json.loads(json.dumps(registry))
    broken["attacks"].append(json.loads(json.dumps(broken["attacks"][0])))
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate attack id"):
        load_registry(str(path))


def test_loader_rejects_an_unmeasured_claim_with_a_verdict(tmp_path, registry):
    """The evidence rule is enforced at load time, not only in this file."""
    broken = json.loads(json.dumps(registry))
    broken["attacks"][0]["detected"] = "caught"
    broken["attacks"][0]["evidence"] = _UNMEASURED_EVIDENCE
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="claims a measurement"):
        load_registry(str(path))


def test_loader_rejects_an_unknown_verdict(tmp_path, registry):
    broken = json.loads(json.dumps(registry))
    broken["attacks"][0]["detected"] = "mostly caught?"
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="detected must be one of"):
        load_registry(str(path))


def test_loader_rejects_an_unknown_target(tmp_path, registry):
    broken = json.loads(json.dumps(registry))
    broken["attacks"][0]["targets"] = ["max_jump_m", "no_such_feature"]
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown targets"):
        load_registry(str(path))


def test_registry_records_at_least_one_defeat(registry):
    """The registry must contain bad news or it is not doing its job.

    Not a tautology worth deleting: a future edit that quietly reclassifies
    the failures upward would pass every other test in this file. This one
    asserts the artifact's purpose -- an evasion surface exists, and this
    document is where it is written down.
    """
    counts = verdict_counts(registry)
    assert counts["passes"] > 0, (
        "no attack in the registry defeats the detector, which is not a "
        "credible result for any scoring function")
