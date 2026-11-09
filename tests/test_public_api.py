"""The package's public surface and the last uncovered defensive branches.

`__init__.py` resolves its exports lazily (PEP 562), which means a typo in
the lazy map is invisible until someone imports that name at runtime. These
tests import every advertised name so a broken entry fails at test time
rather than in a client's first session.

The remaining cases here pin small defensive branches that the main suites
route around: a trip with no fixes inside an otherwise populated account,
a bimodality sample below the minimum size, and negative-infinity formatting
in the report.
"""

import math

import pytest

import tripphysics
from tripphysics.baseline import NaiveBaseline
from tripphysics.report import _fmt_float, render_report

# Every name the package docstring advertises as public API.
PUBLIC_NAMES = (
    "generate_dataset",
    "cell_id_from_xy",
    "extract_features",
    "FEATURE_NAMES",
    "score_account",
    "Baseline",
    "NaiveBaseline",
    "load_scores_csv",
    "evaluate",
    "render_report",
)


# ---------------------------------------------------------------------------
# Lazy public API
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", PUBLIC_NAMES)
def test_public_name_resolves(name):
    # Exercises the PEP 562 __getattr__ hook: a wrong module path or a
    # renamed attribute in the lazy map surfaces here.
    assert getattr(tripphysics, name) is not None


@pytest.mark.parametrize("name", PUBLIC_NAMES)
def test_public_name_is_advertised_in_all(name):
    assert name in tripphysics.__all__


def test_lazy_attribute_is_cached_after_first_access():
    # The hook writes the resolved object into globals(), so the second
    # access must return the identical object rather than re-importing.
    first = tripphysics.evaluate
    second = tripphysics.evaluate
    assert first is second


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="has no attribute"):
        tripphysics.no_such_function


def test_dir_lists_the_public_api():
    listed = dir(tripphysics)
    for name in PUBLIC_NAMES:
        assert name in listed
    assert "__version__" in listed


def test_version_is_exposed():
    assert isinstance(tripphysics.__version__, str)
    assert tripphysics.__version__.count(".") >= 1


def test_lazily_exported_callables_are_the_real_implementations():
    # Guards against the lazy map pointing at a same-named stub in the wrong
    # module: the objects must be the ones the submodules define.
    from tripphysics.eval import evaluate as direct_evaluate
    from tripphysics.features import extract_features as direct_extract
    assert tripphysics.evaluate is direct_evaluate
    assert tripphysics.extract_features is direct_extract


# ---------------------------------------------------------------------------
# Remaining defensive branches
# ---------------------------------------------------------------------------

def _fix(cell_id=1):
    return {"t": 0.0, "x_m": 0.0, "y_m": 0.0, "accuracy_m": 5.0,
            "speed_mps": 0.0, "is_mock": False, "cell_id": cell_id}


def test_naive_baseline_skips_empty_trips_among_populated_ones():
    # A trip with no fixes must be skipped when computing anchor diversity,
    # not counted as an anchor -- otherwise the divisor would be inflated.
    with_empty = {"trips": [
        {"fixes": [_fix(cell_id=1)]},
        {"fixes": []},                    # skipped
        {"fixes": [_fix(cell_id=2)]},
    ]}
    without_empty = {"trips": [
        {"fixes": [_fix(cell_id=1)]},
        {"fixes": [_fix(cell_id=2)]},
    ]}
    assert NaiveBaseline().score(with_empty) == NaiveBaseline().score(without_empty)


def test_bimodality_abstains_on_a_sample_below_the_minimum():
    # _split_separation returns 0.0 below the minimum trip count. Reached via
    # the public path: five scorable trips is one short of the threshold.
    from tripphysics.features import _MIN_TRIPS_BIMODALITY, _split_separation
    import numpy as np

    small = np.array([0.0, 5.0, 0.0, 5.0])  # clearly bimodal, but too small
    assert small.size < _MIN_TRIPS_BIMODALITY
    assert _split_separation(small) == 0.0


def test_bimodality_abstains_on_a_zero_variance_sample():
    from tripphysics.features import _split_separation
    import numpy as np

    assert _split_separation(np.zeros(20)) == 0.0


def test_fmt_float_renders_both_infinities():
    # report.py formats +inf/-inf in words rather than emitting a bare float.
    assert _fmt_float(math.inf) == "+inf"
    assert _fmt_float(-math.inf) == "-inf"
    assert _fmt_float(0.5, 2) == "0.50"


def test_report_renders_negative_infinite_threshold(tmp_path):
    # threshold_at_fp_budget returns -inf when the budget allows flagging
    # every honest account; the markdown must render that, not crash.
    result = {
        "R0": 1.0, "R1": 1.0, "U": 0.0, "budget_fp_per_10k": 100000.0,
        "pr_auc_baseline": 0.5, "pr_auc_pilot": 0.5,
        "precision_at_500_baseline": 0.1, "precision_at_500_pilot": 0.1,
        "n_accounts": 10, "n_shared": 2,
        "threshold_baseline": -math.inf, "threshold_pilot": -math.inf,
    }
    md_path = tmp_path / "report.md"
    render_report(result, {"seed": 1}, str(tmp_path / "report.json"), str(md_path))
    assert "-inf" in md_path.read_text()
