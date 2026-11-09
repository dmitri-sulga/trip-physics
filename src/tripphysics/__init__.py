"""trip-physics: shadow-evaluation harness for profile-sharing detection.

Public API (lazy re-exports; each name resolves on first access so the
package imports cleanly even if an optional submodule is absent):

    generate_dataset   synth.py     -- labelled synthetic trip-log corpus
    extract_features   features.py  -- eleven account-level physics features
    score_account      detector.py  -- physics-first suspicion score in [0, 1]
    Baseline           baseline.py  -- scoring protocol for the incumbent stack
    NaiveBaseline      baseline.py  -- placeholder stand-in; replace with yours
    load_scores_csv    baseline.py  -- inject production scores from a CSV
    evaluate           eval.py      -- R0/R1/U at a fixed false-positive budget
    render_report      report.py    -- JSON + Markdown report artifacts

Determinism contract: every entry point is seed-driven and free of
wall-clock reads; identical inputs reproduce identical outputs byte for
byte. Runtime dependencies: Python 3.11+ and numpy, nothing else.
"""

from __future__ import annotations

from importlib import import_module

__version__ = "0.1.0"

# name -> (relative module, attribute)
_LAZY: dict[str, tuple[str, str]] = {
    "generate_dataset": (".synth", "generate_dataset"),
    "cell_id_from_xy": (".synth", "cell_id_from_xy"),
    "extract_features": (".features", "extract_features"),
    "FEATURE_NAMES": (".features", "FEATURE_NAMES"),
    "score_account": (".detector", "score_account"),
    "Baseline": (".baseline", "Baseline"),
    "NaiveBaseline": (".baseline", "NaiveBaseline"),
    "load_scores_csv": (".baseline", "load_scores_csv"),
    "evaluate": (".eval", "evaluate"),
    "render_report": (".report", "render_report"),
}

__all__ = ["__version__", *sorted(_LAZY)]


def __getattr__(name: str):
    """Resolve public names on demand (PEP 562)."""
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        obj = getattr(import_module(module_name, __name__), attr)
        globals()[name] = obj   # cache: subsequent access skips this hook
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
