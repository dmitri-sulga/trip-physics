"""Determinism: same seed => same dataset, byte-identical reports.

No wall-clock leaks into the pipeline; two full runs with the same seed must
match down to the last byte of the rendered report files.
"""

from tripphysics.baseline import NaiveBaseline
from tripphysics.detector import score_account
from tripphysics.eval import evaluate
from tripphysics.features import extract_features
from tripphysics.report import render_report
from tripphysics.synth import generate_dataset


def _run_pipeline(out_json: str, out_md: str, seed: int) -> dict:
    dataset = generate_dataset(n_accounts=200, shared_fraction=0.05, seed=seed)
    accounts = dataset["accounts"]
    labels = {a["account_id"]: a["is_shared"] for a in accounts}
    naive = NaiveBaseline()
    baseline_scores = {a["account_id"]: naive.score(a) for a in accounts}
    detector_scores = {a["account_id"]: score_account(extract_features(a)) for a in accounts}
    result = evaluate(labels, baseline_scores, detector_scores, budget_fp_per_10k=15.0)
    render_report(result, dataset["meta"], out_json, out_md)
    return result


def test_same_seed_same_dataset():
    a = generate_dataset(n_accounts=60, shared_fraction=0.05, seed=7)
    b = generate_dataset(n_accounts=60, shared_fraction=0.05, seed=7)
    assert a == b


def test_different_seed_different_dataset():
    a = generate_dataset(n_accounts=60, shared_fraction=0.05, seed=7)
    b = generate_dataset(n_accounts=60, shared_fraction=0.05, seed=8)
    assert a != b


def test_same_seed_identical_report_files(tmp_path):
    json_1 = tmp_path / "run1.json"
    md_1 = tmp_path / "run1.md"
    json_2 = tmp_path / "run2.json"
    md_2 = tmp_path / "run2.md"

    result_1 = _run_pipeline(str(json_1), str(md_1), seed=11)
    result_2 = _run_pipeline(str(json_2), str(md_2), seed=11)

    assert result_1 == result_2
    assert json_1.read_bytes() == json_2.read_bytes()
    assert md_1.read_bytes() == md_2.read_bytes()
