"""Pipeline tests: calibration formula, thresholds, decision rule, and a no-leakage check on the real code."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from common import (CONFIGS, decide, logit, make_thresholds, prior_shift, accuracy_weights,
                    corrected_resampled_ttest, load_data)

ROOT = Path(__file__).resolve().parents[1]


def test_prior_shift_hand_examples():
    # equal priors -> unchanged
    assert prior_shift(0.3, 0.5, 0.5) == pytest.approx(0.3)
    # q = 0.5 at source prevalence 0.5, target 0.1: odds 1 * (0.1/0.9) = 1/9 -> p = 0.1
    assert prior_shift(0.5, 0.1, 0.5) == pytest.approx(0.1)
    # q = 0.8 (odds 4), source 0.5, target 0.2 (odds 0.25): new odds 1 -> p = 0.5
    assert prior_shift(0.8, 0.2, 0.5) == pytest.approx(0.5)


def test_prior_shift_is_monotone():
    q = np.linspace(0.01, 0.99, 50)
    assert np.all(np.diff(prior_shift(q, 0.05, 508 / 918)) > 0)


def test_decide_rule():
    out = decide([0.05, 0.5, 0.95], tl=0.2, th=0.8)
    assert list(out) == [0, 2, 1]
    assert decide([0.5], tl=0.9, th=0.1)[0] == 2        # both cuts fire -> Not sure


def test_thresholds_release_rates():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(500, int), np.zeros(400, int)]
    q = np.r_[rng.beta(5, 2, 500), rng.beta(2, 5, 400)]
    t = make_thresholds(q, y)
    for s in (90, 95, 98):
        a = 1 - s / 100
        assert (q[y == 1] < t[f"tl_e{s}"]).mean() <= a + 1e-12          # empirical: at most (1-s) sick below
        assert (q[y == 1] < t[f"tl_c{s}"]).mean() <= a + 1e-12          # conformal: same bound
        assert t[f"tl_c{s}"] <= t[f"tl_e{s}"] + 1e-12                    # conformal is the more cautious cut
    assert (q[y == 0] > t["th_e95"]).mean() <= 0.05 + 1e-12


def test_accuracy_weights_sum_to_one_and_favour_best():
    w = accuracy_weights([0.80, 0.85, 0.86])
    assert w.sum() == pytest.approx(1.0) and w.argmax() == 2


def test_corrected_ttest_returns_p_in_unit_interval():
    d, t, p = corrected_resampled_ttest([.9, .8, .85, .88], [.8, .8, .82, .8], 800, 90)
    assert 0 <= p <= 1


def test_no_leakage_test_labels_never_used_in_fit():
    """Flip every outer-test label; all fitted quantities (probabilities, thresholds) must be identical."""
    spec = importlib.util.spec_from_file_location("train_models", ROOT / "scripts" / "02_train_models.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    X, y = load_data()
    rng = np.random.default_rng(1)
    idx = rng.choice(len(y), 260, replace=False)
    X, y = X.iloc[idx].reset_index(drop=True), y[idx]
    tr, te = np.arange(0, 200), np.arange(200, 260)
    runs = []
    for flip in (False, True):
        y2 = y.copy()
        if flip:
            y2[te] = 1 - y2[te]
        out = {"pred": [], "thr": [], "inner": []}
        tm.fit_one(X, y2, tr, te, "cv", 0, 0, "A_nochol", 1, out)
        runs.append(out)
    for a, b in zip(runs[0]["pred"], runs[1]["pred"]):
        assert np.allclose(a.p_raw, b.p_raw) and np.allclose(a.p_cal, b.p_cal)
    for a, b in zip(runs[0]["thr"], runs[1]["thr"]):
        for k in a:
            if k.startswith(("tl_", "th_")):
                assert a[k] == pytest.approx(b[k])
    for a, b in zip(runs[0]["inner"], runs[1]["inner"]):
        assert a["oof_auc"] == pytest.approx(b["oof_auc"])
