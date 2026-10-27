"""
06_export_models.py - fit the two deployable screening models on all 918 patients and save them.

  Step 1: Group A without cholesterol (config A_nochol), stacking of the six classical models.
  Step 2: all 11 features, zeros -> missing, median imputation, no missing indicator (config F_imp).
Calibration (Platt) and abstention thresholds come from inner out-of-fold predictions only.
Also writes models_dir/expected_outputs.json: five fixed patients and the saved models' outputs,
used by tests/test_app.py to detect any change in the saved models or the app logic.

    python scripts/06_export_models.py [--out DIR] [--jobs N]     (default DIR = app/models)
"""
import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from common import *  # noqa

warnings.filterwarnings("ignore")
sys.path.insert(0, str(ROOT / "app"))

STEPS = {"step1": "A_nochol", "step2": "F_imp"}


def fit_bundle(X, y, cfg, jobs, cols=None, variant=None):
    if cols is None:
        cols, variant = CONFIGS[cfg]
    Xtr = X[cols]
    cv = StratifiedKFold(INNER_SPLITS, shuffle=True, random_state=SEED)
    models, oof = {}, {}
    for name in CLASSICAL:
        est, grid = model_specs()[name]
        pipe = Pipeline([("prep", make_preprocessor(cols, variant)), ("clf", est)])
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=jobs, refit=True).fit(Xtr, y)
        models[name] = gs.best_estimator_
        oof[name] = cross_val_predict(clone(gs.best_estimator_), Xtr, y, cv=cv,
                                      method="predict_proba", n_jobs=jobs)[:, 1]
    Z = np.column_stack([logit(oof[m]) for m in CLASSICAL])
    meta = LogisticRegression(C=1.0, max_iter=5000)
    oof_stack = cross_val_predict(meta, Z, y, cv=cv, method="predict_proba")[:, 1]
    meta.fit(Z, y)
    platt = Platt().fit(oof_stack, y)
    q_oof = platt.predict(oof_stack)
    return {
        "config": cfg, "cols": cols, "variant": variant, "classical": CLASSICAL, "models": models,
        "meta": meta, "platt": platt, "thresholds": make_thresholds(q_oof, y), "pi_source": PI_SOURCE,
        "n_train": int(len(y)), "train_ranges": {c: [float(X[c].min()), float(X[c].max())] for c in NUM_COLS if c in cols},
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, q_oof


def cv_rates():
    """Cross-validated rates shown in the app, read from results/table_system_rates.csv."""
    f = RESULTS / "table_system_rates.csv"
    if not f.exists():
        return {}
    R = pd.read_csv(f)
    r = R[(R.model == "STACK") & (R.step1 == "A_nochol") & (R.system == "two-step/e") & (R.y == 1)]
    return {int(s): float(v) for s, v in zip(r.s1, r.r1_low)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "app" / "models"))
    ap.add_argument("--jobs", type=int, default=-1)
    args, _ = ap.parse_known_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    X, y = load_data()
    qs = {}
    for step, cfg in STEPS.items():
        t0 = time.time()
        b, q = fit_bundle(X, y, cfg, args.jobs)
        b["cv_step1_missed_sick_rate"] = cv_rates() if step == "step1" else {}
        joblib.dump(b, out / f"{step}.joblib")
        qs[step] = q
        print(f"{step}: config {cfg}, thresholds " + ", ".join(f"{k}={v:.3f}" for k, v in b["thresholds"].items()
              if k in ("tl_e95", "tl_e98", "th_e95")) + f" ({time.time() - t0:.0f}s)", flush=True)

    # five fixed patients: spread over the Step 2 score distribution
    import logic
    order = np.argsort(qs["step2"])
    picks = [int(order[int(len(order) * f)]) for f in (0.05, 0.30, 0.50, 0.70, 0.95)]
    fixtures = []
    for i in picks:
        row = X.iloc[i]
        inputs = {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in ALL_FEATURES}
        rec = {"row": i, "inputs": inputs}
        for step, fields in (("step1", logic.FIELDS_STEP1), ("step2", logic.FIELDS_STEP2)):
            r = logic.assess(step, {k: inputs[k] for k in fields}, sens_level=95, pi_target=0.05, models_dir=out)
            rec[step] = {"label": r["label"], "q_source": round(r["q_source"], 6), "risk_target": round(r["risk_target"], 6)}
        fixtures.append(rec)
    json.dump(fixtures, open(out / "expected_outputs.json", "w"), indent=1)
    print("saved ->", out)


if __name__ == "__main__":
    main()
