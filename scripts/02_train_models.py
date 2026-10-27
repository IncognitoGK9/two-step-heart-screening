"""
02_train_models.py - nested repeated cross-validation of 7 single models plus a soft vote and a
stacking combiner, for five feature configurations.

Per outer split everything that is fitted or chosen uses the training part only:
  1. 5-fold inner grid search (accuracy) for each model;
  2. inner out-of-fold probabilities of the tuned model on the training part;
  3. Platt calibration, abstention thresholds and conformal cut-offs from those out-of-fold probabilities;
  4. the stacking combiner (logistic regression on the logits of the six classical models' out-of-fold
     probabilities).
The outer test part is touched once, only to produce probabilities.

Outputs (results/): preds.csv, thresholds.csv, inner_cv.csv
    python 02_train_models.py [--quick] [--jobs N] [--limit-folds K] [--configs F_raw A_chol ...]
"""
import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from sklearn.pipeline import Pipeline

from common import *  # noqa

warnings.filterwarnings("ignore")


def fit_one(X, y, tr, te, split, repeat, fold, cfg, jobs, out):
    cols, variant = CONFIGS[cfg]
    Xtr, Xte, ytr, yte = X.iloc[tr][cols], X.iloc[te][cols], y[tr], y[te]
    cv = StratifiedKFold(INNER_SPLITS, shuffle=True, random_state=SEED + 1000 * repeat + fold)
    oof, tst, inner_acc, params = {}, {}, {}, {}
    for name, (est, grid) in model_specs().items():
        pipe = Pipeline([("prep", make_preprocessor(cols, variant)), ("clf", est)])
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=jobs, refit=True).fit(Xtr, ytr)
        oof[name] = cross_val_predict(clone(gs.best_estimator_), Xtr, ytr, cv=cv,
                                      method="predict_proba", n_jobs=jobs)[:, 1]
        tst[name] = gs.predict_proba(Xte)[:, 1]
        inner_acc[name] = gs.best_score_
        params[name] = json.dumps({k.replace("clf__", ""): str(v) for k, v in gs.best_params_.items()})

    # accuracy-weighted soft vote (original proposal) over the six classical models
    w = accuracy_weights([inner_acc[m] for m in CLASSICAL])
    oof["SOFT"] = np.average([oof[m] for m in CLASSICAL], axis=0, weights=w)
    tst["SOFT"] = np.average([tst[m] for m in CLASSICAL], axis=0, weights=w)
    params["SOFT"] = json.dumps({"tau": TAU, "weights": [round(float(v), 4) for v in w]})

    # stacking: logistic-regression combiner on out-of-fold logits (training part only)
    Z = np.column_stack([logit(oof[m]) for m in CLASSICAL])
    Zte = np.column_stack([logit(tst[m]) for m in CLASSICAL])
    meta = LogisticRegression(C=1.0, max_iter=5000)
    oof["STACK"] = cross_val_predict(meta, Z, ytr, cv=cv, method="predict_proba")[:, 1]
    tst["STACK"] = clone(meta).fit(Z, ytr).predict_proba(Zte)[:, 1]
    params["STACK"] = "{}"

    for name in ALL_MODELS:
        platt = Platt().fit(oof[name], ytr)
        q_oof, q_te = platt.predict(oof[name]), platt.predict(tst[name])
        key = dict(split=split, repeat=repeat, fold=fold, config=cfg, model=name)
        out["pred"].append(pd.DataFrame({**key, "row_id": te, "y": yte, "p_raw": tst[name], "p_cal": q_te}))
        out["thr"].append({**key, **make_thresholds(q_oof, ytr)})
        out["inner"].append({**key, "inner_acc": inner_acc.get(name, np.nan),
                             "oof_auc": roc_auc_score(ytr, oof[name]),
                             "test_acc": float(((tst[name] >= 0.5) == yte).mean()),
                             "n_train": len(tr), "n_test": len(te), "best_params": params[name]})


def save(out):
    pd.concat(out["pred"]).to_csv(RESULTS / "preds.csv", index=False)
    pd.DataFrame(out["thr"]).to_csv(RESULTS / "thresholds.csv", index=False)
    pd.DataFrame(out["inner"]).to_csv(RESULTS / "inner_cv.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 outer repeat")
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--limit-folds", type=int, default=0, help="debug: only the first K outer folds per repeat")
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--no-holdout", action="store_true")
    args = ap.parse_args()
    repeats = 1 if args.quick else args.repeats

    X, y = load_data()
    out = {"pred": [], "thr": [], "inner": []}
    t0 = time.time()
    outer = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=repeats, random_state=SEED)
    n_total = (args.limit_folds or N_SPLITS) * repeats
    done = 0
    for i, (tr, te) in enumerate(outer.split(X, y)):
        r, f = divmod(i, N_SPLITS)
        if args.limit_folds and f >= args.limit_folds:
            continue
        for cfg in args.configs:
            fit_one(X, y, tr, te, "cv", r, f, cfg, args.jobs, out)
        done += 1
        el = time.time() - t0
        last = {d["model"]: d["test_acc"] for d in out["inner"][-len(ALL_MODELS):]}
        print(f"[cv] repeat {r + 1}/{repeats} fold {f + 1}/{N_SPLITS} {el:6.0f}s "
              f"(eta {el / done * (n_total - done):5.0f}s) last config acc: "
              + " ".join(f"{m}={last[m]:.3f}" for m in ("LR", "RF", "GBM", "STACK")), flush=True)
        if f == N_SPLITS - 1:
            save(out)
    if not args.no_holdout:
        idx = np.arange(len(y))
        tr, te = train_test_split(idx, test_size=HOLDOUT_TEST_SIZE, stratify=y, random_state=SEED)
        for cfg in args.configs:
            fit_one(X, y, tr, te, "holdout", 0, 0, cfg, args.jobs, out)
        print(f"[holdout] done ({time.time() - t0:.0f}s)", flush=True)
    save(out)
    print("saved ->", RESULTS / "preds.csv", RESULTS / "thresholds.csv", RESULTS / "inner_cv.csv")


if __name__ == "__main__":
    main()
