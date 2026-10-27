"""
03_ladder.py - ablation ladder: marginal value of each added measurement.
Logistic regression (tuned C) and random forest (fixed settings), 3 x 10 outer CV, zeros -> missing + median.
Output: results/ladder_folds.csv, results/table_ladder.csv
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from common import *  # noqa

warnings.filterwarnings("ignore")


def one(X, y, tr, te, rung, label, cols, model, r, f):
    prep = make_preprocessor(cols, "impute")
    if model == "LR":
        pipe = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=5000))])
        est = GridSearchCV(pipe, {"clf__C": [0.1, 1, 10]},
                           cv=StratifiedKFold(INNER_SPLITS, shuffle=True, random_state=SEED), scoring="accuracy")
    else:
        est = Pipeline([("prep", prep), ("clf", RandomForestClassifier(
            n_estimators=200, min_samples_leaf=3, random_state=SEED, n_jobs=1))])
    est.fit(X.iloc[tr][cols], y[tr])
    p = est.predict_proba(X.iloc[te][cols])[:, 1]
    return dict(rung=label, model=model, repeat=r, fold=f, acc=float(((p >= .5) == y[te]).mean()),
                auroc=float(roc_auc_score(y[te], p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--limit-folds", type=int, default=0)
    ap.add_argument("--configs", nargs="*")
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument("--no-holdout", action="store_true")
    args = ap.parse_args()
    X, y = load_data()
    repeats = 1 if args.quick else args.repeats
    jobs = []
    for i, (tr, te) in enumerate(RepeatedStratifiedKFold(
            n_splits=N_SPLITS, n_repeats=repeats, random_state=SEED).split(X, y)):
        r, f = divmod(i, N_SPLITS)
        if args.limit_folds and f >= args.limit_folds:
            continue
        for k, (label, cols) in enumerate(LADDER):
            for model in ("LR", "RF"):
                jobs.append((X, y, tr, te, k, label, cols, model, r, f))
    rows = Parallel(n_jobs=args.jobs)(delayed(one)(*j) for j in jobs)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "ladder_folds.csv", index=False)
    g = df.groupby(["model", "rung"], sort=True)
    t = g.agg(acc_mean=("acc", "mean"), acc_sd=("acc", "std"),
              auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std")).reset_index()
    t["auroc_gain"] = t.groupby("model").auroc_mean.diff()
    t.to_csv(RESULTS / "table_ladder.csv", index=False)
    print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
