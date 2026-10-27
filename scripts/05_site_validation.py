"""
05_site_validation.py - leave-one-site-out validation on the four UCI source files
(Cleveland, Hungarian, Switzerland, VA Long Beach), 11 of the 14 original attributes mapped to our features.

The Kaggle file already contains these patients, so it cannot serve as external data. Here the two
screening models (Step 1: Group A without cholesterol; Step 2: all 11 features) are trained on three
sites and applied once to the fourth; calibration and thresholds come from the training sites' inner
out-of-fold predictions. Missing values are imputed from the training sites only.

Needs external/processed.{cleveland,hungarian,switzerland,va}.data. Output: results/table_site_validation.csv
"""
import importlib.util
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import *  # noqa

warnings.filterwarnings("ignore")
EXT = ROOT / "external"
SITES = {"cleveland": "Cleveland", "hungarian": "Hungarian", "switzerland": "Switzerland", "va": "VA Long Beach"}
RAW = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]


def load_site(key):
    d = pd.read_csv(EXT / f"processed.{key}.data", header=None, names=RAW, na_values="?")
    out = pd.DataFrame({
        "Age": d.age, "Sex": d.sex.map({1: "M", 0: "F"}), "ChestPainType": d.cp.map({1: "TA", 2: "ATA", 3: "NAP", 4: "ASY"}),
        "RestingBP": d.trestbps, "Cholesterol": d.chol, "FastingBS": d.fbs,
        "RestingECG": d.restecg.map({0: "Normal", 1: "ST", 2: "LVH"}), "MaxHR": d.thalach,
        "ExerciseAngina": d.exang.map({0: "N", 1: "Y"}), "Oldpeak": d.oldpeak,
        "ST_Slope": d.slope.map({1: "Up", 2: "Flat", 3: "Down"}), "site": SITES[key]})
    for c in CAT_COLS:
        out[c] = out[c].astype(object).where(out[c].notna(), np.nan)
    return out, (d.num > 0).astype(int).to_numpy()


def score(b, df):
    Z = np.column_stack([logit(b["models"][m].predict_proba(df[b["cols"]])[:, 1]) for m in b["classical"]])
    return b["platt"].predict(b["meta"].predict_proba(Z)[:, 1])


def main():
    need = [EXT / f"processed.{k}.data" for k in SITES]
    if not all(f.exists() for f in need):
        print("UCI source files not found in external/; site-wise validation skipped.")
        return
    spec = importlib.util.spec_from_file_location("export_models", ROOT / "scripts" / "06_export_models.py")
    em = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(em)
    frames, ys = zip(*[load_site(k) for k in SITES])
    X = pd.concat(frames, ignore_index=True)
    y = np.concatenate(ys)
    rows = []
    for held in SITES.values():
        te = (X.site == held).to_numpy()
        Xtr, ytr, Xte, yte = X[~te].reset_index(drop=True), y[~te], X[te].reset_index(drop=True), y[te]
        b1, _ = em.fit_bundle(Xtr, ytr, "site_step1", -1, cols=GROUP_A_NOCHOL, variant="impute_all")
        b2, _ = em.fit_bundle(Xtr, ytr, "site_step2", -1, cols=ALL_FEATURES, variant="impute_all")
        q1, q2 = score(b1, Xte), score(b2, Xte)
        o1 = decide(q1, b1["thresholds"][f"tl_e{S1_PRIMARY}"], b1["thresholds"]["th_e95"])
        o2 = decide(q2, b2["thresholds"][f"tl_e{S2_LEVEL}"], b2["thresholds"]["th_e95"])
        fin = np.where(o1 == 0, 0, o2)
        sick, healthy = yte == 1, yte == 0
        both = sick.any() and healthy.any()
        rows.append(dict(
            held_out_site=held, n=len(yte), n_disease=int(sick.sum()), prevalence=float(yte.mean()),
            n_train=int((~te).sum()),
            auroc_step1=roc_auc_score(yte, q1) if both else np.nan, auroc_step2=roc_auc_score(yte, q2) if both else np.nan,
            step1_sent_home=float((o1 == 0).mean()), step1_sick_sent_home=float((o1[sick] == 0).mean()),
            final_sick_sent_home=float((fin[sick] == 0).mean()), final_healthy_sent_home=float((fin[healthy] == 0).mean()) if healthy.any() else np.nan,
            final_not_sure=float((fin == 2).mean()), final_high=float((fin == 1).mean()),
            acc_step2_forced=float(((q2 >= .5) == yte).mean())))
    T = pd.DataFrame(rows)
    T.to_csv(RESULTS / "table_site_validation.csv", index=False)
    print(T.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
