"""
04_evaluate.py - every table, test and figure from results/*.csv.
Outputs: table_main.csv, table_holdout.csv, table_tests.csv, table_confusion.csv, table_system_rates.csv,
         table_system.csv, table_mechanism.csv, auto_decisions.json, summary.md, figures (300 dpi).
"""
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

import plot_style as ps
from common import *  # noqa

warnings.filterwarnings("ignore")
ps.apply()

P = pd.read_csv(RESULTS / "preds.csv")
T = pd.read_csv(RESULTS / "thresholds.csv")
I = pd.read_csv(RESULTS / "inner_cv.csv")
KEY = ["split", "repeat", "fold", "config", "model"]
Pcv, Tcv = P[P.split == "cv"], T[T.split == "cv"]
N_REP = Pcv.repeat.nunique()
CONFS = [c for c in CONFIGS if c in set(P.config)]
n_train, n_test = I[I.split == "cv"].n_train.mean(), I[I.split == "cv"].n_test.mean()


def md(df, nd=3):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v)
                                         for v in r) + " |")
    return "\n".join(lines)


def ece(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(bins) if (idx == b).any())


def metrics(g):
    y, p, q = g.y.to_numpy(), g.p_raw.to_numpy(), g.p_cal.to_numpy()
    pred = p >= 0.5
    return dict(acc=(pred == y).mean(), sens=(pred & (y == 1)).sum() / (y == 1).sum(),
                spec=(~pred & (y == 0)).sum() / (y == 0).sum(), auroc=roc_auc_score(y, p),
                brier_raw=((p - y) ** 2).mean(), brier_cal=((q - y) ** 2).mean())


# ---------------------------------------------------------------- 1. main comparison table
FM = pd.DataFrame([{**dict(zip(KEY, k)), **metrics(g)} for k, g in Pcv.groupby(KEY, sort=False)])
FM.to_csv(RESULTS / "fold_metrics.csv", index=False)
E = pd.DataFrame([{"config": k[0], "model": k[1], "repeat": k[2], "ece_raw": ece(g.p_raw.to_numpy(), g.y.to_numpy()),
                   "ece_cal": ece(g.p_cal.to_numpy(), g.y.to_numpy())}
                  for k, g in Pcv.groupby(["config", "model", "repeat"], sort=False)])
agg = FM.groupby(["config", "model"]).agg(
    acc_mean=("acc", "mean"), acc_sd=("acc", "std"), auroc=("auroc", "mean"), sens=("sens", "mean"),
    spec=("spec", "mean"), brier_raw=("brier_raw", "mean"), brier_cal=("brier_cal", "mean")).reset_index()
agg = agg.merge(E.groupby(["config", "model"])[["ece_raw", "ece_cal"]].mean().reset_index(), on=["config", "model"])
agg["name"] = agg.model.map(MODEL_NAMES)
agg["acc_mean"] *= 100
agg["acc_sd"] *= 100
agg.to_csv(RESULTS / "table_main.csv", index=False)

# holdout
H = P[P.split == "holdout"]
if len(H):
    hold = pd.DataFrame([{**dict(zip(KEY, k)), **metrics(g)} for k, g in H.groupby(KEY, sort=False)])
    hold.to_csv(RESULTS / "table_holdout.csv", index=False)

# ---------------------------------------------------------------- 2. significance tests
test_rows = []
for cfg in CONFS:
    A = FM[FM.config == cfg].pivot(index=["repeat", "fold"], columns="model", values="acc")
    chi, pf = stats.friedmanchisquare(*[A[m] for m in ALL_MODELS])
    best_single = A[SINGLE].mean().idxmax()
    test_rows.append(dict(config=cfg, contrast=f"Friedman over {len(ALL_MODELS)} methods", mean_diff=np.nan,
                          stat=chi, p_corrected_t=np.nan, p_wilcoxon=pf))
    pairs = [("STACK", best_single), ("STACK", "RF"), ("SOFT", "RF"), ("SOFT", best_single), ("STACK", "SOFT")]
    for a, b in dict.fromkeys(pairs):
        if a == b:
            continue
        d, t, pt = corrected_resampled_ttest(A[a], A[b], n_train, n_test)
        try:
            pw = stats.wilcoxon(A[a], A[b]).pvalue
        except ValueError:
            pw = 1.0
        test_rows.append(dict(config=cfg, contrast=f"{a} vs {b}", mean_diff=d * 100, stat=t,
                              p_corrected_t=pt, p_wilcoxon=pw))
pd.DataFrame(test_rows).to_csv(RESULTS / "table_tests.csv", index=False)

# ---------------------------------------------------------------- 3. best method and confusion matrix
ref = agg[agg.config == "F_raw"].sort_values("acc_mean", ascending=False).reset_index(drop=True)
best = ref.model[0]
if len(ref) > 1 and ref.acc_mean[0] - ref.acc_mean[1] < 0.1:
    top = ref[ref.acc_mean >= ref.acc_mean[0] - 0.1]
    best = top.sort_values("auroc", ascending=False).model.iloc[0]
cm_df = Pcv[(Pcv.config == "F_raw") & (Pcv.model == best) & (Pcv.repeat == 0)]
tn = int(((cm_df.p_raw < .5) & (cm_df.y == 0)).sum()); fp = int(((cm_df.p_raw >= .5) & (cm_df.y == 0)).sum())
fn = int(((cm_df.p_raw < .5) & (cm_df.y == 1)).sum()); tp = int(((cm_df.p_raw >= .5) & (cm_df.y == 1)).sum())
pd.DataFrame([dict(method=best, config="F_raw", repeat=1, TN=tn, FP=fp, FN=fn, TP=tp)]).to_csv(
    RESULTS / "table_confusion.csv", index=False)
fig, ax = plt.subplots(figsize=(3.4, 3.0))
ax.imshow([[tn, fp], [fn, tp]], cmap=ps.SEQUENTIAL)
for (i, j), v in np.ndenumerate([[tn, fp], [fn, tp]]):
    ax.text(j, i, str(v), ha="center", va="center", fontsize=12, color="white" if v > 250 else ps.INK)
ax.set_xticks([0, 1], ["Normal", "Heart disease"]); ax.set_yticks([0, 1], ["Normal", "Heart disease"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.grid(False)
ax.set_title(MODEL_NAMES[best] + ", pooled test folds", fontsize=9)
fig.savefig(FIGURES / "fig_confusion.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.2, 3.0))
r = ref.sort_values("acc_mean")
ax.barh(r.name, r.acc_mean, xerr=r.acc_sd, color=[ps.ORANGE if m in ("STACK", "SOFT") else ps.BLUE for m in r.model],
        error_kw=dict(ecolor=ps.INK2, lw=0.8))
ax.set_xlim(78, 92); ax.set_xlabel("Accuracy (%), mean and s.d. over outer folds (all 11 features)")
fig.savefig(FIGURES / "fig_accuracy.png"); plt.close(fig)

# ---------------------------------------------------------------- 4. reliability
fig, ax = plt.subplots(figsize=(3.4, 3.0))
g = Pcv[(Pcv.config == STEP2_CONFIG) & (Pcv.model == "STACK")]
for col, lab, c in (("p_raw", "Stacking output", ps.ORANGE), ("p_cal", "After Platt calibration", ps.BLUE)):
    b = np.minimum((g[col] * 10).astype(int), 9)
    d = g.groupby(b).agg(x=(col, "mean"), y=("y", "mean"))
    ax.plot(d.x, d.y, "o-", ms=3, lw=1, color=c, label=lab)
ax.plot([0, 1], [0, 1], color=ps.GREY, lw=0.8)
ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed fraction with disease"); ax.legend(loc="upper left")
fig.savefig(FIGURES / "fig_reliability.png"); plt.close(fig)

# ---------------------------------------------------------------- 5. ladder figure
try:
    L = pd.read_csv(RESULTS / "table_ladder.csv")
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for m, c in (("LR", ps.BLUE), ("RF", ps.ORANGE)):
        d = L[L.model == m]
        ax.errorbar(range(len(d)), d.auroc_mean, yerr=d.auroc_sd, fmt="o-", ms=3, lw=1, color=c,
                    capsize=2, label=MODEL_NAMES[m])
    ax.set_xticks(range(len(LADDER)), [l.split(" ", 1)[1] if i else l.split(" ", 1)[1] for i, (l, _) in enumerate(LADDER)],
                  rotation=20, ha="right")
    ax.set_ylabel("AUROC (outer folds)"); ax.legend()
    fig.savefig(FIGURES / "fig_ladder.png"); plt.close(fig)
except FileNotFoundError:
    L = None

# ---------------------------------------------------------------- 6. automatic decisions
auto = {"best_method_rule": "highest mean outer-CV accuracy, F_raw; <0.1 point gap broken by AUROC", "best_method": best}
if {"A_chol", "A_nochol"} <= set(CONFS):
    au = agg.set_index(["config", "model"]).auroc
    gain = float(au["A_chol", "STACK"] - au["A_nochol", "STACK"])
    auto["cholesterol_auroc_gain_stack"] = gain
    auto["step1_config"] = "A_chol" if gain >= 0.01 else "A_nochol"
    auto["step1_rule"] = "A_chol only if it raises STACK AUROC by >= 0.01, else A_nochol"
else:
    auto["step1_config"] = CONFS[0]
oa = I[(I.split == "cv") & (I.config == STEP2_CONFIG)].groupby("model").oof_auc.mean()
best_single_oof = oa[SINGLE].idxmax()
auto["proposed_model"] = "STACK" if oa["STACK"] >= oa[best_single_oof] else best_single_oof
auto["proposed_rule"] = "STACK if its inner out-of-fold AUROC (F_imp) >= best single model, else best single"
auto["oof_auc_F_imp"] = {k: round(float(v), 4) for k, v in oa.items()}
S1CFG = auto["step1_config"]

# ---------------------------------------------------------------- 7. two-step simulation
def class_rates(y, o1, final, tests, meta):
    rows = []
    for yv in (0, 1):
        mk = y == yv
        rows.append(dict(meta, y=yv, n=mk.sum() / N_REP, r1_low=(o1[mk] == 0).mean(), r1_high=(o1[mk] == 1).mean(),
                         r1_ns=(o1[mk] == 2).mean(), rf_low=(final[mk] == 0).mean(), rf_high=(final[mk] == 1).mean(),
                         rf_ns=(final[mk] == 2).mean(), tests=tests[mk].mean()))
    return rows


def thr_cols(df, suf):
    return {c[:-len(suf)]: df[c].to_numpy() for c in df.columns if c.startswith(("tl_", "th_")) and c.endswith(suf)}


rate_rows = []
for m in ALL_MODELS:
    b = Pcv[(Pcv.config == STEP2_CONFIG) & (Pcv.model == m)].merge(
        Tcv[(Tcv.config == STEP2_CONFIG) & (Tcv.model == m)], on=KEY).drop(columns=KEY[3:])
    for c1 in [c for c in ("A_chol", "A_nochol") if c in CONFS]:
        a = Pcv[(Pcv.config == c1) & (Pcv.model == m)].merge(
            Tcv[(Tcv.config == c1) & (Tcv.model == m)], on=KEY).drop(columns=KEY[3:])
        d = a.merge(b, on=["split", "repeat", "fold", "row_id"], suffixes=("_1", "_2"))
        y = d.y_1.to_numpy()
        t1 = thr_cols(d, "_1")
        t2 = thr_cols(d, "_2")
        q1, q2 = d.p_cal_1.to_numpy(), d.p_cal_2.to_numpy()
        for mech in ("e", "c"):
            o2 = decide(q2, t2[f"tl_{mech}{S2_LEVEL}"], t2[f"th_{mech}95"])
            for s1 in (90, 95, 98):
                o1 = decide(q1, t1[f"tl_{mech}{s1}"], t1[f"th_{mech}95"])
                final = np.where(o1 == 0, 0, o2)
                rate_rows += class_rates(y, o1, final, (o1 != 0).astype(float),
                                         dict(model=m, step1=c1, system=f"two-step/{mech}", s1=s1))
        # baselines (forced 0.5 decisions on the uncalibrated output, as a standard classifier)
        f2 = (d.p_raw_2.to_numpy() >= .5).astype(int)
        f1 = (d.p_raw_1.to_numpy() >= .5).astype(int)
        o1 = np.where(f1 == 0, 0, 2)
        rate_rows += class_rates(y, o1, np.where(o1 == 0, 0, f2), (o1 != 0).astype(float),
                                 dict(model=m, step1=c1, system="baseline: fixed 0.5 cutoff at Step 1", s1=0))
    rate_rows += class_rates(y, np.full(len(y), 2), f2, np.ones(len(y)),
                             dict(model=m, step1="-", system="baseline: test everyone, forced 0.5", s1=0))
    for mech in ("e", "c"):
        o2 = decide(q2, t2[f"tl_{mech}{S2_LEVEL}"], t2[f"th_{mech}95"])
        rate_rows += class_rates(y, np.full(len(y), 2), o2, np.ones(len(y)),
                                 dict(model=m, step1="-", system=f"baseline: test everyone, Step 2 only/{mech}", s1=0))
R = pd.DataFrame(rate_rows)
R.to_csv(RESULTS / "table_system_rates.csv", index=False)

meta_cols = ["model", "step1", "system", "s1"]
R0, R1 = R[R.y == 0].set_index(meta_cols), R[R.y == 1].set_index(meta_cols)
sc_rows = []
for sname, pi in SCENARIOS.items():
    sick, healthy = 1000 * pi, 1000 * (1 - pi)
    for idx in R1.index:
        a, h = R1.loc[idx], R0.loc[idx]
        flagged = sick * a.rf_high + healthy * h.rf_high
        home = sick * a.rf_low + healthy * h.rf_low
        lo, hi = wilson(a.rf_low * a.n, a.n)
        sc_rows.append(dict(zip(meta_cols, idx), scenario=sname, pi=pi,
                            tests_per_1000=sick * a.tests + healthy * h.tests,
                            sent_home_step1_per_1000=sick * a.r1_low + healthy * h.r1_low,
                            missed_step1_per_1000=sick * a.r1_low,
                            missed_final_per_1000=sick * a.rf_low,
                            missed_final_lo=sick * lo, missed_final_hi=sick * hi,
                            final_sens=1 - a.rf_low, final_home_rate_healthy=h.rf_low,
                            high_risk_per_1000=flagged, ppv_high=sick * a.rf_high / flagged if flagged else np.nan,
                            npv_low=healthy * h.rf_low / home if home else np.nan,
                            not_sure_final_per_1000=sick * a.rf_ns + healthy * h.rf_ns,
                            cov_sick_step1=1 - a.r1_low, cov_healthy_step1=1 - h.r1_high))
S = pd.DataFrame(sc_rows)
S.to_csv(RESULTS / "table_system.csv", index=False)

# mechanism comparison (decision 6) for the proposed model, Step 1 variant chosen above, s1 = 95
pm = auto["proposed_model"]
mech_rows = []
for mech in ("e", "c"):
    a = R1.loc[(pm, S1CFG, f"two-step/{mech}", S1_PRIMARY)]
    h = R0.loc[(pm, S1CFG, f"two-step/{mech}", S1_PRIMARY)]
    ns_overall = PI_SOURCE * a.r1_ns + (1 - PI_SOURCE) * h.r1_ns
    mech_rows.append(dict(mechanism="two-threshold (empirical)" if mech == "e" else "Mondrian conformal",
                          cov_sick_step1=1 - a.r1_low, cov_healthy_step1=1 - h.r1_high,
                          not_sure_step1=ns_overall, cov_sick_final=1 - a.rf_low, not_sure_final=PI_SOURCE * a.rf_ns + (1 - PI_SOURCE) * h.rf_ns))
MECH = pd.DataFrame(mech_rows)
MECH.to_csv(RESULTS / "table_mechanism.csv", index=False)
ok = MECH[(MECH.cov_sick_step1 >= .94) & (MECH.cov_healthy_step1 >= .94)]
if len(ok) == 0:
    chosen = "two-threshold (empirical)"
elif len(ok) == 1 or abs(ok.not_sure_step1.iloc[0] - ok.not_sure_step1.iloc[-1]) < 0.005:
    chosen = ok.mechanism.iloc[0]
else:
    chosen = ok.sort_values("not_sure_step1").mechanism.iloc[0]
auto["mechanism"] = chosen
auto["mechanism_rule"] = "both class coverages >= 0.94; smaller Step 1 not-sure share; tie (<0.5 point) -> two-threshold"
MECH_CODE = "c" if chosen == "Mondrian conformal" else "e"
json.dump(auto, open(RESULTS / "auto_decisions.json", "w"), indent=2)

# trade-off figure at the reference scenario
sub = S[(S.scenario == "5% (reference)")]
fig, ax = plt.subplots(figsize=(5.2, 3.2))
for m, c in ((pm, ps.ORANGE), ("RF", ps.BLUE)):
    d = sub[(sub.model == m) & (sub.step1 == S1CFG) & (sub.system == f"two-step/{MECH_CODE}")]
    ax.plot(d.tests_per_1000, d.missed_final_per_1000, "o-", ms=4, lw=1, color=c, label=f"{MODEL_NAMES[m]}, two-step (90/95/98%)")
d = sub[(sub.model == pm) & (sub.system == "baseline: fixed 0.5 cutoff at Step 1") & (sub.step1 == S1CFG)]
ax.scatter(d.tests_per_1000, d.missed_final_per_1000, marker="s", color=ps.GREY, label="Fixed 0.5 cutoff at Step 1")
d = sub[(sub.model == pm) & (sub.system == "baseline: test everyone, forced 0.5")]
ax.scatter(d.tests_per_1000, d.missed_final_per_1000, marker="^", color=ps.INK2, label="Test everyone")
ax.set_xlabel("Patients needing equipment tests, per 1,000 screened"); ax.set_ylabel("Sick patients sent home, per 1,000 screened")
ax.legend(fontsize=7); fig.savefig(FIGURES / "fig_tradeoff.png"); plt.close(fig)

# ---------------------------------------------------------------- 8. summary.md
out = [f"# Results summary\n\nProtocol: {N_REP} x {N_SPLITS} stratified outer folds ({N_REP * N_SPLITS} test folds), {INNER_SPLITS}-fold inner tuning, seed {SEED}. "
       f"All fitted or chosen quantities (tuning, stacking, calibration, thresholds) use training folds only.\n"]
for cfg in CONFS:
    t = agg[agg.config == cfg].sort_values("acc_mean", ascending=False)
    t = t.assign(Accuracy=t.acc_mean.map("{:.2f}".format) + " ± " + t.acc_sd.map("{:.2f}".format))[
        ["name", "Accuracy", "auroc", "sens", "spec", "brier_raw", "brier_cal", "ece_raw", "ece_cal"]]
    out.append(f"## Config {cfg}: {', '.join(CONFIGS[cfg][0]) if len(CONFIGS[cfg][0]) < 11 else 'all 11 features'} ({CONFIGS[cfg][1]})\n\n" + md(t, 3) + "\n")
out.append("## Tests (accuracy per outer fold)\n\n" + md(pd.DataFrame(test_rows), 4) + "\n")
out.append(f"## Best method (declared rule)\n\n{auto['best_method_rule']}: **{MODEL_NAMES[best]}**; pooled confusion matrix, repeat 1: TN={tn}, FP={fp}, FN={fn}, TP={tp}.\n")
if L is not None:
    out.append("## Ablation ladder (AUROC)\n\n" + md(L, 4) + "\n")
out.append("## Automatic decisions\n\n```\n" + json.dumps(auto, indent=2) + "\n```\n")
out.append("## Mechanism comparison (proposed model, Step 1, 95% release)\n\n" + md(MECH, 4) + "\n")
hl = S[(S.scenario == "5% (reference)") & (S.step1.isin([S1CFG, "-"])) & (S.model.isin([pm, "RF", "LR"])) &
       (S.system.isin([f"two-step/{MECH_CODE}", "baseline: fixed 0.5 cutoff at Step 1", "baseline: test everyone, forced 0.5",
                       f"baseline: test everyone, Step 2 only/{MECH_CODE}"])) & (S.s1.isin([0, 90, 95, 98]))]
out.append("## System at the 5% reference scenario (per 1,000 screened)\n\n" + md(hl[[
    "model", "system", "s1", "tests_per_1000", "missed_step1_per_1000", "missed_final_per_1000", "final_sens",
    "high_risk_per_1000", "ppv_high", "not_sure_final_per_1000"]], 3) + "\n")
(RESULTS / "summary.md").write_text("\n".join(out))
print("\n".join(out)[:6000])
