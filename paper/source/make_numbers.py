"""make_numbers.py - every number in the paper, read from results/*.csv and dataset/heart.csv -> numbers.json"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from common import MODEL_NAMES, SCENARIOS
R = ROOT / "results"
f = lambda x, d=2: f"{x:.{d}f}".replace("-", "−")
pct = lambda x, d=1: f"{100 * x:.{d}f}"
def p_fmt(p):
    if p < 0.001:
        e = int(np.floor(np.log10(p))); m = p / 10 ** e
        return f"{m:.1f} × 10^{e}^".replace("-", "−")
    return f"{p:.3f}"

M, H = pd.read_csv(R / "table_main.csv"), pd.read_csv(R / "table_holdout.csv")
T, S = pd.read_csv(R / "table_tests.csv"), pd.read_csv(R / "table_system.csv")
Mech, Site = pd.read_csv(R / "table_mechanism.csv"), pd.read_csv(R / "table_site_validation.csv")
L, CM = pd.read_csv(R / "table_ladder.csv"), pd.read_csv(R / "table_confusion.csv")
A = json.load(open(R / "auto_decisions.json"))
d = pd.read_csv(ROOT / "dataset" / "heart.csv")
N = {}
z = d.Cholesterol == 0
N["data"] = dict(n=len(d), pos=int(d.HeartDisease.sum()), neg=int((d.HeartDisease == 0).sum()), pos_pct=pct(d.HeartDisease.mean()),
                 zero=int(z.sum()), zero_pct=pct(z.mean()), zero_prev=pct(d[z].HeartDisease.mean()), rest_prev=pct(d[~z].HeartDisease.mean()),
                 bp0=int((d.RestingBP == 0).sum()), age_min=int(d.Age.min()), age_max=int(d.Age.max()))
order = ["LR", "DT", "RF", "KNN", "SVM", "GBM", "MLP", "SOFT", "STACK"]
def row(cfg, m): return M[(M.config == cfg) & (M.model == m)].iloc[0]
N["main"] = {}
for m in order:
    r, ri = row("F_raw", m), row("F_ind", m)
    h = H[(H.config == "F_raw") & (H.model == m)].iloc[0]
    N["main"][m] = dict(name=MODEL_NAMES[m], acc=f(r.acc_mean), sd=f(r.acc_sd), auroc=f(r.auroc, 3), sens=f(r.sens, 3), spec=f(r.spec, 3),
                        brier=f(r.brier_cal, 3), ece_raw=f(r.ece_raw, 3), ece_cal=f(r.ece_cal, 3), ho=f(100 * h.acc), acc_ind=f(ri.acc_mean), sd_ind=f(ri.acc_sd),
                        accf=r.acc_mean)
best = max(order, key=lambda m: N["main"][m]["accf"])
N["best_cv"] = best
tf = lambda c: T[(T.config == "F_raw") & (T.contrast == c)].iloc[0]
fr = T[(T.config == "F_raw") & T.contrast.str.startswith("Friedman")].iloc[0]
N["tests"] = dict(chi2=f(fr.stat), p_friedman=p_fmt(fr.p_wilcoxon))
for k, c in (("stack_rf", "STACK vs RF"), ("soft_rf", "SOFT vs RF"), ("stack_soft", "STACK vs SOFT")):
    r = tf(c); N["tests"][k] = dict(diff=f(r.mean_diff), t=f(r.stat), p=f(r.p_corrected_t, 2), pw=f(r.p_wilcoxon, 2))
cm = CM.iloc[0]
tp, tn, fp, fn = int(cm.TP), int(cm.TN), int(cm.FP), int(cm.FN)
N["conf"] = dict(method=MODEL_NAMES[cm.method], key=cm.method, TP=tp, TN=tn, FP=fp, FN=fn, acc=pct((tp + tn) / (tp + tn + fp + fn)),
                 sens=pct(tp / (tp + fn)), spec=pct(tn / (tn + fp)))
def tier(cfg):
    o = {}
    for m in ("STACK", "LR", "RF"):
        r = row(cfg, m); o[m] = dict(acc=f"{r.acc_mean:.2f} ± {r.acc_sd:.2f}", auroc=f(r.auroc, 3))
    return o
N["tier"] = {c: tier(c) for c in ("A_nochol", "A_chol", "F_imp")}
N["chol_gain"] = f(A["cholesterol_auroc_gain_stack"], 3)
lad = {}
for m in ("LR", "RF"):
    t = L[L.model == m].reset_index(drop=True)
    lad[m] = [dict(rung=r.rung, auroc=f(r.auroc_mean, 3), gain=("" if pd.isna(r.auroc_gain) else f"{r.auroc_gain:+.3f}".replace("-", "−"))) for r in t.itertuples()]
N["ladder"] = lad
st = row("F_imp", "STACK")
N["calib"] = dict(stack_ece_raw=f(st.ece_raw, 3), stack_ece_cal=f(st.ece_cal, 3), stack_brier_raw=f(st.brier_raw, 3), stack_brier_cal=f(st.brier_cal, 3),
                  knn_ece_raw=f(row("F_raw", "KNN").ece_raw, 3), knn_ece_cal=f(row("F_raw", "KNN").ece_cal, 3))
N["mech"] = {("e" if "two" in r.mechanism else "c"): dict(cov_sick=pct(r.cov_sick_step1), cov_healthy=pct(r.cov_healthy_step1), ns=pct(r.not_sure_step1),
             ns_final=pct(r.not_sure_final)) for r in Mech.itertuples()}
N["auto"] = dict(step1=A["step1_config"], proposed=A["proposed_model"], mechanism=A["mechanism"], best=A["best_method"],
                 oof=A["oof_auc_F_imp"])
def sysrow(model, step1, system, s1, scen):
    return S[(S.model == model) & (S.step1 == step1) & (S.system == system) & (S.s1 == s1) & (S.scenario == scen)].iloc[0]
REF = "5% (reference)"
def line(label, r, ci=False):
    miss = f"{r.missed_final_per_1000:.1f}" + (f" ({r.missed_final_lo:.1f}–{r.missed_final_hi:.1f})" if ci else "")
    return [label, f"{r.tests_per_1000:.0f}", miss, pct(r.final_sens), f"{r.high_risk_per_1000:.0f}", f(r.ppv_high), f"{r.not_sure_final_per_1000:.0f}"]
sys_rows = []
for s1 in (90, 95, 98):
    sys_rows.append(line(f"Two-step, Step 1 release at {s1}% sensitivity", sysrow("STACK", "A_nochol", "two-step/e", s1, REF), ci=True))
sys_rows.append(line("Fixed 0.5 cutoff at Step 1, forced 0.5 at Step 2", sysrow("STACK", "A_nochol", "baseline: fixed 0.5 cutoff at Step 1", 0, REF), ci=True))
sys_rows.append(line("Test everyone, forced 0.5", sysrow("STACK", "-", "baseline: test everyone, forced 0.5", 0, REF), ci=True))
sys_rows.append(line("Test everyone, Step 2 with not-sure band", sysrow("STACK", "-", "baseline: test everyone, Step 2 only/e", 0, REF), ci=True))
N["sys_rows"] = sys_rows
prim = sysrow("STACK", "A_nochol", "two-step/e", 95, REF)
N["prim"] = dict(tests=f"{prim.tests_per_1000:.0f}", missed=f"{prim.missed_final_per_1000:.1f}", lo=f"{prim.missed_final_lo:.1f}", hi=f"{prim.missed_final_hi:.1f}",
                 sens=pct(prim.final_sens), missed1=f"{prim.missed_step1_per_1000:.1f}", ppv=f(prim.ppv_high), high=f"{prim.high_risk_per_1000:.0f}",
                 ns=f"{prim.not_sure_final_per_1000:.0f}", npv=f(prim.npv_low, 3))
fixed = sysrow("STACK", "A_nochol", "baseline: fixed 0.5 cutoff at Step 1", 0, REF); tall = sysrow("STACK", "-", "baseline: test everyone, forced 0.5", 0, REF)
tns = sysrow("STACK", "-", "baseline: test everyone, Step 2 only/e", 0, REF)
N["base"] = dict(fixed_missed=f"{fixed.missed_final_per_1000:.1f}", fixed_tests=f"{fixed.tests_per_1000:.0f}", all_missed=f"{tall.missed_final_per_1000:.1f}",
                 ns_missed=f"{tns.missed_final_per_1000:.1f}", ns_ns=f"{tns.not_sure_final_per_1000:.0f}")
RT = pd.read_csv(R / "table_system_rates.csv")
rr = RT[(RT.model == "STACK") & (RT.step1 == "A_nochol") & (RT.system == "two-step/e") & (RT.s1 == 95)].set_index("y")
N["rates"] = dict(sick_home1=pct(rr.loc[1, "r1_low"]), healthy_home1=pct(rr.loc[0, "r1_low"]), sick_tests=pct(rr.loc[1, "tests"]), healthy_tests=pct(rr.loc[0, "tests"]),
                  sick_home_final=pct(rr.loc[1, "rf_low"]), healthy_home_final=pct(rr.loc[0, "rf_low"]), ns_sick=pct(rr.loc[1, "rf_ns"]), ns_healthy=pct(rr.loc[0, "rf_ns"]))
sc_rows = []
for sname in SCENARIOS:
    r = sysrow("STACK", "A_nochol", "two-step/e", 95, sname)
    sc_rows.append([sname.replace(" (reference)", " (reference)").replace(" (dataset)", " (dataset)"), f"{r.tests_per_1000:.0f}",
                    f"{r.missed_final_per_1000:.1f} ({r.missed_final_lo:.1f}–{r.missed_final_hi:.1f})", f"{r.high_risk_per_1000:.0f}", f(r.ppv_high), f(r.npv_low, 3), f"{r.not_sure_final_per_1000:.0f}"])
N["scen_rows"] = sc_rows
site_rows = []
for r in Site.itertuples():
    site_rows.append([r.held_out_site, f"{r.n} ({pct(r.prevalence, 0)}%)", f(r.auroc_step1, 3), f(r.auroc_step2, 3), pct(r.step1_sick_sent_home), pct(r.final_healthy_sent_home), pct(r.final_not_sure)])
N["site_rows"] = site_rows
N["site"] = dict(a1_min=f(Site.auroc_step1.min(), 3), a1_max=f(Site.auroc_step1.max(), 3), a2_min=f(Site.auroc_step2.min(), 3), a2_max=f(Site.auroc_step2.max(), 3),
                 s1_max=pct(Site.step1_sick_sent_home.max()), s1_max_site=Site.loc[Site.step1_sick_sent_home.idxmax(), "held_out_site"],
                 s1_min=pct(Site.step1_sick_sent_home.min()))
log = open(R / "run_log.txt").read()
import re
N["run"] = dict(versions=re.search(r"python [^\n]+", log).group(0), cores=re.search(r"cpu cores: (\d+)", log).group(1))
json.dump(N, open(Path(__file__).with_name("numbers.json"), "w"), indent=1, ensure_ascii=False)
print("ok", list(N))
print(json.dumps({k: N[k] for k in ("best_cv", "conf", "prim", "base", "rates", "site", "run", "tests", "auto", "chol_gain")}, indent=1, ensure_ascii=False))

# ---- extra facts: feature table counts and cholesterol zeros by UCI site
N = json.load(open(Path(__file__).with_name("numbers.json")))
vc = lambda c: ", ".join(f"{k} ({v})" for k, v in d[c].value_counts().items())
N["feat"] = dict(Sex=vc("Sex"), CP=vc("ChestPainType"), FBS=vc("FastingBS"), ECG=vc("RestingECG"), EA=vc("ExerciseAngina"), SL=vc("ST_Slope"),
                 age=f"{d.Age.min()} – {d.Age.max()}", bp=f"{d.RestingBP.min()} – {d.RestingBP.max()}", chol=f"{d.Cholesterol.min()} – {d.Cholesterol.max()}",
                 hr=f"{d.MaxHR.min()} – {d.MaxHR.max()}", op=f"{d.Oldpeak.min():.1f} – {d.Oldpeak.max():.1f}".replace("-", "−"))
zs = {}
for k in ("cleveland", "hungarian", "switzerland", "va"):
    p = ROOT / "external" / f"processed.{k}.data"
    if p.exists():
        t = pd.read_csv(p, header=None, na_values="?")
        zs[k] = int((t[4] == 0).sum())
N["zero_by_site"] = zs
json.dump(N, open(Path(__file__).with_name("numbers.json"), "w"), indent=1, ensure_ascii=False)
print(N["feat"], zs)
