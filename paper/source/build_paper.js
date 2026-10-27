// Generates the class-assignment paper. Every number comes from numbers.json (make_numbers.py) which reads results/*.csv.
const fs = require('fs');
const path = require('path');
const H = require('./helpers.js');
const REFS = require('./refs.js');
const N = require('./numbers.json');
const { d, P: P0, H1, H2, BUL: BUL0, NUM: NUM0, figure, table, tcaption, eqImg, SINGLE, scaleW } = H;
const { Document, Packer, Paragraph, TextRun, AlignmentType, LevelFormat, Footer, PageNumber, BorderStyle, Table, TableRow, TableCell, WidthType, ShadingType } = d;
const FIG = path.join(__dirname, '..', '..', 'figures') + path.sep;
const SRC = __dirname + path.sep;

const order = [];
function cite(text) {
  return text.replace(/\{([A-Z0-9_,]+)\}/g, (_, keys) => {
    const nums = keys.split(',').map(k => {
      if (!REFS[k]) throw new Error('unknown ref ' + k);
      if (!order.includes(k)) order.push(k);
      return order.indexOf(k) + 1;
    }).sort((a, b) => a - b);
    return '[' + nums.join(', ') + ']';
  });
}
const P = (t, o) => P0(cite(t), o);
const BUL = (t) => BUL0(cite(t));
const NUM = (t, r) => NUM0(cite(t), r);
const T = (rows, w, o) => table(rows.map(r => r.map(c => cite(String(c)))), w, o);

const m = N.main, dd = N.data, pr = N.prim, bs = N.base, cf = N.conf, tt = N.tests, rt = N.rates;
const sub = (a, b) => (parseFloat(a) - parseFloat(b)).toFixed(1);
const hs = N.site_rows.map(r => parseFloat(r[5])), ns = N.site_rows.map(r => parseFloat(r[6]));
const hsMin = Math.min(...hs).toFixed(1), hsMax = Math.max(...hs).toFixed(1), nsMin = Math.min(...ns).toFixed(1), nsMax = Math.max(...ns).toFixed(1);
const B = [];

// ------------------------------------------------------------------ title block
B.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80, ...SINGLE },
  children: [new TextRun({ text: 'Two-Step Heart Disease Screening with a “Not Sure” Option: Classical Classifiers, Stacking and Prevalence-Adjusted Referral Rules', bold: true, size: 28 })] }));
B.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 20, ...SINGLE }, children: [new TextRun({ text: 'Ombiro Zablon B. Handson 225446  ·  Lawrence Kasera 111780', size: 21 })] }));
B.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 20, ...SINGLE }, children: [new TextRun({ text: 'School of Computing and Engineering Sciences, Strathmore University, Nairobi, Kenya', italics: true, size: 19 })] }));
B.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160, ...SINGLE }, children: [new TextRun({ text: 'Advanced Machine Learning - Class Assignment  ·  October 2026', size: 19 })] }));

// ------------------------------------------------------------------ abstract
const ABS = `Heart disease is a leading cause of death, and where specialists and exercise-test equipment are limited, patients must first be sorted with inexpensive measurements. Using the Heart Failure Prediction dataset (${dd.n} patients, 11 features), we compare seven classical and neural classifiers and two ensembles on accuracy under nested, repeated stratified cross-validation (3 × 10 outer folds, 5-fold inner tuning). The accuracy-weighted soft vote was highest (${m.SOFT.acc} ± ${m.SOFT.sd} %), but the leading methods did not differ significantly (soft vote versus random forest, corrected _t_-test _p_ = ${tt.soft_rf.p}). We therefore propose a two-step screening system built on stacking. Step 1 uses five inexpensive inputs and returns Low risk, High risk or Not sure; Step 2 adds the examination results. Cut-offs are set from training-fold out-of-fold scores only, and results are adjusted to the target prevalence. At an assumed 5 % prevalence, the system needed ${pr.tests} equipment tests per 1,000 patients instead of 1,000 and sent ${pr.missed} sick patients per 1,000 home (95 % interval ${pr.lo}–${pr.hi}), against ${bs.fixed_missed} for a fixed 0.5 cutoff at Step 1; ${pr.ns} per 1,000 remained “not sure”. In leave-one-site-out checks the Step 1 cut-off did not transfer reliably (up to ${N.site.s1_max} % of sick patients sent home at one site). The data are a single, referral-enriched, historical sample, and the system has no clinical validation.`;
B.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 60, ...SINGLE }, indent: { left: 300, right: 300 },
  border: { top: { style: BorderStyle.SINGLE, size: 4, color: '888888', space: 6 } },
  children: [new TextRun({ text: 'Abstract — ', bold: true, size: 19 }), ...H.runs(ABS, { size: 19 })] }));
B.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 160, ...SINGLE }, indent: { left: 300, right: 300 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: '888888', space: 6 } },
  children: [new TextRun({ text: 'Keywords — ', bold: true, size: 19 }), new TextRun({ text: 'heart disease screening; classical machine learning; stacking; abstention; prevalence shift; nested cross-validation.', size: 19 })] }));

// ------------------------------------------------------------------ 1 introduction
B.push(H1('1. Introduction'));
B.push(P(`Cardiovascular diseases caused an estimated 19.8 million deaths in 2022, about 32 % of all deaths worldwide {WHO}. About 5 in 100 adults in the United States (4.9 %) report coronary heart disease {CDC}. Many cases are found late, although the risk is visible in routine measurements: age, blood pressure, chest-pain history, blood sugar and cholesterol. We take as a design premise, not as a measured fact, that in many clinics a heart specialist and exercise-test equipment are not available to every patient with a complaint, so the first decision is who needs the equipment test and how soon. Risk charts address a version of this problem and have been applied in Kenya {WHORC,VUSI}; they give one score from population cohorts and no way to say that a patient cannot be classified from the inputs at hand.`));
B.push(P(`Machine-learning studies of the Heart Failure Prediction dataset report accuracies of 85–93 % {ZENG,QUMS,ZHANG}, mostly from one random split. Accuracy, however, treats the two errors alike. In screening, the dangerous error is a sick patient sent home. The question is therefore not only which classifier is most accurate, but whether a classifier can be arranged so that cheap measurements decide first, uncertain patients receive more tests, and the number of sick patients sent home is controlled.`));
B.push(P(`**Problem.** Given the 11 clinical inputs of ${dd.n} patients, classify heart disease, compare supervised methods fairly, and test an improvement. **Proposed solution.** A two-step screening system (Fig. 1): Step 1 uses five inputs available in a basic clinic and returns Low risk (home), High risk (urgent tests) or Not sure (tests, normal priority); Step 2 adds the examination results and returns Low risk, High risk or Still not sure (specialist decides). Our contributions are:`, { before: 60 }));
B.push(NUM(`A leakage-free benchmark of nine methods (seven single models, a soft vote and stacking) with corrected significance tests {NADEAU,DEMSAR}.`));
B.push(NUM(`A two-step system whose calibration, cut-offs and “not sure” bands come from training-fold out-of-fold scores, with a pre-specified rule for each design choice and prevalence-adjusted results {SAERENS}.`));
B.push(NUM(`An evaluation against two baselines, five prevalence scenarios and leave-one-site-out validation on the original UCI files {UCI}, reporting where the approach fails.`));
B.push(P(`Section 2 reviews related work, Section 3 gives the methodology (training dataset, problem, models, train/test split and the screening system), Section 4 the results, Section 5 discusses limits and Section 6 concludes. The code is at https://github.com/IncognitoGK9/two-step-heart-screening and the application is hosted at https://two-step-heart-screening.streamlit.app/.`));

// ------------------------------------------------------------------ 2 related works
B.push(H1('2. Related Works'));
B.push(P(`Three recent studies use this dataset. Zeng {ZENG} compared logistic regression, decision tree, random forest, _k_-NN and SVM and reported random forest best at 88 %. Qumsiyeh et al. {QUMS} imputed zero cholesterol and blood-pressure values and reported hold-out accuracies of 85.1–91.7 %. Zhang et al. {ZHANG} reported up to 92.6 % with a deep network. All three rely on a single split and report accuracy only. For tabular data of this size, tree ensembles generally match or exceed neural networks {GRINSZ}, which is why we include a small neural network only as a check. Table 1 places the other building blocks.`));
B.push(P(`The methods differ in what they assume. Logistic regression is a linear, interpretable baseline {COX}; a single decision tree is readable but unstable {CART}; a random forest averages many trees to reduce variance {BREIMAN01}; _k_-NN and the RBF-kernel SVM are local and kernel methods that depend on feature scaling {COVER,CORTES}; gradient boosting fits trees in sequence {FRIED01}; stacking learns how to combine the others {WOLPERT}. With 918 patients and 11 mostly low-cardinality inputs, strong models are likely to make correlated errors, so little may be gained by combining them; the results in Section 4 test this.`));
B.push(tcaption('Table 1.', 'Comparative study: related work and how this paper uses it.'));
B.push(T([
  ['Topic', 'Representative work', 'Relevance to this paper'],
  ['Same dataset', 'Zeng {ZENG}; Qumsiyeh et al. {QUMS}; Zhang et al. {ZHANG}', 'Accuracy comparisons on one split; no referral or “not sure” output. We re-run the comparison under repeated nested cross-validation.'],
  ['Risk charts', 'WHO charts {WHORC}; Nairobi application {VUSI}', 'Cheap inputs, one fixed score. Our Step 1 is a learned counterpart with an explicit “not sure” band.'],
  ['Spectrum bias', 'Ransohoff and Feinstein {RANSOHOFF}', 'Test performance depends on the patient mix. Our sample is referral-enriched; we report prevalence scenarios.'],
  ['Prior shift, calibration', 'Saerens et al. {SAERENS}; Van Calster et al. {VANCALSTER}; Guo et al. {GUO}', 'Odds are re-weighted to a new prevalence; probabilities must be calibrated to be read as risk.'],
  ['Abstention', 'Vovk et al. {VOVK}; Angelopoulos and Bates {ANGEL}', 'Conformal cut-offs give coverage guarantees; compared here with an empirical two-threshold rule.'],
  ['Ensembles', 'Wolpert {WOLPERT}', 'Stacking learns how to combine models; used as our proposed scorer.'],
], scaleW([1250, 2850, 4200]), { size: 16, keep: true }));
B.push(P(`In short, prior work on this dataset ranks classifiers by accuracy; the screening literature supplies calibration, prevalence adjustment and abstention. We combine the two and measure the result in screening terms: tests needed, sick patients sent home and patients left “not sure”.`, { before: 80 }));

// ------------------------------------------------------------------ 3 dataset and problem
B.push(H1('3. Methodology'));
B.push(H2('3.1 Training dataset'));
B.push(P(`The dataset {FEDE} pools several 1980s cohorts, including the four held in the UCI repository {UCI}. It has ${dd.n} rows, 11 predictors and a binary target, _HeartDisease_, equal to 1 for ${dd.pos} patients (${dd.pos_pct} %) and 0 for ${dd.neg}. Table 2 lists the predictors and their group: Group A is what a basic clinic can record, Group B needs an ECG or exercise test.`));
B.push(tcaption('Table 2.', `Predictors (n = ${dd.n}). Group A: basic clinic; Group B: examination tests.`));
const F = N.feat;
B.push(T([
  ['Feature', 'Group', 'Description', 'Values / range'],
  ['Age', 'A', 'Age (years)', F.age],
  ['Sex', 'A', 'Sex', F.Sex],
  ['ChestPainType', 'A', 'Chest pain type', F.CP],
  ['RestingBP', 'A', 'Resting blood pressure (mm Hg)', `${F.bp} (${dd.bp0} zero)`],
  ['FastingBS', 'A', 'Fasting blood sugar > 120 mg/dl', F.FBS],
  ['Cholesterol', 'A*', 'Serum cholesterol (mg/dl)', `${F.chol} (${dd.zero} zeros)`],
  ['RestingECG', 'B', 'Resting ECG result', F.ECG],
  ['MaxHR', 'B', 'Maximum heart rate (bpm)', F.hr],
  ['ExerciseAngina', 'B', 'Exercise-induced angina', F.EA],
  ['Oldpeak', 'B', 'ST depression (mm)', F.op],
  ['ST_Slope', 'B', 'Peak-exercise ST slope', F.SL],
], scaleW([1500, 650, 3000, 3150]), { size: 16, keep: true }));
const zs = N.zero_by_site;
B.push(P(`A* marks cholesterol, which needs a blood test and is used only in Step 2. ${dd.zero} patients (${dd.zero_pct} %) have Cholesterol = 0, a placeholder for an unmeasured value, and one has RestingBP = 0. Zero cholesterol is associated with disease in ${dd.zero_prev} % of these patients against ${dd.rest_prev} % of the rest. In the UCI files all ${dd.zero} zeros come from two sites (Switzerland ${zs.switzerland}, VA Long Beach ${zs.va}), so the zero indicates the recording site as much as a clinical fact; we therefore treat zeros as missing, impute the training-fold median, and add no missing-value indicator {SPERRIN}.`, { before: 100 }));
B.push(H2('3.2 Problem statement'));
B.push(P(`With inputs **x** and label _y_ ∈ {0, 1}, we first seek a classifier maximising accuracy,`));
B.push(eqImg(SRC + 'eq1.png', 1));
B.push(P(`following the assignment, and then a screening rule that maps **x** to {Low, High, Not sure} so that few sick patients are sent home and few healthy patients receive the equipment test.`));

// ------------------------------------------------------------------ 4 methodology
B.push(H2('3.3 Pre-processing and input sets'));
B.push(P(`Categorical inputs are one-hot encoded and numeric inputs standardised inside a scikit-learn pipeline {SKLEARN}, fitted on training data only. Five configurations are used: **F_raw** (all 11 inputs, zeros kept as values, the common practice and the main comparison), **F_ind** (zeros imputed plus a missing-value flag), **F_imp** (zeros imputed; used for Step 2), **A_nochol** (Group A without cholesterol; used for Step 1) and **A_chol** (Group A with cholesterol). A_chol replaces A_nochol at Step 1 only if it raises the stacking AUROC by at least 0.01 (pre-specified).`));
B.push(H2('3.4 Classifiers'));
B.push(P(`Seven single models are tuned by 5-fold inner grid search on accuracy: logistic regression (LR) {COX}, decision tree (DT) {CART}, random forest (RF) {BREIMAN01}, _k_-NN {COVER}, RBF-kernel SVM {CORTES}, histogram gradient boosting (GBM) {FRIED01} and a small multilayer perceptron (MLP). Two ensembles combine the six classical models. The **accuracy-weighted soft vote** (SOFT) sets each model’s weight by a softmax of its inner accuracy _a_~m~ with temperature τ = 0.01, fixed before any outer evaluation,`));
B.push(eqImg(SRC + 'eq2.png', 2));
B.push(P(`and the vote is the weighted average of the models’ probabilities, thresholded at 0.5,`));
B.push(eqImg(SRC + 'eq3.png', 3));
B.push(P(`**Stacking** (STACK) fits a logistic-regression combiner on the logits of the six models’ inner out-of-fold probabilities _p̃_~m~ {WOLPERT},`));
B.push(eqImg(SRC + 'eq4.png', 4));
B.push(P(`so the combiner never sees a probability from a model that was fitted on the same patient.`));
B.push(tcaption('Table 3.', 'Hyper-parameter grids searched by the inner 5-fold cross-validation (scripts/common.py); other settings are scikit-learn defaults.'));
B.push(T([
  ['Model', 'Hyper-parameters searched', 'Grid size'],
  ['LR', 'C ∈ {0.01, 0.1, 1, 10, 100}', '5'],
  ['DT', 'max_depth ∈ {3, 4, 5, 6, 8, None}; min_samples_leaf ∈ {1, 5, 10, 20}', '24'],
  ['RF', '200 trees; max_features ∈ {sqrt, 0.5}; min_samples_leaf ∈ {1, 3}', '4'],
  ['k-NN', 'k ∈ {3, 5, 7, 9, 11, 15, 21, 25}; weights ∈ {uniform, distance}', '16'],
  ['SVM', 'RBF kernel; C ∈ {0.1, 1, 10}; γ ∈ {scale, 0.01, 0.1}', '9'],
  ['GBM', 'max_leaf_nodes ∈ {7, 15}; learning_rate ∈ {0.03, 0.1}; max_iter ∈ {100, 200}', '8'],
  ['MLP', 'hidden layers ∈ {(16), (32, 16)}; L2 penalty ∈ {0.001, 0.01}', '4'],
], scaleW([900, 6500, 900]), { size: 16, keep: true }));
B.push(H2('3.5 Evaluation protocol and train/test split'));
B.push(P(`The outer loop is stratified 10-fold cross-validation repeated 3 times (30 test folds of about 92 patients). Inside each training fold, a stratified 5-fold grid search selects hyper-parameters, produces the out-of-fold probabilities used for stacking, calibration and cut-offs, and the model is refitted and evaluated once on the held-out fold, so tuning never touches test data {CAWLEY}. A single stratified 80/20 hold-out (734 training, 184 test patients) is also reported for comparison with the literature. The random seed is 42. A unit test flips the test labels and checks that predictions do not change, which would expose any leakage. The “best” method is fixed before the results: the highest mean outer-fold accuracy on F_raw, ties of under 0.1 point broken by AUROC.`));
B.push(H2('3.6 Two-step screening system'));
B.push(...figure(SRC + 'fig_flow.png', 5.7, 'Figure 1.', 'Two-step screening system. The dangerous error is a sick patient sent home at Step 1.'));
B.push(P(`Each step scores a patient with STACK, then applies Platt scaling {PLATT} fitted on out-of-fold scores, and finally a prior-shift correction {SAERENS} from the dataset prevalence π~s~ = ${dd.pos_pct} % to a target prevalence π~t~,`));
B.push(eqImg(SRC + 'eq5.png', 5));
B.push(P(`where _s_ is the stacking score. Sensitivity and specificity do not change under this correction (assuming the class-conditional distributions stay fixed); predictive values, flag rates and counts per 1,000 do. We report π~t~ = 2, 5, 10, 20 % and the dataset’s own ${dd.pos_pct} %; 5 % is the reference, close to the US figure {CDC}. The decision rule is`));
B.push(eqImg(SRC + 'eq6.png', 6));
B.push(P(`where _t_~L~ is the largest score with at most 1 − _s_ of the sick out-of-fold patients below it (release level _s_ = 95 % at Step 1, with 90 and 98 % as sensitivity analysis; 98 % at Step 2) and _t_~H~ the smallest score with at most 5 % of the healthy patients above it. Patients between the cut-offs, or where the cut-offs cross, are Not sure. A conformal alternative sets the cut-offs by class-conditional (Mondrian) conformal calibration with the finite-sample correction {VOVK,ANGEL}; the mechanism is chosen by a pre-specified rule: both coverages at least 0.94, then the smaller Step 1 “not sure” share, a tie (under 0.5 point) going to the simpler two-threshold rule. Cholesterol is optional at Step 2: empty or zero is treated as missing and imputed. Intervals are 95 % Wilson intervals on class-conditional rates {WILSON}.`));
B.push(H2('3.7 Statistical testing'));
B.push(P(`Equal performance of the nine methods is tested with the Friedman test on per-fold ranks {FRIEDMAN,DEMSAR}. Pairwise differences use the corrected resampled _t_-test {NADEAU}, with _d_~j~ the per-fold difference, d̄ its mean, σ̂² its variance and _J_ = 30,`));
B.push(eqImg(SRC + 'eq7.png', 7));
B.push(P(`compared with Student’s _t_ on 29 degrees of freedom; Wilcoxon _p_-values are also computed. α = 0.05.`));

// ------------------------------------------------------------------ 5 results
B.push(H1('4. Results'));
B.push(P(`Software: ${N.run.versions.replace('python','Python')}. The full pipeline ran in 679 s on a 12-core Mac (terminal log); the earlier five-model benchmark is reproduced within 0.2 accuracy points (for example RF 87.43 against 87.32 %; docs/DECISIONS.md).`));
B.push(H2('4.1 Classification results for all methods'));
const ord = ['LR', 'DT', 'RF', 'KNN', 'SVM', 'GBM', 'MLP', 'SOFT', 'STACK'];
const nm = { SOFT: 'Soft vote (first proposal)', STACK: 'Stacking (final proposal)' };
const bestI = ord.indexOf(N.best_cv) + 1;
const rows3 = [['Method', 'CV acc. (%)', 'Hold-out (%)', 'AUROC', 'Sens.', 'Spec.', 'Brier', 'ECE raw → cal.']]
  .concat(ord.map(k => [nm[k] || m[k].name, `${m[k].acc} ± ${m[k].sd}`, m[k].ho, m[k].auroc, m[k].sens, m[k].spec, m[k].brier, `${m[k].ece_raw} → ${m[k].ece_cal}`]));
rows3[bestI][1] = '**' + m[N.best_cv].acc + '** ± ' + m[N.best_cv].sd;
B.push(tcaption('Table 4.', `All methods on the 11 inputs (F_raw): mean ± sd over 30 outer folds, 80/20 hold-out accuracy, and other metrics (fold means). Best CV accuracy in bold. Source: results/table_main.csv, table_holdout.csv.`));
B.push(T(rows3, scaleW([2150, 1200, 850, 800, 650, 650, 650, 1350]), { size: 16, ruleAbove: [8], shadeRows: [8, 9], keep: true }));
B.push(P(`The best method under the rule in §3.5 is the accuracy-weighted soft vote (${m.SOFT.acc} %), followed by stacking (${m.STACK.acc} %) and random forest (${m.RF.acc} %); the decision tree trails (${m.DT.acc} %). The Friedman test rejects equal performance (χ² = ${tt.chi2}, _p_ = ${tt.p_friedman}), driven mainly by the tree. Among the leading methods the differences are not significant: soft vote versus random forest Δ = ${tt.soft_rf.diff} points (_p_ = ${tt.soft_rf.p}); stacking versus random forest Δ = ${tt.stack_rf.diff} (_p_ = ${tt.stack_rf.p}); stacking versus soft vote Δ = ${tt.stack_soft.diff} (_p_ = ${tt.stack_soft.p}). Hold-out accuracies are 2–4 points higher than the cross-validated means for every model except the tree, which is 5 points lower, and they reorder the models, as expected from a 184-patient test set (binomial standard error about 2.4 points). On this evidence accuracy cannot separate the strong methods. Handling of zero cholesterol matters little for the ensembles: the soft vote reaches ${m.SOFT.acc} % raw, ${m.SOFT.acc_ind} % with the indicator, and 87.40 % with imputation alone (results/summary.md); only logistic regression gains clearly from the indicator (${m.LR.acc} to ${m.LR.acc_ind} %).`, { before: 100 }));
B.push(...figure(SRC + 'fig_cm_rel.png', 5.6, 'Figure 2.', `(a) Confusion matrix of the best method (${cf.method}), pooled over the test folds of one repetition (n = ${dd.n}): accuracy ${cf.acc} %, sensitivity ${cf.sens} %, specificity ${cf.spec} %. (b) Reliability of the stacking score before and after Platt scaling.`));
B.push(H2('4.2 Which inputs matter, and the choice of Step 1'));
const LL = N.ladder.LR, RR = N.ladder.RF;
B.push(P(`Adding inputs in clinical order (Fig. 3) shows that chest pain gives the first large gain (LR AUROC ${LL[0].auroc} to ${LL[1].auroc}), resting blood pressure and the blood tests add little (${LL[2].gain} and ${LL[3].gain} for LR; ${RR[2].gain} and ${RR[3].gain} for RF), the resting ECG adds nothing (${LL[4].gain} and ${RR[4].gain}), and the exercise group adds the most (${LL[5].gain} and ${RR[5].gain}). Step 1 with five Group A inputs reaches STACK AUROC ${N.tier.A_nochol.STACK.auroc} (accuracy ${N.tier.A_nochol.STACK.acc} %); adding cholesterol raises AUROC to ${N.tier.A_chol.STACK.auroc} (accuracy ${N.tier.A_chol.STACK.acc} %), a gain of ${N.chol_gain}, below the 0.01 threshold, so Step 1 excludes it. Step 2 with all inputs reaches AUROC ${N.tier.F_imp.STACK.auroc} (accuracy ${N.tier.F_imp.STACK.acc} %). Step 1 alone is therefore a weak classifier, which is why a “not sure” option is needed.`));
B.push(...figure(FIG + 'fig_ladder.png', 4.3, 'Figure 3.', 'Outer-fold AUROC (mean ± sd) as inputs are added, for logistic regression and random forest.'));
B.push(H2('4.3 Two-step system'));
B.push(P(`Table 5 gives the system at the 5 % reference prevalence; each row adds or removes one component, so the table is also the ablation of the proposal. Per 1,000 patients screened, the two-step system with Step 1 released at 95 % sensitivity sends ${pr.tests} patients for equipment tests (${(1000 - pr.tests)} fewer than testing everyone) and sends ${pr.missed} sick patients home (${pr.missed1} at Step 1), final sensitivity ${pr.sens} %. The fixed 0.5 cutoff at Step 1 needs fewer tests (${bs.fixed_tests}) but misses ${bs.fixed_missed}, about ${(parseFloat(bs.fixed_missed) / parseFloat(pr.missed)).toFixed(1)} times as many. Testing everyone and keeping the “not sure” band misses fewer (${bs.ns_missed}) but needs 1,000 tests and leaves ${bs.ns_ns} patients not sure. Raising the Step 1 release level from 90 to 98 % lowers missed cases (${N.sys_rows[0][2].split(' ')[0]} to ${N.sys_rows[2][2].split(' ')[0]}) and raises tests (${N.sys_rows[0][1]} to ${N.sys_rows[2][1]}). Of every 1,000 patients, ${pr.high} are called High risk with positive predictive value ${pr.ppv}, and ${pr.ns} end Not sure. At Step 1, ${rt.sick_home1} % of sick and ${rt.healthy_home1} % of healthy patients are sent home; overall ${rt.healthy_home_final} % of healthy patients go home and ${rt.ns_healthy} % are left not sure.`));
B.push(tcaption('Table 5.', 'Systems at 5 % prevalence, per 1,000 patients screened (stacking scorer). Source: results/table_system.csv.'));
B.push(T([['System', 'Tests', 'Sick sent home (95 % interval)', 'Sens. (%)', 'High risk', 'PPV', 'Not sure']].concat(N.sys_rows),
  scaleW([3300, 700, 1700, 800, 750, 600, 750]), { size: 16, boldRows: [2], ruleAbove: [4], keep: true }));
B.push(...figure(FIG + 'fig_tradeoff.png', 4.3, 'Figure 4.', 'Sick patients sent home against patients needing equipment tests, per 1,000 at 5 % prevalence: two-step systems at Step 1 release levels of 90, 95 and 98 % (stacking and random forest) and the two baselines.'));
B.push(P(`Prevalence changes the predictive values and counts, not the rates (Table 6). At 2 % prevalence the system sends home ${N.scen_rows[0][2].split(' ')[0]} sick patients per 1,000 with PPV ${N.scen_rows[0][4]}; at 20 %, ${N.scen_rows[3][2].split(' ')[0]} with PPV ${N.scen_rows[3][4]}. Using the dataset’s own prevalence (${dd.pos_pct} %) would overstate PPV (${N.scen_rows[4][4]}) for a general clinic.`, { before: 100 }));
B.push(tcaption('Table 6.', 'Two-step system (Step 1 release 95 %) by assumed prevalence, per 1,000 patients.'));
B.push(T([['Prevalence', 'Tests', 'Sick sent home (95 % interval)', 'High risk', 'PPV', 'NPV', 'Not sure']].concat(N.scen_rows),
  scaleW([1700, 700, 2200, 900, 700, 700, 800]), { size: 16, boldRows: [2], keep: true }));
B.push(H2('4.4 Calibration and the “not sure” mechanism'));
B.push(P(`Platt scaling changes little: stacking Brier ${N.calib.stack_brier_raw} before and ${N.calib.stack_brier_cal} after, ECE ${N.calib.stack_ece_raw} and ${N.calib.stack_ece_cal} on F_imp; for _k_-NN it worsens ECE (${N.calib.knn_ece_raw} to ${N.calib.knn_ece_cal}). The two cut-off mechanisms were close at Step 1: the empirical two-threshold rule leaves ${N.mech.e.cov_sick} % of sick patients out of the Low group and ${N.mech.e.cov_healthy} % of healthy ones out of the High group, with ${N.mech.e.ns} % Not sure; the conformal rule gives ${N.mech.c.cov_sick} %, ${N.mech.c.cov_healthy} % and ${N.mech.c.ns} %. The gap in the “not sure” share is under 2 points, and the pre-specified rule selects the two-threshold rule, which also has the smaller final “not sure” share (${N.mech.e.ns_final} against ${N.mech.c.ns_final} %).`));
B.push(H2('4.5 Leave-one-site-out validation'));
B.push(P(`To test transfer to a new hospital, models were trained on three UCI sites and applied to the fourth, with cut-offs from the training sites (Table 7; results/table_site_validation.csv). AUROC ranges from ${N.site.a1_min} to ${N.site.a1_max} at Step 1 and ${N.site.a2_min} to ${N.site.a2_max} at Step 2. The Step 1 cut-off, designed to send at most 5 % of sick patients home, sends ${N.site.s1_max} % home at the ${N.site.s1_max_site} site, where prevalence is lowest; Step 2 releases no further sick patients at any site. Healthy patients sent home range from ${hsMin} % to ${hsMax} % and the Not sure share from ${nsMin} % to ${nsMax} %.`));
B.push(tcaption('Table 7.', 'Leave-one-site-out results (stacking, Step 1 release 95 %).'));
B.push(T([['Held-out site', 'Patients (prev.)', 'AUROC Step 1', 'AUROC Step 2', 'Sick sent home (%)', 'Healthy sent home (%)', 'Not sure (%)']].concat(N.site_rows),
  scaleW([1500, 1300, 950, 950, 1100, 1250, 1000]), { size: 16, keep: true }));

B.push(H2('4.6 Web application'));
B.push(P(`The system is available as a web application at https://two-step-heart-screening.streamlit.app/, built with Streamlit from the code in the submitted archive. Fig. 5 shows it on a phone-sized screen, the setting of a basic clinic. Each screen asks only for the inputs of its step, rejects values outside plausible ranges, and returns Low risk, High risk or Not sure with a plain instruction. It also shows the estimated risk at the chosen prevalence and, at Step 1, the share of sick patients that cross-validation sent home. The sidebar sets the assumed prevalence and the Step 1 safety level, and cholesterol can be marked as not measured at Step 2. The values in the figure are the form’s default example values, not a real patient. The application carries the warning that it is a research prototype and not for clinical use.`));
B.push(...figure(FIG + 'fig_app_mobile.png', 5.7, 'Figure 5.', 'Mobile view of the application, left to right: title, warning and Step 1 form; Step 1 result (Not sure, send for the Step 2 tests); Step 2 result (Not sure, refer to a specialist).'));

// ------------------------------------------------------------------ 6 discussion
B.push(H1('5. Discussion and Limitations'));
B.push(P(`**What the evidence supports.** Accuracy does not separate the strong methods: the best (soft vote, ${m.SOFT.acc} %) is not significantly better than random forest or stacking. The value of the proposal is in screening measures, not accuracy: against a fixed 0.5 cutoff at Step 1 it sends home about ${(parseFloat(bs.fixed_missed) / parseFloat(pr.missed)).toFixed(1)} times fewer sick patients, and against testing everyone it needs ${(1000 - pr.tests)} fewer equipment tests per 1,000 at a cost of ${sub(pr.missed, bs.ns_missed)} extra sick patients sent home per 1,000. This is a trade-off, not a free gain, and about one patient in three (${pr.ns} per 1,000) still ends Not sure. The input ladder agrees with clinical practice: chest pain and the exercise test carry the signal, the resting ECG adds nothing in this data.`));
B.push(P(`**Why abstention.** A forced 0.5 decision after testing everyone misses ${bs.all_missed} sick patients per 1,000; adding the “not sure” band lowers this to ${bs.ns_missed} by moving ${bs.ns_ns} patients to the specialist. Abstention does not remove error; it moves it into a referral queue whose size is the price. In the two-step system, the final outcomes among sick patients are ${(100 - parseFloat(rt.sick_home_final) - parseFloat(rt.ns_sick)).toFixed(1)} % High risk, ${rt.ns_sick} % Not sure and ${rt.sick_home_final} % Low risk.`));
B.push(P(`**Why accuracy ties.** All models see the same 11 inputs, and the input ladder shows that most of the signal sits in a few of them (chest pain, exercise test), so a likely explanation is that the strong models make largely the same errors and a combination has little to correct. We did not measure error overlap directly (not yet measured).`));
B.push(P(`**Limitations.** (1) One dataset, pooled from 1980s cohorts, with no clinical validation; the web application is a research prototype. (2) The sample is referral-enriched, so spectrum bias {RANSOHOFF} applies: prior-shift correction fixes the prevalence, not the case mix. (3) Chest-pain type depends on how the question is asked. (4) The cholesterol signal is entangled with the recording site (§3.1). (5) The 5 % reference is a US figure {CDC}; no Kenyan prevalence is used. (6) Leave-one-site-out results show the Step 1 cut-off does not transfer cleanly, so local recalibration would be needed. (7) The “not sure” share of about one third is high.`));

// ------------------------------------------------------------------ 7 conclusion
B.push(H1('6. Conclusion'));
B.push(P(`Under repeated nested cross-validation, the soft vote, stacking and random forest are statistically tied on accuracy (about ${m.SOFT.acc} %), and only the decision tree is clearly worse. A two-step screening system built on stacking answers a different question: at 5 % prevalence it needs ${pr.tests} tests per 1,000 patients and sends ${pr.missed} sick patients home, between the cheap fixed-cutoff pathway (${bs.fixed_tests} tests, ${bs.fixed_missed} missed) and testing everyone (1,000 tests, ${bs.all_missed} missed with a forced decision). The approach fails to transfer cleanly across sites and leaves a third of patients not sure, so it supports a research prototype, not clinical use. Next steps are external validation on a local cohort and recalibration of the Step 1 cut-off per site.`));

// ------------------------------------------------------------------ references
B.push(H1('References'));
const refParas = () => order.map((k, i) => new Paragraph({ spacing: { after: 40, ...SINGLE }, indent: { left: 420, hanging: 420 }, alignment: AlignmentType.LEFT,
  children: [new TextRun({ text: `[${i + 1}]\t`, size: 17 }), ...H.runs(REFS[k], { size: 17 })], tabStops: [{ type: 'left', position: 420 }] }));
const REF_AT = B.length; B.push(null);
const APP = [];
APP.push(H1('Appendix A. Reproducibility'));
APP.push(P(`The code, results and paper source are available at https://github.com/IncognitoGK9/two-step-heart-screening and in the submitted archive. The web application is hosted on Streamlit Community Cloud at https://two-step-heart-screening.streamlit.app/. The full study runs in about 11 minutes (679 s) on a 12-core Apple M3 Pro Mac; the README in the repository gives the commands. Library versions are pinned in requirements.txt and the random seed is 42.`));

const unused = Object.keys(REFS).filter(k => !order.includes(k));
if (unused.length) { console.error('UNUSED REFS:', unused.join(', ')); process.exit(1); }
const body = B.slice(0, REF_AT).concat(refParas(), APP);

const doc = new Document({
  creator: 'Handson & Kasera', title: 'Two-Step Heart Disease Screening',
  styles: {
    default: { document: { run: { font: 'Times New Roman', size: 22 }, paragraph: { spacing: { line: 360, lineRule: 'auto', after: 0 } } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 24, bold: true, font: 'Times New Roman' }, paragraph: { spacing: { before: 200, after: 0 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 22, bold: true, italics: true, font: 'Times New Roman' }, paragraph: { spacing: { before: 140, after: 0 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: 'bullets', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 500, hanging: 260 } } } }] },
    { reference: 'nums', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '(%1)', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 500, hanging: 380 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 2160, right: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, spacing: SINGLE, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18 })] })] }) },
    children: body,
  }],
});
Packer.toBuffer(doc).then(buf => { fs.writeFileSync(process.argv[2] || 'paper.docx', buf); console.log('refs used:', order.length); });
