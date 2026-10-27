"""
common.py (v3) - configuration, feature groups, preprocessing, models, calibration,
abstention thresholds and statistics shared by every script.

Dataset: fedesoriano (2021), Heart Failure Prediction Dataset, Kaggle (918 patients, 11 features).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "dataset" / "heart.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# ----------------------------------------------------------------- protocol
SEED = 42
N_SPLITS = 10            # outer stratified folds
N_REPEATS = 3            # outer repeats -> 30 test folds
INNER_SPLITS = 5         # inner folds: tuning, out-of-fold predictions, calibration, thresholds
HOLDOUT_TEST_SIZE = 0.20
TAU = 0.01               # temperature of the accuracy-weighted soft vote (fixed a priori, as in the original paper)
SENS_LEVELS = (0.90, 0.95, 0.98)   # Step 1 release sensitivity sweep (pre-specified primary: 0.95)
S1_PRIMARY = 95
S2_LEVEL = 98            # Step 2 release sensitivity (pre-specified)
ALPHA_HIGH = 0.05        # at most 5% of healthy patients may be labelled High risk
SCENARIOS = {"2%": 0.02, "5% (reference)": 0.05, "10%": 0.10, "20%": 0.20,
             "55.3% (dataset)": 508 / 918}
PI_SOURCE = 508 / 918

# ----------------------------------------------------------------- features
TARGET = "HeartDisease"
CAT_COLS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
NUM_COLS = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
ZERO_AS_MISSING = ["Cholesterol", "RestingBP"]
GROUP_A = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS"]
GROUP_A_NOCHOL = [c for c in GROUP_A if c != "Cholesterol"]
GROUP_B = ["RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"]
ALL_FEATURES = GROUP_A + GROUP_B
LADDER = [
    ("1 Age, sex", ["Age", "Sex"]),
    ("2 + chest pain", ["Age", "Sex", "ChestPainType"]),
    ("3 + resting BP", ["Age", "Sex", "ChestPainType", "RestingBP"]),
    ("4 + blood tests", GROUP_A),
    ("5 + resting ECG", GROUP_A + ["RestingECG"]),
    ("6 + exercise test", ALL_FEATURES),
]

# config -> (columns, pre-processing variant)
#   raw        zeros kept as values (common practice)
#   impute     zeros -> missing, median imputation (no indicator)
#   impute_ind zeros -> missing, median imputation + missing indicator (original proposed pre-processing)
CONFIGS = {
    "F_raw": (ALL_FEATURES, "raw"),
    "F_ind": (ALL_FEATURES, "impute_ind"),
    "F_imp": (ALL_FEATURES, "impute"),       # Step 2 of the screening system
    "A_chol": (GROUP_A, "impute"),           # Step 1 with cholesterol (empty = median)
    "A_nochol": (GROUP_A_NOCHOL, "raw"),     # Step 1 without cholesterol
}
STEP2_CONFIG = "F_imp"

CLASSICAL = ["LR", "DT", "RF", "KNN", "SVM", "GBM"]
SINGLE = CLASSICAL + ["MLP"]
ALL_MODELS = SINGLE + ["SOFT", "STACK"]
MODEL_NAMES = {
    "LR": "Logistic regression", "DT": "Decision tree", "RF": "Random forest",
    "KNN": "k-NN", "SVM": "SVM (RBF)", "GBM": "Gradient boosting", "MLP": "MLP (small)",
    "SOFT": "Accuracy-weighted soft vote", "STACK": "Stacking (LR combiner)",
}


def load_data():
    df = pd.read_csv(DATA_FILE)
    return df.drop(columns=[TARGET]), df[TARGET].to_numpy()


def zeros_to_nan(X):
    X = X.copy()
    for c in ZERO_AS_MISSING:
        if c in X.columns:
            X[c] = X[c].replace(0, np.nan)
    return X


def make_preprocessor(cols, variant):
    num = [c for c in NUM_COLS if c in cols]
    cat = [c for c in CAT_COLS if c in cols]
    ohe = OneHotEncoder(handle_unknown="ignore")
    if variant == "raw":
        return ColumnTransformer([("num", StandardScaler(), num), ("cat", ohe, cat)])
    ind = variant == "impute_ind"
    if variant == "impute_all":
        numpipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        catpipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", ohe)])
        ct = ColumnTransformer([("num", numpipe, num), ("cat", catpipe, cat)])
        return Pipeline([("zeros", FunctionTransformer(zeros_to_nan)), ("ct", ct)])
    numpipe = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=ind)),
                        ("scale", StandardScaler())])
    ct = ColumnTransformer([("num", numpipe, num), ("cat", ohe, cat)])
    return Pipeline([("zeros", FunctionTransformer(zeros_to_nan)), ("ct", ct)])


def model_specs():
    return {
        "LR": (LogisticRegression(max_iter=5000), {"clf__C": [0.01, 0.1, 1, 10, 100]}),
        "DT": (DecisionTreeClassifier(random_state=SEED),
               {"clf__max_depth": [3, 4, 5, 6, 8, None], "clf__min_samples_leaf": [1, 5, 10, 20]}),
        "RF": (RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=1),
               {"clf__max_features": ["sqrt", 0.5], "clf__min_samples_leaf": [1, 3]}),
        "KNN": (KNeighborsClassifier(),
                {"clf__n_neighbors": [3, 5, 7, 9, 11, 15, 21, 25], "clf__weights": ["uniform", "distance"]}),
        "SVM": (SVC(kernel="rbf", probability=True, random_state=SEED),
                {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01, 0.1]}),
        "GBM": (HistGradientBoostingClassifier(random_state=SEED),
                {"clf__max_leaf_nodes": [7, 15], "clf__learning_rate": [0.03, 0.1],
                 "clf__max_iter": [100, 200]}),
        "MLP": (MLPClassifier(max_iter=500, random_state=SEED),
                {"clf__hidden_layer_sizes": [(16,), (32, 16)], "clf__alpha": [1e-3, 1e-2]}),
    }


# ------------------------------------------------- combination, calibration
def accuracy_weights(inner_acc, tau=TAU):
    a = np.asarray(inner_acc, dtype=float)
    w = np.exp((a - a.max()) / tau)
    return w / w.sum()


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class Platt:
    """Logistic (Platt) calibration fitted on out-of-fold training predictions only."""

    def fit(self, p, y):
        self.m = LogisticRegression(C=1e4, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
        return self

    def predict(self, p):
        return self.m.predict_proba(logit(p).reshape(-1, 1))[:, 1]


def prior_shift(q, pi_target, pi_source=PI_SOURCE):
    """Move calibrated probabilities from the source prevalence to a target prevalence (Saerens et al., 2002)."""
    q = np.clip(np.asarray(q, dtype=float), 1e-12, 1 - 1e-12)
    odds = q / (1 - q) * (pi_target / (1 - pi_target)) / (pi_source / (1 - pi_source))
    return odds / (1 + odds)


def make_thresholds(q, y):
    """
    Thresholds on calibrated out-of-fold probabilities q (training fold only).
      tl_e<s>: empirical lower cut, <= (1-s) of sick patients fall below it.
      tl_c<s>: Mondrian-conformal lower cut with the finite-sample (+1) correction.
      th_e95 / th_c95: upper cuts, <= 5% of healthy patients above it (empirical / conformal).
    Rule: Low risk if q < tl; High risk if q > th; otherwise (or if both) Not sure.
    """
    q1 = np.sort(q[y == 1])
    q0 = np.sort(q[y == 0])
    n1, n0 = len(q1), len(q0)
    out = {}
    for s in SENS_LEVELS:
        a = round(1 - s, 4)
        tag = int(round(s * 100))
        out[f"tl_e{tag}"] = float(np.quantile(q1, a, method="lower"))
        m = int(np.floor(a * (n1 + 1))) - 1
        out[f"tl_c{tag}"] = float(q1[m]) if m >= 0 else -np.inf
    out["th_e95"] = float(np.quantile(q0, 1 - ALPHA_HIGH, method="higher"))
    m0 = int(np.floor(ALPHA_HIGH * (n0 + 1))) - 1
    out["th_c95"] = float(q0[n0 - 1 - m0]) if m0 >= 0 else np.inf
    return out


def decide(q, tl, th):
    """0 = Low risk, 1 = High risk, 2 = Not sure (also when both cuts fire)."""
    q = np.asarray(q)
    low, high = q < tl, q > th
    return np.where(low & ~high, 0, np.where(high & ~low, 1, 2))


# ------------------------------------------------------------- statistics
def corrected_resampled_ttest(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test."""
    from scipy import stats
    d = np.asarray(a) - np.asarray(b)
    var = d.var(ddof=1)
    if var == 0:
        return float(d.mean()), np.nan, 1.0
    t = d.mean() / np.sqrt((1.0 / len(d) + n_test / n_train) * var)
    return float(d.mean()), float(t), float(2 * stats.t.sf(abs(t), df=len(d) - 1))


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))
