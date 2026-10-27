"""
logic.py - screening logic used by the Streamlit app and by the tests (no Streamlit import here).

Step 1 uses Group A without cholesterol. Step 2 uses all 11 measurements; an empty cholesterol field
(or 0) is treated as "not measured" and median-imputed, with no missing indicator.
"""
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import GROUP_A_NOCHOL, ALL_FEATURES, S2_LEVEL, decide, logit, prior_shift  # noqa: E402

MODELS_DIR = Path(os.environ.get("HF_MODELS_DIR", ROOT / "app" / "models"))
FIELDS_STEP1 = [c for c in ["Age", "Sex", "ChestPainType", "RestingBP", "FastingBS"] if c in GROUP_A_NOCHOL]
FIELDS_STEP2 = ALL_FEATURES
OPTIONAL = {"step1": set(), "step2": {"Cholesterol"}}
CHOICES = {"Sex": ["M", "F"], "ChestPainType": ["TA", "ATA", "NAP", "ASY"], "RestingECG": ["Normal", "ST", "LVH"],
           "ExerciseAngina": ["N", "Y"], "ST_Slope": ["Up", "Flat", "Down"]}
HARD_RANGES = {"Age": (18, 100), "RestingBP": (70, 250), "Cholesterol": (100, 600), "MaxHR": (60, 220),
               "Oldpeak": (-3.0, 7.0), "FastingBS": (0, 1)}
LABELS = ["Low risk", "High risk", "Not sure"]


class InputError(ValueError):
    def __init__(self, errors):
        super().__init__("; ".join(errors))
        self.errors = errors


_cache = {}


def load_bundle(step, models_dir=None):
    d = Path(models_dir) if models_dir else MODELS_DIR
    key = (str(d), step)
    if key not in _cache:
        f = d / f"{step}.joblib"
        if not f.exists():
            raise FileNotFoundError(f"{f} not found. Build it with: bash run_local.sh --export")
        _cache[key] = joblib.load(f)
    return _cache[key]


def _missing(v):
    return v is None or (isinstance(v, float) and np.isnan(v)) or v == ""


def validate(step, inputs, models_dir=None):
    """Return (errors, warnings). Errors block the assessment."""
    fields = FIELDS_STEP1 if step == "step1" else FIELDS_STEP2
    errors, warns = [], []
    for k in inputs:
        if k not in fields:
            errors.append(f"{k} is not used at {'Step 1' if step == 'step1' else 'Step 2'}")
    for k in fields:
        v = inputs.get(k)
        if _missing(v):
            if k not in OPTIONAL[step]:
                errors.append(f"{k} is required")
            continue
        if k in CHOICES:
            if v not in CHOICES[k]:
                errors.append(f"{k} must be one of {CHOICES[k]}")
        else:
            try:
                x = float(v)
            except (TypeError, ValueError):
                errors.append(f"{k} must be a number")
                continue
            if k == "Cholesterol" and x == 0:
                warns.append("Cholesterol 0 is treated as 'not measured'.")
                continue
            lo, hi = HARD_RANGES[k]
            if not lo <= x <= hi:
                errors.append(f"{k} must be between {lo} and {hi}")
    if not errors:
        try:
            rng = load_bundle(step, models_dir)["train_ranges"]
            for k, (lo, hi) in rng.items():
                v = inputs.get(k)
                if not _missing(v) and not (k == "Cholesterol" and float(v) == 0) and not lo <= float(v) <= hi:
                    warns.append(f"{k} is outside the range of the training data ({lo:g} to {hi:g}); the result is less reliable.")
        except FileNotFoundError:
            pass
    return errors, warns


def to_frame(step, inputs, models_dir=None):
    b = load_bundle(step, models_dir)
    row = {}
    for c in b["cols"]:
        v = inputs.get(c)
        row[c] = np.nan if (_missing(v) and c not in CHOICES) else v
    df = pd.DataFrame([row])[b["cols"]]
    for c in df.columns:
        if c not in CHOICES:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def score(step, inputs, models_dir=None):
    """Calibrated probability of heart disease at the training-data prevalence."""
    b = load_bundle(step, models_dir)
    df = to_frame(step, inputs, models_dir)
    Z = np.column_stack([logit(b["models"][m].predict_proba(df)[:, 1]) for m in b["classical"]])
    return float(b["platt"].predict(b["meta"].predict_proba(Z)[:, 1])[0])


def assess(step, inputs, sens_level=95, pi_target=0.05, models_dir=None):
    errors, warns = validate(step, inputs, models_dir)
    if errors:
        raise InputError(errors)
    b = load_bundle(step, models_dir)
    q = score(step, inputs, models_dir)
    level = sens_level if step == "step1" else S2_LEVEL
    tl, th = b["thresholds"][f"tl_e{level}"], b["thresholds"]["th_e95"]
    cls = int(decide([q], tl, th)[0])
    return {"step": step, "class": cls, "label": LABELS[cls], "q_source": q,
            "risk_target": float(prior_shift(q, pi_target, b["pi_source"])), "tl": tl, "th": th,
            "level": level, "warnings": warns}


ADVICE = {
    ("step1", 0): "No equipment tests needed now. Advise the patient to return if symptoms appear or change.",
    ("step1", 1): "Send for the Step 2 tests (resting ECG and exercise stress test) with urgent priority.",
    ("step1", 2): "Send for the Step 2 tests (resting ECG and exercise stress test) with normal priority.",
    ("step2", 0): "No specialist review needed now. Advise the patient to return if symptoms appear or change.",
    ("step2", 1): "Refer to a heart specialist.",
    ("step2", 2): "The tests do not settle the question. Refer to a heart specialist, who decides.",
}
