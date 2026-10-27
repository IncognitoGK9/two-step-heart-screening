"""App-logic tests: five fixed patients against the saved models, boundary values, missing fields,
and the cholesterol rule. Needs the saved models (bash run_local.sh --export)."""
import json
import os
from pathlib import Path

import pytest

import logic

MODELS = Path(os.environ.get("HF_MODELS_DIR", logic.MODELS_DIR))
needs_models = pytest.mark.skipif(not (MODELS / "step1.joblib").exists(),
                                  reason="saved models not found; run: bash run_local.sh --export")


def fixtures():
    return json.load(open(MODELS / "expected_outputs.json"))


def sub(inputs, fields):
    return {k: inputs[k] for k in fields}


@needs_models
def test_five_fixed_patients_match_saved_models():
    fx = fixtures()
    assert len(fx) == 5
    for rec in fx:
        for step, fields in (("step1", logic.FIELDS_STEP1), ("step2", logic.FIELDS_STEP2)):
            r = logic.assess(step, sub(rec["inputs"], fields), 95, 0.05, models_dir=MODELS)
            exp = rec[step]
            assert r["label"] == exp["label"]
            assert r["q_source"] == pytest.approx(exp["q_source"], abs=1e-4)
            assert r["risk_target"] == pytest.approx(exp["risk_target"], abs=1e-4)


@needs_models
def test_probabilities_in_unit_interval_and_risk_falls_with_lower_prevalence():
    rec = fixtures()[2]["inputs"]
    hi = logic.assess("step2", sub(rec, logic.FIELDS_STEP2), 95, 0.20, models_dir=MODELS)
    lo = logic.assess("step2", sub(rec, logic.FIELDS_STEP2), 95, 0.02, models_dir=MODELS)
    assert 0 <= lo["q_source"] <= 1
    assert lo["risk_target"] < hi["risk_target"]


@needs_models
def test_cholesterol_not_measured_equals_zero_and_is_not_a_signal():
    rec = sub(fixtures()[3]["inputs"], logic.FIELDS_STEP2)
    a = logic.assess("step2", {**rec, "Cholesterol": None}, 95, 0.05, models_dir=MODELS)
    b = logic.assess("step2", {**rec, "Cholesterol": 0}, 95, 0.05, models_dir=MODELS)
    assert a["q_source"] == pytest.approx(b["q_source"]) and a["label"] == b["label"]
    # an empty field must not make everyone High risk: compare with a typical measured value
    c = logic.assess("step2", {**rec, "Cholesterol": 220}, 95, 0.05, models_dir=MODELS)
    assert abs(a["q_source"] - c["q_source"]) < 0.25


@needs_models
def test_step1_rejects_fields_it_does_not_use():
    rec = sub(fixtures()[0]["inputs"], logic.FIELDS_STEP1)
    for extra in ({"Cholesterol": 200}, {"MaxHR": 150}):
        with pytest.raises(logic.InputError):
            logic.assess("step1", {**rec, **extra}, 95, 0.05, models_dir=MODELS)


@needs_models
@pytest.mark.parametrize("field,value", [("Age", 17), ("Age", 101), ("RestingBP", 69), ("RestingBP", 251),
                                          ("FastingBS", 2), ("Sex", "X"), ("ChestPainType", "ZZ")])
def test_step1_boundary_values_rejected(field, value):
    rec = sub(fixtures()[0]["inputs"], logic.FIELDS_STEP1)
    with pytest.raises(logic.InputError):
        logic.assess("step1", {**rec, field: value}, 95, 0.05, models_dir=MODELS)


@needs_models
@pytest.mark.parametrize("field,value", [("Age", 18), ("Age", 100), ("RestingBP", 70), ("RestingBP", 250)])
def test_step1_boundary_values_accepted(field, value):
    rec = sub(fixtures()[0]["inputs"], logic.FIELDS_STEP1)
    r = logic.assess("step1", {**rec, field: value}, 95, 0.05, models_dir=MODELS)
    assert r["label"] in logic.LABELS


@needs_models
@pytest.mark.parametrize("step,fields", [("step1", logic.FIELDS_STEP1), ("step2", logic.FIELDS_STEP2)])
def test_missing_required_field_rejected(step, fields):
    rec = sub(fixtures()[0]["inputs"], fields)
    for k in fields:
        if k == "Cholesterol":
            continue
        bad = {x: v for x, v in rec.items() if x != k}
        with pytest.raises(logic.InputError):
            logic.assess(step, bad, 95, 0.05, models_dir=MODELS)


@needs_models
def test_higher_safety_level_never_sends_more_patients_home():
    for rec in fixtures():
        inp = sub(rec["inputs"], logic.FIELDS_STEP1)
        low = [logic.assess("step1", inp, s, 0.05, models_dir=MODELS)["label"] == "Low risk" for s in (90, 95, 98)]
        assert low[0] >= low[1] >= low[2]


def test_streamlit_app_renders():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(logic.ROOT) / "app" / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
