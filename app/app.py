"""Streamlit app: two-step heart disease screening. Research prototype, not for clinical use.
Run:  bash run_local.sh --app   (or: streamlit run app/app.py)"""
import streamlit as st

import logic

st.set_page_config(page_title="Heart screening prototype", page_icon=None, layout="centered")
st.title("Two-step heart disease screening")
st.warning("Research prototype, not for clinical use. Built from 918 patients in one public dataset; "
           "it has not been validated on patients in a clinic.")

step_name = st.sidebar.radio("Screen", ["Step 1: basic clinic", "Step 2: hospital"])
step = "step1" if step_name.startswith("Step 1") else "step2"
prev = st.sidebar.selectbox("Assumed share of the people you screen who have heart disease", [2, 5, 10, 20], index=1,
                            format_func=lambda x: f"{x}%")
level = 95
if step == "step1":
    level = st.sidebar.selectbox("Step 1 safety level (share of sick patients the step must not send home)",
                                 [90, 95, 98], index=1, format_func=lambda x: f"{x}%")

try:
    bundle = logic.load_bundle(step)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

CP = {"TA": "Typical angina", "ATA": "Atypical angina", "NAP": "Non-anginal pain", "ASY": "No chest pain (asymptomatic)"}
ECG = {"Normal": "Normal", "ST": "ST-T wave abnormality", "LVH": "Left ventricular hypertrophy"}
SLOPE = {"Up": "Upsloping", "Flat": "Flat", "Down": "Downsloping"}

st.header("Step 1: basic clinic" if step == "step1" else "Step 2: hospital")
st.caption("Measurements: age, sex, chest pain, resting blood pressure, fasting blood sugar." if step == "step1" else
           "All 11 measurements. Cholesterol may be left unmeasured.")

with st.form("patient"):
    c1, c2 = st.columns(2)
    inputs = {}
    inputs["Age"] = c1.number_input("Age (years)", 0, 120, 50)
    inputs["Sex"] = c2.selectbox("Sex", logic.CHOICES["Sex"], format_func=lambda x: {"M": "Male", "F": "Female"}[x])
    inputs["ChestPainType"] = c1.selectbox("Chest pain type", logic.CHOICES["ChestPainType"], format_func=CP.get, index=3)
    inputs["RestingBP"] = c2.number_input("Resting blood pressure (mmHg)", 0, 300, 130)
    inputs["FastingBS"] = int(c1.checkbox("Fasting blood sugar above 120 mg/dL"))
    if step == "step2":
        not_measured = c2.checkbox("Cholesterol not measured")
        chol = c2.number_input("Cholesterol (mg/dL)", 0, 700, 220)
        inputs["Cholesterol"] = None if not_measured else chol
        inputs["RestingECG"] = c1.selectbox("Resting ECG", logic.CHOICES["RestingECG"], format_func=ECG.get)
        inputs["MaxHR"] = c2.number_input("Maximum heart rate reached (beats/min)", 0, 260, 140)
        inputs["ExerciseAngina"] = c1.selectbox("Chest pain during exercise", logic.CHOICES["ExerciseAngina"],
                                                format_func=lambda x: {"N": "No", "Y": "Yes"}[x])
        inputs["Oldpeak"] = c2.number_input("ST depression during exercise (Oldpeak)", -5.0, 10.0, 0.0, step=0.1)
        inputs["ST_Slope"] = c1.selectbox("ST slope at peak exercise", logic.CHOICES["ST_Slope"], format_func=SLOPE.get)
    go = st.form_submit_button("Assess")

if go:
    try:
        r = logic.assess(step, inputs, sens_level=level, pi_target=prev / 100)
    except logic.InputError as e:
        for m in e.errors:
            st.error(m)
        st.stop()
    box = {0: st.success, 1: st.error, 2: st.warning}[r["class"]]
    box(f"**{r['label']}**")
    st.write(logic.ADVICE[(step, r["class"])])
    st.metric(f"Estimated risk, if {prev}% of the people screened have heart disease", f"{r['risk_target'] * 100:.1f}%")
    st.caption(f"Estimated risk in the training data (55% with heart disease): {r['q_source'] * 100:.0f}%.")
    for w in r["warnings"]:
        st.info(w)
    if step == "step1":
        miss = bundle.get("cv_step1_missed_sick_rate", {}).get(level)
        if miss is not None:
            st.caption(f"In cross-validation, about {miss * 100:.0f} of every 100 people with heart disease were "
                       f"classified Low risk at this step and sent home.")
    else:
        st.caption("Step 2 sends home only patients it rates Low risk at a 98% safety level.")

v = bundle["versions"]
st.divider()
st.caption(f"Model built {bundle['built']} with scikit-learn {v['scikit-learn']}, Python {v['python']}; "
           f"trained on {bundle['n_train']} patients.")
