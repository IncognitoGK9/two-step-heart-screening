# Results summary

Protocol: 3 x 10 stratified outer folds (30 test folds), 5-fold inner tuning, seed 42. All fitted or chosen quantities (tuning, stacking, calibration, thresholds) use training folds only.

## Config F_raw: all 11 features (raw)

| name | Accuracy | auroc | sens | spec | brier_raw | brier_cal | ece_raw | ece_cal |
|---|---|---|---|---|---|---|---|---|
| Accuracy-weighted soft vote | 87.72 ± 3.14 | 0.934 | 0.909 | 0.837 | 0.096 | 0.096 | 0.023 | 0.017 |
| Stacking (LR combiner) | 87.54 ± 3.04 | 0.935 | 0.908 | 0.835 | 0.096 | 0.096 | 0.017 | 0.018 |
| Random forest | 87.43 ± 3.54 | 0.931 | 0.914 | 0.825 | 0.100 | 0.100 | 0.033 | 0.029 |
| Gradient boosting | 87.00 ± 3.23 | 0.931 | 0.898 | 0.835 | 0.098 | 0.099 | 0.021 | 0.018 |
| SVM (RBF) | 86.78 ± 3.59 | 0.930 | 0.906 | 0.820 | 0.101 | 0.101 | 0.017 | 0.018 |
| k-NN | 86.71 ± 3.48 | 0.925 | 0.896 | 0.831 | 0.103 | 0.111 | 0.038 | 0.080 |
| Logistic regression | 86.42 ± 3.23 | 0.926 | 0.894 | 0.827 | 0.104 | 0.105 | 0.023 | 0.012 |
| MLP (small) | 86.16 ± 3.87 | 0.930 | 0.889 | 0.828 | 0.100 | 0.100 | 0.039 | 0.031 |
| Decision tree | 83.98 ± 3.28 | 0.898 | 0.867 | 0.807 | 0.121 | 0.122 | 0.045 | 0.047 |

## Config F_ind: all 11 features (impute_ind)

| name | Accuracy | auroc | sens | spec | brier_raw | brier_cal | ece_raw | ece_cal |
|---|---|---|---|---|---|---|---|---|
| Accuracy-weighted soft vote | 87.76 ± 3.01 | 0.934 | 0.914 | 0.833 | 0.096 | 0.096 | 0.026 | 0.025 |
| Stacking (LR combiner) | 87.62 ± 3.27 | 0.935 | 0.910 | 0.834 | 0.096 | 0.096 | 0.022 | 0.022 |
| Gradient boosting | 87.33 ± 3.28 | 0.932 | 0.904 | 0.836 | 0.097 | 0.098 | 0.014 | 0.018 |
| Logistic regression | 87.32 ± 3.02 | 0.930 | 0.907 | 0.832 | 0.102 | 0.101 | 0.027 | 0.024 |
| Random forest | 87.32 ± 3.44 | 0.931 | 0.916 | 0.820 | 0.100 | 0.099 | 0.036 | 0.030 |
| SVM (RBF) | 86.85 ± 3.13 | 0.929 | 0.906 | 0.823 | 0.101 | 0.101 | 0.023 | 0.024 |
| k-NN | 86.06 ± 3.35 | 0.923 | 0.900 | 0.811 | 0.105 | 0.113 | 0.032 | 0.076 |
| MLP (small) | 85.94 ± 3.54 | 0.926 | 0.886 | 0.826 | 0.105 | 0.104 | 0.043 | 0.025 |
| Decision tree | 84.02 ± 3.20 | 0.898 | 0.867 | 0.807 | 0.119 | 0.121 | 0.041 | 0.042 |

## Config F_imp: all 11 features (impute)

| name | Accuracy | auroc | sens | spec | brier_raw | brier_cal | ece_raw | ece_cal |
|---|---|---|---|---|---|---|---|---|
| Accuracy-weighted soft vote | 87.40 ± 3.81 | 0.930 | 0.911 | 0.828 | 0.100 | 0.099 | 0.025 | 0.024 |
| Stacking (LR combiner) | 87.32 ± 3.52 | 0.929 | 0.906 | 0.833 | 0.100 | 0.100 | 0.025 | 0.028 |
| Random forest | 87.10 ± 4.05 | 0.928 | 0.912 | 0.820 | 0.103 | 0.103 | 0.036 | 0.034 |
| MLP (small) | 86.85 ± 3.55 | 0.924 | 0.895 | 0.836 | 0.103 | 0.104 | 0.041 | 0.039 |
| SVM (RBF) | 86.56 ± 3.85 | 0.924 | 0.909 | 0.812 | 0.105 | 0.105 | 0.024 | 0.022 |
| Gradient boosting | 86.27 ± 3.46 | 0.926 | 0.890 | 0.829 | 0.102 | 0.101 | 0.028 | 0.036 |
| Logistic regression | 86.13 ± 3.37 | 0.923 | 0.889 | 0.827 | 0.106 | 0.106 | 0.026 | 0.025 |
| k-NN | 85.51 ± 3.78 | 0.913 | 0.901 | 0.798 | 0.112 | 0.120 | 0.042 | 0.078 |
| Decision tree | 83.99 ± 3.38 | 0.904 | 0.883 | 0.786 | 0.118 | 0.118 | 0.027 | 0.026 |

## Config A_chol: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS (impute)

| name | Accuracy | auroc | sens | spec | brier_raw | brier_cal | ece_raw | ece_cal |
|---|---|---|---|---|---|---|---|---|
| Stacking (LR combiner) | 79.27 ± 3.73 | 0.858 | 0.849 | 0.723 | 0.151 | 0.151 | 0.024 | 0.026 |
| MLP (small) | 78.98 ± 3.18 | 0.849 | 0.848 | 0.717 | 0.155 | 0.155 | 0.038 | 0.028 |
| Gradient boosting | 78.97 ± 3.13 | 0.853 | 0.858 | 0.705 | 0.152 | 0.152 | 0.039 | 0.031 |
| Accuracy-weighted soft vote | 78.83 ± 3.59 | 0.858 | 0.850 | 0.712 | 0.151 | 0.151 | 0.027 | 0.032 |
| Logistic regression | 78.61 ± 3.53 | 0.857 | 0.837 | 0.724 | 0.153 | 0.153 | 0.025 | 0.023 |
| Random forest | 78.54 ± 3.76 | 0.851 | 0.844 | 0.712 | 0.155 | 0.155 | 0.042 | 0.036 |
| SVM (RBF) | 77.23 ± 3.63 | 0.839 | 0.841 | 0.687 | 0.161 | 0.161 | 0.021 | 0.022 |
| k-NN | 76.98 ± 4.28 | 0.841 | 0.865 | 0.652 | 0.163 | 0.166 | 0.048 | 0.068 |
| Decision tree | 76.80 ± 3.33 | 0.834 | 0.841 | 0.677 | 0.164 | 0.166 | 0.047 | 0.049 |

## Config A_nochol: Age, Sex, ChestPainType, RestingBP, FastingBS (raw)

| name | Accuracy | auroc | sens | spec | brier_raw | brier_cal | ece_raw | ece_cal |
|---|---|---|---|---|---|---|---|---|
| Logistic regression | 78.14 ± 3.70 | 0.855 | 0.829 | 0.722 | 0.154 | 0.154 | 0.027 | 0.023 |
| Gradient boosting | 77.96 ± 2.88 | 0.851 | 0.850 | 0.693 | 0.156 | 0.155 | 0.025 | 0.028 |
| MLP (small) | 77.92 ± 3.32 | 0.848 | 0.841 | 0.702 | 0.158 | 0.157 | 0.032 | 0.026 |
| Accuracy-weighted soft vote | 77.89 ± 3.02 | 0.855 | 0.841 | 0.702 | 0.153 | 0.154 | 0.038 | 0.035 |
| Stacking (LR combiner) | 77.78 ± 3.31 | 0.853 | 0.832 | 0.711 | 0.154 | 0.154 | 0.034 | 0.031 |
| k-NN | 77.20 ± 4.01 | 0.837 | 0.873 | 0.647 | 0.161 | 0.162 | 0.043 | 0.044 |
| Decision tree | 77.09 ± 3.14 | 0.837 | 0.840 | 0.685 | 0.161 | 0.163 | 0.042 | 0.046 |
| SVM (RBF) | 77.01 ± 3.50 | 0.837 | 0.840 | 0.684 | 0.165 | 0.165 | 0.032 | 0.037 |
| Random forest | 76.25 ± 3.83 | 0.838 | 0.818 | 0.693 | 0.163 | 0.161 | 0.056 | 0.041 |

## Tests (accuracy per outer fold)

| config | contrast | mean_diff | stat | p_corrected_t | p_wilcoxon |
|---|---|---|---|---|---|
| F_raw | Friedman over 9 methods | nan | 56.2474 | nan | 0.0000 |
| F_raw | STACK vs RF | 0.1115 | 0.1955 | 0.8464 | 0.7777 |
| F_raw | SOFT vs RF | 0.2926 | 0.4485 | 0.6571 | 0.6851 |
| F_raw | STACK vs SOFT | -0.1812 | -0.4446 | 0.6599 | 0.4539 |
| F_ind | Friedman over 9 methods | nan | 77.1455 | nan | 0.0000 |
| F_ind | STACK vs GBM | 0.2879 | 0.3789 | 0.7075 | 0.6835 |
| F_ind | STACK vs RF | 0.2926 | 0.5091 | 0.6146 | 0.0640 |
| F_ind | SOFT vs RF | 0.4380 | 0.6571 | 0.5163 | 0.0506 |
| F_ind | SOFT vs GBM | 0.4332 | 0.4812 | 0.6340 | 0.3895 |
| F_ind | STACK vs SOFT | -0.1453 | -0.2741 | 0.7859 | 0.4574 |
| F_imp | Friedman over 9 methods | nan | 48.1677 | nan | 0.0000 |
| F_imp | STACK vs RF | 0.2198 | 0.2764 | 0.7842 | 0.5253 |
| F_imp | SOFT vs RF | 0.2922 | 0.4311 | 0.6696 | 0.4584 |
| F_imp | STACK vs SOFT | -0.0725 | -0.1308 | 0.8968 | 0.9476 |
| A_chol | Friedman over 9 methods | nan | 43.4973 | nan | 0.0000 |
| A_chol | STACK vs MLP | 0.2879 | 0.3451 | 0.7325 | 0.6081 |
| A_chol | STACK vs RF | 0.7290 | 0.7266 | 0.4733 | 0.1114 |
| A_chol | SOFT vs RF | 0.2914 | 0.3796 | 0.7070 | 0.2012 |
| A_chol | SOFT vs MLP | -0.1497 | -0.1561 | 0.8770 | 0.5288 |
| A_chol | STACK vs SOFT | 0.4376 | 0.5605 | 0.5794 | 0.2925 |
| A_nochol | Friedman over 9 methods | nan | 24.0855 | nan | 0.0022 |
| A_nochol | STACK vs LR | -0.3619 | -0.5372 | 0.5952 | 0.2944 |
| A_nochol | STACK vs RF | 1.5261 | 1.4843 | 0.1485 | 0.0031 |
| A_nochol | SOFT vs RF | 1.6332 | 1.3984 | 0.1726 | 0.0046 |
| A_nochol | SOFT vs LR | -0.2548 | -0.2911 | 0.7730 | 0.6253 |
| A_nochol | STACK vs SOFT | -0.1071 | -0.1733 | 0.8636 | 0.8152 |

## Best method (declared rule)

highest mean outer-CV accuracy, F_raw; <0.1 point gap broken by AUROC: **Accuracy-weighted soft vote**; pooled confusion matrix, repeat 1: TN=343, FP=67, FN=45, TP=463.

## Ablation ladder (AUROC)

| model | rung | acc_mean | acc_sd | auroc_mean | auroc_sd | auroc_gain |
|---|---|---|---|---|---|---|
| LR | 1 Age, sex | 0.6863 | 0.0361 | 0.7321 | 0.0502 | nan |
| LR | 2 + chest pain | 0.7814 | 0.0340 | 0.8441 | 0.0398 | 0.1120 |
| LR | 3 + resting BP | 0.7796 | 0.0398 | 0.8436 | 0.0389 | -0.0005 |
| LR | 4 + blood tests | 0.7861 | 0.0348 | 0.8573 | 0.0370 | 0.0137 |
| LR | 5 + resting ECG | 0.7865 | 0.0342 | 0.8561 | 0.0378 | -0.0012 |
| LR | 6 + exercise test | 0.8609 | 0.0337 | 0.9225 | 0.0261 | 0.0665 |
| RF | 1 Age, sex | 0.6710 | 0.0403 | 0.6905 | 0.0484 | nan |
| RF | 2 + chest pain | 0.7680 | 0.0389 | 0.8125 | 0.0401 | 0.1219 |
| RF | 3 + resting BP | 0.7600 | 0.0417 | 0.8255 | 0.0396 | 0.0130 |
| RF | 4 + blood tests | 0.7864 | 0.0367 | 0.8521 | 0.0363 | 0.0266 |
| RF | 5 + resting ECG | 0.7977 | 0.0349 | 0.8498 | 0.0405 | -0.0023 |
| RF | 6 + exercise test | 0.8718 | 0.0416 | 0.9291 | 0.0282 | 0.0793 |

## Automatic decisions

```
{
  "best_method_rule": "highest mean outer-CV accuracy, F_raw; <0.1 point gap broken by AUROC",
  "best_method": "SOFT",
  "cholesterol_auroc_gain_stack": 0.005441575003985233,
  "step1_config": "A_nochol",
  "step1_rule": "A_chol only if it raises STACK AUROC by >= 0.01, else A_nochol",
  "proposed_model": "STACK",
  "proposed_rule": "STACK if its inner out-of-fold AUROC (F_imp) >= best single model, else best single",
  "oof_auc_F_imp": {
    "DT": 0.8949,
    "GBM": 0.922,
    "KNN": 0.9128,
    "LR": 0.9215,
    "MLP": 0.9174,
    "RF": 0.9226,
    "SOFT": 0.9275,
    "STACK": 0.9246,
    "SVM": 0.9207
  },
  "mechanism": "two-threshold (empirical)",
  "mechanism_rule": "both class coverages >= 0.94; smaller Step 1 not-sure share; tie (<0.5 point) -> two-threshold"
}
```

## Mechanism comparison (proposed model, Step 1, 95% release)

| mechanism | cov_sick_step1 | cov_healthy_step1 | not_sure_step1 | cov_sick_final | not_sure_final |
|---|---|---|---|---|---|
| two-threshold (empirical) | 0.9600 | 0.9504 | 0.5236 | 0.9416 | 0.3391 |
| Mondrian conformal | 0.9600 | 0.9561 | 0.5338 | 0.9469 | 0.3631 |

## System at the 5% reference scenario (per 1,000 screened)

| model | system | s1 | tests_per_1000 | missed_step1_per_1000 | missed_final_per_1000 | final_sens | high_risk_per_1000 | ppv_high | not_sure_final_per_1000 |
|---|---|---|---|---|---|---|---|---|---|
| LR | two-step/e | 90 | 408.662 | 5.118 | 5.479 | 0.890 | 78.481 | 0.380 | 298.153 |
| LR | two-step/e | 95 | 536.409 | 2.493 | 3.248 | 0.935 | 78.547 | 0.381 | 340.481 |
| LR | two-step/e | 98 | 691.617 | 0.984 | 2.034 | 0.959 | 78.613 | 0.381 | 381.020 |
| LR | baseline: fixed 0.5 cutoff at Step 1 | 0 | 305.616 | 8.530 | 11.089 | 0.778 | 158.626 | 0.245 | 0.000 |
| LR | baseline: test everyone, forced 0.5 | 0 | 1000.000 | 0.000 | 5.545 | 0.889 | 208.968 | 0.213 | 0.000 |
| LR | baseline: test everyone, Step 2 only/e | 0 | 1000.000 | 0.000 | 1.148 | 0.977 | 78.613 | 0.381 | 433.654 |
| RF | two-step/e | 90 | 416.714 | 4.790 | 5.479 | 0.890 | 71.682 | 0.386 | 245.481 |
| RF | two-step/e | 95 | 520.386 | 2.297 | 3.215 | 0.936 | 73.587 | 0.381 | 294.498 |
| RF | two-step/e | 98 | 668.250 | 1.181 | 2.198 | 0.956 | 73.686 | 0.382 | 337.896 |
| RF | baseline: fixed 0.5 cutoff at Step 1 | 0 | 332.091 | 9.088 | 11.253 | 0.775 | 163.869 | 0.236 | 0.000 |
| RF | baseline: test everyone, forced 0.5 | 0 | 1000.000 | 0.000 | 4.396 | 0.912 | 216.295 | 0.211 | 0.000 |
| RF | baseline: test everyone, Step 2 only/e | 0 | 1000.000 | 0.000 | 1.181 | 0.976 | 73.686 | 0.382 | 401.474 |
| STACK | two-step/e | 90 | 408.152 | 4.856 | 5.217 | 0.896 | 78.285 | 0.378 | 283.938 |
| STACK | two-step/e | 95 | 520.682 | 2.001 | 2.920 | 0.942 | 78.449 | 0.380 | 331.640 |
| STACK | two-step/e | 98 | 706.161 | 1.115 | 2.198 | 0.956 | 78.449 | 0.380 | 377.931 |
| STACK | baseline: fixed 0.5 cutoff at Step 1 | 0 | 316.560 | 8.399 | 10.925 | 0.781 | 155.701 | 0.251 | 0.000 |
| STACK | baseline: test everyone, forced 0.5 | 0 | 1000.000 | 0.000 | 4.692 | 0.906 | 204.414 | 0.222 | 0.000 |
| STACK | baseline: test everyone, Step 2 only/e | 0 | 1000.000 | 0.000 | 1.148 | 0.977 | 78.449 | 0.380 | 433.045 |
