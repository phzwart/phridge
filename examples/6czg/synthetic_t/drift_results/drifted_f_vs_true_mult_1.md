# Held-Out Log-Likelihood Comparison: `drifted_f` vs `true_model`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `-0.3427` nats/refl
- **Model B NLL**: `-0.1658` nats/refl
- **Difference (Gain $\Delta$)**: `-0.1769` nats/refl (`+0.1769` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-159.19` nats
- **Uncertainty**: Bootstrap SE = `0.0428` (95% CI: `[-0.2679, -0.1020]`) | Naive SE = `0.0121` (Ratio: `3.54`x)
- **Win Fraction $P(d_h > 0)$**: `18.3%` (165 wins, 735 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `1.71e-86`
- **Wilcoxon Signed-Rank $p$-value**: `6.95e-69`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.2389` | `-0.0620` | `+0.1769` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.1455` | `+0.0314` | `+0.1769` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`true_model`) | Model B (`drifted_f`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0037` | `-0.0021` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `-0.2392` | `-0.1374` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1035` | `-0.0284` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.4315 | -1.1717 | -0.2599 | 0.0267 | 2.3% | 1/42 |
| 1 | 6.10 | 4.86 | 46 | 0.0711 | 0.7383 | -0.6671 | 0.1055 | 0.0% | 0/46 |
| 2 | 4.83 | 4.21 | 45 | -0.6230 | -0.2846 | -0.3384 | 0.0379 | 0.0% | 0/45 |
| 3 | 4.20 | 3.81 | 46 | -0.1966 | 0.1409 | -0.3375 | 0.0431 | 0.0% | 0/46 |
| 4 | 3.80 | 3.53 | 46 | -0.1032 | 0.3360 | -0.4391 | 0.0502 | 0.0% | 0/46 |
| 5 | 3.53 | 3.32 | 46 | 0.1441 | 0.5134 | -0.3693 | 0.0453 | 4.3% | 2/44 |
| 6 | 3.31 | 3.16 | 44 | -0.4083 | -0.1772 | -0.2310 | 0.0250 | 6.8% | 3/41 |
| 7 | 3.14 | 3.02 | 45 | 0.1935 | 0.3757 | -0.1822 | 0.0440 | 13.3% | 6/39 |
| 8 | 3.01 | 2.89 | 46 | -0.4553 | -0.2550 | -0.2004 | 0.0285 | 13.0% | 6/40 |
| 9 | 2.88 | 2.79 | 43 | -0.4898 | -0.2745 | -0.2153 | 0.0383 | 11.6% | 5/38 |
| 10 | 2.79 | 2.70 | 45 | -0.7168 | -0.5564 | -0.1604 | 0.0251 | 11.1% | 5/40 |
| 11 | 2.70 | 2.62 | 45 | -0.3674 | -0.2123 | -0.1551 | 0.0251 | 8.9% | 4/41 |
| 12 | 2.62 | 2.55 | 44 | -0.4573 | -0.3805 | -0.0768 | 0.0292 | 34.1% | 15/29 |
| 13 | 2.55 | 2.49 | 47 | -0.3809 | -0.2769 | -0.1041 | 0.0328 | 23.4% | 11/36 |
| 14 | 2.49 | 2.43 | 43 | -0.6402 | -0.5808 | -0.0595 | 0.0259 | 34.9% | 15/28 |
| 15 | 2.43 | 2.38 | 43 | -0.0180 | -0.0855 | +0.0675 | 0.0493 | 44.2% | 19/24 |
| 16 | 2.38 | 2.33 | 44 | -0.3516 | -0.4058 | +0.0542 | 0.0561 | 45.5% | 20/24 |
| 17 | 2.32 | 2.28 | 47 | -0.1726 | -0.2335 | +0.0609 | 0.0576 | 36.2% | 17/30 |
| 18 | 2.28 | 2.24 | 48 | -0.3085 | -0.3376 | +0.0291 | 0.0374 | 37.5% | 18/30 |
| 19 | 2.24 | 2.20 | 44 | -0.2307 | -0.2973 | +0.0666 | 0.0576 | 40.9% | 18/26 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-13, 1, 8) | 4.90 | 286007.4 | 1880.0 | 152.1 | 610.7 | 652.8 | 7.590 | 10.971 | -3.381 |
| (9, 5, 4) | 5.27 | 126633.4 | 1439.0 | 88.0 | 404.6 | 441.8 | 3.128 | 5.785 | -2.656 |
| (14, 2, 4) | 5.33 | 191022.5 | 1524.0 | 125.3 | 496.8 | 532.5 | 4.500 | 6.846 | -2.346 |
| (1, 7, 2) | 4.96 | 78873.8 | 1370.1 | 57.6 | 319.2 | 350.1 | 2.108 | 4.126 | -2.018 |
| (27, 7, 12) | 2.34 | 12520.4 | 99.1 | 126.3 | 127.1 | 116.3 | 2.595 | 0.749 | +1.846 |
| (30, 2, 14) | 2.32 | 10789.6 | 173.0 | 62.4 | 119.2 | 111.8 | 2.741 | 0.938 | +1.803 |
| (2, 2, 14) | 4.11 | 122539.2 | 1141.1 | 107.4 | 400.6 | 426.7 | 4.566 | 6.316 | -1.750 |
| (-14, 2, 12) | 3.78 | 72853.3 | 380.7 | 191.4 | 304.8 | 329.7 | 1.800 | 3.461 | -1.662 |
| (14, 4, 0) | 5.00 | 55794.1 | 534.0 | 104.5 | 266.4 | 293.5 | 1.302 | 2.888 | -1.586 |
| (5, 9, 4) | 3.70 | 52903.9 | 744.2 | 71.1 | 258.4 | 281.4 | 1.245 | 2.722 | -1.477 |
| (35, 1, 10) | 2.23 | 7867.0 | 108.4 | 72.6 | 101.2 | 95.5 | 2.302 | 0.875 | +1.427 |
| (-16, 4, 12) | 3.38 | 80193.7 | 611.6 | 131.1 | 321.5 | 340.6 | 4.017 | 5.415 | -1.398 |
| (0, 4, 12) | 4.33 | 80152.9 | 414.4 | 193.4 | 322.3 | 346.6 | 1.643 | 2.919 | -1.276 |
| (32, 6, 2) | 2.41 | 13761.9 | 200.9 | 68.5 | 133.9 | 130.8 | 2.646 | 1.412 | +1.234 |
| (6, 14, 10) | 2.29 | 8012.3 | 146.5 | 54.7 | 101.6 | 93.4 | 1.877 | 0.661 | +1.216 |
| (-9, 5, 4) | 5.30 | 73520.8 | 1046.7 | 70.2 | 310.4 | 334.0 | 2.034 | 3.235 | -1.201 |
| (8, 2, 0) | 9.10 | 78201.9 | 374.4 | 208.9 | 319.1 | 352.7 | 0.596 | 1.740 | -1.144 |
| (-8, 8, 6) | 3.78 | 27261.2 | 590.2 | 46.2 | 187.2 | 207.5 | 0.515 | 1.658 | -1.142 |
| (-14, 4, 2) | 4.94 | 126721.0 | 1288.0 | 98.4 | 405.2 | 428.7 | 3.404 | 4.499 | -1.094 |
| (17, 5, 0) | 4.08 | 59851.4 | 763.7 | 78.4 | 272.0 | 291.1 | 1.507 | 2.598 | -1.092 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `true_model` | 1.51 | 0.9998 | 199.18862156908295 | 1000.0 | 1.000 |
| `drifted_f` | 6.08 | 0.9990 | 199.1739897783472 | 1000.0 | 1.000 |