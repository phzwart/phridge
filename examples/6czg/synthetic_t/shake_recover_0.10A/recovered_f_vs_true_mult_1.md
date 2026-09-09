# Held-Out Log-Likelihood Comparison: `recovered_f` vs `deposited`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `-0.3427` nats/refl
- **Model B NLL**: `-0.1152` nats/refl
- **Difference (Gain $\Delta$)**: `-0.2275` nats/refl (`+0.2275` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-204.71` nats
- **Uncertainty**: Bootstrap SE = `0.0230` (95% CI: `[-0.2744, -0.1843]`) | Naive SE = `0.0137` (Ratio: `1.68`x)
- **Win Fraction $P(d_h > 0)$**: `19.9%` (179 wins, 721 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `1.01e-77`
- **Wilcoxon Signed-Rank $p$-value**: `2.15e-68`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.3234` | `-0.0959` | `+0.2275` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.1744` | `+0.0530` | `+0.2275` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`deposited`) | Model B (`recovered_f`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0037` | `-0.0086` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `-0.2392` | `-0.1113` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1035` | `-0.0039` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.4315 | -1.1361 | -0.2954 | 0.0295 | 4.7% | 2/41 |
| 1 | 6.10 | 4.86 | 46 | 0.0711 | 0.4904 | -0.4193 | 0.0626 | 4.3% | 2/44 |
| 2 | 4.83 | 4.21 | 45 | -0.6230 | -0.2718 | -0.3512 | 0.0461 | 2.2% | 1/44 |
| 3 | 4.20 | 3.81 | 46 | -0.1966 | 0.0778 | -0.2744 | 0.0590 | 13.0% | 6/40 |
| 4 | 3.80 | 3.53 | 46 | -0.1032 | 0.3022 | -0.4054 | 0.0724 | 6.5% | 3/43 |
| 5 | 3.53 | 3.32 | 46 | 0.1441 | 0.3819 | -0.2378 | 0.0643 | 26.1% | 12/34 |
| 6 | 3.31 | 3.16 | 44 | -0.4083 | -0.1149 | -0.2934 | 0.0510 | 18.2% | 8/36 |
| 7 | 3.14 | 3.02 | 45 | 0.1935 | 0.4019 | -0.2084 | 0.0864 | 22.2% | 10/35 |
| 8 | 3.01 | 2.89 | 46 | -0.4553 | -0.1634 | -0.2920 | 0.0575 | 19.6% | 9/37 |
| 9 | 2.88 | 2.79 | 43 | -0.4898 | -0.1657 | -0.3242 | 0.0578 | 9.3% | 4/39 |
| 10 | 2.79 | 2.70 | 45 | -0.7168 | -0.5322 | -0.1846 | 0.0432 | 13.3% | 6/39 |
| 11 | 2.70 | 2.62 | 45 | -0.3674 | -0.1535 | -0.2139 | 0.0499 | 24.4% | 11/34 |
| 12 | 2.62 | 2.55 | 44 | -0.4573 | -0.3432 | -0.1141 | 0.0509 | 38.6% | 17/27 |
| 13 | 2.55 | 2.49 | 47 | -0.3809 | -0.1688 | -0.2122 | 0.0478 | 21.3% | 10/37 |
| 14 | 2.49 | 2.43 | 43 | -0.6402 | -0.5181 | -0.1221 | 0.0417 | 32.6% | 14/29 |
| 15 | 2.43 | 2.38 | 43 | -0.0180 | 0.1050 | -0.1230 | 0.0684 | 30.2% | 13/30 |
| 16 | 2.38 | 2.33 | 44 | -0.3516 | -0.2061 | -0.1455 | 0.0767 | 31.8% | 14/30 |
| 17 | 2.32 | 2.28 | 47 | -0.1726 | -0.0985 | -0.0741 | 0.0687 | 27.7% | 13/34 |
| 18 | 2.28 | 2.24 | 48 | -0.3085 | -0.1482 | -0.1603 | 0.0548 | 22.9% | 11/37 |
| 19 | 2.24 | 2.20 | 44 | -0.2307 | -0.1397 | -0.0910 | 0.0721 | 29.5% | 13/31 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (9, 3, 14) | 3.66 | 132444.0 | 1109.9 | 119.3 | 413.5 | 445.8 | 3.779 | 5.906 | -2.127 |
| (2, 2, 14) | 4.11 | 122539.2 | 1141.1 | 107.4 | 400.6 | 434.5 | 4.566 | 6.611 | -2.046 |
| (1, 7, 2) | 4.96 | 78873.8 | 1370.1 | 57.6 | 319.2 | 354.5 | 2.108 | 4.079 | -1.971 |
| (20, 2, 14) | 2.94 | 7282.9 | 148.1 | 49.2 | 98.9 | 119.5 | 0.223 | 2.124 | -1.900 |
| (14, 2, 4) | 5.33 | 191022.5 | 1524.0 | 125.3 | 496.8 | 535.5 | 4.500 | 6.297 | -1.797 |
| (27, 7, 12) | 2.34 | 12520.4 | 99.1 | 126.3 | 127.1 | 118.2 | 2.595 | 0.849 | +1.746 |
| (-12, 6, 14) | 3.11 | 10869.7 | 108.5 | 100.2 | 117.3 | 138.1 | 0.429 | 2.157 | -1.727 |
| (30, 2, 14) | 2.32 | 10789.6 | 173.0 | 62.4 | 119.2 | 113.5 | 2.741 | 1.071 | +1.670 |
| (-12, 4, 8) | 4.46 | 74634.5 | 1028.6 | 72.6 | 313.4 | 345.7 | 1.598 | 3.165 | -1.567 |
| (27, 1, 2) | 3.11 | 66716.5 | 854.0 | 78.1 | 291.6 | 290.9 | 3.791 | 2.226 | +1.566 |
| (16, 8, 17) | 2.43 | 164.0 | 48.4 | 3.4 | 12.8 | 25.9 | -1.779 | -0.280 | -1.499 |
| (26, 4, 2) | 3.04 | 6856.0 | 44.3 | 154.8 | 94.2 | 112.0 | 0.169 | 1.643 | -1.474 |
| (0, 4, 12) | 4.33 | 80152.9 | 414.4 | 193.4 | 322.3 | 352.3 | 1.643 | 3.074 | -1.431 |
| (21, 3, 2) | 3.79 | 56287.7 | 956.4 | 58.9 | 271.5 | 297.2 | 1.661 | 3.088 | -1.427 |
| (-12, 8, 6) | 3.51 | 50266.4 | 424.9 | 118.3 | 256.7 | 279.2 | 2.703 | 4.102 | -1.398 |
| (-13, 11, 2) | 2.87 | 1897.8 | 55.9 | 33.9 | 49.8 | 65.1 | -0.612 | 0.712 | -1.323 |
| (-13, 1, 8) | 4.90 | 286007.4 | 1880.0 | 152.1 | 610.7 | 649.8 | 7.590 | 8.911 | -1.321 |
| (7, 1, 22) | 2.63 | 3182.2 | 67.1 | 47.4 | 64.2 | 76.7 | -0.016 | 1.283 | -1.298 |
| (-7, 11, 10) | 2.76 | 3379.5 | 27.9 | 121.1 | 65.9 | 80.2 | -0.317 | 0.950 | -1.267 |
| (23, 3, 8) | 3.17 | 4655.6 | 64.4 | 72.3 | 76.7 | 96.2 | -0.522 | 0.733 | -1.255 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `deposited` | 1.51 | 0.9998 | 199.18862156908295 | 1000.0 | 1.000 |
| `recovered_f` | 9.01 | 0.9975 | 199.16575536516672 | 1000.0 | 1.000 |