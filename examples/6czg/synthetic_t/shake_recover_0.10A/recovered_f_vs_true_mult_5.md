# Held-Out Log-Likelihood Comparison: `recovered_f` vs `deposited`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `-0.1720` nats/refl
- **Model B NLL**: `0.0272` nats/refl
- **Difference (Gain $\Delta$)**: `-0.1992` nats/refl (`+0.1992` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-179.28` nats
- **Uncertainty**: Bootstrap SE = `0.0279` (95% CI: `[-0.2549, -0.1484]`) | Naive SE = `0.0137` (Ratio: `2.03`x)
- **Win Fraction $P(d_h > 0)$**: `23.4%` (211 wins, 689 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `7.77e-60`
- **Wilcoxon Signed-Rank $p$-value**: `5.54e-57`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.2834` | `-0.0842` | `+0.1992` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.1522` | `+0.0470` | `+0.1992` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`deposited`) | Model B (`recovered_f`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `+0.0054` | `+0.0026` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `-0.0890` | `0.0152` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0830` | `+0.0120` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.4783 | -1.1116 | -0.3667 | 0.0306 | 4.7% | 2/41 |
| 1 | 6.10 | 4.86 | 46 | 0.0574 | 0.4356 | -0.3781 | 0.0535 | 10.9% | 5/41 |
| 2 | 4.83 | 4.21 | 45 | -0.5641 | -0.2124 | -0.3517 | 0.0380 | 8.9% | 4/41 |
| 3 | 4.20 | 3.81 | 46 | -0.1786 | 0.0843 | -0.2629 | 0.0518 | 19.6% | 9/37 |
| 4 | 3.80 | 3.53 | 46 | -0.1109 | 0.2902 | -0.4012 | 0.0608 | 6.5% | 3/43 |
| 5 | 3.53 | 3.32 | 46 | 0.2014 | 0.4276 | -0.2262 | 0.0657 | 17.4% | 8/38 |
| 6 | 3.31 | 3.16 | 44 | -0.3773 | -0.0985 | -0.2789 | 0.0477 | 11.4% | 5/39 |
| 7 | 3.14 | 3.02 | 45 | 0.2134 | 0.4505 | -0.2371 | 0.0818 | 22.2% | 10/35 |
| 8 | 3.01 | 2.89 | 46 | -0.3379 | -0.1021 | -0.2359 | 0.0593 | 19.6% | 9/37 |
| 9 | 2.88 | 2.79 | 43 | -0.3773 | -0.0817 | -0.2956 | 0.0684 | 14.0% | 6/37 |
| 10 | 2.79 | 2.70 | 45 | -0.6225 | -0.4282 | -0.1942 | 0.0468 | 13.3% | 6/39 |
| 11 | 2.70 | 2.62 | 45 | -0.1486 | -0.0193 | -0.1293 | 0.0527 | 33.3% | 15/30 |
| 12 | 2.62 | 2.55 | 44 | -0.2226 | -0.1177 | -0.1049 | 0.0533 | 31.8% | 14/30 |
| 13 | 2.55 | 2.49 | 47 | -0.1495 | 0.0007 | -0.1502 | 0.0501 | 31.9% | 15/32 |
| 14 | 2.49 | 2.43 | 43 | -0.5134 | -0.3932 | -0.1201 | 0.0402 | 30.2% | 13/30 |
| 15 | 2.43 | 2.38 | 43 | 0.1633 | 0.2795 | -0.1162 | 0.0550 | 32.6% | 14/29 |
| 16 | 2.38 | 2.33 | 44 | 0.0364 | 0.1678 | -0.1314 | 0.0670 | 40.9% | 18/26 |
| 17 | 2.32 | 2.28 | 47 | 0.1084 | 0.1472 | -0.0388 | 0.0443 | 36.2% | 17/30 |
| 18 | 2.28 | 2.24 | 48 | 0.4964 | 0.4684 | +0.0281 | 0.0856 | 39.6% | 19/29 |
| 19 | 2.24 | 2.20 | 44 | 0.2361 | 0.2401 | -0.0040 | 0.0829 | 43.2% | 19/25 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-15, 5, 23) | 2.25 | 913.7 | 371.5 | 2.5 | 59.7 | 54.5 | 6.542 | 3.371 | +3.171 |
| (-14, 6, 23) | 2.22 | 2169.1 | 327.5 | 6.6 | 65.1 | 59.5 | 3.299 | 1.284 | +2.015 |
| (9, 3, 14) | 3.66 | 131464.6 | 5549.5 | 23.7 | 410.6 | 444.9 | 3.440 | 5.386 | -1.946 |
| (2, 2, 14) | 4.11 | 121519.8 | 5705.5 | 21.3 | 397.8 | 432.6 | 3.965 | 5.671 | -1.706 |
| (16, 14, 2) | 2.27 | 4477.3 | 849.0 | 5.3 | 85.4 | 75.9 | 2.542 | 0.943 | +1.599 |
| (-12, 6, 14) | 3.11 | 11836.3 | 542.5 | 21.8 | 116.6 | 138.9 | 0.117 | 1.703 | -1.586 |
| (26, 4, 2) | 3.04 | 6863.8 | 221.5 | 31.0 | 93.5 | 112.0 | 0.105 | 1.614 | -1.510 |
| (1, 7, 2) | 4.96 | 84293.1 | 6850.5 | 12.3 | 317.0 | 353.6 | 1.218 | 2.699 | -1.481 |
| (20, 2, 14) | 2.94 | 7117.5 | 740.5 | 9.6 | 98.2 | 116.5 | 0.262 | 1.735 | -1.472 |
| (-16, 4, 12) | 3.38 | 73827.9 | 3058.0 | 24.1 | 319.4 | 326.5 | 5.458 | 4.005 | +1.452 |
| (16, 8, 17) | 2.43 | 56.3 | 242.0 | 0.2 | 12.7 | 27.4 | -1.063 | 0.350 | -1.413 |
| (-12, 8, 6) | 3.51 | 50830.6 | 2124.5 | 23.9 | 254.9 | 278.7 | 2.245 | 3.635 | -1.390 |
| (-7, 11, 10) | 2.76 | 3473.2 | 139.5 | 24.9 | 65.5 | 81.0 | -0.364 | 0.931 | -1.295 |
| (-13, 11, 2) | 2.87 | 1975.1 | 279.5 | 7.1 | 49.5 | 65.6 | -0.616 | 0.660 | -1.276 |
| (8, 2, 18) | 3.10 | 1705.4 | 533.5 | 3.2 | 58.2 | 73.8 | 0.114 | 1.373 | -1.258 |
| (21, 3, 2) | 3.79 | 52173.5 | 4782.0 | 10.9 | 269.5 | 296.2 | 2.152 | 3.406 | -1.254 |
| (-12, 4, 8) | 4.46 | 77267.5 | 5143.0 | 15.0 | 311.3 | 343.4 | 1.187 | 2.439 | -1.252 |
| (-13, 1, 8) | 4.90 | 298524.3 | 9400.0 | 31.8 | 606.0 | 646.0 | 4.638 | 5.871 | -1.233 |
| (-20, 2, 14) | 2.98 | 13465.1 | 447.5 | 30.1 | 140.0 | 127.4 | 1.571 | 0.346 | +1.224 |
| (-23, 7, 7) | 2.82 | 115.3 | 253.5 | 0.5 | 32.0 | 20.1 | 0.180 | -1.017 | +1.197 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `deposited` | 7.06 | 0.9953 | 7.176088803437241 | 2.3277404455528554 | 1.000 |
| `recovered_f` | 11.90 | 0.9934 | 6.864132184791917 | 2.3333425651996667 | 1.000 |