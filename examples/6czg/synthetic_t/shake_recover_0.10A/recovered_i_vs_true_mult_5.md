# Held-Out Log-Likelihood Comparison: `recovered_i` vs `deposited`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `-0.1720` nats/refl
- **Model B NLL**: `0.0271` nats/refl
- **Difference (Gain $\Delta$)**: `-0.1991` nats/refl (`+0.1991` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-179.17` nats
- **Uncertainty**: Bootstrap SE = `0.0283` (95% CI: `[-0.2561, -0.1475]`) | Naive SE = `0.0137` (Ratio: `2.07`x)
- **Win Fraction $P(d_h > 0)$**: `24.0%` (216 wins, 684 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `2.69e-57`
- **Wilcoxon Signed-Rank $p$-value**: `8.11e-57`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.2835` | `-0.0845` | `+0.1991` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.1521` | `+0.0470` | `+0.1991` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`deposited`) | Model B (`recovered_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `+0.0054` | `+0.0027` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `-0.0890` | `0.0157` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0830` | `+0.0114` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.4783 | -1.1097 | -0.3685 | 0.0301 | 4.7% | 2/41 |
| 1 | 6.10 | 4.86 | 46 | 0.0574 | 0.4426 | -0.3851 | 0.0536 | 10.9% | 5/41 |
| 2 | 4.83 | 4.21 | 45 | -0.5641 | -0.2134 | -0.3507 | 0.0395 | 8.9% | 4/41 |
| 3 | 4.20 | 3.81 | 46 | -0.1786 | 0.0869 | -0.2655 | 0.0512 | 21.7% | 10/36 |
| 4 | 3.80 | 3.53 | 46 | -0.1109 | 0.2907 | -0.4017 | 0.0608 | 10.9% | 5/41 |
| 5 | 3.53 | 3.32 | 46 | 0.2014 | 0.4380 | -0.2367 | 0.0631 | 17.4% | 8/38 |
| 6 | 3.31 | 3.16 | 44 | -0.3773 | -0.1058 | -0.2715 | 0.0484 | 11.4% | 5/39 |
| 7 | 3.14 | 3.02 | 45 | 0.2134 | 0.4543 | -0.2409 | 0.0825 | 22.2% | 10/35 |
| 8 | 3.01 | 2.89 | 46 | -0.3379 | -0.0964 | -0.2415 | 0.0593 | 21.7% | 10/36 |
| 9 | 2.88 | 2.79 | 43 | -0.3773 | -0.0830 | -0.2944 | 0.0679 | 14.0% | 6/37 |
| 10 | 2.79 | 2.70 | 45 | -0.6225 | -0.4268 | -0.1957 | 0.0468 | 13.3% | 6/39 |
| 11 | 2.70 | 2.62 | 45 | -0.1486 | -0.0221 | -0.1265 | 0.0523 | 35.6% | 16/29 |
| 12 | 2.62 | 2.55 | 44 | -0.2226 | -0.1240 | -0.0986 | 0.0532 | 34.1% | 15/29 |
| 13 | 2.55 | 2.49 | 47 | -0.1495 | -0.0011 | -0.1484 | 0.0493 | 29.8% | 14/33 |
| 14 | 2.49 | 2.43 | 43 | -0.5134 | -0.3935 | -0.1199 | 0.0400 | 30.2% | 13/30 |
| 15 | 2.43 | 2.38 | 43 | 0.1633 | 0.2715 | -0.1082 | 0.0532 | 27.9% | 12/31 |
| 16 | 2.38 | 2.33 | 44 | 0.0364 | 0.1668 | -0.1304 | 0.0683 | 40.9% | 18/26 |
| 17 | 2.32 | 2.28 | 47 | 0.1084 | 0.1475 | -0.0391 | 0.0443 | 38.3% | 18/29 |
| 18 | 2.28 | 2.24 | 48 | 0.4964 | 0.4649 | +0.0315 | 0.0858 | 39.6% | 19/29 |
| 19 | 2.24 | 2.20 | 44 | 0.2361 | 0.2364 | -0.0003 | 0.0812 | 45.5% | 20/24 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-15, 5, 23) | 2.25 | 913.7 | 371.5 | 2.5 | 59.7 | 54.5 | 6.542 | 3.376 | +3.166 |
| (-14, 6, 23) | 2.22 | 2169.1 | 327.5 | 6.6 | 65.1 | 59.4 | 3.299 | 1.280 | +2.019 |
| (9, 3, 14) | 3.66 | 131464.6 | 5549.5 | 23.7 | 410.6 | 444.9 | 3.440 | 5.380 | -1.940 |
| (-12, 6, 14) | 3.11 | 11836.3 | 542.5 | 21.8 | 116.6 | 139.4 | 0.117 | 1.760 | -1.643 |
| (2, 2, 14) | 4.11 | 121519.8 | 5705.5 | 21.3 | 397.8 | 431.8 | 3.965 | 5.575 | -1.610 |
| (16, 14, 2) | 2.27 | 4477.3 | 849.0 | 5.3 | 85.4 | 76.1 | 2.542 | 0.960 | +1.583 |
| (26, 4, 2) | 3.04 | 6863.8 | 221.5 | 31.0 | 93.5 | 112.4 | 0.105 | 1.673 | -1.568 |
| (1, 7, 2) | 4.96 | 84293.1 | 6850.5 | 12.3 | 317.0 | 354.1 | 1.218 | 2.737 | -1.519 |
| (20, 2, 14) | 2.94 | 7117.5 | 740.5 | 9.6 | 98.2 | 116.7 | 0.262 | 1.755 | -1.492 |
| (-12, 8, 6) | 3.51 | 50830.6 | 2124.5 | 23.9 | 254.9 | 278.9 | 2.245 | 3.668 | -1.423 |
| (-12, 4, 8) | 4.46 | 77267.5 | 5143.0 | 15.0 | 311.3 | 344.6 | 1.187 | 2.521 | -1.333 |
| (16, 8, 17) | 2.43 | 56.3 | 242.0 | 0.2 | 12.7 | 26.8 | -1.063 | 0.261 | -1.324 |
| (-7, 11, 10) | 2.76 | 3473.2 | 139.5 | 24.9 | 65.5 | 81.1 | -0.364 | 0.940 | -1.304 |
| (-16, 4, 12) | 3.38 | 73827.9 | 3058.0 | 24.1 | 319.4 | 327.6 | 5.458 | 4.154 | +1.303 |
| (-13, 11, 2) | 2.87 | 1975.1 | 279.5 | 7.1 | 49.5 | 65.7 | -0.616 | 0.675 | -1.290 |
| (-13, 1, 8) | 4.90 | 298524.3 | 9400.0 | 31.8 | 606.0 | 645.8 | 4.638 | 5.857 | -1.219 |
| (8, 2, 18) | 3.10 | 1705.4 | 533.5 | 3.2 | 58.2 | 73.4 | 0.114 | 1.331 | -1.217 |
| (3, 5, 24) | 2.33 | 304.7 | 203.0 | 1.5 | 26.9 | 35.1 | -0.151 | 1.062 | -1.213 |
| (21, 3, 14) | 2.82 | 3891.0 | 427.0 | 9.1 | 70.1 | 84.8 | -0.107 | 1.103 | -1.210 |
| (-20, 2, 14) | 2.98 | 13465.1 | 447.5 | 30.1 | 140.0 | 128.1 | 1.571 | 0.370 | +1.201 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `deposited` | 7.06 | 0.9953 | 7.176088803437241 | 2.3277404455528554 | 1.000 |
| `recovered_i` | 11.83 | 0.9935 | 6.877894534466397 | 2.337291788101025 | 1.000 |