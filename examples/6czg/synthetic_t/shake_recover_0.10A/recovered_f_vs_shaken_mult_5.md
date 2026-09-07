# Held-Out Log-Likelihood Comparison: `recovered_f` vs `shaken`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.1721` nats/refl
- **Model B NLL**: `0.0272` nats/refl
- **Difference (Gain $\Delta$)**: `+0.1449` nats/refl (`-0.1449` nats NLL reduction)
- **Estimated Log Bayes Factor**: `+130.41` nats
- **Uncertainty**: Bootstrap SE = `0.0596` (95% CI: `[+0.0205, +0.2489]`) | Naive SE = `0.0286` (Ratio: `2.09`x)
- **Win Fraction $P(d_h > 0)$**: `63.1%` (568 wins, 332 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `3.32e-15`
- **Wilcoxon Signed-Rank $p$-value**: `5.41e-13`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `-0.0753` | `-0.0696` | `-0.1449` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `-0.2016` | `+0.0567` | `-0.1449` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`shaken`) | Model B (`recovered_f`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `+0.0074` | `+0.0026` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.2456` | `0.0152` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0735` | `+0.0120` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.1991 | -1.1116 | -0.0875 | 0.3346 | 23.3% | 10/33 |
| 1 | 6.10 | 4.86 | 46 | -0.1378 | 0.4356 | -0.5734 | 0.1343 | 21.7% | 10/36 |
| 2 | 4.83 | 4.21 | 45 | -0.4892 | -0.2124 | -0.2768 | 0.0897 | 26.7% | 12/33 |
| 3 | 4.20 | 3.81 | 46 | 0.1913 | 0.0843 | +0.1070 | 0.1054 | 52.2% | 24/22 |
| 4 | 3.80 | 3.53 | 46 | 0.1253 | 0.2902 | -0.1650 | 0.1263 | 39.1% | 18/28 |
| 5 | 3.53 | 3.32 | 46 | 0.3683 | 0.4276 | -0.0592 | 0.1664 | 50.0% | 23/23 |
| 6 | 3.31 | 3.16 | 44 | -0.0418 | -0.0985 | +0.0567 | 0.0735 | 56.8% | 25/19 |
| 7 | 3.14 | 3.02 | 45 | 0.4607 | 0.4505 | +0.0102 | 0.0893 | 53.3% | 24/21 |
| 8 | 3.01 | 2.89 | 46 | 0.0838 | -0.1021 | +0.1859 | 0.0710 | 69.6% | 32/14 |
| 9 | 2.88 | 2.79 | 43 | 0.1855 | -0.0817 | +0.2672 | 0.0956 | 69.8% | 30/13 |
| 10 | 2.79 | 2.70 | 45 | -0.1219 | -0.4282 | +0.3064 | 0.0719 | 71.1% | 32/13 |
| 11 | 2.70 | 2.62 | 45 | 0.3076 | -0.0193 | +0.3269 | 0.1106 | 73.3% | 33/12 |
| 12 | 2.62 | 2.55 | 44 | 0.0992 | -0.1177 | +0.2168 | 0.0773 | 68.2% | 30/14 |
| 13 | 2.55 | 2.49 | 47 | 0.3528 | 0.0007 | +0.3521 | 0.0951 | 80.9% | 38/9 |
| 14 | 2.49 | 2.43 | 43 | 0.1297 | -0.3932 | +0.5229 | 0.0727 | 93.0% | 40/3 |
| 15 | 2.43 | 2.38 | 43 | 0.6826 | 0.2795 | +0.4031 | 0.1019 | 83.7% | 36/7 |
| 16 | 2.38 | 2.33 | 44 | 0.5173 | 0.1678 | +0.3495 | 0.0933 | 75.0% | 33/11 |
| 17 | 2.32 | 2.28 | 47 | 0.5385 | 0.1472 | +0.3913 | 0.0657 | 83.0% | 39/8 |
| 18 | 2.28 | 2.24 | 48 | 0.6784 | 0.4684 | +0.2100 | 0.0938 | 81.2% | 39/9 |
| 19 | 2.24 | 2.20 | 44 | 0.6299 | 0.2401 | +0.3898 | 0.0812 | 90.9% | 40/4 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (3, 1, 0) | 22.09 | 19735.3 | 907.5 | 21.7 | 190.4 | 175.0 | 13.430 | -0.274 | +13.704 |
| (20, 4, 7) | 3.47 | 471.7 | 177.5 | 2.7 | 80.0 | 32.7 | 3.683 | -1.210 | +4.893 |
| (14, 2, 4) | 5.33 | 184945.9 | 7620.0 | 24.3 | 469.3 | 530.9 | 1.866 | 5.819 | -3.953 |
| (9, 3, 14) | 3.66 | 131464.6 | 5549.5 | 23.7 | 399.0 | 444.9 | 1.648 | 5.386 | -3.739 |
| (-18, 8, 15) | 2.52 | 1690.8 | 288.5 | 5.9 | 80.6 | 48.9 | 3.547 | -0.007 | +3.554 |
| (-13, 1, 8) | 4.90 | 298524.3 | 9400.0 | 31.8 | 595.0 | 646.0 | 2.530 | 5.871 | -3.341 |
| (0, 4, 12) | 4.33 | 73099.5 | 2072.0 | 35.3 | 294.1 | 348.0 | 0.678 | 3.682 | -3.005 |
| (-16, 4, 12) | 3.38 | 73827.9 | 3058.0 | 24.1 | 289.0 | 326.5 | 1.225 | 4.005 | -2.781 |
| (-30, 2, 14) | 2.35 | 1474.5 | 210.0 | 7.0 | 69.4 | 40.9 | 2.533 | -0.212 | +2.745 |
| (-5, 9, 1) | 3.82 | 3872.4 | 457.5 | 8.5 | 111.5 | 81.2 | 2.504 | -0.105 | +2.609 |
| (-18, 4, 10) | 3.43 | 48176.8 | 3216.5 | 15.0 | 260.8 | 283.0 | 2.284 | 4.871 | -2.587 |
| (-7, 1, 22) | 2.64 | 4453.4 | 389.5 | 11.4 | 110.3 | 87.5 | 3.897 | 1.358 | +2.539 |
| (-12, 8, 6) | 3.51 | 50830.6 | 2124.5 | 23.9 | 248.3 | 278.7 | 1.169 | 3.635 | -2.466 |
| (15, 9, 12) | 2.69 | 502.9 | 159.0 | 3.2 | 55.1 | 29.5 | 1.620 | -0.794 | +2.414 |
| (-29, 9, 3) | 2.33 | -225.8 | 139.5 | -1.6 | 31.1 | 10.1 | 3.468 | 1.086 | +2.382 |
| (7, 9, 19) | 2.39 | 1554.3 | 297.0 | 5.2 | 68.9 | 45.6 | 2.336 | 0.040 | +2.297 |
| (-15, 9, 19) | 2.26 | -986.1 | 181.0 | -5.4 | 17.0 | 4.6 | 13.035 | 15.171 | -2.136 |
| (-17, 1, 9) | 3.99 | 30373.9 | 1000.5 | 30.4 | 230.3 | 212.0 | 3.647 | 1.573 | +2.074 |
| (19, 1, 8) | 3.78 | 38637.3 | 2592.0 | 14.9 | 260.6 | 240.9 | 3.690 | 1.655 | +2.034 |
| (18, 2, 18) | 2.66 | 11891.4 | 347.0 | 34.3 | 150.1 | 128.9 | 3.516 | 1.491 | +2.025 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `shaken` | 19.76 | 0.9863 | 5.716430828222444 | 1.845924405170934 | 1.000 |
| `recovered_f` | 11.90 | 0.9934 | 6.864132184791917 | 2.3333425651996667 | 1.000 |