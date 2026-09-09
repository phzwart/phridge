# Held-Out Log-Likelihood Comparison: `recovered_i` vs `deposited`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `-0.3063` nats/refl
- **Model B NLL**: `-0.0900` nats/refl
- **Difference (Gain $\Delta$)**: `-0.2163` nats/refl (`+0.2163` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-194.70` nats
- **Uncertainty**: Bootstrap SE = `0.0247` (95% CI: `[-0.2681, -0.1721]`) | Naive SE = `0.0141` (Ratio: `1.76`x)
- **Win Fraction $P(d_h > 0)$**: `21.8%` (196 wins, 704 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `7.38e-68`
- **Wilcoxon Signed-Rank $p$-value**: `2.90e-63`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.2994` | `-0.0831` | `+0.2163` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.1663` | `+0.0500` | `+0.2163` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`deposited`) | Model B (`recovered_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0046` | `-0.0070` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `-0.2196` | `-0.0895` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0868` | `-0.0005` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.4352 | -1.1491 | -0.2860 | 0.0298 | 4.7% | 2/41 |
| 1 | 6.10 | 4.86 | 46 | 0.0534 | 0.4751 | -0.4217 | 0.0638 | 4.3% | 2/44 |
| 2 | 4.83 | 4.21 | 45 | -0.6110 | -0.2684 | -0.3426 | 0.0465 | 2.2% | 1/44 |
| 3 | 4.20 | 3.81 | 46 | -0.1932 | 0.0853 | -0.2785 | 0.0602 | 15.2% | 7/39 |
| 4 | 3.80 | 3.53 | 46 | -0.0816 | 0.3228 | -0.4044 | 0.0722 | 8.7% | 4/42 |
| 5 | 3.53 | 3.32 | 46 | 0.1297 | 0.3676 | -0.2379 | 0.0600 | 26.1% | 12/34 |
| 6 | 3.31 | 3.16 | 44 | -0.3544 | -0.0825 | -0.2719 | 0.0558 | 18.2% | 8/36 |
| 7 | 3.14 | 3.02 | 45 | 0.2389 | 0.4434 | -0.2045 | 0.0917 | 20.0% | 9/36 |
| 8 | 3.01 | 2.89 | 46 | -0.4545 | -0.1577 | -0.2968 | 0.0541 | 15.2% | 7/39 |
| 9 | 2.88 | 2.79 | 43 | -0.4664 | -0.1416 | -0.3248 | 0.0543 | 11.6% | 5/38 |
| 10 | 2.79 | 2.70 | 45 | -0.6087 | -0.4319 | -0.1768 | 0.0434 | 17.8% | 8/37 |
| 11 | 2.70 | 2.62 | 45 | -0.3627 | -0.1663 | -0.1964 | 0.0459 | 22.2% | 10/35 |
| 12 | 2.62 | 2.55 | 44 | -0.4188 | -0.2991 | -0.1197 | 0.0537 | 38.6% | 17/27 |
| 13 | 2.55 | 2.49 | 47 | -0.3601 | -0.1600 | -0.2001 | 0.0464 | 21.3% | 10/37 |
| 14 | 2.49 | 2.43 | 43 | -0.6115 | -0.5165 | -0.0950 | 0.0490 | 44.2% | 19/24 |
| 15 | 2.43 | 2.38 | 43 | 0.0250 | 0.1251 | -0.1000 | 0.0664 | 37.2% | 16/27 |
| 16 | 2.38 | 2.33 | 44 | -0.2699 | -0.1727 | -0.0972 | 0.0931 | 29.5% | 13/31 |
| 17 | 2.32 | 2.28 | 47 | -0.1212 | -0.0663 | -0.0548 | 0.0669 | 25.5% | 12/35 |
| 18 | 2.28 | 2.24 | 48 | -0.1720 | -0.0634 | -0.1086 | 0.0618 | 35.4% | 17/31 |
| 19 | 2.24 | 2.20 | 44 | -0.1449 | -0.0420 | -0.1029 | 0.0730 | 38.6% | 17/27 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (27, 7, 12) | 2.34 | 11715.6 | 148.6 | 78.8 | 127.0 | 117.7 | 3.749 | 1.027 | +2.722 |
| (2, 2, 14) | 4.11 | 124622.5 | 1711.6 | 72.8 | 400.3 | 435.0 | 3.747 | 5.977 | -2.230 |
| (9, 3, 14) | 3.66 | 132472.5 | 1664.9 | 79.6 | 413.2 | 445.0 | 3.637 | 5.731 | -2.094 |
| (27, 1, 2) | 3.11 | 63525.2 | 1281.0 | 49.6 | 291.4 | 290.4 | 4.781 | 2.749 | +2.031 |
| (1, 7, 2) | 4.96 | 79170.5 | 2055.1 | 38.5 | 319.0 | 353.8 | 1.908 | 3.780 | -1.872 |
| (14, 2, 4) | 5.33 | 185493.8 | 2286.0 | 81.1 | 496.4 | 535.1 | 5.002 | 6.773 | -1.771 |
| (-12, 6, 14) | 3.11 | 10697.2 | 162.8 | 65.7 | 117.3 | 138.1 | 0.480 | 2.208 | -1.727 |
| (20, 2, 14) | 2.94 | 7592.6 | 222.2 | 34.2 | 98.8 | 119.0 | 0.092 | 1.818 | -1.726 |
| (-13, 1, 8) | 4.90 | 286163.9 | 2820.0 | 101.5 | 610.3 | 649.7 | 6.878 | 8.473 | -1.595 |
| (26, 4, 2) | 3.04 | 6957.2 | 66.4 | 104.7 | 94.1 | 113.3 | 0.125 | 1.692 | -1.566 |
| (-12, 4, 8) | 4.46 | 75875.1 | 1542.9 | 49.2 | 313.1 | 345.6 | 1.398 | 2.957 | -1.559 |
| (21, 3, 2) | 3.79 | 57625.2 | 1434.6 | 40.2 | 271.3 | 298.5 | 1.383 | 2.918 | -1.535 |
| (16, 8, 17) | 2.43 | 150.4 | 72.6 | 2.1 | 12.8 | 25.4 | -1.749 | -0.303 | -1.446 |
| (-12, 8, 6) | 3.51 | 50749.5 | 637.3 | 79.6 | 256.5 | 278.8 | 2.377 | 3.778 | -1.400 |
| (0, 4, 12) | 4.33 | 80890.8 | 621.6 | 130.1 | 322.0 | 351.6 | 1.508 | 2.903 | -1.395 |
| (30, 2, 14) | 2.32 | 10907.6 | 259.5 | 42.0 | 119.1 | 113.3 | 2.404 | 1.033 | +1.371 |
| (-18, 4, 10) | 3.43 | 53950.7 | 964.9 | 55.9 | 264.3 | 285.9 | 2.613 | 3.952 | -1.339 |
| (16, 14, 2) | 2.27 | 5454.2 | 254.7 | 21.4 | 85.9 | 77.1 | 1.806 | 0.514 | +1.293 |
| (-27, 1, 2) | 3.12 | 25268.3 | 600.2 | 42.1 | 183.0 | 200.6 | 1.900 | 3.185 | -1.285 |
| (10, 12, 14) | 2.32 | 9087.6 | 127.8 | 71.1 | 108.7 | 102.3 | 2.051 | 0.783 | +1.268 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `deposited` | 2.09 | 0.9997 | 11.120828743810849 | 15.070314747558738 | 1.000 |
| `recovered_i` | 8.94 | 0.9976 | 7.144230624326415 | 7.062454964122224 | 1.000 |