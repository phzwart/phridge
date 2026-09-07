# Held-Out Log-Likelihood Comparison: `recovered_f` vs `shaken`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `1.8739` nats/refl
- **Model B NLL**: `1.8455` nats/refl
- **Difference (Gain $\Delta$)**: `+0.0284` nats/refl (`-0.0284` nats NLL reduction)
- **Estimated Log Bayes Factor**: `+25.56` nats
- **Uncertainty**: Bootstrap SE = `0.0082` (95% CI: `[+0.0132, +0.0458]`) | Naive SE = `0.0071` (Ratio: `1.16`x)
- **Win Fraction $P(d_h > 0)$**: `55.8%` (502 wins, 398 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `5.89e-04`
- **Wilcoxon Signed-Rank $p$-value**: `5.04e-04`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `-0.0283` | `-0.0001` | `-0.0284` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `-0.0296` | `+0.0012` | `-0.0284` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`shaken`) | Model B (`recovered_f`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0273` | `-0.0274` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `1.9998` | `1.9824` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1259` | `-0.1369` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -0.3625 | -0.3752 | +0.0127 | 0.0456 | 53.5% | 23/20 |
| 1 | 6.10 | 4.86 | 46 | 1.5566 | 1.5142 | +0.0424 | 0.0336 | 56.5% | 26/20 |
| 2 | 4.83 | 4.21 | 45 | 1.0070 | 0.9570 | +0.0500 | 0.0311 | 60.0% | 27/18 |
| 3 | 4.20 | 3.81 | 46 | 1.8132 | 1.8121 | +0.0012 | 0.0334 | 52.2% | 24/22 |
| 4 | 3.80 | 3.53 | 46 | 1.7713 | 1.7924 | -0.0212 | 0.0293 | 45.7% | 21/25 |
| 5 | 3.53 | 3.32 | 46 | 1.9361 | 1.9192 | +0.0169 | 0.0283 | 43.5% | 20/26 |
| 6 | 3.31 | 3.16 | 44 | 1.5288 | 1.5062 | +0.0226 | 0.0203 | 52.3% | 23/21 |
| 7 | 3.14 | 3.02 | 45 | 2.2596 | 2.2610 | -0.0014 | 0.0266 | 55.6% | 25/20 |
| 8 | 3.01 | 2.89 | 46 | 1.8125 | 1.6949 | +0.1176 | 0.0447 | 67.4% | 31/15 |
| 9 | 2.88 | 2.79 | 43 | 1.9492 | 1.8932 | +0.0560 | 0.0367 | 62.8% | 27/16 |
| 10 | 2.79 | 2.70 | 45 | 1.8767 | 1.8092 | +0.0675 | 0.0391 | 66.7% | 30/15 |
| 11 | 2.70 | 2.62 | 45 | 1.9291 | 1.8606 | +0.0685 | 0.0343 | 64.4% | 29/16 |
| 12 | 2.62 | 2.55 | 44 | 2.0614 | 1.9980 | +0.0635 | 0.0369 | 56.8% | 25/19 |
| 13 | 2.55 | 2.49 | 47 | 1.9414 | 1.8930 | +0.0484 | 0.0218 | 66.0% | 31/16 |
| 14 | 2.49 | 2.43 | 43 | 2.2774 | 2.2267 | +0.0507 | 0.0262 | 65.1% | 28/15 |
| 15 | 2.43 | 2.38 | 43 | 2.3573 | 2.3517 | +0.0056 | 0.0377 | 58.1% | 25/18 |
| 16 | 2.38 | 2.33 | 44 | 2.2771 | 2.3124 | -0.0353 | 0.0232 | 43.2% | 19/25 |
| 17 | 2.32 | 2.28 | 47 | 2.3002 | 2.3083 | -0.0081 | 0.0153 | 46.8% | 22/25 |
| 18 | 2.28 | 2.24 | 48 | 2.5202 | 2.5016 | +0.0186 | 0.0212 | 47.9% | 23/25 |
| 19 | 2.24 | 2.20 | 44 | 2.5748 | 2.5833 | -0.0084 | 0.0194 | 52.3% | 23/21 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (6, 2, 2) | 10.31 | 10499.1 | 1030.0 | 10.2 | 102.1 | 116.6 | -2.272 | -1.053 | -1.219 |
| (-13, 7, 14) | 2.92 | 15310.3 | 4250.0 | 3.6 | 75.2 | 95.7 | 2.706 | 1.598 | +1.107 |
| (18, 2, 18) | 2.66 | 6044.3 | 6940.0 | 0.9 | 135.1 | 107.4 | 2.843 | 1.907 | +0.936 |
| (28, 6, 6) | 2.59 | 17610.3 | 4650.0 | 3.8 | 95.6 | 111.3 | 2.756 | 1.885 | +0.870 |
| (13, 11, 2) | 2.87 | 3646.0 | 1960.0 | 1.9 | 26.8 | 67.5 | 1.052 | 0.202 | +0.850 |
| (-13, 5, 16) | 2.96 | 30241.3 | 5360.0 | 5.6 | 77.7 | 99.0 | 6.074 | 5.235 | +0.839 |
| (20, 4, 7) | 3.47 | 5775.2 | 3550.0 | 1.6 | 70.7 | 32.8 | 0.281 | 1.065 | -0.783 |
| (-12, 8, 19) | 2.41 | 18115.2 | 7820.0 | 2.3 | 58.6 | 89.7 | 3.683 | 2.911 | +0.772 |
| (-16, 8, 13) | 2.74 | 12645.1 | 5550.0 | 2.3 | 29.7 | 62.6 | 3.076 | 2.324 | +0.752 |
| (34, 0, 7) | 2.39 | 15835.7 | 5250.0 | 3.0 | 55.2 | 26.9 | 4.098 | 4.836 | -0.737 |
| (-13, 1, 20) | 2.72 | -4721.7 | 5680.0 | -0.8 | 71.4 | 89.1 | 2.586 | 3.316 | -0.730 |
| (-3, 5, 11) | 4.26 | 6656.7 | 2680.0 | 2.5 | 38.8 | 55.5 | 0.594 | -0.133 | +0.727 |
| (15, 7, 18) | 2.47 | 19932.4 | 6800.0 | 2.9 | 90.7 | 110.0 | 3.278 | 2.559 | +0.719 |
| (-13, 9, 11) | 2.87 | 14178.0 | 4470.0 | 3.2 | 43.8 | 65.2 | 3.713 | 3.013 | +0.700 |
| (24, 4, 11) | 2.79 | 6477.3 | 3380.0 | 1.9 | 30.9 | 54.6 | 1.800 | 1.103 | +0.697 |
| (-22, 2, 17) | 2.59 | -5348.6 | 4480.0 | -1.2 | 39.8 | 62.0 | 2.378 | 3.074 | -0.696 |
| (19, 9, 0) | 2.95 | -8204.7 | 10170.0 | -0.8 | 127.9 | 143.9 | 3.767 | 4.454 | -0.687 |
| (-11, 5, 4) | 4.93 | 17117.3 | 8620.0 | 2.0 | 76.1 | 111.0 | 1.422 | 0.737 | +0.685 |
| (13, 11, 0) | 2.88 | 3394.9 | 2530.0 | 1.3 | 46.3 | 81.7 | 0.471 | 1.142 | -0.671 |
| (0, 2, 15) | 3.87 | -3337.5 | 9190.0 | -0.4 | 97.7 | 118.3 | 1.907 | 2.567 | -0.660 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `shaken` | 123.97 | 0.5611 | 6.658136633124035 | 1.2936131050297361 | 1.000 |
| `recovered_f` | 121.12 | 0.5774 | 7.097634274753213 | 1.4165830643757904 | 1.000 |