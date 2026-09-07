# Held-Out Log-Likelihood Comparison: `rerefined_deposited_i` vs `deposited_6czg`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.5269` nats/refl
- **Model B NLL**: `0.5155` nats/refl
- **Difference (Gain $\Delta$)**: `+0.0114` nats/refl (`-0.0114` nats NLL reduction)
- **Estimated Log Bayes Factor**: `+10.25` nats
- **Uncertainty**: Bootstrap SE = `0.0185` (95% CI: `[-0.0274, +0.0456]`) | Naive SE = `0.0112` (Ratio: `1.64`x)
- **Win Fraction $P(d_h > 0)$**: `50.6%` (455 wins, 445 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `7.64e-01`
- **Wilcoxon Signed-Rank $p$-value**: `7.56e-01`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `-0.0045` | `-0.0069` | `-0.0114` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `-0.0169` | `+0.0055` | `-0.0114` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`deposited_6czg`) | Model B (`rerefined_deposited_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0014` | `-0.0057` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.5231` | `0.4771` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `+0.0038` | `+0.0384` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -0.8390 | -0.6518 | -0.1872 | 0.0491 | 14.0% | 6/37 |
| 1 | 6.10 | 4.86 | 46 | 0.4423 | 0.5578 | -0.1155 | 0.0570 | 26.1% | 12/34 |
| 2 | 4.83 | 4.21 | 45 | 0.0393 | 0.1311 | -0.0918 | 0.0495 | 17.8% | 8/37 |
| 3 | 4.20 | 3.81 | 46 | 0.4607 | 0.4895 | -0.0288 | 0.0246 | 30.4% | 14/32 |
| 4 | 3.80 | 3.53 | 46 | 0.7685 | 0.7417 | +0.0268 | 0.0682 | 26.1% | 12/34 |
| 5 | 3.53 | 3.32 | 46 | 0.9284 | 0.9456 | -0.0172 | 0.0463 | 32.6% | 15/31 |
| 6 | 3.31 | 3.16 | 44 | 0.5496 | 0.4538 | +0.0958 | 0.0703 | 56.8% | 25/19 |
| 7 | 3.14 | 3.02 | 45 | 1.4329 | 1.2456 | +0.1873 | 0.0625 | 68.9% | 31/14 |
| 8 | 3.01 | 2.89 | 46 | 0.4290 | 0.3844 | +0.0446 | 0.0490 | 45.7% | 21/25 |
| 9 | 2.88 | 2.79 | 43 | 0.6684 | 0.6411 | +0.0274 | 0.0360 | 67.4% | 29/14 |
| 10 | 2.79 | 2.70 | 45 | 0.2783 | 0.2178 | +0.0605 | 0.0491 | 53.3% | 24/21 |
| 11 | 2.70 | 2.62 | 45 | 0.5405 | 0.5430 | -0.0025 | 0.0254 | 55.6% | 25/20 |
| 12 | 2.62 | 2.55 | 44 | 0.4655 | 0.3494 | +0.1161 | 0.0637 | 65.9% | 29/15 |
| 13 | 2.55 | 2.49 | 47 | 0.5095 | 0.5134 | -0.0039 | 0.0251 | 57.4% | 27/20 |
| 14 | 2.49 | 2.43 | 43 | 0.2470 | 0.2453 | +0.0017 | 0.0378 | 58.1% | 25/18 |
| 15 | 2.43 | 2.38 | 43 | 0.7808 | 0.7117 | +0.0691 | 0.0276 | 81.4% | 35/8 |
| 16 | 2.38 | 2.33 | 44 | 0.7066 | 0.8002 | -0.0936 | 0.0521 | 54.5% | 24/20 |
| 17 | 2.32 | 2.28 | 47 | 0.9569 | 0.9154 | +0.0416 | 0.0446 | 63.8% | 30/17 |
| 18 | 2.28 | 2.24 | 48 | 0.4733 | 0.4099 | +0.0634 | 0.0489 | 70.8% | 34/14 |
| 19 | 2.24 | 2.20 | 44 | 0.6249 | 0.5952 | +0.0297 | 0.0364 | 65.9% | 29/15 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-19, 7, 4) | 3.27 | 65816.4 | 797.1 | 82.6 | 179.2 | 205.8 | 6.660 | 4.164 | +2.496 |
| (28, 6, 6) | 2.59 | 1729.0 | 46.5 | 37.2 | 100.1 | 78.7 | 3.555 | 1.540 | +2.015 |
| (-21, 3, 2) | 3.80 | 50411.9 | 629.1 | 80.1 | 123.8 | 138.5 | 6.345 | 4.517 | +1.828 |
| (-19, 9, 8) | 2.75 | 14480.0 | 198.7 | 72.9 | 51.1 | 65.0 | 6.495 | 4.803 | +1.692 |
| (28, 0, 0) | 3.03 | 82866.5 | 1460.6 | 56.7 | 175.4 | 197.8 | 9.200 | 7.526 | +1.673 |
| (14, 2, 4) | 5.33 | 126810.0 | 1524.0 | 83.2 | 436.4 | 508.9 | 1.948 | 3.561 | -1.613 |
| (-8, 12, 14) | 2.36 | 338.5 | 29.4 | 11.5 | 48.2 | 58.6 | 1.189 | 2.789 | -1.599 |
| (37, 1, 0) | 2.29 | 6425.9 | 107.4 | 59.8 | 55.9 | 43.7 | 3.383 | 4.943 | -1.560 |
| (24, 0, 10) | 3.02 | 5069.1 | 134.8 | 37.6 | 182.5 | 160.4 | 4.100 | 2.582 | +1.518 |
| (14, 10, 12) | 2.59 | 140.1 | 27.1 | 5.2 | 62.8 | 48.3 | 2.200 | 0.798 | +1.402 |
| (-7, 7, 2) | 4.61 | 66075.7 | 781.0 | 84.6 | 175.6 | 197.0 | 3.794 | 2.477 | +1.317 |
| (-10, 0, 12) | 4.31 | 4234.8 | 122.6 | 34.5 | 213.2 | 206.0 | 3.337 | 2.131 | +1.206 |
| (-23, 3, 8) | 3.20 | 22889.6 | 300.5 | 76.2 | 100.9 | 121.6 | 3.029 | 1.855 | +1.174 |
| (12, 10, 2) | 3.14 | 37671.5 | 459.8 | 81.9 | 156.3 | 177.8 | 3.868 | 2.698 | +1.170 |
| (-23, 1, 4) | 3.58 | 21340.5 | 304.5 | 70.1 | 91.8 | 119.6 | 2.374 | 1.231 | +1.143 |
| (-13, 1, 8) | 4.90 | 159005.0 | 1880.0 | 84.6 | 534.8 | 580.2 | 3.701 | 4.821 | -1.120 |
| (20, 8, 0) | 3.06 | 77200.7 | 929.6 | 83.0 | 185.5 | 194.2 | 11.719 | 10.620 | +1.099 |
| (2, 14, 12) | 2.24 | 160.0 | 25.0 | 6.4 | 33.2 | 18.8 | 0.519 | -0.551 | +1.071 |
| (4, 12, 13) | 2.46 | 6027.1 | 91.4 | 65.9 | 53.1 | 43.6 | 2.319 | 3.375 | -1.055 |
| (-33, 3, 9) | 2.36 | 4035.6 | 78.1 | 51.7 | 36.5 | 29.7 | 2.601 | 3.622 | -1.021 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `deposited_6czg` | 42.89 | 0.8845 | 4.829139997587939 | 6.78336178322578 | 1.000 |
| `rerefined_deposited_i` | 42.98 | 0.8876 | 5.033316793443273 | 6.792880528577561 | 1.000 |