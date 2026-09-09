# Held-Out Log-Likelihood Comparison: `rerefined_i` vs `deposited_1ee2`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 1994 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 997
- **Working Reflections (|Work|)**: 101881
- **Model A NLL**: `0.1112` nats/refl
- **Model B NLL**: `0.2259` nats/refl
- **Difference (Gain $\Delta$)**: `-0.1147` nats/refl (`+0.1147` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-228.70` nats
- **Uncertainty**: Bootstrap SE = `0.0107` (95% CI: `[-0.1380, -0.0945]`) | Naive SE = `0.0066` (Ratio: `1.62`x)
- **Win Fraction $P(d_h > 0)$**: `24.5%` (488 wins, 1506 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `2.71e-120`
- **Wilcoxon Signed-Rank $p$-value**: `3.60e-96`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.1355` | `-0.0208` | `+0.1147` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.1007` | `+0.0140` | `+0.1147` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`deposited_1ee2`) | Model B (`rerefined_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0050` | `0.0050` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `+0.0038` | `+0.0036` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.0911` | `0.1410` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `+0.0201` | `+0.0849` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 15.05 | 4.22 | 96 | 0.2118 | 0.3540 | -0.1422 | 0.0268 | 20.8% | 20/76 |
| 1 | 4.20 | 3.36 | 98 | 0.2149 | 0.3060 | -0.0911 | 0.0266 | 24.5% | 24/74 |
| 2 | 3.34 | 2.92 | 105 | 0.3984 | 0.5366 | -0.1382 | 0.0365 | 22.9% | 24/81 |
| 3 | 2.92 | 2.65 | 94 | -0.0361 | 0.1446 | -0.1807 | 0.0236 | 13.8% | 13/81 |
| 4 | 2.65 | 2.46 | 103 | 0.0026 | 0.2003 | -0.1976 | 0.0300 | 16.5% | 17/86 |
| 5 | 2.46 | 2.32 | 100 | -0.0320 | 0.1102 | -0.1422 | 0.0214 | 18.0% | 18/82 |
| 6 | 2.31 | 2.20 | 101 | 0.0268 | 0.1930 | -0.1662 | 0.0238 | 17.8% | 18/83 |
| 7 | 2.20 | 2.10 | 99 | -0.0210 | 0.1350 | -0.1560 | 0.0189 | 10.1% | 10/89 |
| 8 | 2.10 | 2.02 | 97 | 0.1481 | 0.2424 | -0.0943 | 0.0303 | 21.6% | 21/76 |
| 9 | 2.02 | 1.95 | 104 | -0.1039 | 0.0668 | -0.1707 | 0.0191 | 16.3% | 17/87 |
| 10 | 1.95 | 1.89 | 98 | 0.0828 | 0.1815 | -0.0987 | 0.0255 | 21.4% | 21/77 |
| 11 | 1.89 | 1.83 | 105 | -0.0388 | 0.0552 | -0.0940 | 0.0241 | 26.7% | 28/77 |
| 12 | 1.83 | 1.78 | 103 | 0.0135 | 0.1028 | -0.0894 | 0.0327 | 26.2% | 27/76 |
| 13 | 1.78 | 1.74 | 91 | 0.1114 | 0.2049 | -0.0935 | 0.0302 | 25.3% | 23/68 |
| 14 | 1.74 | 1.70 | 96 | 0.1838 | 0.2414 | -0.0576 | 0.0292 | 29.2% | 28/68 |
| 15 | 1.70 | 1.66 | 110 | 0.1815 | 0.2209 | -0.0394 | 0.0296 | 42.7% | 47/63 |
| 16 | 1.66 | 1.63 | 96 | 0.3789 | 0.4485 | -0.0696 | 0.0443 | 33.3% | 32/64 |
| 17 | 1.63 | 1.60 | 108 | 0.0904 | 0.1526 | -0.0622 | 0.0280 | 32.4% | 35/73 |
| 18 | 1.60 | 1.57 | 88 | 0.2442 | 0.3170 | -0.0728 | 0.0451 | 38.6% | 34/54 |
| 19 | 1.57 | 1.54 | 102 | 0.1941 | 0.3313 | -0.1371 | 0.0346 | 30.4% | 31/71 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-22, 4, 8) | 2.48 | 2723.6 | 193.6 | 14.1 | 68.8 | 79.5 | 1.735 | 3.568 | -1.833 |
| (-17, 18, 10) | 2.52 | 670.8 | 50.1 | 13.4 | 35.1 | 45.8 | -0.090 | 1.459 | -1.549 |
| (-27, 29, 23) | 1.54 | 253.1 | 16.2 | 15.6 | 14.3 | 11.2 | 0.571 | 2.074 | -1.503 |
| (-17, 23, 25) | 2.05 | 955.2 | 99.6 | 9.6 | 45.0 | 41.7 | 2.643 | 1.179 | +1.463 |
| (-26, 1, 35) | 1.82 | 473.2 | 74.2 | 6.4 | 31.8 | 29.1 | 2.500 | 1.089 | +1.411 |
| (15, 26, 28) | 1.72 | 54.7 | 22.6 | 2.4 | 12.7 | 16.3 | 0.378 | 1.777 | -1.399 |
| (13, 39, 12) | 1.63 | 640.0 | 36.7 | 17.4 | 23.8 | 20.5 | 0.748 | 2.065 | -1.318 |
| (-15, 9, 20) | 2.94 | 3417.6 | 376.4 | 9.1 | 74.5 | 86.2 | 0.882 | 2.156 | -1.273 |
| (31, 11, 4) | 1.65 | 146.3 | 9.7 | 15.0 | 10.6 | 6.8 | -0.196 | 1.030 | -1.226 |
| (5, 13, 46) | 1.77 | 72.1 | 18.1 | 4.0 | 16.5 | 13.7 | 1.181 | -0.033 | +1.214 |
| (10, 43, 8) | 1.59 | 347.8 | 37.3 | 9.3 | 16.5 | 13.8 | 0.771 | 1.967 | -1.196 |
| (-34, 1, 26) | 1.57 | 998.8 | 16.4 | 60.8 | 32.5 | 28.9 | 0.832 | 2.001 | -1.168 |
| (7, 1, 22) | 3.33 | 3389.1 | 236.1 | 14.4 | 73.0 | 85.7 | 0.424 | 1.585 | -1.161 |
| (-1, 33, 25) | 1.90 | 233.9 | 48.4 | 4.8 | 20.3 | 25.1 | 0.051 | 1.212 | -1.161 |
| (33, 12, 4) | 1.55 | 284.0 | 10.8 | 26.4 | 15.3 | 12.9 | 0.609 | 1.740 | -1.130 |
| (17, 37, 7) | 1.64 | 21.5 | 20.7 | 1.0 | 10.7 | 13.1 | 0.996 | 2.124 | -1.128 |
| (19, 32, 1) | 1.77 | 1080.4 | 159.9 | 6.8 | 30.5 | 25.5 | 0.799 | 1.920 | -1.121 |
| (16, 26, 34) | 1.57 | 308.8 | 21.2 | 14.6 | 23.6 | 22.7 | 2.819 | 1.726 | +1.093 |
| (-32, 13, 8) | 1.64 | 391.3 | 12.8 | 30.5 | 17.9 | 15.1 | 0.513 | 1.600 | -1.086 |
| (-27, 22, 26) | 1.66 | 339.1 | 16.6 | 20.4 | 24.1 | 21.8 | 1.631 | 0.546 | +1.085 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `deposited_1ee2` | 24.27 | 0.9525 | 199.2148458113208 | 321.3156706038993 | 1.000 |
| `rerefined_i` | 26.86 | 0.9466 | 199.2150577093264 | 343.5923592085409 | 1.000 |