# Held-Out Log-Likelihood Comparison: `drifted_f` vs `true_model`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `1.1555` nats/refl
- **Model B NLL**: `1.1672` nats/refl
- **Difference (Gain $\Delta$)**: `-0.0118` nats/refl (`+0.0118` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-10.58` nats
- **Uncertainty**: Bootstrap SE = `0.0067` (95% CI: `[-0.0256, +0.0008]`) | Naive SE = `0.0071` (Ratio: `0.94`x)
- **Win Fraction $P(d_h > 0)$**: `43.7%` (393 wins, 507 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `1.62e-04`
- **Wilcoxon Signed-Rank $p$-value**: `6.44e-04`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.0159` | `-0.0042` | `+0.0118` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.0128` | `-0.0011` | `+0.0118` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`true_model`) | Model B (`drifted_f`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0069` | `-0.0098` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `1.3184` | `1.3204` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1629` | `-0.1532` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -0.8849 | -0.8279 | -0.0570 | 0.0410 | 30.2% | 13/30 |
| 1 | 6.10 | 4.86 | 46 | 0.6455 | 0.6968 | -0.0513 | 0.0279 | 47.8% | 22/24 |
| 2 | 4.83 | 4.21 | 45 | 0.4586 | 0.4582 | +0.0004 | 0.0633 | 33.3% | 15/30 |
| 3 | 4.20 | 3.81 | 46 | 0.9819 | 0.9989 | -0.0170 | 0.0242 | 37.0% | 17/29 |
| 4 | 3.80 | 3.53 | 46 | 1.2523 | 1.2784 | -0.0261 | 0.0257 | 41.3% | 19/27 |
| 5 | 3.53 | 3.32 | 46 | 1.3688 | 1.3896 | -0.0208 | 0.0238 | 45.7% | 21/25 |
| 6 | 3.31 | 3.16 | 44 | 1.1069 | 1.0832 | +0.0237 | 0.0288 | 45.5% | 20/24 |
| 7 | 3.14 | 3.02 | 45 | 1.3792 | 1.4203 | -0.0412 | 0.0263 | 44.4% | 20/25 |
| 8 | 3.01 | 2.89 | 46 | 0.7427 | 0.7962 | -0.0535 | 0.0406 | 41.3% | 19/27 |
| 9 | 2.88 | 2.79 | 43 | 0.9986 | 1.0529 | -0.0543 | 0.0321 | 48.8% | 21/22 |
| 10 | 2.79 | 2.70 | 45 | 0.8386 | 0.8186 | +0.0200 | 0.0267 | 48.9% | 22/23 |
| 11 | 2.70 | 2.62 | 45 | 1.4649 | 1.4472 | +0.0177 | 0.0300 | 51.1% | 23/22 |
| 12 | 2.62 | 2.55 | 44 | 1.3107 | 1.3211 | -0.0104 | 0.0295 | 38.6% | 17/27 |
| 13 | 2.55 | 2.49 | 47 | 1.4068 | 1.3951 | +0.0118 | 0.0248 | 36.2% | 17/30 |
| 14 | 2.49 | 2.43 | 43 | 1.1857 | 1.1691 | +0.0166 | 0.0241 | 48.8% | 21/22 |
| 15 | 2.43 | 2.38 | 43 | 1.7405 | 1.7745 | -0.0340 | 0.0275 | 41.9% | 18/25 |
| 16 | 2.38 | 2.33 | 44 | 1.4157 | 1.4172 | -0.0015 | 0.0304 | 25.0% | 11/33 |
| 17 | 2.32 | 2.28 | 47 | 1.8088 | 1.8251 | -0.0164 | 0.0187 | 51.1% | 24/23 |
| 18 | 2.28 | 2.24 | 48 | 1.8633 | 1.8195 | +0.0437 | 0.0322 | 56.2% | 27/21 |
| 19 | 2.24 | 2.20 | 44 | 1.9094 | 1.8991 | +0.0103 | 0.0198 | 59.1% | 26/18 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-3, 5, 11) | 4.26 | 10860.2 | 1340.0 | 8.1 | 59.3 | 61.9 | 5.953 | 3.425 | +2.528 |
| (8, 12, 16) | 2.25 | -1066.5 | 1855.0 | -0.6 | 64.9 | 54.7 | 4.214 | 2.933 | +1.281 |
| (-1, 5, 4) | 6.37 | 22235.6 | 3590.0 | 6.2 | 182.7 | 180.6 | 2.165 | 1.156 | +1.009 |
| (13, 11, 2) | 2.87 | 2913.6 | 980.0 | 3.0 | 53.3 | 71.5 | -0.278 | 0.674 | -0.952 |
| (-26, 8, 2) | 2.62 | 5215.5 | 1475.0 | 3.5 | 22.9 | 32.0 | 3.826 | 2.880 | +0.945 |
| (11, 7, 14) | 2.98 | 16005.2 | 2775.0 | 5.8 | 120.3 | 107.3 | 0.264 | 1.117 | -0.853 |
| (24, 0, 10) | 3.02 | 49740.7 | 6740.0 | 7.4 | 192.8 | 179.1 | 2.407 | 3.255 | -0.848 |
| (-10, 10, 13) | 2.66 | -51.7 | 1855.0 | -0.0 | 56.5 | 67.2 | 1.605 | 2.416 | -0.810 |
| (-30, 2, 4) | 2.75 | 33.8 | 1710.0 | 0.0 | 49.7 | 31.9 | 0.843 | 0.100 | +0.744 |
| (2, 14, 12) | 2.24 | 70.6 | 1250.0 | 0.1 | 34.8 | 46.3 | 1.038 | 1.763 | -0.725 |
| (8, 4, 16) | 3.25 | 8413.7 | 12420.0 | 0.7 | 178.9 | 162.0 | 2.902 | 2.179 | +0.722 |
| (6, 0, 7) | 7.24 | 3275.8 | 1265.0 | 2.6 | 22.1 | 28.6 | -0.236 | -0.918 | +0.681 |
| (26, 4, 16) | 2.35 | 283.7 | 2170.0 | 0.1 | 66.9 | 59.7 | 2.563 | 1.883 | +0.680 |
| (24, 6, 6) | 2.89 | -5949.6 | 5975.0 | -1.0 | 89.6 | 72.0 | 3.432 | 2.754 | +0.678 |
| (-11, 3, 7) | 5.17 | 1039.4 | 1790.0 | 0.6 | 64.9 | 73.2 | 0.125 | 0.761 | -0.636 |
| (-12, 2, 7) | 5.23 | 4948.1 | 1855.0 | 2.7 | 76.3 | 88.3 | -0.860 | -0.241 | -0.620 |
| (-27, 1, 6) | 3.00 | 21460.1 | 5610.0 | 3.8 | 132.7 | 118.0 | 0.981 | 1.595 | -0.614 |
| (-15, 7, 18) | 2.50 | 4465.8 | 2290.0 | 2.0 | 88.8 | 77.3 | 1.652 | 1.049 | +0.604 |
| (27, 7, 12) | 2.34 | 5610.5 | 4955.0 | 1.1 | 117.0 | 108.3 | 2.967 | 2.374 | +0.594 |
| (-13, 5, 20) | 2.54 | 7058.8 | 1940.0 | 3.6 | 32.5 | 41.4 | 3.950 | 3.359 | +0.591 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `true_model` | 73.06 | 0.7118 | 8.316639414194936 | 2.2151127829374033 | 1.000 |
| `drifted_f` | 73.82 | 0.7100 | 9.144780111604435 | 2.6785037744361655 | 1.000 |