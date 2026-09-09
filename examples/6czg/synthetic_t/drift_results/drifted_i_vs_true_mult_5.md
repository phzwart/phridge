# Held-Out Log-Likelihood Comparison: `drifted_i` vs `true_model`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `-0.1720` nats/refl
- **Model B NLL**: `-0.0241` nats/refl
- **Difference (Gain $\Delta$)**: `-0.1478` nats/refl (`+0.1478` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-133.06` nats
- **Uncertainty**: Bootstrap SE = `0.0462` (95% CI: `[-0.2464, -0.0671]`) | Naive SE = `0.0120` (Ratio: `3.84`x)
- **Win Fraction $P(d_h > 0)$**: `25.1%` (226 wins, 674 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `2.04e-52`
- **Wilcoxon Signed-Rank $p$-value**: `1.14e-48`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.2106` | `-0.0627` | `+0.1478` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.1196` | `+0.0282` | `+0.1478` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`true_model`) | Model B (`drifted_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `+0.0054` | `+0.0067` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `-0.0890` | `-0.0135` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0830` | `-0.0106` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.4783 | -1.1259 | -0.3524 | 0.0268 | 2.3% | 1/42 |
| 1 | 6.10 | 4.86 | 46 | 0.0574 | 0.6839 | -0.6265 | 0.0844 | 2.2% | 1/45 |
| 2 | 4.83 | 4.21 | 45 | -0.5641 | -0.2092 | -0.3549 | 0.0288 | 0.0% | 0/45 |
| 3 | 4.20 | 3.81 | 46 | -0.1786 | 0.1290 | -0.3076 | 0.0386 | 6.5% | 3/43 |
| 4 | 3.80 | 3.53 | 46 | -0.1109 | 0.3052 | -0.4161 | 0.0408 | 0.0% | 0/46 |
| 5 | 3.53 | 3.32 | 46 | 0.2014 | 0.5440 | -0.3426 | 0.0411 | 4.3% | 2/44 |
| 6 | 3.31 | 3.16 | 44 | -0.3773 | -0.1753 | -0.2021 | 0.0240 | 4.5% | 2/42 |
| 7 | 3.14 | 3.02 | 45 | 0.2134 | 0.3866 | -0.1732 | 0.0528 | 11.1% | 5/40 |
| 8 | 3.01 | 2.89 | 46 | -0.3379 | -0.1849 | -0.1530 | 0.0323 | 21.7% | 10/36 |
| 9 | 2.88 | 2.79 | 43 | -0.3773 | -0.1694 | -0.2080 | 0.0413 | 16.3% | 7/36 |
| 10 | 2.79 | 2.70 | 45 | -0.6225 | -0.4700 | -0.1525 | 0.0278 | 11.1% | 5/40 |
| 11 | 2.70 | 2.62 | 45 | -0.1486 | -0.0532 | -0.0954 | 0.0401 | 33.3% | 15/30 |
| 12 | 2.62 | 2.55 | 44 | -0.2226 | -0.1814 | -0.0412 | 0.0342 | 38.6% | 17/27 |
| 13 | 2.55 | 2.49 | 47 | -0.1495 | -0.0959 | -0.0536 | 0.0364 | 31.9% | 15/32 |
| 14 | 2.49 | 2.43 | 43 | -0.5134 | -0.5051 | -0.0083 | 0.0286 | 41.9% | 18/25 |
| 15 | 2.43 | 2.38 | 43 | 0.1633 | 0.0784 | +0.0850 | 0.0403 | 60.5% | 26/17 |
| 16 | 2.38 | 2.33 | 44 | 0.0364 | -0.0392 | +0.0755 | 0.0454 | 52.3% | 23/21 |
| 17 | 2.32 | 2.28 | 47 | 0.1084 | 0.0017 | +0.1068 | 0.0347 | 59.6% | 28/19 |
| 18 | 2.28 | 2.24 | 48 | 0.4964 | 0.3460 | +0.1504 | 0.0687 | 52.1% | 25/23 |
| 19 | 2.24 | 2.20 | 44 | 0.2361 | 0.1172 | +0.1189 | 0.0617 | 52.3% | 23/21 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-15, 5, 23) | 2.25 | 913.7 | 371.5 | 2.5 | 59.7 | 54.0 | 6.542 | 3.504 | +3.038 |
| (-13, 1, 8) | 4.90 | 298524.3 | 9400.0 | 31.8 | 606.0 | 650.6 | 4.638 | 7.387 | -2.749 |
| (14, 2, 4) | 5.33 | 184945.9 | 7620.0 | 24.3 | 493.3 | 532.4 | 4.862 | 6.873 | -2.011 |
| (1, 7, 2) | 4.96 | 84293.1 | 6850.5 | 12.3 | 317.0 | 354.1 | 1.218 | 3.031 | -1.814 |
| (-14, 6, 23) | 2.22 | 2169.1 | 327.5 | 6.6 | 65.1 | 60.2 | 3.299 | 1.547 | +1.752 |
| (9, 5, 4) | 5.27 | 118950.6 | 7195.0 | 16.5 | 401.8 | 436.7 | 3.823 | 5.565 | -1.742 |
| (2, 2, 14) | 4.11 | 121519.8 | 5705.5 | 21.3 | 397.8 | 425.6 | 3.965 | 5.546 | -1.581 |
| (12, 6, 14) | 3.08 | 23153.4 | 1496.5 | 15.5 | 186.3 | 182.4 | 3.623 | 2.048 | +1.575 |
| (32, 6, 2) | 2.41 | 12757.4 | 1004.5 | 12.7 | 133.0 | 125.7 | 2.805 | 1.342 | +1.463 |
| (14, 4, 0) | 5.00 | 54687.4 | 2670.0 | 20.5 | 264.8 | 292.4 | 1.440 | 2.861 | -1.421 |
| (-14, 2, 12) | 3.78 | 76665.2 | 1903.5 | 40.3 | 302.8 | 326.9 | 1.160 | 2.423 | -1.263 |
| (11, 5, 4) | 4.90 | 42974.9 | 2010.5 | 21.4 | 230.0 | 253.4 | 0.768 | 1.886 | -1.118 |
| (5, 9, 4) | 3.70 | 54523.7 | 3721.0 | 14.7 | 256.6 | 279.1 | 0.919 | 2.036 | -1.117 |
| (16, 14, 2) | 2.27 | 4477.3 | 849.0 | 5.3 | 85.4 | 80.4 | 2.542 | 1.439 | +1.103 |
| (-8, 8, 6) | 3.78 | 27671.9 | 2951.0 | 9.4 | 186.0 | 208.6 | 0.434 | 1.529 | -1.094 |
| (-16, 4, 12) | 3.38 | 73827.9 | 3058.0 | 24.1 | 319.4 | 338.2 | 5.458 | 6.521 | -1.064 |
| (-14, 4, 2) | 4.94 | 119246.2 | 6440.0 | 18.5 | 402.2 | 429.8 | 4.151 | 5.210 | -1.059 |
| (7, 7, 11) | 3.52 | 16206.3 | 1156.5 | 14.0 | 145.0 | 160.8 | 0.653 | 1.677 | -1.024 |
| (9, 3, 14) | 3.66 | 131464.6 | 5549.5 | 23.7 | 410.6 | 432.2 | 3.440 | 4.409 | -0.969 |
| (18, 4, 10) | 3.39 | 19180.2 | 768.0 | 25.0 | 151.6 | 166.7 | 0.448 | 1.414 | -0.966 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `true_model` | 7.06 | 0.9953 | 7.176088803437241 | 2.3277404455528554 | 1.000 |
| `drifted_i` | 9.77 | 0.9945 | 7.771953734155296 | 2.617201349742824 | 1.000 |