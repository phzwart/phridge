# Held-Out Log-Likelihood Comparison: `drifted_f` vs `true_model`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `-0.1720` nats/refl
- **Model B NLL**: `-0.0244` nats/refl
- **Difference (Gain $\Delta$)**: `-0.1476` nats/refl (`+0.1476` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-132.81` nats
- **Uncertainty**: Bootstrap SE = `0.0450` (95% CI: `[-0.2433, -0.0676]`) | Naive SE = `0.0119` (Ratio: `3.79`x)
- **Win Fraction $P(d_h > 0)$**: `25.4%` (229 wins, 671 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `5.30e-51`
- **Wilcoxon Signed-Rank $p$-value**: `1.83e-49`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.2099` | `-0.0623` | `+0.1476` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.1197` | `+0.0278` | `+0.1476` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`true_model`) | Model B (`drifted_f`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `+0.0054` | `+0.0064` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `-0.0890` | `-0.0138` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0830` | `-0.0106` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.4783 | -1.1236 | -0.3547 | 0.0257 | 2.3% | 1/42 |
| 1 | 6.10 | 4.86 | 46 | 0.0574 | 0.6592 | -0.6018 | 0.0799 | 2.2% | 1/45 |
| 2 | 4.83 | 4.21 | 45 | -0.5641 | -0.2257 | -0.3384 | 0.0270 | 0.0% | 0/45 |
| 3 | 4.20 | 3.81 | 46 | -0.1786 | 0.1242 | -0.3028 | 0.0342 | 4.3% | 2/44 |
| 4 | 3.80 | 3.53 | 46 | -0.1109 | 0.3018 | -0.4128 | 0.0417 | 2.2% | 1/45 |
| 5 | 3.53 | 3.32 | 46 | 0.2014 | 0.5394 | -0.3380 | 0.0400 | 4.3% | 2/44 |
| 6 | 3.31 | 3.16 | 44 | -0.3773 | -0.1638 | -0.2135 | 0.0238 | 4.5% | 2/42 |
| 7 | 3.14 | 3.02 | 45 | 0.2134 | 0.4003 | -0.1869 | 0.0536 | 11.1% | 5/40 |
| 8 | 3.01 | 2.89 | 46 | -0.3379 | -0.1908 | -0.1472 | 0.0334 | 21.7% | 10/36 |
| 9 | 2.88 | 2.79 | 43 | -0.3773 | -0.1691 | -0.2083 | 0.0414 | 14.0% | 6/37 |
| 10 | 2.79 | 2.70 | 45 | -0.6225 | -0.4673 | -0.1552 | 0.0283 | 13.3% | 6/39 |
| 11 | 2.70 | 2.62 | 45 | -0.1486 | -0.0467 | -0.1019 | 0.0388 | 35.6% | 16/29 |
| 12 | 2.62 | 2.55 | 44 | -0.2226 | -0.1701 | -0.0525 | 0.0342 | 36.4% | 16/28 |
| 13 | 2.55 | 2.49 | 47 | -0.1495 | -0.1021 | -0.0474 | 0.0369 | 31.9% | 15/32 |
| 14 | 2.49 | 2.43 | 43 | -0.5134 | -0.4996 | -0.0138 | 0.0286 | 44.2% | 19/24 |
| 15 | 2.43 | 2.38 | 43 | 0.1633 | 0.0774 | +0.0859 | 0.0420 | 60.5% | 26/17 |
| 16 | 2.38 | 2.33 | 44 | 0.0364 | -0.0422 | +0.0785 | 0.0445 | 56.8% | 25/19 |
| 17 | 2.32 | 2.28 | 47 | 0.1084 | 0.0045 | +0.1039 | 0.0367 | 57.4% | 27/20 |
| 18 | 2.28 | 2.24 | 48 | 0.4964 | 0.3483 | +0.1482 | 0.0697 | 54.2% | 26/22 |
| 19 | 2.24 | 2.20 | 44 | 0.2361 | 0.1246 | +0.1115 | 0.0637 | 52.3% | 23/21 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-15, 5, 23) | 2.25 | 913.7 | 371.5 | 2.5 | 59.7 | 53.9 | 6.542 | 3.467 | +3.075 |
| (-13, 1, 8) | 4.90 | 298524.3 | 9400.0 | 31.8 | 606.0 | 648.6 | 4.638 | 7.126 | -2.488 |
| (-14, 6, 23) | 2.22 | 2169.1 | 327.5 | 6.6 | 65.1 | 60.1 | 3.299 | 1.515 | +1.784 |
| (14, 2, 4) | 5.33 | 184945.9 | 7620.0 | 24.3 | 493.3 | 530.5 | 4.862 | 6.644 | -1.781 |
| (1, 7, 2) | 4.96 | 84293.1 | 6850.5 | 12.3 | 317.0 | 353.1 | 1.218 | 2.951 | -1.733 |
| (14, 4, 0) | 5.00 | 54687.4 | 2670.0 | 20.5 | 264.8 | 294.7 | 1.440 | 3.064 | -1.624 |
| (12, 6, 14) | 3.08 | 23153.4 | 1496.5 | 15.5 | 186.3 | 182.0 | 3.623 | 2.012 | +1.611 |
| (9, 5, 4) | 5.27 | 118950.6 | 7195.0 | 16.5 | 401.8 | 435.2 | 3.823 | 5.409 | -1.586 |
| (32, 6, 2) | 2.41 | 12757.4 | 1004.5 | 12.7 | 133.0 | 125.3 | 2.805 | 1.308 | +1.497 |
| (2, 2, 14) | 4.11 | 121519.8 | 5705.5 | 21.3 | 397.8 | 424.2 | 3.965 | 5.374 | -1.408 |
| (-14, 2, 12) | 3.78 | 76665.2 | 1903.5 | 40.3 | 302.8 | 327.8 | 1.160 | 2.509 | -1.349 |
| (-14, 4, 2) | 4.94 | 119246.2 | 6440.0 | 18.5 | 402.2 | 430.9 | 4.151 | 5.337 | -1.186 |
| (16, 14, 2) | 2.27 | 4477.3 | 849.0 | 5.3 | 85.4 | 80.0 | 2.542 | 1.384 | +1.159 |
| (7, 7, 11) | 3.52 | 16206.3 | 1156.5 | 14.0 | 145.0 | 162.2 | 0.653 | 1.809 | -1.155 |
| (5, 9, 4) | 3.70 | 54523.7 | 3721.0 | 14.7 | 256.6 | 278.4 | 0.919 | 1.993 | -1.073 |
| (18, 4, 10) | 3.39 | 19180.2 | 768.0 | 25.0 | 151.6 | 167.7 | 0.448 | 1.505 | -1.057 |
| (-25, 11, 8) | 2.23 | 6437.4 | 408.0 | 15.8 | 94.9 | 91.5 | 2.448 | 1.396 | +1.052 |
| (-8, 8, 6) | 3.78 | 27671.9 | 2951.0 | 9.4 | 186.0 | 207.7 | 0.434 | 1.478 | -1.044 |
| (11, 5, 4) | 4.90 | 42974.9 | 2010.5 | 21.4 | 230.0 | 252.1 | 0.768 | 1.790 | -1.022 |
| (-16, 4, 12) | 3.38 | 73827.9 | 3058.0 | 24.1 | 319.4 | 337.9 | 5.458 | 6.474 | -1.017 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `true_model` | 7.06 | 0.9953 | 7.176088803437241 | 2.3277404455528554 | 1.000 |
| `drifted_f` | 9.77 | 0.9943 | 7.780059205817361 | 2.6242456572623 | 1.000 |