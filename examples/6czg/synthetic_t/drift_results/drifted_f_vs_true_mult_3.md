# Held-Out Log-Likelihood Comparison: `drifted_f` vs `true_model`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `-0.2639` nats/refl
- **Model B NLL**: `-0.1002` nats/refl
- **Difference (Gain $\Delta$)**: `-0.1637` nats/refl (`+0.1637` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-147.32` nats
- **Uncertainty**: Bootstrap SE = `0.0390` (95% CI: `[-0.2442, -0.0956]`) | Naive SE = `0.0113` (Ratio: `3.43`x)
- **Win Fraction $P(d_h > 0)$**: `19.2%` (173 wins, 727 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `2.09e-81`
- **Wilcoxon Signed-Rank $p$-value**: `6.98e-68`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.2208` | `-0.0571` | `+0.1637` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.1358` | `+0.0279` | `+0.1637` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`true_model`) | Model B (`drifted_f`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0030` | `-0.0030` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `-0.1682` | `-0.0777` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0957` | `-0.0224` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.4259 | -1.1585 | -0.2673 | 0.0252 | 2.3% | 1/42 |
| 1 | 6.10 | 4.86 | 46 | 0.0724 | 0.6892 | -0.6168 | 0.0929 | 2.2% | 1/45 |
| 2 | 4.83 | 4.21 | 45 | -0.5815 | -0.2707 | -0.3108 | 0.0304 | 0.0% | 0/45 |
| 3 | 4.20 | 3.81 | 46 | -0.1444 | 0.1728 | -0.3171 | 0.0380 | 0.0% | 0/46 |
| 4 | 3.80 | 3.53 | 46 | -0.0121 | 0.3794 | -0.3915 | 0.0449 | 0.0% | 0/46 |
| 5 | 3.53 | 3.32 | 46 | 0.1315 | 0.4422 | -0.3108 | 0.0399 | 8.7% | 4/42 |
| 6 | 3.31 | 3.16 | 44 | -0.4101 | -0.1860 | -0.2241 | 0.0239 | 4.5% | 2/42 |
| 7 | 3.14 | 3.02 | 45 | 0.3277 | 0.4757 | -0.1480 | 0.0421 | 20.0% | 9/36 |
| 8 | 3.01 | 2.89 | 46 | -0.4378 | -0.2495 | -0.1882 | 0.0300 | 15.2% | 7/39 |
| 9 | 2.88 | 2.79 | 43 | -0.4125 | -0.2250 | -0.1875 | 0.0435 | 16.3% | 7/36 |
| 10 | 2.79 | 2.70 | 45 | -0.6341 | -0.4966 | -0.1374 | 0.0251 | 13.3% | 6/39 |
| 11 | 2.70 | 2.62 | 45 | -0.2527 | -0.1144 | -0.1384 | 0.0306 | 22.2% | 10/35 |
| 12 | 2.62 | 2.55 | 44 | -0.3635 | -0.2854 | -0.0782 | 0.0301 | 20.5% | 9/35 |
| 13 | 2.55 | 2.49 | 47 | -0.2721 | -0.1953 | -0.0768 | 0.0375 | 21.3% | 10/37 |
| 14 | 2.49 | 2.43 | 43 | -0.5054 | -0.4651 | -0.0403 | 0.0272 | 37.2% | 16/27 |
| 15 | 2.43 | 2.38 | 43 | 0.1747 | 0.0716 | +0.1030 | 0.0570 | 46.5% | 20/23 |
| 16 | 2.38 | 2.33 | 44 | -0.3149 | -0.2950 | -0.0199 | 0.0458 | 38.6% | 17/27 |
| 17 | 2.32 | 2.28 | 47 | -0.0469 | -0.0686 | +0.0217 | 0.0535 | 36.2% | 17/30 |
| 18 | 2.28 | 2.24 | 48 | -0.1876 | -0.2188 | +0.0312 | 0.0329 | 39.6% | 19/29 |
| 19 | 2.24 | 2.20 | 44 | -0.0691 | -0.1096 | +0.0405 | 0.0610 | 40.9% | 18/26 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-13, 1, 8) | 4.90 | 286196.6 | 5640.0 | 50.7 | 609.8 | 647.9 | 6.748 | 9.235 | -2.486 |
| (9, 5, 4) | 5.27 | 125566.9 | 4317.0 | 29.1 | 403.8 | 439.9 | 2.971 | 5.303 | -2.332 |
| (14, 2, 4) | 5.33 | 173387.8 | 4572.0 | 37.9 | 495.9 | 532.3 | 7.202 | 9.388 | -2.186 |
| (1, 7, 2) | 4.96 | 78832.9 | 4110.3 | 19.2 | 318.6 | 351.9 | 1.900 | 3.965 | -2.065 |
| (35, 1, 10) | 2.23 | 7399.3 | 325.2 | 22.8 | 101.0 | 94.3 | 2.956 | 1.006 | +1.950 |
| (14, 4, 0) | 5.00 | 55468.7 | 1602.0 | 34.6 | 265.8 | 296.0 | 1.267 | 3.042 | -1.774 |
| (30, 2, 14) | 2.32 | 10623.7 | 519.0 | 20.5 | 119.0 | 109.9 | 2.656 | 0.904 | +1.752 |
| (2, 2, 14) | 4.11 | 126040.1 | 3423.3 | 36.8 | 399.8 | 425.5 | 3.390 | 4.983 | -1.594 |
| (27, 7, 12) | 2.34 | 12562.0 | 297.3 | 42.3 | 126.9 | 115.1 | 2.295 | 0.773 | +1.522 |
| (5, 9, 4) | 3.70 | 49441.7 | 2232.6 | 22.1 | 258.0 | 280.3 | 1.774 | 3.193 | -1.419 |
| (-14, 2, 12) | 3.78 | 72507.4 | 1142.1 | 63.5 | 304.2 | 327.5 | 1.752 | 3.160 | -1.407 |
| (32, 6, 2) | 2.41 | 13800.6 | 602.7 | 22.9 | 133.6 | 128.5 | 2.331 | 1.150 | +1.181 |
| (-32, 4, 4) | 2.51 | 9557.5 | 326.1 | 29.3 | 115.2 | 111.1 | 2.220 | 1.058 | +1.162 |
| (-9, 5, 4) | 5.30 | 77244.7 | 3140.1 | 24.6 | 309.8 | 334.5 | 1.368 | 2.529 | -1.161 |
| (18, 4, 10) | 3.39 | 17035.1 | 460.8 | 37.0 | 152.5 | 167.8 | 1.178 | 2.281 | -1.104 |
| (-16, 4, 12) | 3.38 | 80259.0 | 1834.8 | 43.7 | 320.8 | 338.5 | 3.631 | 4.719 | -1.089 |
| (-14, 4, 2) | 4.94 | 127430.9 | 3864.0 | 33.0 | 404.5 | 428.4 | 2.978 | 4.064 | -1.086 |
| (8, 2, 0) | 9.10 | 78362.5 | 1123.2 | 69.8 | 318.5 | 351.0 | 0.568 | 1.596 | -1.028 |
| (12, 6, 14) | 3.08 | 25625.0 | 897.9 | 28.5 | 187.1 | 184.0 | 2.466 | 1.446 | +1.021 |
| (0, 4, 12) | 4.33 | 81200.9 | 1243.2 | 65.3 | 321.6 | 343.7 | 1.431 | 2.436 | -1.005 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `true_model` | 4.17 | 0.9987 | 199.20962731196659 | 579.3022966013517 | 1.000 |
| `drifted_f` | 7.92 | 0.9978 | 199.20328969208404 | 653.8373836511294 | 1.000 |