# Held-Out Log-Likelihood Comparison: `drifted_i` vs `true_model`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `-0.2639` nats/refl
- **Model B NLL**: `-0.0993` nats/refl
- **Difference (Gain $\Delta$)**: `-0.1646` nats/refl (`+0.1646` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-148.11` nats
- **Uncertainty**: Bootstrap SE = `0.0397` (95% CI: `[-0.2474, -0.0950]`) | Naive SE = `0.0115` (Ratio: `3.46`x)
- **Win Fraction $P(d_h > 0)$**: `19.3%` (174 wins, 726 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `8.76e-81`
- **Wilcoxon Signed-Rank $p$-value**: `1.87e-67`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.2224` | `-0.0578` | `+0.1646` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.1363` | `+0.0283` | `+0.1646` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`true_model`) | Model B (`drifted_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0030` | `-0.0034` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `-0.1682` | `-0.0749` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0957` | `-0.0244` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.4259 | -1.1502 | -0.2757 | 0.0249 | 2.3% | 1/42 |
| 1 | 6.10 | 4.86 | 46 | 0.0724 | 0.6928 | -0.6204 | 0.0958 | 2.2% | 1/45 |
| 2 | 4.83 | 4.21 | 45 | -0.5815 | -0.2566 | -0.3249 | 0.0326 | 0.0% | 0/45 |
| 3 | 4.20 | 3.81 | 46 | -0.1444 | 0.1741 | -0.3185 | 0.0387 | 0.0% | 0/46 |
| 4 | 3.80 | 3.53 | 46 | -0.0121 | 0.3880 | -0.4001 | 0.0453 | 0.0% | 0/46 |
| 5 | 3.53 | 3.32 | 46 | 0.1315 | 0.4467 | -0.3152 | 0.0404 | 8.7% | 4/42 |
| 6 | 3.31 | 3.16 | 44 | -0.4101 | -0.1850 | -0.2251 | 0.0245 | 2.3% | 1/43 |
| 7 | 3.14 | 3.02 | 45 | 0.3277 | 0.4783 | -0.1506 | 0.0402 | 20.0% | 9/36 |
| 8 | 3.01 | 2.89 | 46 | -0.4378 | -0.2503 | -0.1875 | 0.0297 | 19.6% | 9/37 |
| 9 | 2.88 | 2.79 | 43 | -0.4125 | -0.2231 | -0.1893 | 0.0439 | 14.0% | 6/37 |
| 10 | 2.79 | 2.70 | 45 | -0.6341 | -0.4945 | -0.1395 | 0.0256 | 13.3% | 6/39 |
| 11 | 2.70 | 2.62 | 45 | -0.2527 | -0.1184 | -0.1343 | 0.0302 | 24.4% | 11/34 |
| 12 | 2.62 | 2.55 | 44 | -0.3635 | -0.2936 | -0.0700 | 0.0301 | 18.2% | 8/36 |
| 13 | 2.55 | 2.49 | 47 | -0.2721 | -0.1977 | -0.0743 | 0.0379 | 21.3% | 10/37 |
| 14 | 2.49 | 2.43 | 43 | -0.5054 | -0.4711 | -0.0344 | 0.0271 | 37.2% | 16/27 |
| 15 | 2.43 | 2.38 | 43 | 0.1747 | 0.0682 | +0.1065 | 0.0570 | 46.5% | 20/23 |
| 16 | 2.38 | 2.33 | 44 | -0.3149 | -0.2950 | -0.0199 | 0.0464 | 40.9% | 18/26 |
| 17 | 2.32 | 2.28 | 47 | -0.0469 | -0.0693 | +0.0223 | 0.0531 | 36.2% | 17/30 |
| 18 | 2.28 | 2.24 | 48 | -0.1876 | -0.2217 | +0.0341 | 0.0329 | 39.6% | 19/29 |
| 19 | 2.24 | 2.20 | 44 | -0.0691 | -0.1120 | +0.0430 | 0.0608 | 40.9% | 18/26 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-13, 1, 8) | 4.90 | 286196.6 | 5640.0 | 50.7 | 609.8 | 648.8 | 6.748 | 9.348 | -2.600 |
| (9, 5, 4) | 5.27 | 125566.9 | 4317.0 | 29.1 | 403.8 | 440.7 | 2.971 | 5.380 | -2.408 |
| (14, 2, 4) | 5.33 | 173387.8 | 4572.0 | 37.9 | 495.9 | 533.6 | 7.202 | 9.553 | -2.350 |
| (1, 7, 2) | 4.96 | 78832.9 | 4110.3 | 19.2 | 318.6 | 352.2 | 1.900 | 3.982 | -2.082 |
| (35, 1, 10) | 2.23 | 7399.3 | 325.2 | 22.8 | 101.0 | 94.3 | 2.956 | 1.003 | +1.953 |
| (30, 2, 14) | 2.32 | 10623.7 | 519.0 | 20.5 | 119.0 | 110.2 | 2.656 | 0.921 | +1.735 |
| (14, 4, 0) | 5.00 | 55468.7 | 1602.0 | 34.6 | 265.8 | 294.9 | 1.267 | 2.931 | -1.664 |
| (2, 2, 14) | 4.11 | 126040.1 | 3423.3 | 36.8 | 399.8 | 425.2 | 3.390 | 4.921 | -1.532 |
| (27, 7, 12) | 2.34 | 12562.0 | 297.3 | 42.3 | 126.9 | 115.0 | 2.295 | 0.775 | +1.521 |
| (5, 9, 4) | 3.70 | 49441.7 | 2232.6 | 22.1 | 258.0 | 281.2 | 1.774 | 3.274 | -1.500 |
| (-14, 2, 12) | 3.78 | 72507.4 | 1142.1 | 63.5 | 304.2 | 326.6 | 1.752 | 3.066 | -1.314 |
| (32, 6, 2) | 2.41 | 13800.6 | 602.7 | 22.9 | 133.6 | 127.9 | 2.331 | 1.101 | +1.229 |
| (-32, 4, 4) | 2.51 | 9557.5 | 326.1 | 29.3 | 115.2 | 110.7 | 2.220 | 1.025 | +1.195 |
| (-14, 4, 2) | 4.94 | 127430.9 | 3864.0 | 33.0 | 404.5 | 428.8 | 2.978 | 4.093 | -1.114 |
| (18, 4, 10) | 3.39 | 17035.1 | 460.8 | 37.0 | 152.5 | 167.9 | 1.178 | 2.290 | -1.112 |
| (-8, 8, 6) | 3.78 | 27640.2 | 1770.6 | 15.6 | 186.9 | 208.1 | 0.446 | 1.534 | -1.088 |
| (8, 2, 0) | 9.10 | 78362.5 | 1123.2 | 69.8 | 318.5 | 351.5 | 0.568 | 1.618 | -1.050 |
| (-9, 5, 4) | 5.30 | 77244.7 | 3140.1 | 24.6 | 309.8 | 333.1 | 1.368 | 2.411 | -1.044 |
| (17, 5, 0) | 4.08 | 53884.0 | 2291.1 | 23.5 | 271.4 | 290.7 | 2.646 | 3.678 | -1.033 |
| (-16, 4, 12) | 3.38 | 80259.0 | 1834.8 | 43.7 | 320.8 | 338.1 | 3.631 | 4.643 | -1.012 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `true_model` | 4.17 | 0.9987 | 199.20962731196659 | 579.3022966013517 | 1.000 |
| `drifted_i` | 7.95 | 0.9978 | 199.20381643503458 | 654.2130007284266 | 1.000 |