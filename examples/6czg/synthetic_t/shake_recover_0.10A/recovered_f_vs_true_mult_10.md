# Held-Out Log-Likelihood Comparison: `recovered_f` vs `deposited`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.0322` nats/refl
- **Model B NLL**: `0.1982` nats/refl
- **Difference (Gain $\Delta$)**: `-0.1660` nats/refl (`+0.1660` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-149.43` nats
- **Uncertainty**: Bootstrap SE = `0.0287` (95% CI: `[-0.2236, -0.1112]`) | Naive SE = `0.0162` (Ratio: `1.77`x)
- **Win Fraction $P(d_h > 0)$**: `26.1%` (235 wins, 665 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `3.05e-48`
- **Wilcoxon Signed-Rank $p$-value**: `3.82e-44`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.2627` | `-0.0966` | `+0.1660` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.1238` | `+0.0423` | `+0.1660` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`deposited`) | Model B (`recovered_f`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0071` | `-0.0166` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.1228` | `0.2112` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0907` | `-0.0130` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.5441 | -1.1711 | -0.3730 | 0.1660 | 18.6% | 8/35 |
| 1 | 6.10 | 4.86 | 46 | -0.1528 | 0.2362 | -0.3890 | 0.0360 | 6.5% | 3/43 |
| 2 | 4.83 | 4.21 | 45 | -0.6179 | -0.2664 | -0.3516 | 0.0520 | 13.3% | 6/39 |
| 3 | 4.20 | 3.81 | 46 | -0.1780 | 0.0629 | -0.2408 | 0.0501 | 15.2% | 7/39 |
| 4 | 3.80 | 3.53 | 46 | 0.0820 | 0.3658 | -0.2837 | 0.0674 | 15.2% | 7/39 |
| 5 | 3.53 | 3.32 | 46 | 0.3107 | 0.5017 | -0.1910 | 0.0581 | 19.6% | 9/37 |
| 6 | 3.31 | 3.16 | 44 | -0.1737 | 0.0312 | -0.2049 | 0.0497 | 22.7% | 10/34 |
| 7 | 3.14 | 3.02 | 45 | 0.3932 | 0.6351 | -0.2419 | 0.0715 | 17.8% | 8/37 |
| 8 | 3.01 | 2.89 | 46 | -0.2065 | 0.0427 | -0.2492 | 0.0476 | 19.6% | 9/37 |
| 9 | 2.88 | 2.79 | 43 | 0.1173 | 0.3000 | -0.1827 | 0.0917 | 23.3% | 10/33 |
| 10 | 2.79 | 2.70 | 45 | -0.3108 | -0.1725 | -0.1383 | 0.0472 | 26.7% | 12/33 |
| 11 | 2.70 | 2.62 | 45 | 0.1873 | 0.2859 | -0.0986 | 0.0463 | 37.8% | 17/28 |
| 12 | 2.62 | 2.55 | 44 | 0.1010 | 0.2022 | -0.1012 | 0.0484 | 25.0% | 11/33 |
| 13 | 2.55 | 2.49 | 47 | 0.0881 | 0.2176 | -0.1295 | 0.0469 | 25.5% | 12/35 |
| 14 | 2.49 | 2.43 | 43 | -0.1044 | -0.0347 | -0.0697 | 0.0412 | 37.2% | 16/27 |
| 15 | 2.43 | 2.38 | 43 | 0.4917 | 0.4289 | +0.0627 | 0.0868 | 30.2% | 13/30 |
| 16 | 2.38 | 2.33 | 44 | 0.3501 | 0.4187 | -0.0686 | 0.0549 | 38.6% | 17/27 |
| 17 | 2.32 | 2.28 | 47 | 0.7179 | 0.6905 | +0.0274 | 0.0807 | 44.7% | 21/26 |
| 18 | 2.28 | 2.24 | 48 | 0.3901 | 0.5064 | -0.1163 | 0.0544 | 41.7% | 20/28 |
| 19 | 2.24 | 2.20 | 44 | 0.6173 | 0.5883 | +0.0290 | 0.0942 | 43.2% | 19/25 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (3, 1, 0) | 22.09 | 19917.5 | 1815.0 | 11.0 | 176.3 | 179.2 | 3.700 | -0.226 | +3.925 |
| (-5, 13, 13) | 2.32 | -560.0 | 338.0 | -1.7 | 38.6 | 29.1 | 7.011 | 3.796 | +3.215 |
| (-27, 11, 4) | 2.22 | 1793.3 | 497.0 | 3.6 | 68.9 | 60.4 | 4.779 | 1.843 | +2.936 |
| (-5, 3, 4) | 8.14 | 27472.1 | 5651.0 | 4.9 | 232.4 | 243.5 | 3.792 | 1.687 | +2.105 |
| (-21, 3, 14) | 2.86 | 291.3 | 1066.0 | 0.3 | 88.0 | 90.2 | 8.008 | 6.000 | +2.009 |
| (0, 4, 1) | 8.73 | 4.5 | 131.0 | 0.0 | 6.5 | 17.2 | -4.092 | -2.136 | -1.956 |
| (-1, 1, 3) | 17.00 | -15.7 | 299.0 | -0.1 | 9.8 | 10.9 | -3.940 | -2.126 | -1.814 |
| (-16, 4, 12) | 3.38 | 66572.1 | 6116.0 | 10.9 | 315.7 | 320.3 | 5.947 | 4.158 | +1.789 |
| (-10, 2, 2) | 7.44 | 22735.9 | 2999.0 | 7.6 | 192.1 | 198.2 | 2.218 | 0.431 | +1.788 |
| (-19, 1, 21) | 2.41 | 6678.8 | 1013.0 | 6.6 | 104.2 | 93.6 | 2.759 | 1.013 | +1.747 |
| (-9, 13, 10) | 2.39 | -189.1 | 223.0 | -0.8 | 28.9 | 21.2 | 2.668 | 0.947 | +1.721 |
| (6, 2, 0) | 11.04 | 59213.2 | 8776.0 | 6.7 | 293.5 | 286.5 | 2.189 | 0.481 | +1.707 |
| (14, 8, 18) | 2.42 | 4573.6 | 736.0 | 6.2 | 87.7 | 75.0 | 2.248 | 0.573 | +1.675 |
| (27, 11, 4) | 2.22 | 3271.2 | 607.0 | 5.4 | 74.2 | 60.6 | 2.248 | 0.586 | +1.663 |
| (-12, 6, 14) | 3.11 | 11109.8 | 1085.0 | 10.2 | 115.2 | 139.0 | 0.264 | 1.910 | -1.647 |
| (2, 10, 12) | 2.87 | 887.2 | 458.0 | 1.9 | 38.2 | 56.7 | -0.604 | 0.927 | -1.531 |
| (26, 4, 2) | 3.04 | 6535.1 | 443.0 | 14.8 | 92.7 | 112.1 | 0.203 | 1.727 | -1.523 |
| (1, 1, 3) | 16.91 | 238.8 | 82.0 | 2.9 | 18.8 | 16.3 | -4.304 | -2.808 | -1.496 |
| (8, 6, 11) | 3.72 | 592.2 | 381.0 | 1.6 | 29.9 | 54.9 | -1.918 | -0.450 | -1.468 |
| (26, 4, 8) | 2.82 | 1150.5 | 482.0 | 2.4 | 50.5 | 65.3 | 0.295 | 1.688 | -1.393 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `deposited` | 12.69 | 0.9925 | 199.00140959097453 | 495.1435794009256 | 1.000 |
| `recovered_f` | 15.95 | 0.9903 | 199.11732842991213 | 510.6639779862454 | 1.000 |