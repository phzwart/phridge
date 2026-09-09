# Held-Out Log-Likelihood Comparison: `recovered_i` vs `shaken`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.5133` nats/refl
- **Model B NLL**: `0.1092` nats/refl
- **Difference (Gain $\Delta$)**: `+0.4041` nats/refl (`-0.4041` nats NLL reduction)
- **Estimated Log Bayes Factor**: `+363.68` nats
- **Uncertainty**: Bootstrap SE = `0.0328` (95% CI: `[+0.3354, +0.4632]`) | Naive SE = `0.0299` (Ratio: `1.09`x)
- **Win Fraction $P(d_h > 0)$**: `78.1%` (703 wins, 197 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `2.64e-67`
- **Wilcoxon Signed-Rank $p$-value**: `5.04e-61`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `-0.2915` | `-0.1126` | `-0.4041` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `-0.5951` | `+0.1910` | `-0.4041` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`shaken`) | Model B (`recovered_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0016` | `-0.0150` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.6325` | `0.1175` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1192` | `-0.0082` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -0.9799 | -1.3448 | +0.3649 | 0.2330 | 48.8% | 21/22 |
| 1 | 6.10 | 4.86 | 46 | 0.2179 | 0.1245 | +0.0934 | 0.1498 | 69.6% | 32/14 |
| 2 | 4.83 | 4.21 | 45 | -0.0830 | -0.3320 | +0.2491 | 0.0963 | 77.8% | 35/10 |
| 3 | 4.20 | 3.81 | 46 | 0.5694 | 0.1042 | +0.4652 | 0.1082 | 89.1% | 41/5 |
| 4 | 3.80 | 3.53 | 46 | 0.3855 | 0.2894 | +0.0961 | 0.1586 | 69.6% | 32/14 |
| 5 | 3.53 | 3.32 | 46 | 1.0151 | 0.3773 | +0.6378 | 0.2104 | 67.4% | 31/15 |
| 6 | 3.31 | 3.16 | 44 | 0.2189 | -0.0311 | +0.2500 | 0.0775 | 77.3% | 34/10 |
| 7 | 3.14 | 3.02 | 45 | 0.7936 | 0.4884 | +0.3052 | 0.1320 | 73.3% | 33/12 |
| 8 | 3.01 | 2.89 | 46 | 0.4360 | -0.0336 | +0.4696 | 0.1014 | 82.6% | 38/8 |
| 9 | 2.88 | 2.79 | 43 | 0.4830 | 0.1163 | +0.3667 | 0.1096 | 72.1% | 31/12 |
| 10 | 2.79 | 2.70 | 45 | 0.2318 | -0.2168 | +0.4487 | 0.0842 | 86.7% | 39/6 |
| 11 | 2.70 | 2.62 | 45 | 0.7177 | 0.1585 | +0.5592 | 0.1166 | 88.9% | 40/5 |
| 12 | 2.62 | 2.55 | 44 | 0.5188 | -0.0506 | +0.5694 | 0.0906 | 81.8% | 36/8 |
| 13 | 2.55 | 2.49 | 47 | 0.5943 | 0.1964 | +0.3979 | 0.1090 | 80.9% | 38/9 |
| 14 | 2.49 | 2.43 | 43 | 0.5277 | 0.0484 | +0.4793 | 0.1162 | 83.7% | 36/7 |
| 15 | 2.43 | 2.38 | 43 | 0.9433 | 0.5557 | +0.3876 | 0.1528 | 86.0% | 37/6 |
| 16 | 2.38 | 2.33 | 44 | 0.8943 | 0.2707 | +0.6236 | 0.1591 | 81.8% | 36/8 |
| 17 | 2.32 | 2.28 | 47 | 0.9407 | 0.4876 | +0.4531 | 0.1025 | 80.9% | 38/9 |
| 18 | 2.28 | 2.24 | 48 | 0.8278 | 0.4108 | +0.4170 | 0.0964 | 77.1% | 37/11 |
| 19 | 2.24 | 2.20 | 44 | 0.9322 | 0.4757 | +0.4565 | 0.1185 | 86.4% | 38/6 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (3, 1, 0) | 22.09 | 21820.7 | 544.5 | 40.1 | 203.9 | 199.4 | 9.335 | 0.518 | +8.817 |
| (20, 4, 7) | 3.47 | 514.5 | 106.5 | 4.8 | 127.7 | 36.2 | 5.261 | -1.027 | +6.287 |
| (9, 3, 14) | 3.66 | 120355.5 | 3329.7 | 36.1 | 363.4 | 441.8 | 1.395 | 6.699 | -5.304 |
| (-1, 1, 3) | 17.00 | 95.8 | 89.7 | 1.1 | 57.5 | 29.7 | 1.767 | -2.691 | +4.458 |
| (-16, 4, 12) | 3.38 | 80259.0 | 1834.8 | 43.7 | 224.5 | 309.0 | 5.738 | 1.426 | +4.312 |
| (14, 2, 4) | 5.33 | 173387.8 | 4572.0 | 37.9 | 411.1 | 507.6 | 1.646 | 5.749 | -4.103 |
| (27, 7, 12) | 2.34 | 12562.0 | 297.3 | 42.3 | 75.4 | 118.6 | 5.078 | 1.145 | +3.933 |
| (-20, 2, 14) | 2.98 | 15360.2 | 268.5 | 57.2 | 51.6 | 116.0 | 4.328 | 0.645 | +3.682 |
| (-30, 2, 14) | 2.35 | 1239.0 | 126.0 | 9.8 | 95.5 | 41.6 | 3.709 | 0.056 | +3.653 |
| (16, 8, 17) | 2.43 | -222.1 | 145.2 | -1.5 | 4.9 | 42.1 | 1.389 | 4.952 | -3.563 |
| (-30, 4, 4) | 2.66 | 15718.2 | 530.4 | 29.6 | 79.2 | 130.6 | 4.363 | 0.963 | +3.400 |
| (-29, 9, 3) | 2.33 | 115.7 | 83.7 | 1.4 | 63.5 | 21.3 | 2.718 | -0.586 | +3.304 |
| (2, 2, 14) | 4.11 | 126040.1 | 3423.3 | 36.8 | 423.6 | 437.6 | 2.317 | 5.479 | -3.162 |
| (-12, 8, 19) | 2.41 | 5774.6 | 234.6 | 24.6 | 31.4 | 83.9 | 3.913 | 0.754 | +3.159 |
| (-24, 2, 2) | 3.45 | 54725.0 | 2310.3 | 23.7 | 182.6 | 246.4 | 4.173 | 1.047 | +3.127 |
| (-13, 1, 8) | 4.90 | 286196.6 | 5640.0 | 50.7 | 553.8 | 619.2 | 1.722 | 4.843 | -3.122 |
| (-18, 4, 10) | 3.43 | 49670.1 | 1929.9 | 25.7 | 239.6 | 288.1 | 1.349 | 4.406 | -3.057 |
| (16, 10, 13) | 2.46 | 107.3 | 81.6 | 1.3 | 62.6 | 20.2 | 1.953 | -0.962 | +2.915 |
| (-25, 11, 8) | 2.23 | 7111.2 | 244.8 | 29.0 | 56.1 | 92.3 | 3.994 | 1.101 | +2.893 |
| (-13, 5, 13) | 3.33 | 12210.2 | 483.0 | 25.3 | 48.1 | 122.2 | 3.195 | 0.400 | +2.795 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `shaken` | 34.91 | 0.9604 | 199.07025649898966 | 1000.0 | 1.000 |
| `recovered_i` | 16.73 | 0.9900 | 199.1474825067724 | 1000.0 | 1.000 |