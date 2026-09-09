# Held-Out Log-Likelihood Comparison: `recovered_f` vs `shaken`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.5008` nats/refl
- **Model B NLL**: `0.0782` nats/refl
- **Difference (Gain $\Delta$)**: `+0.4226` nats/refl (`-0.4226` nats NLL reduction)
- **Estimated Log Bayes Factor**: `+380.34` nats
- **Uncertainty**: Bootstrap SE = `0.0329` (95% CI: `[+0.3525, +0.4818]`) | Naive SE = `0.0297` (Ratio: `1.11`x)
- **Win Fraction $P(d_h > 0)$**: `79.1%` (712 wins, 188 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `2.15e-72`
- **Wilcoxon Signed-Rank $p$-value**: `2.29e-61`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `-0.2996` | `-0.1230` | `-0.4226` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `-0.6376` | `+0.2150` | `-0.4226` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`shaken`) | Model B (`recovered_f`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `+0.0019` | `-0.0132` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.6082` | `0.0841` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1074` | `-0.0059` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -0.9646 | -1.3488 | +0.3842 | 0.2545 | 46.5% | 20/23 |
| 1 | 6.10 | 4.86 | 46 | 0.2139 | 0.1057 | +0.1082 | 0.1293 | 69.6% | 32/14 |
| 2 | 4.83 | 4.21 | 45 | -0.1279 | -0.3643 | +0.2365 | 0.0954 | 75.6% | 34/11 |
| 3 | 4.20 | 3.81 | 46 | 0.5317 | 0.0322 | +0.4995 | 0.1143 | 93.5% | 43/3 |
| 4 | 3.80 | 3.53 | 46 | 0.4041 | 0.2538 | +0.1502 | 0.1304 | 76.1% | 35/11 |
| 5 | 3.53 | 3.32 | 46 | 0.9915 | 0.3236 | +0.6680 | 0.1975 | 73.9% | 34/12 |
| 6 | 3.31 | 3.16 | 44 | 0.1961 | -0.0296 | +0.2257 | 0.0817 | 75.0% | 33/11 |
| 7 | 3.14 | 3.02 | 45 | 0.8541 | 0.4722 | +0.3820 | 0.1391 | 77.8% | 35/10 |
| 8 | 3.01 | 2.89 | 46 | 0.4128 | -0.0395 | +0.4523 | 0.1053 | 84.8% | 39/7 |
| 9 | 2.88 | 2.79 | 43 | 0.5190 | 0.1385 | +0.3806 | 0.1158 | 74.4% | 32/11 |
| 10 | 2.79 | 2.70 | 45 | 0.2947 | -0.1685 | +0.4632 | 0.0840 | 91.1% | 41/4 |
| 11 | 2.70 | 2.62 | 45 | 0.6344 | 0.0818 | +0.5526 | 0.1150 | 88.9% | 40/5 |
| 12 | 2.62 | 2.55 | 44 | 0.4994 | -0.0677 | +0.5671 | 0.0881 | 84.1% | 37/7 |
| 13 | 2.55 | 2.49 | 47 | 0.6294 | 0.1690 | +0.4604 | 0.1065 | 78.7% | 37/10 |
| 14 | 2.49 | 2.43 | 43 | 0.4747 | -0.0418 | +0.5166 | 0.1188 | 79.1% | 34/9 |
| 15 | 2.43 | 2.38 | 43 | 0.8787 | 0.4620 | +0.4167 | 0.1448 | 88.4% | 38/5 |
| 16 | 2.38 | 2.33 | 44 | 0.8894 | 0.2526 | +0.6369 | 0.1562 | 81.8% | 36/8 |
| 17 | 2.32 | 2.28 | 47 | 0.8515 | 0.3545 | +0.4970 | 0.1005 | 76.6% | 36/11 |
| 18 | 2.28 | 2.24 | 48 | 0.8068 | 0.4141 | +0.3926 | 0.0941 | 77.1% | 37/11 |
| 19 | 2.24 | 2.20 | 44 | 0.9454 | 0.4764 | +0.4690 | 0.1294 | 88.6% | 39/5 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (3, 1, 0) | 22.09 | 21112.0 | 272.2 | 77.5 | 204.5 | 198.9 | 10.709 | 0.574 | +10.135 |
| (20, 4, 7) | 3.47 | 531.1 | 53.2 | 10.0 | 127.8 | 35.0 | 5.198 | -1.076 | +6.274 |
| (-16, 4, 12) | 3.38 | 80054.6 | 917.4 | 87.3 | 224.8 | 308.4 | 5.703 | 1.418 | +4.286 |
| (-1, 1, 3) | 17.00 | 198.2 | 44.8 | 4.4 | 57.6 | 29.3 | 1.144 | -2.818 | +3.961 |
| (-30, 2, 14) | 2.35 | 1122.6 | 63.0 | 17.8 | 95.6 | 41.4 | 3.859 | 0.058 | +3.801 |
| (-20, 2, 14) | 2.98 | 15598.6 | 134.2 | 116.2 | 51.6 | 117.0 | 4.380 | 0.641 | +3.739 |
| (-30, 4, 4) | 2.66 | 15866.4 | 265.2 | 59.8 | 79.2 | 130.7 | 4.471 | 0.968 | +3.502 |
| (-29, 9, 3) | 2.33 | -64.8 | 41.8 | -1.5 | 63.6 | 20.8 | 5.662 | 2.179 | +3.483 |
| (-12, 8, 19) | 2.41 | 6035.3 | 117.3 | 51.5 | 31.4 | 84.3 | 4.204 | 0.755 | +3.449 |
| (2, 2, 14) | 4.11 | 124622.5 | 1711.6 | 72.8 | 423.9 | 437.9 | 2.396 | 5.821 | -3.425 |
| (27, 7, 12) | 2.34 | 11715.6 | 148.6 | 78.8 | 75.4 | 119.0 | 4.487 | 1.140 | +3.347 |
| (-24, 2, 2) | 3.45 | 55504.6 | 1155.1 | 48.0 | 182.7 | 246.6 | 4.351 | 1.036 | +3.315 |
| (-13, 1, 8) | 4.90 | 286163.9 | 2820.0 | 101.5 | 553.9 | 620.1 | 1.727 | 4.925 | -3.198 |
| (16, 10, 13) | 2.46 | 34.6 | 40.8 | 0.8 | 62.6 | 19.9 | 2.193 | -0.888 | +3.081 |
| (27, 1, 2) | 3.11 | 63525.2 | 1281.0 | 49.6 | 213.4 | 272.0 | 4.388 | 1.360 | +3.027 |
| (9, 3, 14) | 3.66 | 132472.5 | 1664.9 | 79.6 | 363.5 | 441.5 | 1.724 | 4.713 | -2.989 |
| (-25, 11, 8) | 2.23 | 6973.5 | 122.4 | 57.0 | 56.1 | 92.1 | 3.934 | 1.102 | +2.833 |
| (15, 7, 4) | 3.64 | 72767.8 | 1165.5 | 62.4 | 212.6 | 287.3 | 3.508 | 0.826 | +2.682 |
| (16, 8, 17) | 2.43 | 150.4 | 72.6 | 2.1 | 4.9 | 41.7 | -0.970 | 1.663 | -2.633 |
| (-19, 1, 20) | 2.49 | -39.2 | 56.9 | -0.7 | 49.8 | 5.9 | 2.138 | -0.490 | +2.628 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `shaken` | 35.13 | 0.9592 | 3.1024938534637583 | 1.791144890696521 | 1.000 |
| `recovered_f` | 15.78 | 0.9921 | 4.7118256357385295 | 3.474313210162355 | 1.000 |