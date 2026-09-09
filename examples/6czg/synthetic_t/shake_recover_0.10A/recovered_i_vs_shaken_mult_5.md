# Held-Out Log-Likelihood Comparison: `recovered_i` vs `shaken`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.1721` nats/refl
- **Model B NLL**: `0.0271` nats/refl
- **Difference (Gain $\Delta$)**: `+0.1450` nats/refl (`-0.1450` nats NLL reduction)
- **Estimated Log Bayes Factor**: `+130.52` nats
- **Uncertainty**: Bootstrap SE = `0.0601` (95% CI: `[+0.0192, +0.2507]`) | Naive SE = `0.0287` (Ratio: `2.09`x)
- **Win Fraction $P(d_h > 0)$**: `63.6%` (572 wins, 328 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `3.65e-16`
- **Wilcoxon Signed-Rank $p$-value**: `5.34e-13`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `-0.0748` | `-0.0702` | `-0.1450` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `-0.2017` | `+0.0567` | `-0.1450` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`shaken`) | Model B (`recovered_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `+0.0074` | `+0.0027` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.2456` | `0.0157` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0735` | `+0.0114` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.1991 | -1.1097 | -0.0893 | 0.3344 | 23.3% | 10/33 |
| 1 | 6.10 | 4.86 | 46 | -0.1378 | 0.4426 | -0.5804 | 0.1356 | 21.7% | 10/36 |
| 2 | 4.83 | 4.21 | 45 | -0.4892 | -0.2134 | -0.2758 | 0.0897 | 24.4% | 11/34 |
| 3 | 4.20 | 3.81 | 46 | 0.1913 | 0.0869 | +0.1044 | 0.1083 | 52.2% | 24/22 |
| 4 | 3.80 | 3.53 | 46 | 0.1253 | 0.2907 | -0.1654 | 0.1280 | 39.1% | 18/28 |
| 5 | 3.53 | 3.32 | 46 | 0.3683 | 0.4380 | -0.0697 | 0.1678 | 50.0% | 23/23 |
| 6 | 3.31 | 3.16 | 44 | -0.0418 | -0.1058 | +0.0640 | 0.0730 | 59.1% | 26/18 |
| 7 | 3.14 | 3.02 | 45 | 0.4607 | 0.4543 | +0.0064 | 0.0907 | 53.3% | 24/21 |
| 8 | 3.01 | 2.89 | 46 | 0.0838 | -0.0964 | +0.1802 | 0.0709 | 67.4% | 31/15 |
| 9 | 2.88 | 2.79 | 43 | 0.1855 | -0.0830 | +0.2684 | 0.0960 | 69.8% | 30/13 |
| 10 | 2.79 | 2.70 | 45 | -0.1219 | -0.4268 | +0.3050 | 0.0723 | 71.1% | 32/13 |
| 11 | 2.70 | 2.62 | 45 | 0.3076 | -0.0221 | +0.3298 | 0.1117 | 77.8% | 35/10 |
| 12 | 2.62 | 2.55 | 44 | 0.0992 | -0.1240 | +0.2231 | 0.0766 | 70.5% | 31/13 |
| 13 | 2.55 | 2.49 | 47 | 0.3528 | -0.0011 | +0.3539 | 0.0957 | 83.0% | 39/8 |
| 14 | 2.49 | 2.43 | 43 | 0.1297 | -0.3935 | +0.5231 | 0.0723 | 93.0% | 40/3 |
| 15 | 2.43 | 2.38 | 43 | 0.6826 | 0.2715 | +0.4111 | 0.1010 | 83.7% | 36/7 |
| 16 | 2.38 | 2.33 | 44 | 0.5173 | 0.1668 | +0.3505 | 0.0936 | 75.0% | 33/11 |
| 17 | 2.32 | 2.28 | 47 | 0.5385 | 0.1475 | +0.3910 | 0.0667 | 85.1% | 40/7 |
| 18 | 2.28 | 2.24 | 48 | 0.6784 | 0.4649 | +0.2135 | 0.0940 | 81.2% | 39/9 |
| 19 | 2.24 | 2.20 | 44 | 0.6299 | 0.2364 | +0.3935 | 0.0799 | 90.9% | 40/4 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (3, 1, 0) | 22.09 | 19735.3 | 907.5 | 21.7 | 190.4 | 174.8 | 13.430 | -0.276 | +13.706 |
| (20, 4, 7) | 3.47 | 471.7 | 177.5 | 2.7 | 80.0 | 32.7 | 3.683 | -1.212 | +4.895 |
| (14, 2, 4) | 5.33 | 184945.9 | 7620.0 | 24.3 | 469.3 | 531.6 | 1.866 | 5.892 | -4.026 |
| (9, 3, 14) | 3.66 | 131464.6 | 5549.5 | 23.7 | 399.0 | 444.9 | 1.648 | 5.380 | -3.732 |
| (-18, 8, 15) | 2.52 | 1690.8 | 288.5 | 5.9 | 80.6 | 48.2 | 3.547 | -0.049 | +3.597 |
| (-13, 1, 8) | 4.90 | 298524.3 | 9400.0 | 31.8 | 595.0 | 645.8 | 2.530 | 5.857 | -3.327 |
| (0, 4, 12) | 4.33 | 73099.5 | 2072.0 | 35.3 | 294.1 | 348.0 | 0.678 | 3.685 | -3.007 |
| (-16, 4, 12) | 3.38 | 73827.9 | 3058.0 | 24.1 | 289.0 | 327.6 | 1.225 | 4.154 | -2.930 |
| (-30, 2, 14) | 2.35 | 1474.5 | 210.0 | 7.0 | 69.4 | 41.4 | 2.533 | -0.196 | +2.729 |
| (-5, 9, 1) | 3.82 | 3872.4 | 457.5 | 8.5 | 111.5 | 80.5 | 2.504 | -0.137 | +2.641 |
| (-7, 1, 22) | 2.64 | 4453.4 | 389.5 | 11.4 | 110.3 | 87.1 | 3.897 | 1.319 | +2.578 |
| (-18, 4, 10) | 3.43 | 48176.8 | 3216.5 | 15.0 | 260.8 | 282.4 | 2.284 | 4.796 | -2.512 |
| (-12, 8, 6) | 3.51 | 50830.6 | 2124.5 | 23.9 | 248.3 | 278.9 | 1.169 | 3.668 | -2.500 |
| (15, 9, 12) | 2.69 | 502.9 | 159.0 | 3.2 | 55.1 | 29.0 | 1.620 | -0.818 | +2.438 |
| (-29, 9, 3) | 2.33 | -225.8 | 139.5 | -1.6 | 31.1 | 10.3 | 3.468 | 1.105 | +2.363 |
| (7, 9, 19) | 2.39 | 1554.3 | 297.0 | 5.2 | 68.9 | 45.5 | 2.336 | 0.031 | +2.305 |
| (9, 5, 4) | 5.27 | 118950.6 | 7195.0 | 16.5 | 379.0 | 422.0 | 1.487 | 3.624 | -2.137 |
| (-15, 9, 19) | 2.26 | -986.1 | 181.0 | -5.4 | 17.0 | 4.6 | 13.035 | 15.172 | -2.137 |
| (-17, 1, 9) | 3.99 | 30373.9 | 1000.5 | 30.4 | 230.3 | 211.2 | 3.647 | 1.518 | +2.129 |
| (19, 1, 8) | 3.78 | 38637.3 | 2592.0 | 14.9 | 260.6 | 239.6 | 3.690 | 1.570 | +2.119 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `shaken` | 19.76 | 0.9863 | 5.716430828222444 | 1.845924405170934 | 1.000 |
| `recovered_i` | 11.83 | 0.9935 | 6.877894534466397 | 2.337291788101025 | 1.000 |