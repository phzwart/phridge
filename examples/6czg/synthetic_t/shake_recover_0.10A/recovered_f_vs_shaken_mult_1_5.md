# Held-Out Log-Likelihood Comparison: `recovered_f` vs `shaken`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.0760` nats/refl
- **Model B NLL**: `-0.0906` nats/refl
- **Difference (Gain $\Delta$)**: `+0.1666` nats/refl (`-0.1666` nats NLL reduction)
- **Estimated Log Bayes Factor**: `+149.92` nats
- **Uncertainty**: Bootstrap SE = `0.0722` (95% CI: `[+0.0152, +0.2931]`) | Naive SE = `0.0263` (Ratio: `2.75`x)
- **Win Fraction $P(d_h > 0)$**: `64.9%` (584 wins, 316 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `3.06e-19`
- **Wilcoxon Signed-Rank $p$-value**: `3.80e-17`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `-0.0737` | `-0.0929` | `-0.1666` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `-0.2352` | `+0.0686` | `-0.1666` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`shaken`) | Model B (`recovered_f`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0077` | `-0.0069` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.1674` | `-0.0903` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0914` | `-0.0003` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.4221 | -1.1499 | -0.2722 | 0.1533 | 25.6% | 11/32 |
| 1 | 6.10 | 4.86 | 46 | -0.1650 | 0.4732 | -0.6382 | 0.1624 | 19.6% | 9/37 |
| 2 | 4.83 | 4.21 | 45 | -0.5081 | -0.2710 | -0.2371 | 0.0860 | 22.2% | 10/35 |
| 3 | 4.20 | 3.81 | 46 | 0.1925 | 0.0829 | +0.1095 | 0.1121 | 47.8% | 22/24 |
| 4 | 3.80 | 3.53 | 46 | 0.0923 | 0.3225 | -0.2302 | 0.1401 | 37.0% | 17/29 |
| 5 | 3.53 | 3.32 | 46 | 0.3255 | 0.3667 | -0.0412 | 0.1614 | 50.0% | 23/23 |
| 6 | 3.31 | 3.16 | 44 | -0.0978 | -0.0852 | -0.0126 | 0.0779 | 54.5% | 24/20 |
| 7 | 3.14 | 3.02 | 45 | 0.3939 | 0.4448 | -0.0509 | 0.1052 | 51.1% | 23/22 |
| 8 | 3.01 | 2.89 | 46 | 0.0086 | -0.1578 | +0.1664 | 0.0782 | 71.7% | 33/13 |
| 9 | 2.88 | 2.79 | 43 | 0.1465 | -0.1419 | +0.2884 | 0.0948 | 69.8% | 30/13 |
| 10 | 2.79 | 2.70 | 45 | -0.1206 | -0.4332 | +0.3126 | 0.0756 | 77.8% | 35/10 |
| 11 | 2.70 | 2.62 | 45 | 0.1743 | -0.1669 | +0.3412 | 0.1018 | 71.1% | 32/13 |
| 12 | 2.62 | 2.55 | 44 | -0.0564 | -0.2991 | +0.2427 | 0.0807 | 86.4% | 38/6 |
| 13 | 2.55 | 2.49 | 47 | 0.2747 | -0.1596 | +0.4343 | 0.0986 | 85.1% | 40/7 |
| 14 | 2.49 | 2.43 | 43 | 0.0968 | -0.5171 | +0.6139 | 0.0918 | 93.0% | 40/3 |
| 15 | 2.43 | 2.38 | 43 | 0.5437 | 0.1270 | +0.4167 | 0.0997 | 86.0% | 37/6 |
| 16 | 2.38 | 2.33 | 44 | 0.2809 | -0.1734 | +0.4543 | 0.1038 | 81.8% | 36/8 |
| 17 | 2.32 | 2.28 | 47 | 0.3974 | -0.0667 | +0.4642 | 0.0680 | 85.1% | 40/7 |
| 18 | 2.28 | 2.24 | 48 | 0.3704 | -0.0619 | +0.4323 | 0.0724 | 91.7% | 44/4 |
| 19 | 2.24 | 2.20 | 44 | 0.5138 | -0.0430 | +0.5568 | 0.0712 | 90.9% | 40/4 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (3, 1, 0) | 22.09 | 21112.0 | 272.2 | 77.5 | 193.1 | 175.4 | 5.059 | -0.333 | +5.392 |
| (20, 4, 7) | 3.47 | 531.1 | 53.2 | 10.0 | 80.5 | 32.3 | 3.591 | -1.241 | +4.831 |
| (14, 2, 4) | 5.33 | 185493.8 | 2286.0 | 81.1 | 472.2 | 534.9 | 1.998 | 6.767 | -4.769 |
| (-13, 1, 8) | 4.90 | 286163.9 | 2820.0 | 101.5 | 599.0 | 649.6 | 3.946 | 8.474 | -4.528 |
| (-18, 8, 15) | 2.52 | 1442.8 | 86.6 | 16.7 | 81.0 | 49.0 | 4.392 | 0.142 | +4.251 |
| (9, 3, 14) | 3.66 | 132472.5 | 1664.9 | 79.6 | 401.4 | 445.0 | 1.691 | 5.735 | -4.044 |
| (-30, 2, 14) | 2.35 | 1122.6 | 63.0 | 17.8 | 69.8 | 42.4 | 3.559 | 0.076 | +3.484 |
| (16, 10, 13) | 2.46 | 34.6 | 40.8 | 0.8 | 38.4 | 14.9 | 1.399 | -1.405 | +2.804 |
| (-29, 9, 3) | 2.33 | -64.8 | 41.8 | -1.5 | 31.2 | 10.6 | 3.653 | 0.883 | +2.770 |
| (-5, 9, 1) | 3.82 | 3795.9 | 137.2 | 27.7 | 112.2 | 82.8 | 2.656 | 0.001 | +2.655 |
| (-12, 8, 6) | 3.51 | 50749.5 | 637.3 | 79.6 | 249.8 | 278.8 | 1.220 | 3.782 | -2.562 |
| (0, 4, 12) | 4.33 | 80890.8 | 621.6 | 130.1 | 295.8 | 351.4 | 0.423 | 2.890 | -2.467 |
| (1, 7, 2) | 4.96 | 79170.5 | 2055.1 | 38.5 | 314.5 | 354.0 | 1.363 | 3.807 | -2.444 |
| (-7, 1, 22) | 2.64 | 4794.6 | 116.9 | 41.0 | 110.9 | 88.7 | 3.650 | 1.222 | +2.428 |
| (-18, 4, 10) | 3.43 | 53950.7 | 964.9 | 55.9 | 262.2 | 286.0 | 1.546 | 3.970 | -2.424 |
| (-13, 15, 1) | 2.21 | 60.3 | 42.4 | 1.4 | 30.9 | 14.6 | 1.366 | -0.919 | +2.286 |
| (35, 1, 9) | 2.26 | 342.6 | 53.6 | 6.4 | 47.3 | 29.5 | 2.512 | 0.258 | +2.254 |
| (16, 8, 11) | 2.86 | 269.4 | 44.6 | 6.0 | 48.0 | 22.8 | 0.897 | -1.322 | +2.219 |
| (-20, 4, 15) | 2.77 | -129.7 | 56.1 | -2.3 | 34.5 | 6.4 | 4.195 | 1.982 | +2.213 |
| (14, 2, 12) | 3.73 | 10405.0 | 446.1 | 23.3 | 155.5 | 127.9 | 2.437 | 0.278 | +2.159 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `shaken` | 17.82 | 0.9895 | 4.270775575139874 | 2.808886449306507 | 1.000 |
| `recovered_f` | 8.91 | 0.9977 | 7.157700865338023 | 7.086207309349849 | 1.000 |