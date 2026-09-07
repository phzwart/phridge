# Held-Out Log-Likelihood Comparison: `drifted_f` vs `true_model`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `-0.3063` nats/refl
- **Model B NLL**: `-0.1402` nats/refl
- **Difference (Gain $\Delta$)**: `-0.1661` nats/refl (`+0.1661` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-149.50` nats
- **Uncertainty**: Bootstrap SE = `0.0421` (95% CI: `[-0.2545, -0.0911]`) | Naive SE = `0.0121` (Ratio: `3.49`x)
- **Win Fraction $P(d_h > 0)$**: `20.2%` (182 wins, 718 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `6.43e-76`
- **Wilcoxon Signed-Rank $p$-value**: `5.44e-64`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.2179` | `-0.0518` | `+0.1661` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.1366` | `+0.0295` | `+0.1661` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`true_model`) | Model B (`drifted_f`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0046` | `-0.0019` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `-0.2196` | `-0.1172` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0868` | `-0.0230` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.4352 | -1.1867 | -0.2485 | 0.0243 | 2.3% | 1/42 |
| 1 | 6.10 | 4.86 | 46 | 0.0534 | 0.6890 | -0.6356 | 0.1035 | 2.2% | 1/45 |
| 2 | 4.83 | 4.21 | 45 | -0.6110 | -0.2898 | -0.3212 | 0.0387 | 0.0% | 0/45 |
| 3 | 4.20 | 3.81 | 46 | -0.1932 | 0.1279 | -0.3211 | 0.0435 | 0.0% | 0/46 |
| 4 | 3.80 | 3.53 | 46 | -0.0816 | 0.3634 | -0.4450 | 0.0508 | 2.2% | 1/45 |
| 5 | 3.53 | 3.32 | 46 | 0.1297 | 0.4618 | -0.3321 | 0.0437 | 4.3% | 2/44 |
| 6 | 3.31 | 3.16 | 44 | -0.3544 | -0.1321 | -0.2223 | 0.0266 | 4.5% | 2/42 |
| 7 | 3.14 | 3.02 | 45 | 0.2389 | 0.4409 | -0.2019 | 0.0387 | 11.1% | 5/40 |
| 8 | 3.01 | 2.89 | 46 | -0.4545 | -0.2584 | -0.1962 | 0.0274 | 13.0% | 6/40 |
| 9 | 2.88 | 2.79 | 43 | -0.4664 | -0.2446 | -0.2218 | 0.0392 | 14.0% | 6/37 |
| 10 | 2.79 | 2.70 | 45 | -0.6087 | -0.4607 | -0.1481 | 0.0252 | 17.8% | 8/37 |
| 11 | 2.70 | 2.62 | 45 | -0.3627 | -0.2241 | -0.1386 | 0.0248 | 8.9% | 4/41 |
| 12 | 2.62 | 2.55 | 44 | -0.4188 | -0.3518 | -0.0670 | 0.0310 | 31.8% | 14/30 |
| 13 | 2.55 | 2.49 | 47 | -0.3601 | -0.2755 | -0.0845 | 0.0310 | 23.4% | 11/36 |
| 14 | 2.49 | 2.43 | 43 | -0.6115 | -0.5768 | -0.0347 | 0.0261 | 48.8% | 21/22 |
| 15 | 2.43 | 2.38 | 43 | 0.0250 | -0.0454 | +0.0704 | 0.0434 | 51.2% | 22/21 |
| 16 | 2.38 | 2.33 | 44 | -0.2699 | -0.3314 | +0.0615 | 0.0758 | 43.2% | 19/25 |
| 17 | 2.32 | 2.28 | 47 | -0.1212 | -0.1745 | +0.0534 | 0.0503 | 40.4% | 19/28 |
| 18 | 2.28 | 2.24 | 48 | -0.1720 | -0.2414 | +0.0694 | 0.0468 | 41.7% | 20/28 |
| 19 | 2.24 | 2.20 | 44 | -0.1449 | -0.2031 | +0.0582 | 0.0528 | 45.5% | 20/24 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-13, 1, 8) | 4.90 | 286163.9 | 2820.0 | 101.5 | 610.3 | 649.7 | 6.878 | 9.830 | -2.952 |
| (27, 7, 12) | 2.34 | 11715.6 | 148.6 | 78.8 | 127.0 | 116.3 | 3.749 | 0.912 | +2.838 |
| (14, 2, 4) | 5.33 | 185493.8 | 2286.0 | 81.1 | 496.4 | 533.7 | 5.002 | 7.595 | -2.593 |
| (9, 5, 4) | 5.27 | 124478.5 | 2158.5 | 57.7 | 404.3 | 440.2 | 3.214 | 5.697 | -2.482 |
| (1, 7, 2) | 4.96 | 79170.5 | 2055.1 | 38.5 | 319.0 | 350.6 | 1.908 | 3.934 | -2.025 |
| (2, 2, 14) | 4.11 | 124622.5 | 1711.6 | 72.8 | 400.3 | 426.1 | 3.747 | 5.499 | -1.752 |
| (-14, 2, 12) | 3.78 | 72266.5 | 571.1 | 126.6 | 304.6 | 329.1 | 1.850 | 3.498 | -1.648 |
| (5, 9, 4) | 3.70 | 52203.8 | 1116.3 | 46.8 | 258.2 | 282.1 | 1.330 | 2.916 | -1.586 |
| (14, 4, 0) | 5.00 | 55763.6 | 801.0 | 69.6 | 266.2 | 293.1 | 1.231 | 2.749 | -1.518 |
| (30, 2, 14) | 2.32 | 10907.6 | 259.5 | 42.0 | 119.1 | 111.8 | 2.404 | 0.922 | +1.482 |
| (-9, 5, 4) | 5.30 | 73539.8 | 1570.0 | 46.8 | 310.2 | 336.7 | 1.888 | 3.347 | -1.459 |
| (-8, 8, 6) | 3.78 | 25716.6 | 885.3 | 29.0 | 187.1 | 207.9 | 0.795 | 2.067 | -1.273 |
| (-16, 4, 12) | 3.38 | 80054.6 | 917.4 | 87.3 | 321.2 | 339.3 | 3.765 | 5.031 | -1.266 |
| (-18, 4, 10) | 3.43 | 53950.7 | 964.9 | 55.9 | 264.3 | 281.9 | 2.613 | 3.878 | -1.265 |
| (0, 4, 12) | 4.33 | 80890.8 | 621.6 | 130.1 | 322.0 | 346.2 | 1.508 | 2.756 | -1.248 |
| (-14, 4, 2) | 4.94 | 125909.8 | 1932.0 | 65.2 | 404.9 | 429.0 | 3.272 | 4.462 | -1.190 |
| (32, 4, 12) | 2.25 | 3048.8 | 165.5 | 18.4 | 69.8 | 68.2 | 2.684 | 1.550 | +1.135 |
| (-12, 4, 8) | 4.46 | 75875.1 | 1542.9 | 49.2 | 313.1 | 336.0 | 1.398 | 2.493 | -1.096 |
| (10, 12, 14) | 2.32 | 9087.6 | 127.8 | 71.1 | 108.7 | 104.6 | 2.051 | 0.977 | +1.074 |
| (17, 5, 0) | 4.08 | 57159.6 | 1145.6 | 49.9 | 271.8 | 290.6 | 1.925 | 2.998 | -1.073 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `true_model` | 2.09 | 0.9997 | 11.120828743810849 | 15.070314747558738 | 1.000 |
| `drifted_f` | 6.27 | 0.9989 | 8.304382684723622 | 9.062916033705973 | 1.000 |