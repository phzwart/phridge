# Held-Out Log-Likelihood Comparison: `drifted_i` vs `true_model`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `-0.3063` nats/refl
- **Model B NLL**: `-0.1394` nats/refl
- **Difference (Gain $\Delta$)**: `-0.1669` nats/refl (`+0.1669` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-150.22` nats
- **Uncertainty**: Bootstrap SE = `0.0427` (95% CI: `[-0.2567, -0.0907]`) | Naive SE = `0.0122` (Ratio: `3.50`x)
- **Win Fraction $P(d_h > 0)$**: `20.3%` (183 wins, 717 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `2.53e-75`
- **Wilcoxon Signed-Rank $p$-value**: `1.23e-63`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.2192` | `-0.0523` | `+0.1669` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.1372` | `+0.0297` | `+0.1669` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`true_model`) | Model B (`drifted_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0046` | `-0.0019` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `-0.2196` | `-0.1159` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0868` | `-0.0235` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.4352 | -1.1846 | -0.2506 | 0.0246 | 2.3% | 1/42 |
| 1 | 6.10 | 4.86 | 46 | 0.0534 | 0.6964 | -0.6430 | 0.1051 | 2.2% | 1/45 |
| 2 | 4.83 | 4.21 | 45 | -0.6110 | -0.2838 | -0.3272 | 0.0392 | 0.0% | 0/45 |
| 3 | 4.20 | 3.81 | 46 | -0.1932 | 0.1322 | -0.3254 | 0.0451 | 0.0% | 0/46 |
| 4 | 3.80 | 3.53 | 46 | -0.0816 | 0.3690 | -0.4506 | 0.0514 | 2.2% | 1/45 |
| 5 | 3.53 | 3.32 | 46 | 0.1297 | 0.4661 | -0.3364 | 0.0448 | 4.3% | 2/44 |
| 6 | 3.31 | 3.16 | 44 | -0.3544 | -0.1346 | -0.2198 | 0.0266 | 4.5% | 2/42 |
| 7 | 3.14 | 3.02 | 45 | 0.2389 | 0.4402 | -0.2012 | 0.0387 | 11.1% | 5/40 |
| 8 | 3.01 | 2.89 | 46 | -0.4545 | -0.2572 | -0.1973 | 0.0276 | 15.2% | 7/39 |
| 9 | 2.88 | 2.79 | 43 | -0.4664 | -0.2437 | -0.2226 | 0.0392 | 14.0% | 6/37 |
| 10 | 2.79 | 2.70 | 45 | -0.6087 | -0.4608 | -0.1479 | 0.0253 | 17.8% | 8/37 |
| 11 | 2.70 | 2.62 | 45 | -0.3627 | -0.2237 | -0.1391 | 0.0250 | 8.9% | 4/41 |
| 12 | 2.62 | 2.55 | 44 | -0.4188 | -0.3545 | -0.0643 | 0.0312 | 31.8% | 14/30 |
| 13 | 2.55 | 2.49 | 47 | -0.3601 | -0.2756 | -0.0845 | 0.0312 | 23.4% | 11/36 |
| 14 | 2.49 | 2.43 | 43 | -0.6115 | -0.5787 | -0.0329 | 0.0262 | 51.2% | 22/21 |
| 15 | 2.43 | 2.38 | 43 | 0.0250 | -0.0474 | +0.0725 | 0.0437 | 51.2% | 22/21 |
| 16 | 2.38 | 2.33 | 44 | -0.2699 | -0.3326 | +0.0627 | 0.0761 | 43.2% | 19/25 |
| 17 | 2.32 | 2.28 | 47 | -0.1212 | -0.1762 | +0.0551 | 0.0508 | 38.3% | 18/29 |
| 18 | 2.28 | 2.24 | 48 | -0.1720 | -0.2435 | +0.0716 | 0.0473 | 41.7% | 20/28 |
| 19 | 2.24 | 2.20 | 44 | -0.1449 | -0.2048 | +0.0599 | 0.0529 | 45.5% | 20/24 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-13, 1, 8) | 4.90 | 286163.9 | 2820.0 | 101.5 | 610.3 | 650.1 | 6.878 | 9.884 | -3.006 |
| (27, 7, 12) | 2.34 | 11715.6 | 148.6 | 78.8 | 127.0 | 116.2 | 3.749 | 0.901 | +2.848 |
| (14, 2, 4) | 5.33 | 185493.8 | 2286.0 | 81.1 | 496.4 | 534.3 | 5.002 | 7.675 | -2.673 |
| (9, 5, 4) | 5.27 | 124478.5 | 2158.5 | 57.7 | 404.3 | 440.3 | 3.214 | 5.704 | -2.490 |
| (1, 7, 2) | 4.96 | 79170.5 | 2055.1 | 38.5 | 319.0 | 350.8 | 1.908 | 3.946 | -2.037 |
| (2, 2, 14) | 4.11 | 124622.5 | 1711.6 | 72.8 | 400.3 | 426.5 | 3.747 | 5.550 | -1.803 |
| (-14, 2, 12) | 3.78 | 72266.5 | 571.1 | 126.6 | 304.6 | 329.4 | 1.850 | 3.523 | -1.672 |
| (5, 9, 4) | 3.70 | 52203.8 | 1116.3 | 46.8 | 258.2 | 282.1 | 1.330 | 2.915 | -1.586 |
| (30, 2, 14) | 2.32 | 10907.6 | 259.5 | 42.0 | 119.1 | 111.6 | 2.404 | 0.912 | +1.492 |
| (-9, 5, 4) | 5.30 | 73539.8 | 1570.0 | 46.8 | 310.2 | 336.8 | 1.888 | 3.351 | -1.462 |
| (14, 4, 0) | 5.00 | 55763.6 | 801.0 | 69.6 | 266.2 | 292.3 | 1.231 | 2.678 | -1.447 |
| (-16, 4, 12) | 3.38 | 80054.6 | 917.4 | 87.3 | 321.2 | 339.8 | 3.765 | 5.103 | -1.338 |
| (-8, 8, 6) | 3.78 | 25716.6 | 885.3 | 29.0 | 187.1 | 208.2 | 0.795 | 2.090 | -1.295 |
| (-18, 4, 10) | 3.43 | 53950.7 | 964.9 | 55.9 | 264.3 | 281.9 | 2.613 | 3.879 | -1.266 |
| (0, 4, 12) | 4.33 | 80890.8 | 621.6 | 130.1 | 322.0 | 346.3 | 1.508 | 2.766 | -1.258 |
| (-14, 4, 2) | 4.94 | 125909.8 | 1932.0 | 65.2 | 404.9 | 429.4 | 3.272 | 4.506 | -1.234 |
| (32, 4, 12) | 2.25 | 3048.8 | 165.5 | 18.4 | 69.8 | 68.1 | 2.684 | 1.521 | +1.164 |
| (2, 10, 4) | 3.42 | 30854.4 | 510.8 | 60.4 | 201.7 | 217.5 | 1.741 | 2.871 | -1.130 |
| (17, 5, 0) | 4.08 | 57159.6 | 1145.6 | 49.9 | 271.8 | 291.0 | 1.925 | 3.039 | -1.114 |
| (-12, 4, 8) | 4.46 | 75875.1 | 1542.9 | 49.2 | 313.1 | 336.3 | 1.398 | 2.511 | -1.113 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `true_model` | 2.09 | 0.9997 | 11.120828743810849 | 15.070314747558738 | 1.000 |
| `drifted_i` | 6.33 | 0.9988 | 8.285002301397121 | 9.027584986864827 | 1.000 |