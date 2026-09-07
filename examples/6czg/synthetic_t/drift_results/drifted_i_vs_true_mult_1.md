# Held-Out Log-Likelihood Comparison: `drifted_i` vs `true_model`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `-0.3427` nats/refl
- **Model B NLL**: `-0.1654` nats/refl
- **Difference (Gain $\Delta$)**: `-0.1772` nats/refl (`+0.1772` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-159.51` nats
- **Uncertainty**: Bootstrap SE = `0.0430` (95% CI: `[-0.2691, -0.1020]`) | Naive SE = `0.0122` (Ratio: `3.54`x)
- **Win Fraction $P(d_h > 0)$**: `18.6%` (167 wins, 733 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `3.34e-85`
- **Wilcoxon Signed-Rank $p$-value**: `1.66e-68`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.2395` | `-0.0623` | `+0.1772` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.1457` | `+0.0315` | `+0.1772` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`true_model`) | Model B (`drifted_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0037` | `-0.0023` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `-0.2392` | `-0.1364` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1035` | `-0.0291` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.4315 | -1.1652 | -0.2663 | 0.0280 | 2.3% | 1/42 |
| 1 | 6.10 | 4.86 | 46 | 0.0711 | 0.7409 | -0.6698 | 0.1063 | 0.0% | 0/46 |
| 2 | 4.83 | 4.21 | 45 | -0.6230 | -0.2806 | -0.3424 | 0.0386 | 0.0% | 0/45 |
| 3 | 4.20 | 3.81 | 46 | -0.1966 | 0.1375 | -0.3341 | 0.0433 | 0.0% | 0/46 |
| 4 | 3.80 | 3.53 | 46 | -0.1032 | 0.3333 | -0.4365 | 0.0503 | 0.0% | 0/46 |
| 5 | 3.53 | 3.32 | 46 | 0.1441 | 0.5155 | -0.3714 | 0.0469 | 6.5% | 3/43 |
| 6 | 3.31 | 3.16 | 44 | -0.4083 | -0.1769 | -0.2313 | 0.0251 | 6.8% | 3/41 |
| 7 | 3.14 | 3.02 | 45 | 0.1935 | 0.3802 | -0.1867 | 0.0447 | 15.6% | 7/38 |
| 8 | 3.01 | 2.89 | 46 | -0.4553 | -0.2536 | -0.2018 | 0.0287 | 13.0% | 6/40 |
| 9 | 2.88 | 2.79 | 43 | -0.4898 | -0.2734 | -0.2165 | 0.0387 | 11.6% | 5/38 |
| 10 | 2.79 | 2.70 | 45 | -0.7168 | -0.5562 | -0.1606 | 0.0250 | 11.1% | 5/40 |
| 11 | 2.70 | 2.62 | 45 | -0.3674 | -0.2136 | -0.1538 | 0.0251 | 8.9% | 4/41 |
| 12 | 2.62 | 2.55 | 44 | -0.4573 | -0.3832 | -0.0741 | 0.0293 | 34.1% | 15/29 |
| 13 | 2.55 | 2.49 | 47 | -0.3809 | -0.2768 | -0.1041 | 0.0329 | 23.4% | 11/36 |
| 14 | 2.49 | 2.43 | 43 | -0.6402 | -0.5818 | -0.0584 | 0.0260 | 34.9% | 15/28 |
| 15 | 2.43 | 2.38 | 43 | -0.0180 | -0.0871 | +0.0692 | 0.0496 | 44.2% | 19/24 |
| 16 | 2.38 | 2.33 | 44 | -0.3516 | -0.4057 | +0.0541 | 0.0560 | 45.5% | 20/24 |
| 17 | 2.32 | 2.28 | 47 | -0.1726 | -0.2345 | +0.0619 | 0.0576 | 36.2% | 17/30 |
| 18 | 2.28 | 2.24 | 48 | -0.3085 | -0.3387 | +0.0301 | 0.0375 | 37.5% | 18/30 |
| 19 | 2.24 | 2.20 | 44 | -0.2307 | -0.2982 | +0.0675 | 0.0579 | 40.9% | 18/26 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-13, 1, 8) | 4.90 | 286007.4 | 1880.0 | 152.1 | 610.7 | 653.0 | 7.590 | 10.985 | -3.395 |
| (9, 5, 4) | 5.27 | 126633.4 | 1439.0 | 88.0 | 404.6 | 442.3 | 3.128 | 5.838 | -2.710 |
| (14, 2, 4) | 5.33 | 191022.5 | 1524.0 | 125.3 | 496.8 | 532.5 | 4.500 | 6.836 | -2.336 |
| (1, 7, 2) | 4.96 | 78873.8 | 1370.1 | 57.6 | 319.2 | 350.2 | 2.108 | 4.134 | -2.025 |
| (27, 7, 12) | 2.34 | 12520.4 | 99.1 | 126.3 | 127.1 | 116.3 | 2.595 | 0.748 | +1.846 |
| (30, 2, 14) | 2.32 | 10789.6 | 173.0 | 62.4 | 119.2 | 111.7 | 2.741 | 0.932 | +1.810 |
| (2, 2, 14) | 4.11 | 122539.2 | 1141.1 | 107.4 | 400.6 | 427.0 | 4.566 | 6.348 | -1.782 |
| (-14, 2, 12) | 3.78 | 72853.3 | 380.7 | 191.4 | 304.8 | 329.5 | 1.800 | 3.441 | -1.641 |
| (14, 4, 0) | 5.00 | 55794.1 | 534.0 | 104.5 | 266.4 | 293.7 | 1.302 | 2.899 | -1.597 |
| (-16, 4, 12) | 3.38 | 80193.7 | 611.6 | 131.1 | 321.5 | 341.4 | 4.017 | 5.552 | -1.535 |
| (5, 9, 4) | 3.70 | 52903.9 | 744.2 | 71.1 | 258.4 | 281.3 | 1.245 | 2.707 | -1.461 |
| (35, 1, 10) | 2.23 | 7867.0 | 108.4 | 72.6 | 101.2 | 95.4 | 2.302 | 0.869 | +1.433 |
| (0, 4, 12) | 4.33 | 80152.9 | 414.4 | 193.4 | 322.3 | 346.6 | 1.643 | 2.921 | -1.277 |
| (32, 6, 2) | 2.41 | 13761.9 | 200.9 | 68.5 | 133.9 | 130.7 | 2.646 | 1.394 | +1.252 |
| (8, 2, 0) | 9.10 | 78201.9 | 374.4 | 208.9 | 319.1 | 354.4 | 0.596 | 1.821 | -1.225 |
| (6, 14, 10) | 2.29 | 8012.3 | 146.5 | 54.7 | 101.6 | 93.3 | 1.877 | 0.661 | +1.216 |
| (-14, 4, 2) | 4.94 | 126721.0 | 1288.0 | 98.4 | 405.2 | 429.5 | 3.404 | 4.581 | -1.176 |
| (-9, 5, 4) | 5.30 | 73520.8 | 1046.7 | 70.2 | 310.4 | 333.5 | 2.034 | 3.179 | -1.145 |
| (-8, 8, 6) | 3.78 | 27261.2 | 590.2 | 46.2 | 187.2 | 207.2 | 0.515 | 1.640 | -1.124 |
| (-12, 4, 8) | 4.46 | 74634.5 | 1028.6 | 72.6 | 313.4 | 336.2 | 1.598 | 2.705 | -1.107 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `true_model` | 1.51 | 0.9998 | 199.18862156908295 | 1000.0 | 1.000 |
| `drifted_i` | 6.14 | 0.9990 | 199.17395004567555 | 1000.0 | 1.000 |