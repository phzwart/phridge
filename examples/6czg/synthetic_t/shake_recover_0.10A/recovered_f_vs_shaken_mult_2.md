# Held-Out Log-Likelihood Comparison: `recovered_f` vs `shaken`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.0954` nats/refl
- **Model B NLL**: `-0.0815` nats/refl
- **Difference (Gain $\Delta$)**: `+0.1769` nats/refl (`-0.1769` nats NLL reduction)
- **Estimated Log Bayes Factor**: `+159.21` nats
- **Uncertainty**: Bootstrap SE = `0.0678` (95% CI: `[+0.0365, +0.2967]`) | Naive SE = `0.0304` (Ratio: `2.23`x)
- **Win Fraction $P(d_h > 0)$**: `64.3%` (579 wins, 321 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `6.36e-18`
- **Wilcoxon Signed-Rank $p$-value**: `3.19e-16`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `-0.0788` | `-0.0981` | `-0.1769` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `-0.2347` | `+0.0578` | `-0.1769` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`shaken`) | Model B (`recovered_f`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `+0.0053` | `-0.0064` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.1796` | `-0.0736` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0843` | `-0.0079` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.1294 | -1.1260 | -0.0034 | 0.3550 | 27.9% | 12/31 |
| 1 | 6.10 | 4.86 | 46 | -0.1815 | 0.4466 | -0.6280 | 0.1576 | 21.7% | 10/36 |
| 2 | 4.83 | 4.21 | 45 | -0.4982 | -0.2612 | -0.2370 | 0.0841 | 20.0% | 9/36 |
| 3 | 4.20 | 3.81 | 46 | 0.2194 | 0.1135 | +0.1060 | 0.1175 | 50.0% | 23/23 |
| 4 | 3.80 | 3.53 | 46 | 0.0671 | 0.2889 | -0.2218 | 0.1293 | 37.0% | 17/29 |
| 5 | 3.53 | 3.32 | 46 | 0.4016 | 0.4743 | -0.0727 | 0.1682 | 45.7% | 21/25 |
| 6 | 3.31 | 3.16 | 44 | -0.1115 | -0.1173 | +0.0058 | 0.0720 | 54.5% | 24/20 |
| 7 | 3.14 | 3.02 | 45 | 0.4129 | 0.4108 | +0.0021 | 0.1041 | 51.1% | 23/22 |
| 8 | 3.01 | 2.89 | 46 | 0.0146 | -0.1741 | +0.1887 | 0.0754 | 73.9% | 34/12 |
| 9 | 2.88 | 2.79 | 43 | 0.1288 | -0.1732 | +0.3021 | 0.1009 | 72.1% | 31/12 |
| 10 | 2.79 | 2.70 | 45 | -0.2176 | -0.5170 | +0.2994 | 0.0694 | 80.0% | 36/9 |
| 11 | 2.70 | 2.62 | 45 | 0.1946 | -0.1431 | +0.3377 | 0.1066 | 68.9% | 31/14 |
| 12 | 2.62 | 2.55 | 44 | 0.0107 | -0.2305 | +0.2413 | 0.0768 | 79.5% | 35/9 |
| 13 | 2.55 | 2.49 | 47 | 0.2220 | -0.1384 | +0.3604 | 0.0961 | 76.6% | 36/11 |
| 14 | 2.49 | 2.43 | 43 | 0.0841 | -0.5274 | +0.6115 | 0.0815 | 95.3% | 41/2 |
| 15 | 2.43 | 2.38 | 43 | 0.5460 | 0.1342 | +0.4118 | 0.1103 | 88.4% | 38/5 |
| 16 | 2.38 | 2.33 | 44 | 0.3219 | -0.0746 | +0.3965 | 0.1060 | 75.0% | 33/11 |
| 17 | 2.32 | 2.28 | 47 | 0.4281 | -0.0275 | +0.4556 | 0.0746 | 85.1% | 40/7 |
| 18 | 2.28 | 2.24 | 48 | 0.3765 | -0.0742 | +0.4507 | 0.0650 | 93.8% | 45/3 |
| 19 | 2.24 | 2.20 | 44 | 0.5520 | -0.0125 | +0.5646 | 0.0795 | 90.9% | 40/4 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (3, 1, 0) | 22.09 | 20748.0 | 363.0 | 57.2 | 193.2 | 176.0 | 14.374 | -0.306 | +14.679 |
| (20, 4, 7) | 3.47 | 527.6 | 71.0 | 7.4 | 80.6 | 32.8 | 3.664 | -1.222 | +4.886 |
| (-13, 1, 8) | 4.90 | 285707.9 | 3760.0 | 76.0 | 599.9 | 649.3 | 4.107 | 8.626 | -4.520 |
| (14, 2, 4) | 5.33 | 190733.7 | 3048.0 | 62.6 | 472.5 | 533.8 | 1.662 | 5.974 | -4.312 |
| (-18, 8, 15) | 2.52 | 1729.7 | 115.4 | 15.0 | 81.1 | 49.3 | 3.748 | -0.057 | +3.804 |
| (9, 3, 14) | 3.66 | 133223.3 | 2219.8 | 60.0 | 401.8 | 444.6 | 1.649 | 5.360 | -3.711 |
| (-30, 2, 14) | 2.35 | 1114.0 | 84.0 | 13.3 | 69.8 | 42.3 | 3.446 | 0.084 | +3.362 |
| (-5, 9, 1) | 3.82 | 3722.7 | 183.0 | 20.3 | 112.3 | 81.9 | 2.769 | -0.023 | +2.792 |
| (-12, 8, 6) | 3.51 | 49710.6 | 849.8 | 58.5 | 250.0 | 278.7 | 1.333 | 4.123 | -2.790 |
| (16, 10, 13) | 2.46 | 28.6 | 54.4 | 0.5 | 38.5 | 15.5 | 1.429 | -1.193 | +2.622 |
| (-7, 1, 22) | 2.64 | 4716.2 | 155.8 | 30.3 | 111.0 | 88.0 | 3.808 | 1.209 | +2.599 |
| (7, 9, 19) | 2.39 | 1547.8 | 118.8 | 13.0 | 69.4 | 45.6 | 2.542 | -0.040 | +2.582 |
| (0, 4, 12) | 4.33 | 80210.9 | 828.8 | 96.8 | 296.0 | 351.2 | 0.431 | 2.890 | -2.460 |
| (-13, 15, 1) | 2.21 | 0.4 | 56.6 | 0.0 | 30.9 | 14.6 | 2.167 | -0.290 | +2.457 |
| (-29, 9, 3) | 2.33 | -6.0 | 55.8 | -0.1 | 31.3 | 11.6 | 1.451 | -0.953 | +2.403 |
| (1, 7, 2) | 4.96 | 81943.9 | 2740.2 | 29.9 | 314.7 | 355.3 | 1.137 | 3.509 | -2.372 |
| (-14, 4, 2) | 4.94 | 125171.6 | 2576.0 | 48.6 | 392.3 | 428.3 | 1.790 | 4.025 | -2.235 |
| (-18, 4, 10) | 3.43 | 54968.0 | 1286.6 | 42.7 | 262.2 | 285.4 | 1.435 | 3.648 | -2.214 |
| (16, 8, 11) | 2.86 | 271.2 | 59.4 | 4.6 | 48.1 | 23.5 | 0.911 | -1.293 | +2.204 |
| (-16, 4, 12) | 3.38 | 79184.0 | 1223.2 | 64.7 | 290.6 | 329.5 | 1.171 | 3.374 | -2.203 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `shaken` | 18.25 | 0.9880 | 199.07281049923373 | 1000.0 | 1.000 |
| `recovered_f` | 9.27 | 0.9972 | 199.16826138527713 | 976.906380046205 | 1.000 |