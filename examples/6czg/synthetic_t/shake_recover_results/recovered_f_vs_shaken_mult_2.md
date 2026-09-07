# Held-Out Log-Likelihood Comparison: `recovered_f` vs `shaken`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.5073` nats/refl
- **Model B NLL**: `0.0833` nats/refl
- **Difference (Gain $\Delta$)**: `+0.4240` nats/refl (`-0.4240` nats NLL reduction)
- **Estimated Log Bayes Factor**: `+381.56` nats
- **Uncertainty**: Bootstrap SE = `0.0297` (95% CI: `[+0.3595, +0.4787]`) | Naive SE = `0.0301` (Ratio: `0.99`x)
- **Win Fraction $P(d_h > 0)$**: `79.0%` (711 wins, 189 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `8.11e-72`
- **Wilcoxon Signed-Rank $p$-value**: `4.62e-62`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `-0.3006` | `-0.1233` | `-0.4240` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `-0.6335` | `+0.2096` | `-0.4240` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`shaken`) | Model B (`recovered_f`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `+0.0010` | `-0.0152` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.6174` | `0.0982` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1101` | `-0.0149` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -0.9306 | -1.3363 | +0.4057 | 0.2631 | 51.2% | 22/21 |
| 1 | 6.10 | 4.86 | 46 | 0.2346 | 0.0903 | +0.1443 | 0.1285 | 69.6% | 32/14 |
| 2 | 4.83 | 4.21 | 45 | -0.1162 | -0.3622 | +0.2460 | 0.0989 | 77.8% | 35/10 |
| 3 | 4.20 | 3.81 | 46 | 0.5309 | 0.0558 | +0.4751 | 0.1300 | 91.3% | 42/4 |
| 4 | 3.80 | 3.53 | 46 | 0.3989 | 0.2362 | +0.1627 | 0.1297 | 73.9% | 34/12 |
| 5 | 3.53 | 3.32 | 46 | 1.0475 | 0.4064 | +0.6411 | 0.1982 | 73.9% | 34/12 |
| 6 | 3.31 | 3.16 | 44 | 0.2204 | -0.0285 | +0.2489 | 0.0785 | 79.5% | 35/9 |
| 7 | 3.14 | 3.02 | 45 | 0.8945 | 0.4593 | +0.4352 | 0.1523 | 80.0% | 36/9 |
| 8 | 3.01 | 2.89 | 46 | 0.4205 | -0.0506 | +0.4711 | 0.1046 | 80.4% | 37/9 |
| 9 | 2.88 | 2.79 | 43 | 0.5091 | 0.1065 | +0.4026 | 0.1163 | 74.4% | 32/11 |
| 10 | 2.79 | 2.70 | 45 | 0.2124 | -0.2621 | +0.4746 | 0.0820 | 91.1% | 41/4 |
| 11 | 2.70 | 2.62 | 45 | 0.6360 | 0.0850 | +0.5510 | 0.1091 | 91.1% | 41/4 |
| 12 | 2.62 | 2.55 | 44 | 0.5543 | -0.0018 | +0.5561 | 0.0901 | 81.8% | 36/8 |
| 13 | 2.55 | 2.49 | 47 | 0.5677 | 0.1523 | +0.4154 | 0.1054 | 76.6% | 36/11 |
| 14 | 2.49 | 2.43 | 43 | 0.4693 | -0.0666 | +0.5359 | 0.1071 | 81.4% | 35/8 |
| 15 | 2.43 | 2.38 | 43 | 0.8826 | 0.4675 | +0.4151 | 0.1511 | 86.0% | 37/6 |
| 16 | 2.38 | 2.33 | 44 | 0.9111 | 0.3398 | +0.5712 | 0.1704 | 75.0% | 33/11 |
| 17 | 2.32 | 2.28 | 47 | 0.8619 | 0.3796 | +0.4823 | 0.0994 | 83.0% | 39/8 |
| 18 | 2.28 | 2.24 | 48 | 0.8110 | 0.4105 | +0.4005 | 0.0893 | 77.1% | 37/11 |
| 19 | 2.24 | 2.20 | 44 | 0.9532 | 0.4988 | +0.4544 | 0.1322 | 84.1% | 37/7 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (3, 1, 0) | 22.09 | 20748.0 | 363.0 | 57.2 | 204.4 | 199.6 | 10.908 | 0.734 | +10.174 |
| (20, 4, 7) | 3.47 | 527.6 | 71.0 | 7.4 | 127.9 | 35.5 | 5.205 | -1.058 | +6.263 |
| (-1, 1, 3) | 17.00 | 46.2 | 59.8 | 0.8 | 57.6 | 28.7 | 2.223 | -2.658 | +4.881 |
| (2, 2, 14) | 4.11 | 118570.2 | 2282.2 | 52.0 | 424.3 | 438.5 | 2.823 | 7.255 | -4.432 |
| (-16, 4, 12) | 3.38 | 79184.0 | 1223.2 | 64.7 | 224.8 | 309.7 | 5.535 | 1.505 | +4.031 |
| (27, 7, 12) | 2.34 | 12356.0 | 198.2 | 62.3 | 75.5 | 118.9 | 4.919 | 1.129 | +3.791 |
| (-30, 2, 14) | 2.35 | 1114.0 | 84.0 | 13.3 | 95.7 | 41.6 | 3.837 | 0.083 | +3.754 |
| (-20, 2, 14) | 2.98 | 15534.1 | 179.0 | 86.8 | 51.6 | 116.7 | 4.319 | 0.647 | +3.671 |
| (27, 1, 2) | 3.11 | 66291.8 | 1708.0 | 38.8 | 213.3 | 272.3 | 4.924 | 1.318 | +3.606 |
| (-13, 1, 8) | 4.90 | 285707.9 | 3760.0 | 76.0 | 554.9 | 621.1 | 1.712 | 5.192 | -3.480 |
| (-12, 8, 19) | 2.41 | 6067.2 | 156.4 | 38.8 | 31.5 | 84.1 | 4.202 | 0.749 | +3.453 |
| (-29, 9, 3) | 2.33 | -6.0 | 55.8 | -0.1 | 63.6 | 21.4 | 3.637 | 0.201 | +3.436 |
| (-30, 4, 4) | 2.66 | 15495.0 | 353.6 | 43.8 | 79.3 | 130.7 | 4.269 | 0.940 | +3.328 |
| (-24, 2, 2) | 3.45 | 54428.5 | 1540.2 | 35.3 | 182.9 | 247.3 | 4.137 | 1.034 | +3.103 |
| (16, 10, 13) | 2.46 | 28.6 | 54.4 | 0.5 | 62.7 | 20.1 | 2.287 | -0.727 | +3.014 |
| (15, 7, 4) | 3.64 | 74074.4 | 1554.0 | 47.7 | 212.7 | 287.1 | 3.651 | 0.806 | +2.846 |
| (9, 3, 14) | 3.66 | 133223.3 | 2219.8 | 60.0 | 364.0 | 441.8 | 1.752 | 4.584 | -2.832 |
| (-25, 11, 8) | 2.23 | 6968.1 | 163.2 | 42.7 | 56.2 | 92.3 | 3.887 | 1.098 | +2.788 |
| (16, 8, 17) | 2.43 | 115.4 | 96.8 | 1.2 | 4.9 | 42.3 | -0.869 | 1.901 | -2.770 |
| (-13, 5, 13) | 3.33 | 12117.1 | 322.0 | 37.6 | 48.1 | 122.1 | 3.151 | 0.398 | +2.754 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `shaken` | 35.56 | 0.9567 | 55.6323169985576 | 638.4082824385907 | 1.000 |
| `recovered_f` | 16.08 | 0.9909 | 199.07228966685224 | 1000.0 | 1.000 |