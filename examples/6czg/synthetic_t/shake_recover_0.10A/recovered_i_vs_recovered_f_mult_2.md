# Held-Out Log-Likelihood Comparison: `recovered_i` vs `recovered_f`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `-0.0815` nats/refl
- **Model B NLL**: `-0.0807` nats/refl
- **Difference (Gain $\Delta$)**: `-0.0008` nats/refl (`+0.0008` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-0.72` nats
- **Uncertainty**: Bootstrap SE = `0.0004` (95% CI: `[-0.0015, -0.0001]`) | Naive SE = `0.0003` (Ratio: `1.21`x)
- **Win Fraction $P(d_h > 0)$**: `41.2%` (371 wins, 529 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `1.55e-07`
- **Wilcoxon Signed-Rank $p$-value**: `1.34e-04`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.0008` | `-0.0000` | `+0.0008` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.0008` | `-0.0000` | `+0.0008` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`recovered_f`) | Model B (`recovered_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0064` | `-0.0063` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `-0.0736` | `-0.0727` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.0079` | `-0.0080` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -1.1260 | -1.1230 | -0.0030 | 0.0010 | 27.9% | 12/31 |
| 1 | 6.10 | 4.86 | 46 | 0.4466 | 0.4493 | -0.0027 | 0.0028 | 37.0% | 17/29 |
| 2 | 4.83 | 4.21 | 45 | -0.2612 | -0.2625 | +0.0013 | 0.0013 | 53.3% | 24/21 |
| 3 | 4.20 | 3.81 | 46 | 0.1135 | 0.1178 | -0.0043 | 0.0018 | 30.4% | 14/32 |
| 4 | 3.80 | 3.53 | 46 | 0.2889 | 0.2905 | -0.0016 | 0.0021 | 43.5% | 20/26 |
| 5 | 3.53 | 3.32 | 46 | 0.4743 | 0.4754 | -0.0010 | 0.0016 | 50.0% | 23/23 |
| 6 | 3.31 | 3.16 | 44 | -0.1173 | -0.1146 | -0.0028 | 0.0013 | 31.8% | 14/30 |
| 7 | 3.14 | 3.02 | 45 | 0.4108 | 0.4122 | -0.0014 | 0.0016 | 42.2% | 19/26 |
| 8 | 3.01 | 2.89 | 46 | -0.1741 | -0.1733 | -0.0009 | 0.0009 | 43.5% | 20/26 |
| 9 | 2.88 | 2.79 | 43 | -0.1732 | -0.1727 | -0.0005 | 0.0006 | 41.9% | 18/25 |
| 10 | 2.79 | 2.70 | 45 | -0.5170 | -0.5166 | -0.0004 | 0.0006 | 40.0% | 18/27 |
| 11 | 2.70 | 2.62 | 45 | -0.1431 | -0.1430 | -0.0000 | 0.0009 | 42.2% | 19/26 |
| 12 | 2.62 | 2.55 | 44 | -0.2305 | -0.2299 | -0.0006 | 0.0008 | 43.2% | 19/25 |
| 13 | 2.55 | 2.49 | 47 | -0.1384 | -0.1378 | -0.0006 | 0.0009 | 42.6% | 20/27 |
| 14 | 2.49 | 2.43 | 43 | -0.5274 | -0.5264 | -0.0010 | 0.0007 | 32.6% | 14/29 |
| 15 | 2.43 | 2.38 | 43 | 0.1342 | 0.1313 | +0.0029 | 0.0016 | 55.8% | 24/19 |
| 16 | 2.38 | 2.33 | 44 | -0.0746 | -0.0738 | -0.0007 | 0.0011 | 45.5% | 20/24 |
| 17 | 2.32 | 2.28 | 47 | -0.0275 | -0.0282 | +0.0007 | 0.0011 | 42.6% | 20/27 |
| 18 | 2.28 | 2.24 | 48 | -0.0742 | -0.0749 | +0.0007 | 0.0008 | 39.6% | 19/29 |
| 19 | 2.24 | 2.20 | 44 | -0.0125 | -0.0126 | +0.0000 | 0.0009 | 38.6% | 17/27 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (14, 2, 4) | 5.33 | 190733.7 | 3048.0 | 62.6 | 533.8 | 534.4 | 5.974 | 6.039 | -0.066 |
| (9, 3, 14) | 3.66 | 133223.3 | 2219.8 | 60.0 | 444.6 | 444.2 | 5.360 | 5.294 | +0.066 |
| (-13, 1, 8) | 4.90 | 285707.9 | 3760.0 | 76.0 | 649.3 | 649.8 | 8.626 | 8.685 | -0.059 |
| (32, 6, 2) | 2.41 | 13874.4 | 401.8 | 34.5 | 141.9 | 141.6 | 2.988 | 2.930 | +0.058 |
| (2, 2, 14) | 4.11 | 118570.2 | 2282.2 | 52.0 | 436.1 | 436.4 | 7.583 | 7.630 | -0.048 |
| (19, 1, 8) | 3.78 | 42115.3 | 1036.8 | 40.6 | 244.0 | 244.8 | 1.352 | 1.398 | -0.046 |
| (-14, 4, 2) | 4.94 | 125171.6 | 2576.0 | 48.6 | 428.3 | 427.9 | 4.025 | 3.986 | +0.039 |
| (14, 4, 0) | 5.00 | 55363.9 | 1068.0 | 51.8 | 292.6 | 292.2 | 2.525 | 2.487 | +0.038 |
| (17, 5, 0) | 4.08 | 57697.2 | 1527.4 | 37.8 | 294.1 | 294.5 | 2.936 | 2.973 | -0.037 |
| (8, 2, 18) | 3.10 | 2227.2 | 213.4 | 10.4 | 72.5 | 72.2 | 0.758 | 0.722 | +0.036 |
| (21, 3, 2) | 3.79 | 57198.4 | 1912.8 | 29.9 | 297.0 | 297.5 | 2.777 | 2.810 | -0.033 |
| (5, 1, 15) | 3.83 | 33547.1 | 589.2 | 56.9 | 217.3 | 217.8 | 1.485 | 1.518 | -0.033 |
| (-7, 5, 4) | 5.66 | 25612.4 | 417.8 | 61.3 | 195.5 | 196.3 | 0.814 | 0.846 | -0.033 |
| (15, 1, 4) | 5.21 | 56033.7 | 1112.4 | 50.4 | 277.1 | 276.6 | 1.372 | 1.342 | +0.030 |
| (0, 4, 12) | 4.33 | 80210.9 | 828.8 | 96.8 | 351.2 | 350.9 | 2.890 | 2.862 | +0.028 |
| (9, 9, 10) | 3.09 | 16207.3 | 254.2 | 63.8 | 154.7 | 155.0 | 1.562 | 1.589 | -0.027 |
| (3, 5, 2) | 6.67 | 21194.5 | 688.2 | 30.8 | 188.7 | 189.6 | 0.281 | 0.307 | -0.026 |
| (-12, 4, 8) | 4.46 | 74425.5 | 2057.2 | 36.2 | 345.1 | 344.9 | 3.039 | 3.014 | +0.025 |
| (-21, 1, 21) | 2.33 | 796.1 | 73.2 | 10.9 | 38.5 | 38.7 | 0.177 | 0.202 | -0.025 |
| (-18, 4, 10) | 3.43 | 54968.0 | 1286.6 | 42.7 | 285.4 | 285.7 | 3.648 | 3.673 | -0.025 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `recovered_f` | 9.27 | 0.9972 | 199.16826138527713 | 976.906380046205 | 1.000 |
| `recovered_i` | 9.32 | 0.9972 | 199.16823180421954 | 977.3151111708368 | 1.000 |