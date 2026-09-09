# Held-Out Log-Likelihood Comparison: `recovered_i` vs `shaken`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.4852` nats/refl
- **Model B NLL**: `0.0568` nats/refl
- **Difference (Gain $\Delta$)**: `+0.4285` nats/refl (`-0.4285` nats NLL reduction)
- **Estimated Log Bayes Factor**: `+385.61` nats
- **Uncertainty**: Bootstrap SE = `0.0323` (95% CI: `[+0.3603, +0.4875]`) | Naive SE = `0.0301` (Ratio: `1.07`x)
- **Win Fraction $P(d_h > 0)$**: `79.3%` (714 wins, 186 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `1.48e-73`
- **Wilcoxon Signed-Rank $p$-value**: `1.76e-61`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `-0.3012` | `-0.1272` | `-0.4285` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `-0.6488` | `+0.2204` | `-0.4285` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`shaken`) | Model B (`recovered_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0004` | `-0.0169` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.5916` | `0.0636` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1064` | `-0.0069` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -0.9612 | -1.3422 | +0.3810 | 0.2465 | 51.2% | 22/21 |
| 1 | 6.10 | 4.86 | 46 | 0.2205 | 0.0899 | +0.1306 | 0.1238 | 69.6% | 32/14 |
| 2 | 4.83 | 4.21 | 45 | -0.1227 | -0.3721 | +0.2494 | 0.1017 | 77.8% | 35/10 |
| 3 | 4.20 | 3.81 | 46 | 0.5531 | 0.0444 | +0.5086 | 0.1211 | 91.3% | 42/4 |
| 4 | 3.80 | 3.53 | 46 | 0.3927 | 0.2443 | +0.1484 | 0.1360 | 73.9% | 34/12 |
| 5 | 3.53 | 3.32 | 46 | 1.0000 | 0.3240 | +0.6760 | 0.1991 | 73.9% | 34/12 |
| 6 | 3.31 | 3.16 | 44 | 0.2061 | -0.0339 | +0.2401 | 0.0837 | 79.5% | 35/9 |
| 7 | 3.14 | 3.02 | 45 | 0.8727 | 0.4456 | +0.4271 | 0.1474 | 80.0% | 36/9 |
| 8 | 3.01 | 2.89 | 46 | 0.4105 | -0.0507 | +0.4612 | 0.1079 | 84.8% | 39/7 |
| 9 | 2.88 | 2.79 | 43 | 0.4852 | 0.1093 | +0.3759 | 0.1187 | 74.4% | 32/11 |
| 10 | 2.79 | 2.70 | 45 | 0.1946 | -0.2713 | +0.4659 | 0.0879 | 91.1% | 41/4 |
| 11 | 2.70 | 2.62 | 45 | 0.6260 | 0.0827 | +0.5433 | 0.1143 | 86.7% | 39/6 |
| 12 | 2.62 | 2.55 | 44 | 0.4910 | -0.0992 | +0.5902 | 0.0915 | 86.4% | 38/6 |
| 13 | 2.55 | 2.49 | 47 | 0.6213 | 0.1606 | +0.4607 | 0.1101 | 78.7% | 37/10 |
| 14 | 2.49 | 2.43 | 43 | 0.5049 | -0.0374 | +0.5423 | 0.1108 | 83.7% | 36/7 |
| 15 | 2.43 | 2.38 | 43 | 0.8590 | 0.4452 | +0.4138 | 0.1494 | 86.0% | 37/6 |
| 16 | 2.38 | 2.33 | 44 | 0.8502 | 0.2390 | +0.6112 | 0.1676 | 79.5% | 35/9 |
| 17 | 2.32 | 2.28 | 47 | 0.8093 | 0.3321 | +0.4772 | 0.1015 | 74.5% | 35/12 |
| 18 | 2.28 | 2.24 | 48 | 0.7615 | 0.3570 | +0.4045 | 0.0956 | 77.1% | 37/11 |
| 19 | 2.24 | 2.20 | 44 | 0.8512 | 0.3823 | +0.4690 | 0.1299 | 86.4% | 38/6 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (3, 1, 0) | 22.09 | 21386.8 | 181.5 | 117.8 | 204.2 | 199.9 | 10.147 | 0.637 | +9.510 |
| (20, 4, 7) | 3.47 | 512.2 | 35.5 | 14.4 | 127.9 | 35.0 | 5.249 | -1.078 | +6.327 |
| (-1, 1, 3) | 17.00 | 136.2 | 29.9 | 4.6 | 57.5 | 27.9 | 1.504 | -2.912 | +4.416 |
| (-16, 4, 12) | 3.38 | 80193.7 | 611.6 | 131.1 | 224.9 | 309.5 | 5.744 | 1.438 | +4.305 |
| (27, 7, 12) | 2.34 | 12520.4 | 99.1 | 126.3 | 75.5 | 119.3 | 5.040 | 1.120 | +3.920 |
| (27, 1, 2) | 3.11 | 66716.5 | 854.0 | 78.1 | 213.6 | 272.3 | 5.030 | 1.304 | +3.726 |
| (2, 2, 14) | 4.11 | 122539.2 | 1141.1 | 107.4 | 424.2 | 438.0 | 2.544 | 6.264 | -3.720 |
| (-20, 2, 14) | 2.98 | 15440.2 | 89.5 | 172.5 | 51.6 | 117.2 | 4.307 | 0.602 | +3.705 |
| (-29, 9, 3) | 2.33 | 10.5 | 27.9 | 0.4 | 63.6 | 21.0 | 3.427 | -0.177 | +3.604 |
| (-30, 2, 14) | 2.35 | 1361.7 | 42.0 | 32.4 | 95.7 | 41.2 | 3.491 | 0.010 | +3.481 |
| (-30, 4, 4) | 2.66 | 15479.9 | 176.8 | 87.6 | 79.3 | 131.1 | 4.271 | 0.922 | +3.349 |
| (-12, 8, 19) | 2.41 | 5907.0 | 78.2 | 75.5 | 31.5 | 84.4 | 4.053 | 0.736 | +3.317 |
| (-13, 1, 8) | 4.90 | 286007.4 | 1880.0 | 152.1 | 554.1 | 620.6 | 1.723 | 4.939 | -3.216 |
| (9, 3, 14) | 3.66 | 132444.0 | 1109.9 | 119.3 | 363.8 | 442.4 | 1.712 | 4.889 | -3.178 |
| (16, 10, 13) | 2.46 | 68.7 | 27.2 | 2.5 | 62.7 | 20.0 | 1.937 | -1.089 | +3.027 |
| (-25, 11, 8) | 2.23 | 7110.7 | 81.6 | 87.1 | 56.2 | 92.6 | 4.012 | 1.078 | +2.934 |
| (21, 3, 2) | 3.79 | 56287.7 | 956.4 | 58.9 | 253.4 | 306.1 | 0.893 | 3.754 | -2.861 |
| (-24, 2, 2) | 3.45 | 52949.2 | 770.1 | 68.8 | 182.8 | 247.6 | 3.903 | 1.056 | +2.847 |
| (-19, 1, 20) | 2.49 | -16.7 | 37.9 | -0.4 | 49.8 | 5.5 | 1.958 | -0.814 | +2.773 |
| (15, 7, 4) | 3.64 | 73017.5 | 777.0 | 94.0 | 212.7 | 287.9 | 3.528 | 0.821 | +2.707 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `shaken` | 35.40 | 0.9573 | 125.85603678607154 | 1000.0 | 1.000 |
| `recovered_i` | 15.85 | 0.9914 | 199.09692149524685 | 1000.0 | 1.000 |