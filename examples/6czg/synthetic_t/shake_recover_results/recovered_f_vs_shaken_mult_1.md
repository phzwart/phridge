# Held-Out Log-Likelihood Comparison: `recovered_f` vs `shaken`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 900 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.4852` nats/refl
- **Model B NLL**: `0.0568` nats/refl
- **Difference (Gain $\Delta$)**: `+0.4285` nats/refl (`-0.4285` nats NLL reduction)
- **Estimated Log Bayes Factor**: `+385.63` nats
- **Uncertainty**: Bootstrap SE = `0.0323` (95% CI: `[+0.3605, +0.4875]`) | Naive SE = `0.0301` (Ratio: `1.07`x)
- **Win Fraction $P(d_h > 0)$**: `79.3%` (714 wins, 186 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `1.48e-73`
- **Wilcoxon Signed-Rank $p$-value**: `1.74e-61`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `-0.3012` | `-0.1272` | `-0.4285` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `-0.6488` | `+0.2203` | `-0.4285` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`shaken`) | Model B (`recovered_f`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0004` | `-0.0169` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.5916` | `0.0635` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `-0.1064` | `-0.0068` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -0.9612 | -1.3429 | +0.3817 | 0.2464 | 51.2% | 22/21 |
| 1 | 6.10 | 4.86 | 46 | 0.2205 | 0.0894 | +0.1312 | 0.1238 | 69.6% | 32/14 |
| 2 | 4.83 | 4.21 | 45 | -0.1227 | -0.3724 | +0.2497 | 0.1018 | 77.8% | 35/10 |
| 3 | 4.20 | 3.81 | 46 | 0.5531 | 0.0441 | +0.5089 | 0.1209 | 91.3% | 42/4 |
| 4 | 3.80 | 3.53 | 46 | 0.3927 | 0.2441 | +0.1487 | 0.1359 | 73.9% | 34/12 |
| 5 | 3.53 | 3.32 | 46 | 1.0000 | 0.3241 | +0.6759 | 0.1990 | 73.9% | 34/12 |
| 6 | 3.31 | 3.16 | 44 | 0.2061 | -0.0336 | +0.2398 | 0.0837 | 79.5% | 35/9 |
| 7 | 3.14 | 3.02 | 45 | 0.8727 | 0.4455 | +0.4271 | 0.1474 | 80.0% | 36/9 |
| 8 | 3.01 | 2.89 | 46 | 0.4105 | -0.0506 | +0.4611 | 0.1078 | 84.8% | 39/7 |
| 9 | 2.88 | 2.79 | 43 | 0.4852 | 0.1094 | +0.3757 | 0.1187 | 74.4% | 32/11 |
| 10 | 2.79 | 2.70 | 45 | 0.1946 | -0.2713 | +0.4660 | 0.0878 | 91.1% | 41/4 |
| 11 | 2.70 | 2.62 | 45 | 0.6260 | 0.0827 | +0.5433 | 0.1143 | 86.7% | 39/6 |
| 12 | 2.62 | 2.55 | 44 | 0.4910 | -0.0990 | +0.5900 | 0.0915 | 86.4% | 38/6 |
| 13 | 2.55 | 2.49 | 47 | 0.6213 | 0.1611 | +0.4603 | 0.1101 | 78.7% | 37/10 |
| 14 | 2.49 | 2.43 | 43 | 0.5049 | -0.0371 | +0.5420 | 0.1108 | 83.7% | 36/7 |
| 15 | 2.43 | 2.38 | 43 | 0.8590 | 0.4457 | +0.4133 | 0.1495 | 86.0% | 37/6 |
| 16 | 2.38 | 2.33 | 44 | 0.8502 | 0.2388 | +0.6114 | 0.1675 | 79.5% | 35/9 |
| 17 | 2.32 | 2.28 | 47 | 0.8093 | 0.3323 | +0.4770 | 0.1015 | 74.5% | 35/12 |
| 18 | 2.28 | 2.24 | 48 | 0.7615 | 0.3568 | +0.4047 | 0.0955 | 77.1% | 37/11 |
| 19 | 2.24 | 2.20 | 44 | 0.8512 | 0.3821 | +0.4691 | 0.1298 | 86.4% | 38/6 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (3, 1, 0) | 22.09 | 21386.8 | 181.5 | 117.8 | 204.2 | 199.9 | 10.147 | 0.643 | +9.504 |
| (20, 4, 7) | 3.47 | 512.2 | 35.5 | 14.4 | 127.9 | 35.1 | 5.249 | -1.078 | +6.326 |
| (-1, 1, 3) | 17.00 | 136.2 | 29.9 | 4.6 | 57.5 | 28.0 | 1.504 | -2.912 | +4.417 |
| (-16, 4, 12) | 3.38 | 80193.7 | 611.6 | 131.1 | 224.9 | 309.6 | 5.744 | 1.441 | +4.303 |
| (27, 7, 12) | 2.34 | 12520.4 | 99.1 | 126.3 | 75.5 | 119.3 | 5.040 | 1.120 | +3.920 |
| (27, 1, 2) | 3.11 | 66716.5 | 854.0 | 78.1 | 213.6 | 272.3 | 5.030 | 1.304 | +3.726 |
| (2, 2, 14) | 4.11 | 122539.2 | 1141.1 | 107.4 | 424.2 | 437.9 | 2.544 | 6.254 | -3.710 |
| (-20, 2, 14) | 2.98 | 15440.2 | 89.5 | 172.5 | 51.6 | 117.3 | 4.307 | 0.601 | +3.706 |
| (-29, 9, 3) | 2.33 | 10.5 | 27.9 | 0.4 | 63.6 | 21.0 | 3.427 | -0.175 | +3.602 |
| (-30, 2, 14) | 2.35 | 1361.7 | 42.0 | 32.4 | 95.7 | 41.2 | 3.491 | 0.010 | +3.480 |
| (-30, 4, 4) | 2.66 | 15479.9 | 176.8 | 87.6 | 79.3 | 131.1 | 4.271 | 0.922 | +3.349 |
| (-12, 8, 19) | 2.41 | 5907.0 | 78.2 | 75.5 | 31.5 | 84.4 | 4.053 | 0.736 | +3.317 |
| (-13, 1, 8) | 4.90 | 286007.4 | 1880.0 | 152.1 | 554.1 | 620.7 | 1.723 | 4.945 | -3.223 |
| (9, 3, 14) | 3.66 | 132444.0 | 1109.9 | 119.3 | 363.8 | 442.5 | 1.712 | 4.896 | -3.184 |
| (16, 10, 13) | 2.46 | 68.7 | 27.2 | 2.5 | 62.7 | 20.0 | 1.937 | -1.094 | +3.031 |
| (-25, 11, 8) | 2.23 | 7110.7 | 81.6 | 87.1 | 56.2 | 92.5 | 4.012 | 1.078 | +2.934 |
| (21, 3, 2) | 3.79 | 56287.7 | 956.4 | 58.9 | 253.4 | 306.0 | 0.893 | 3.743 | -2.849 |
| (-24, 2, 2) | 3.45 | 52949.2 | 770.1 | 68.8 | 182.8 | 247.6 | 3.903 | 1.057 | +2.847 |
| (-19, 1, 20) | 2.49 | -16.7 | 37.9 | -0.4 | 49.8 | 5.6 | 1.958 | -0.813 | +2.772 |
| (15, 7, 4) | 3.64 | 73017.5 | 777.0 | 94.0 | 212.7 | 288.0 | 3.528 | 0.822 | +2.707 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `shaken` | 35.40 | 0.9573 | 125.85603678607154 | 1000.0 | 1.000 |
| `recovered_f` | 15.85 | 0.9914 | 199.09691978653197 | 1000.0 | 1.000 |