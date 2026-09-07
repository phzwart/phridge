# Held-Out Log-Likelihood Comparison: `rerefined_deposited_i` vs `deposited_6czg`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.5269` nats/refl
- **Model B NLL**: `0.5360` nats/refl
- **Difference (Gain $\Delta$)**: `-0.0091` nats/refl (`+0.0091` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-8.19` nats
- **Uncertainty**: Bootstrap SE = `0.0104` (95% CI: `[-0.0292, +0.0109]`) | Naive SE = `0.0108` (Ratio: `0.96`x)
- **Win Fraction $P(d_h > 0)$**: `40.3%` (363 wins, 537 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `7.29e-09`
- **Wilcoxon Signed-Rank $p$-value**: `1.11e-04`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.0105` | `-0.0014` | `+0.0091` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.0084` | `+0.0007` | `+0.0091` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`deposited_6czg`) | Model B (`rerefined_deposited_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0014` | `-0.0020` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.5231` | `0.4706` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `+0.0038` | `+0.0654` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -0.8390 | -0.7817 | -0.0573 | 0.0382 | 16.3% | 7/36 |
| 1 | 6.10 | 4.86 | 46 | 0.4423 | 0.4754 | -0.0332 | 0.0430 | 41.3% | 19/27 |
| 2 | 4.83 | 4.21 | 45 | 0.0393 | 0.0612 | -0.0219 | 0.0476 | 31.1% | 14/31 |
| 3 | 4.20 | 3.81 | 46 | 0.4607 | 0.4576 | +0.0031 | 0.0208 | 37.0% | 17/29 |
| 4 | 3.80 | 3.53 | 46 | 0.7685 | 0.7541 | +0.0144 | 0.0530 | 34.8% | 16/30 |
| 5 | 3.53 | 3.32 | 46 | 0.9284 | 0.9565 | -0.0281 | 0.0397 | 28.3% | 13/33 |
| 6 | 3.31 | 3.16 | 44 | 0.5496 | 0.4628 | +0.0868 | 0.0549 | 54.5% | 24/20 |
| 7 | 3.14 | 3.02 | 45 | 1.4329 | 1.3315 | +0.1014 | 0.0629 | 46.7% | 21/24 |
| 8 | 3.01 | 2.89 | 46 | 0.4290 | 0.4557 | -0.0266 | 0.0597 | 45.7% | 21/25 |
| 9 | 2.88 | 2.79 | 43 | 0.6684 | 0.6858 | -0.0174 | 0.0380 | 39.5% | 17/26 |
| 10 | 2.79 | 2.70 | 45 | 0.2783 | 0.2416 | +0.0367 | 0.0488 | 46.7% | 21/24 |
| 11 | 2.70 | 2.62 | 45 | 0.5405 | 0.6183 | -0.0778 | 0.0400 | 40.0% | 18/27 |
| 12 | 2.62 | 2.55 | 44 | 0.4655 | 0.4632 | +0.0023 | 0.0687 | 38.6% | 17/27 |
| 13 | 2.55 | 2.49 | 47 | 0.5095 | 0.5356 | -0.0261 | 0.0313 | 48.9% | 23/24 |
| 14 | 2.49 | 2.43 | 43 | 0.2470 | 0.2562 | -0.0092 | 0.0393 | 44.2% | 19/24 |
| 15 | 2.43 | 2.38 | 43 | 0.7808 | 0.7668 | +0.0140 | 0.0231 | 44.2% | 19/24 |
| 16 | 2.38 | 2.33 | 44 | 0.7066 | 0.8101 | -0.1035 | 0.0600 | 45.5% | 20/24 |
| 17 | 2.32 | 2.28 | 47 | 0.9569 | 0.9523 | +0.0046 | 0.0411 | 38.3% | 18/29 |
| 18 | 2.28 | 2.24 | 48 | 0.4733 | 0.4847 | -0.0114 | 0.0716 | 47.9% | 23/25 |
| 19 | 2.24 | 2.20 | 44 | 0.6249 | 0.6585 | -0.0335 | 0.0235 | 36.4% | 16/28 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (8, 12, 16) | 2.25 | 372.1 | 37.1 | 10.0 | 62.0 | 76.1 | 3.606 | 6.077 | -2.472 |
| (28, 0, 0) | 3.03 | 82866.5 | 1460.6 | 56.7 | 175.4 | 202.8 | 9.200 | 7.095 | +2.104 |
| (-8, 12, 14) | 2.36 | 338.5 | 29.4 | 11.5 | 48.2 | 65.4 | 1.189 | 3.145 | -1.955 |
| (9, 13, 0) | 2.61 | 157.7 | 26.4 | 6.0 | 46.2 | 67.6 | 0.466 | 2.394 | -1.928 |
| (-19, 9, 8) | 2.75 | 14480.0 | 198.7 | 72.9 | 51.1 | 64.9 | 6.495 | 4.646 | +1.849 |
| (28, 6, 6) | 2.59 | 1729.0 | 46.5 | 37.2 | 100.1 | 86.4 | 3.555 | 1.982 | +1.573 |
| (-27, 1, 6) | 3.00 | 6769.9 | 112.2 | 60.3 | 127.2 | 152.4 | 1.449 | 2.899 | -1.451 |
| (-19, 7, 4) | 3.27 | 65816.4 | 797.1 | 82.6 | 179.2 | 191.7 | 6.660 | 5.287 | +1.373 |
| (-4, 0, 26) | 2.28 | 46.9 | 49.7 | 0.9 | 57.7 | 27.4 | 0.531 | -0.824 | +1.356 |
| (-7, 7, 2) | 4.61 | 66075.7 | 781.0 | 84.6 | 175.6 | 198.5 | 3.794 | 2.508 | +1.286 |
| (30, 4, 0) | 2.70 | 4380.6 | 82.7 | 53.0 | 103.3 | 129.6 | 1.302 | 2.535 | -1.233 |
| (14, 2, 4) | 5.33 | 126810.0 | 1524.0 | 83.2 | 436.4 | 482.2 | 1.948 | 3.139 | -1.191 |
| (4, 12, 13) | 2.46 | 6027.1 | 91.4 | 65.9 | 53.1 | 40.8 | 2.319 | 3.440 | -1.121 |
| (-21, 3, 2) | 3.80 | 50411.9 | 629.1 | 80.1 | 123.8 | 132.3 | 6.345 | 5.256 | +1.089 |
| (-25, 5, 4) | 3.01 | 4773.0 | 81.0 | 58.9 | 83.2 | 116.2 | 0.305 | 1.387 | -1.081 |
| (0, 4, 12) | 4.33 | 31219.3 | 414.4 | 75.3 | 283.4 | 308.5 | 3.614 | 4.690 | -1.076 |
| (20, 8, 0) | 3.06 | 77200.7 | 929.6 | 83.0 | 185.5 | 192.4 | 11.719 | 10.665 | +1.054 |
| (30, 4, 4) | 2.65 | 37.6 | 31.3 | 1.2 | 43.0 | 60.0 | 0.281 | 1.332 | -1.051 |
| (-5, 9, 4) | 3.71 | 43983.3 | 526.8 | 83.5 | 159.2 | 183.1 | 2.742 | 1.712 | +1.030 |
| (-23, 3, 8) | 3.20 | 22889.6 | 300.5 | 76.2 | 100.9 | 118.2 | 3.029 | 2.005 | +1.024 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `deposited_6czg` | 42.89 | 0.8845 | 4.829139997587939 | 6.78336178322578 | 1.000 |
| `rerefined_deposited_i` | 42.97 | 0.8913 | 4.942072234844168 | 6.576974594557595 | 1.000 |