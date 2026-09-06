# Held-Out Log-Likelihood Comparison: `rerefined_deposited_i` vs `rerefined_deposited_f`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 900 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 450
- **Working Reflections (|Work|)**: 8148
- **Model A NLL**: `0.5361` nats/refl
- **Model B NLL**: `0.5360` nats/refl
- **Difference (Gain $\Delta$)**: `+0.0002` nats/refl (`-0.0002` nats NLL reduction)
- **Estimated Log Bayes Factor**: `+0.15` nats
- **Uncertainty**: Bootstrap SE = `0.0001` (95% CI: `[-0.0000, +0.0004]`) | Naive SE = `0.0001` (Ratio: `0.98`x)
- **Win Fraction $P(d_h > 0)$**: `55.4%` (499 wins, 401 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `1.21e-03`
- **Wilcoxon Signed-Rank $p$-value**: `3.76e-02`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `-0.0002` | `+0.0000` | `-0.0002` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `-0.0002` | `+0.0000` | `-0.0002` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`rerefined_deposited_f`) | Model B (`rerefined_deposited_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0111` | `0.0111` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `-0.0021` | `-0.0020` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.4707` | `0.4706` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `+0.0655` | `+0.0654` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 22.09 | 6.18 | 43 | -0.7820 | -0.7817 | -0.0003 | 0.0006 | 41.9% | 18/25 |
| 1 | 6.10 | 4.86 | 46 | 0.4758 | 0.4754 | +0.0004 | 0.0008 | 41.3% | 19/27 |
| 2 | 4.83 | 4.21 | 45 | 0.0608 | 0.0612 | -0.0004 | 0.0005 | 48.9% | 22/23 |
| 3 | 4.20 | 3.81 | 46 | 0.4571 | 0.4576 | -0.0005 | 0.0004 | 41.3% | 19/27 |
| 4 | 3.80 | 3.53 | 46 | 0.7547 | 0.7541 | +0.0005 | 0.0008 | 56.5% | 26/20 |
| 5 | 3.53 | 3.32 | 46 | 0.9570 | 0.9565 | +0.0006 | 0.0005 | 58.7% | 27/19 |
| 6 | 3.31 | 3.16 | 44 | 0.4640 | 0.4628 | +0.0012 | 0.0006 | 59.1% | 26/18 |
| 7 | 3.14 | 3.02 | 45 | 1.3323 | 1.3315 | +0.0008 | 0.0006 | 57.8% | 26/19 |
| 8 | 3.01 | 2.89 | 46 | 0.4563 | 0.4557 | +0.0007 | 0.0005 | 65.2% | 30/16 |
| 9 | 2.88 | 2.79 | 43 | 0.6858 | 0.6858 | -0.0000 | 0.0003 | 48.8% | 21/22 |
| 10 | 2.79 | 2.70 | 45 | 0.2411 | 0.2416 | -0.0005 | 0.0003 | 44.4% | 20/25 |
| 11 | 2.70 | 2.62 | 45 | 0.6185 | 0.6183 | +0.0001 | 0.0003 | 62.2% | 28/17 |
| 12 | 2.62 | 2.55 | 44 | 0.4632 | 0.4632 | -0.0000 | 0.0003 | 61.4% | 27/17 |
| 13 | 2.55 | 2.49 | 47 | 0.5357 | 0.5356 | +0.0001 | 0.0002 | 63.8% | 30/17 |
| 14 | 2.49 | 2.43 | 43 | 0.2562 | 0.2562 | -0.0000 | 0.0003 | 55.8% | 24/19 |
| 15 | 2.43 | 2.38 | 43 | 0.7671 | 0.7668 | +0.0004 | 0.0002 | 69.8% | 30/13 |
| 16 | 2.38 | 2.33 | 44 | 0.8098 | 0.8101 | -0.0002 | 0.0004 | 47.7% | 21/23 |
| 17 | 2.32 | 2.28 | 47 | 0.9528 | 0.9523 | +0.0005 | 0.0003 | 63.8% | 30/17 |
| 18 | 2.28 | 2.24 | 48 | 0.4847 | 0.4847 | -0.0001 | 0.0005 | 62.5% | 30/18 |
| 19 | 2.24 | 2.20 | 44 | 0.6588 | 0.6585 | +0.0003 | 0.0002 | 56.8% | 25/19 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (17, 3, 10) | 3.61 | 22327.3 | 308.6 | 72.4 | 91.4 | 91.8 | 2.450 | 2.429 | +0.021 |
| (2, 6, 0) | 5.83 | 973.3 | 27.6 | 35.3 | 146.8 | 146.4 | 2.663 | 2.643 | +0.021 |
| (8, 12, 16) | 2.25 | 372.1 | 37.1 | 10.0 | 76.1 | 76.1 | 6.059 | 6.077 | -0.018 |
| (-14, 4, 16) | 3.00 | 1316.0 | 45.6 | 28.9 | 78.8 | 78.5 | 0.817 | 0.799 | +0.018 |
| (-15, 1, 4) | 5.26 | 8623.6 | 133.3 | 64.7 | 164.7 | 164.2 | 1.355 | 1.338 | +0.018 |
| (0, 4, 12) | 4.33 | 31219.3 | 414.4 | 75.3 | 308.2 | 308.5 | 4.673 | 4.690 | -0.017 |
| (-23, 5, 2) | 3.26 | 18197.1 | 250.1 | 72.8 | 83.0 | 83.3 | 2.618 | 2.602 | +0.016 |
| (20, 8, 0) | 3.06 | 77200.7 | 929.6 | 83.0 | 192.2 | 192.4 | 10.679 | 10.665 | +0.015 |
| (2, 10, 8) | 3.18 | 32348.8 | 399.0 | 81.1 | 130.8 | 131.0 | 3.343 | 3.328 | +0.014 |
| (-9, 5, 4) | 5.30 | 88155.3 | 1046.7 | 84.2 | 278.5 | 279.0 | 2.231 | 2.218 | +0.012 |
| (3, 1, 0) | 22.09 | 9662.1 | 181.5 | 53.2 | 151.3 | 150.7 | -0.044 | -0.055 | +0.011 |
| (8, 4, 14) | 3.59 | 18678.9 | 264.1 | 70.7 | 94.3 | 94.6 | 1.612 | 1.602 | +0.011 |
| (26, 4, 2) | 3.04 | 1242.4 | 44.3 | 28.0 | 90.2 | 90.0 | 1.372 | 1.362 | +0.011 |
| (-1, 5, 4) | 6.37 | 4681.9 | 71.8 | 65.2 | 164.8 | 165.1 | 1.425 | 1.436 | -0.011 |
| (-8, 8, 6) | 3.78 | 48404.5 | 590.2 | 82.0 | 175.4 | 175.2 | 2.405 | 2.416 | -0.010 |
| (-8, 0, 9) | 5.65 | 422.3 | 39.1 | 10.8 | 106.7 | 107.1 | -0.363 | -0.353 | -0.010 |
| (21, 3, 2) | 3.79 | 77579.9 | 956.4 | 81.1 | 255.6 | 255.9 | 2.054 | 2.044 | +0.010 |
| (6, 2, 2) | 10.31 | 74.6 | 10.3 | 7.2 | 116.8 | 117.1 | 0.296 | 0.305 | -0.009 |
| (-5, 9, 4) | 3.71 | 43983.3 | 526.8 | 83.5 | 183.3 | 183.1 | 1.703 | 1.712 | -0.009 |
| (24, 0, 10) | 3.02 | 5069.1 | 134.8 | 37.6 | 183.2 | 183.3 | 3.806 | 3.815 | -0.009 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `rerefined_deposited_f` | 42.98 | 0.8912 | 4.9406114070797225 | 6.575538616589774 | 1.000 |
| `rerefined_deposited_i` | 42.97 | 0.8913 | 4.942072234844168 | 6.576974594557595 | 1.000 |