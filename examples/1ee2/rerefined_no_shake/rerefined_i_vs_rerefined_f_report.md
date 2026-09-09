# Held-Out Log-Likelihood Comparison: `rerefined_i` vs `rerefined_f`

## 1. Executive Summary

- **Scored Audit Reflections (|A|)**: 1994 (audit set)
- **Nuisance Tune Reflections (|Tune|)**: 997
- **Working Reflections (|Work|)**: 101881
- **Model A NLL**: `0.2273` nats/refl
- **Model B NLL**: `0.2259` nats/refl
- **Difference (Gain $\Delta$)**: `+0.0014` nats/refl (`-0.0014` nats NLL reduction)
- **Estimated Log Bayes Factor**: `+2.85` nats
- **Uncertainty**: Bootstrap SE = `0.0009` (95% CI: `[-0.0005, +0.0031]`) | Naive SE = `0.0009` (Ratio: `1.01`x)
- **Win Fraction $P(d_h > 0)$**: `53.4%` (1065 wins, 929 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `2.49e-03`
- **Wilcoxon Signed-Rank $p$-value**: `1.06e-01`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `-0.0013` | `-0.0001` | `-0.0014` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `-0.0013` | `-0.0001` | `-0.0014` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`rerefined_f`) | Model B (`rerefined_i`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0050` | `0.0050` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `+0.0038` | `+0.0036` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.1421` | `0.1410` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `+0.0852` | `+0.0849` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 15.05 | 4.22 | 96 | 0.3479 | 0.3540 | -0.0061 | 0.0031 | 41.7% | 40/56 |
| 1 | 4.20 | 3.36 | 98 | 0.3027 | 0.3060 | -0.0033 | 0.0030 | 48.0% | 47/51 |
| 2 | 3.34 | 2.92 | 105 | 0.5355 | 0.5366 | -0.0011 | 0.0039 | 56.2% | 59/46 |
| 3 | 2.92 | 2.65 | 94 | 0.1436 | 0.1446 | -0.0010 | 0.0028 | 57.4% | 54/40 |
| 4 | 2.65 | 2.46 | 103 | 0.2035 | 0.2003 | +0.0032 | 0.0034 | 60.2% | 62/41 |
| 5 | 2.46 | 2.32 | 100 | 0.1121 | 0.1102 | +0.0019 | 0.0033 | 57.0% | 57/43 |
| 6 | 2.31 | 2.20 | 101 | 0.1969 | 0.1930 | +0.0039 | 0.0033 | 55.4% | 56/45 |
| 7 | 2.20 | 2.10 | 99 | 0.1378 | 0.1350 | +0.0028 | 0.0036 | 52.5% | 52/47 |
| 8 | 2.10 | 2.02 | 97 | 0.2452 | 0.2424 | +0.0027 | 0.0039 | 51.5% | 50/47 |
| 9 | 2.02 | 1.95 | 104 | 0.0588 | 0.0668 | -0.0081 | 0.0035 | 48.1% | 50/54 |
| 10 | 1.95 | 1.89 | 98 | 0.1812 | 0.1815 | -0.0003 | 0.0045 | 54.1% | 53/45 |
| 11 | 1.89 | 1.83 | 105 | 0.0517 | 0.0552 | -0.0035 | 0.0039 | 57.1% | 60/45 |
| 12 | 1.83 | 1.78 | 103 | 0.1043 | 0.1028 | +0.0014 | 0.0050 | 48.5% | 50/53 |
| 13 | 1.78 | 1.74 | 91 | 0.2071 | 0.2049 | +0.0023 | 0.0041 | 51.6% | 47/44 |
| 14 | 1.74 | 1.70 | 96 | 0.2462 | 0.2414 | +0.0047 | 0.0057 | 53.1% | 51/45 |
| 15 | 1.70 | 1.66 | 110 | 0.2237 | 0.2209 | +0.0028 | 0.0045 | 50.9% | 56/54 |
| 16 | 1.66 | 1.63 | 96 | 0.4563 | 0.4485 | +0.0078 | 0.0051 | 58.3% | 56/40 |
| 17 | 1.63 | 1.60 | 108 | 0.1570 | 0.1526 | +0.0043 | 0.0040 | 56.5% | 61/47 |
| 18 | 1.60 | 1.57 | 88 | 0.3246 | 0.3170 | +0.0076 | 0.0051 | 56.8% | 50/38 |
| 19 | 1.57 | 1.54 | 102 | 0.3387 | 0.3313 | +0.0074 | 0.0056 | 52.9% | 54/48 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-23, 21, 35) | 1.70 | 572.7 | 82.3 | 7.0 | 36.6 | 35.9 | 4.607 | 4.191 | +0.416 |
| (-28, 3, 20) | 1.92 | 536.2 | 95.1 | 5.6 | 37.1 | 37.7 | 2.698 | 2.967 | -0.269 |
| (8, 23, 46) | 1.56 | 16.9 | 16.7 | 1.0 | 11.0 | 10.6 | 1.788 | 1.540 | +0.248 |
| (19, 39, 5) | 1.54 | 316.0 | 33.3 | 9.5 | 13.6 | 13.3 | 1.860 | 2.066 | -0.207 |
| (-12, 33, 33) | 1.69 | 53.9 | 28.3 | 1.9 | 16.2 | 16.6 | 1.736 | 1.942 | -0.205 |
| (1, 32, 42) | 1.56 | 34.4 | 27.7 | 1.2 | 11.1 | 10.6 | 0.908 | 0.706 | +0.202 |
| (20, 13, 28) | 1.78 | 76.6 | 20.2 | 3.8 | 18.8 | 18.3 | 1.576 | 1.376 | +0.200 |
| (22, 25, 6) | 1.82 | 537.8 | 85.6 | 6.3 | 36.1 | 36.5 | 3.130 | 3.324 | -0.194 |
| (-28, 12, 1) | 1.84 | 19.2 | 6.2 | 3.1 | 13.3 | 13.8 | 0.251 | 0.444 | -0.193 |
| (10, 43, 8) | 1.59 | 347.8 | 37.3 | 9.3 | 13.5 | 13.8 | 2.151 | 1.967 | +0.184 |
| (-15, 22, 7) | 2.46 | 2590.9 | 181.0 | 14.3 | 72.2 | 73.1 | 2.231 | 2.411 | -0.181 |
| (-19, 0, 38) | 2.05 | 94.2 | 26.8 | 3.5 | 30.8 | 30.0 | 1.705 | 1.524 | +0.181 |
| (-14, 25, 44) | 1.65 | 421.4 | 20.4 | 20.6 | 28.2 | 28.5 | 2.511 | 2.681 | -0.171 |
| (19, 39, 0) | 1.56 | 335.7 | 31.8 | 10.5 | 23.3 | 23.0 | 1.503 | 1.333 | +0.170 |
| (8, 2, 52) | 1.60 | 103.8 | 15.9 | 6.5 | 1.3 | 1.6 | 3.255 | 3.085 | +0.170 |
| (-10, 18, 17) | 2.94 | 1378.1 | 151.8 | 9.1 | 60.4 | 61.5 | 1.288 | 1.451 | -0.163 |
| (-25, 4, 12) | 2.18 | 203.0 | 18.6 | 10.9 | 29.2 | 28.5 | 0.973 | 0.813 | +0.159 |
| (-10, 37, 15) | 1.81 | 393.1 | 67.5 | 5.8 | 28.0 | 27.5 | 1.284 | 1.125 | +0.159 |
| (12, 12, 32) | 2.06 | 378.5 | 48.6 | 7.8 | 33.2 | 32.6 | 1.676 | 1.522 | +0.154 |
| (2, 1, 46) | 1.93 | 509.3 | 92.0 | 5.5 | 30.1 | 29.3 | 0.709 | 0.557 | +0.153 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `rerefined_f` | 26.92 | 0.9463 | 199.21514291780232 | 344.44254095996763 | 1.000 |
| `rerefined_i` | 26.86 | 0.9466 | 199.2150577093264 | 343.5923592085409 | 1.000 |