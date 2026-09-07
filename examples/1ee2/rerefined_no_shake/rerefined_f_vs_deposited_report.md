# Held-Out Log-Likelihood Comparison: `rerefined_f` vs `deposited_1ee2`

## 1. Executive Summary

- **Scored Test Reflections (|T|)**: 1994 (held-out)
- **Nuisance Tune Reflections (|Tune|)**: 997
- **Working Reflections (|Work|)**: 101881
- **Model A NLL**: `0.1112` nats/refl
- **Model B NLL**: `0.2273` nats/refl
- **Difference (Gain $\Delta$)**: `-0.1161` nats/refl (`+0.1161` nats NLL reduction)
- **Estimated Log Bayes Factor**: `-231.55` nats
- **Uncertainty**: Bootstrap SE = `0.0104` (95% CI: `[-0.1382, -0.0970]`) | Naive SE = `0.0067` (Ratio: `1.56`x)
- **Win Fraction $P(d_h > 0)$**: `23.9%` (477 wins, 1517 losses, 0 ties)
- **McNemar Sign Test $p$-value**: `9.45e-126`
- **Wilcoxon Signed-Rank $p$-value**: `8.13e-98`

## 2. Two-Way Likelihood Decomposition

The comparison is decomposed into structural (atomic coordinates) and error-model ($\sigma_A$, $\nu$, scale) terms:

| Decomposition Line | Structure Term | Error-Model Term | Total $\Delta$ NLL | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Anchored at $\theta_A$** | `+0.1370` | `-0.0209` | `+0.1161` | $B$ vs $A$ under $\theta_A$ + effect of changing error model to $\theta_B$ |
| **Anchored at $\theta_B$** | `+0.1018` | `+0.0143` | `+0.1161` | $B$ vs $A$ under $\theta_B$ + effect of changing error model from $\theta_A$ |

**Structural Robustness**: Robust: coordinate improvement carries the same sign under both error models

## 3. Optimism and Overfitting Diagnosis

| Diagnostic Metric | Model A (`deposited_1ee2`) | Model B (`rerefined_f`) | Description |
| :--- | :---: | :---: | :--- |
| **Free Nuisance Parameters ($p_\theta$)** | `5` | `5` | Total fitted nuisance parameters |
| **Expected In-Sample Optimism ($p_\theta / |\text{tune}|$)** | `0.0050` | `0.0050` | Theoretical shrinkage / optimism |
| **Tune $\to$ Test Gap (Test − Tune NLL)** | `+0.0038` | `+0.0038` | Should be $\approx p_\theta / |\text{tune}|$ if not overfit |
| **Work NLL** | `0.0911` | `0.1421` | NLL on refinement work set |
| **Work $\to$ Test Gap (Test − Work NLL)** | `+0.0201` | `+0.0852` | Generalization gap across refinement |

> **Note**: An improving work NLL together with a worsening test NLL is the hallmark of overfitting, regardless of R-factor or CC values.

## 4. Resolution Shell Breakdown

| Shell | $d_{\text{max}}$ (Å) | $d_{\text{min}}$ (Å) | $N_{\text{test}}$ | NLL(A) | NLL(B) | $\Delta$ Gain | Boot SE | Win % | Wins / Losses |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 15.05 | 4.22 | 96 | 0.2118 | 0.3479 | -0.1360 | 0.0266 | 20.8% | 20/76 |
| 1 | 4.20 | 3.36 | 98 | 0.2149 | 0.3027 | -0.0878 | 0.0261 | 25.5% | 25/73 |
| 2 | 3.34 | 2.92 | 105 | 0.3984 | 0.5355 | -0.1371 | 0.0356 | 21.0% | 22/83 |
| 3 | 2.92 | 2.65 | 94 | -0.0361 | 0.1436 | -0.1797 | 0.0238 | 13.8% | 13/81 |
| 4 | 2.65 | 2.46 | 103 | 0.0026 | 0.2035 | -0.2008 | 0.0300 | 14.6% | 15/88 |
| 5 | 2.46 | 2.32 | 100 | -0.0320 | 0.1121 | -0.1441 | 0.0206 | 16.0% | 16/84 |
| 6 | 2.31 | 2.20 | 101 | 0.0268 | 0.1969 | -0.1701 | 0.0241 | 12.9% | 13/88 |
| 7 | 2.20 | 2.10 | 99 | -0.0210 | 0.1378 | -0.1588 | 0.0197 | 10.1% | 10/89 |
| 8 | 2.10 | 2.02 | 97 | 0.1481 | 0.2452 | -0.0971 | 0.0309 | 22.7% | 22/75 |
| 9 | 2.02 | 1.95 | 104 | -0.1039 | 0.0588 | -0.1627 | 0.0191 | 18.3% | 19/85 |
| 10 | 1.95 | 1.89 | 98 | 0.0828 | 0.1812 | -0.0983 | 0.0261 | 21.4% | 21/77 |
| 11 | 1.89 | 1.83 | 105 | -0.0388 | 0.0517 | -0.0905 | 0.0230 | 27.6% | 29/76 |
| 12 | 1.83 | 1.78 | 103 | 0.0135 | 0.1043 | -0.0908 | 0.0335 | 27.2% | 28/75 |
| 13 | 1.78 | 1.74 | 91 | 0.1114 | 0.2071 | -0.0958 | 0.0313 | 24.2% | 22/69 |
| 14 | 1.74 | 1.70 | 96 | 0.1838 | 0.2462 | -0.0623 | 0.0296 | 29.2% | 28/68 |
| 15 | 1.70 | 1.66 | 110 | 0.1815 | 0.2237 | -0.0422 | 0.0303 | 37.3% | 41/69 |
| 16 | 1.66 | 1.63 | 96 | 0.3789 | 0.4563 | -0.0774 | 0.0446 | 32.3% | 31/65 |
| 17 | 1.63 | 1.60 | 108 | 0.0904 | 0.1570 | -0.0666 | 0.0283 | 32.4% | 35/73 |
| 18 | 1.60 | 1.57 | 88 | 0.2442 | 0.3246 | -0.0804 | 0.0463 | 37.5% | 33/55 |
| 19 | 1.57 | 1.54 | 102 | 0.1941 | 0.3387 | -0.1446 | 0.0361 | 33.3% | 34/68 |

## 5. Top Discordant Reflections (|d_h|)

| $hkl$ | $d$ (Å) | $I_{\text{obs}}$ | $\sigma(I)$ | $I/\sigma$ | $|F_{c,A}|$ | $|F_{c,B}|$ | $\ell_h(A)$ | $\ell_h(B)$ | $d_h$ (Gain) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| (-22, 4, 8) | 2.48 | 2723.6 | 193.6 | 14.1 | 68.8 | 79.3 | 1.735 | 3.518 | -1.783 |
| (-27, 29, 23) | 1.54 | 253.1 | 16.2 | 15.6 | 14.3 | 11.1 | 0.571 | 2.179 | -1.608 |
| (-17, 18, 10) | 2.52 | 670.8 | 50.1 | 13.4 | 35.1 | 46.0 | -0.090 | 1.494 | -1.584 |
| (-26, 1, 35) | 1.82 | 473.2 | 74.2 | 6.4 | 31.8 | 28.6 | 2.500 | 0.955 | +1.546 |
| (-17, 23, 25) | 2.05 | 955.2 | 99.6 | 9.6 | 45.0 | 41.9 | 2.643 | 1.227 | +1.415 |
| (10, 43, 8) | 1.59 | 347.8 | 37.3 | 9.3 | 16.5 | 13.5 | 0.771 | 2.151 | -1.381 |
| (15, 26, 28) | 1.72 | 54.7 | 22.6 | 2.4 | 12.7 | 16.1 | 0.378 | 1.713 | -1.336 |
| (31, 11, 4) | 1.65 | 146.3 | 9.7 | 15.0 | 10.6 | 6.6 | -0.196 | 1.118 | -1.314 |
| (13, 39, 12) | 1.63 | 640.0 | 36.7 | 17.4 | 23.8 | 20.6 | 0.748 | 2.045 | -1.297 |
| (5, 13, 46) | 1.77 | 72.1 | 18.1 | 4.0 | 16.5 | 13.5 | 1.181 | -0.094 | +1.274 |
| (17, 37, 7) | 1.64 | 21.5 | 20.7 | 1.0 | 10.7 | 13.3 | 0.996 | 2.265 | -1.269 |
| (33, 12, 4) | 1.55 | 284.0 | 10.8 | 26.4 | 15.3 | 12.7 | 0.609 | 1.860 | -1.251 |
| (-34, 1, 26) | 1.57 | 998.8 | 16.4 | 60.8 | 32.5 | 28.8 | 0.832 | 2.073 | -1.240 |
| (-15, 9, 20) | 2.94 | 3417.6 | 376.4 | 9.1 | 74.5 | 85.8 | 0.882 | 2.096 | -1.214 |
| (-32, 13, 8) | 1.64 | 391.3 | 12.8 | 30.5 | 17.9 | 14.9 | 0.513 | 1.686 | -1.173 |
| (19, 32, 1) | 1.77 | 1080.4 | 159.9 | 6.8 | 30.5 | 25.4 | 0.799 | 1.943 | -1.145 |
| (-1, 33, 25) | 1.90 | 233.9 | 48.4 | 4.8 | 20.3 | 25.1 | 0.051 | 1.196 | -1.145 |
| (7, 1, 22) | 3.33 | 3389.1 | 236.1 | 14.4 | 73.0 | 85.5 | 0.424 | 1.559 | -1.135 |
| (16, 26, 34) | 1.57 | 308.8 | 21.2 | 14.6 | 23.6 | 22.7 | 2.819 | 1.694 | +1.125 |
| (22, 25, 6) | 1.82 | 537.8 | 85.6 | 6.3 | 36.5 | 36.1 | 4.249 | 3.130 | +1.119 |

## 6. Secondary Reference Metrics

*(Provided for legacy reference only; unweighted R-factors and CC are dominated by strong reflections and do not constitute a proper scoring rule)*

| Model | $R_{\text{free}}$ (%) | $CC_{\text{free}}$ | $\nu$ | $\text{SE}(\nu)$ | Scale $k_F$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `deposited_1ee2` | 24.27 | 0.9525 | 199.2148458113208 | 321.3156706038993 | 1.000 |
| `rerefined_f` | 26.92 | 0.9463 | 199.21514291780232 | 344.44254095996763 | 1.000 |