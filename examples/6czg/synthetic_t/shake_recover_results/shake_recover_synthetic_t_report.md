# Shake-and-Recover Refinement Benchmark on Synthetic Student-t Intensities

## Overview
This study evaluates the comparative recovery performance of **Intensity-based likelihood (`ml_i`)** versus **Amplitude-based likelihood (`ml_f`)** under controlled noise conditions.

### Protocol Details
- **System**: 6CZG (high-resolution model)
- **Perturbation**: 0.20 Å Cartesian RMSD on positions; +/- 20% fractional Gaussian shift on B-factors.
- **Noise Distribution**: Heavy-tailed Student-t error model ($\nu = 7.0$) preserving per-reflection experimental $I/\sigma$.
- **Noise Multipliers**: `[1.0, 1.5, 2.0, 3.0, 5.0, 10.0]` applied to $\sigma_I$.
- **Cross-Validation**: 3-state holdout scheme (`ReflectionSplit` with 2-fold cross-fitting over the 10% test set).
- **Hyperparameters & Regularization**: Physical unit weighting ($w_{\text{xray}}=N_{\text{work}}, w_{\text{geom}}=1.0$) and hierarchical Student-t ADP prior ($
u_{\text{ADP}}=4, \tau_{1-2} < \tau_{1-3} < \tau_{\text{sphere}}$) optimized via empirical Bayes.

---

## Master Performance Table

| Multiplier | Model | Coord RMSD (Å) | MC RMSD (Å) | SC RMSD (Å) | Recovery % | B RMSD (Å²) | $r(B, B_0)$ | $R_{\text{work}}$ | $R_{\text{free}}$ | Bonds (Å) | Angles (°) | $\Delta\text{NLL}_{I - F}$ | Boot SE | Win % ($I > F$) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1x** | Shaken | `0.2484` | `0.2010` | `0.3024` | --- | `5.71` | `0.7746` | `27.44%` | `27.38%` | `0.2865` | `17.39` | --- | --- | --- |
| | `ml_f` (Amp) | `0.1226` | `0.0766` | `0.1669` | `+50.6%` | `2.96` | `0.9178` | `11.72%` | `13.11%` | `0.0092` | `1.19` | *reference* | --- | --- |
| | **`ml_i` (Int)** | **`0.1226`** | **`0.0766`** | **`0.1669`** | **`+50.6%`** | **`2.97`** | **`0.9178`** | `11.72%` | `13.11%` | `0.0092` | `1.19` | **`+0.0000`** | `0.0001` | **`49.7%`** |
| **1.5x** | Shaken | `0.2484` | `0.2010` | `0.3024` | --- | `5.71` | `0.7746` | `27.48%` | `27.40%` | `0.2865` | `17.39` | --- | --- | --- |
| | `ml_f` (Amp) | `0.1227` | `0.0767` | `0.1669` | `+50.6%` | `2.96` | `0.9176` | `11.81%` | `13.19%` | `0.0091` | `1.19` | *reference* | --- | --- |
| | **`ml_i` (Int)** | **`0.1227`** | **`0.0767`** | **`0.1670`** | **`+50.6%`** | **`2.96`** | **`0.9176`** | `11.81%` | `13.20%` | `0.0091` | `1.19` | **`+0.0001`** | `0.0002` | **`49.9%`** |
| **2x** | Shaken | `0.2484` | `0.2010` | `0.3024` | --- | `5.71` | `0.7746` | `27.56%` | `27.64%` | `0.2865` | `17.39` | --- | --- | --- |
| | `ml_f` (Amp) | `0.1227` | `0.0768` | `0.1670` | `+50.6%` | `2.96` | `0.9173` | `11.96%` | `13.35%` | `0.0091` | `1.18` | *reference* | --- | --- |
| | **`ml_i` (Int)** | **`0.1227`** | **`0.0768`** | **`0.1670`** | **`+50.6%`** | **`2.97`** | **`0.9173`** | `11.97%` | `13.35%` | `0.0091` | `1.18` | **`+0.0004`** | `0.0003` | **`48.6%`** |
| **3x** | Shaken | `0.2484` | `0.2010` | `0.3024` | --- | `5.71` | `0.7746` | `27.85%` | `27.48%` | `0.2865` | `17.39` | --- | --- | --- |
| | `ml_f` (Amp) | `0.1229` | `0.0770` | `0.1670` | `+50.5%` | `2.96` | `0.9172` | `12.33%` | `13.79%` | `0.0091` | `1.18` | *reference* | --- | --- |
| | **`ml_i` (Int)** | **`0.1229`** | **`0.0771`** | **`0.1670`** | **`+50.5%`** | **`2.97`** | **`0.9172`** | `12.34%` | `13.80%` | `0.0091` | `1.18` | **`+0.0007`** | `0.0005` | **`46.1%`** |
| **5x** | Shaken | `0.2484` | `0.2010` | `0.3024` | --- | `5.71` | `0.7746` | `28.16%` | `28.43%` | `0.2865` | `17.39` | --- | --- | --- |
| | `ml_f` (Amp) | `0.1233` | `0.0776` | `0.1674` | `+50.4%` | `2.94` | `0.9166` | `13.25%` | `14.72%` | `0.0091` | `1.18` | *reference* | --- | --- |
| | **`ml_i` (Int)** | **`0.1233`** | **`0.0777`** | **`0.1674`** | **`+50.4%`** | **`2.95`** | **`0.9170`** | `13.27%` | `14.74%` | `0.0091` | `1.18` | **`+0.0007`** | `0.0007` | **`46.4%`** |
| **10x** | Shaken | `0.2484` | `0.2010` | `0.3024` | --- | `5.71` | `0.7746` | `29.71%` | `29.71%` | `0.2865` | `17.39` | --- | --- | --- |
| | `ml_f` (Amp) | `0.1245` | `0.0798` | `0.1681` | `+49.9%` | `2.96` | `0.9120` | `16.14%` | `17.81%` | `0.0090` | `1.18` | *reference* | --- | --- |
| | **`ml_i` (Int)** | **`0.1241`** | **`0.0791`** | **`0.1678`** | **`+50.0%`** | **`2.99`** | **`0.9137`** | `16.08%` | `17.74%` | `0.0090` | `1.18` | **`-0.0001`** | `0.0016` | **`50.0%`** |

---

## Statistical Insights

### Noise Multiplier 1x
- **Held-Out NLL Gain ($\Delta$ Gain)**: `+0.0000` nats/reflection (Bootstrap SE: `0.0001`, 95% CI: `[-0.0002, +0.0001]`)
- **Pairwise Win Fraction**: `49.7%` (447 wins vs 453 losses)
- **Sign Tests**: McNemar $p = 8.68e-01$ | Wilcoxon signed-rank $p = 9.75e-01$
- **Two-Way Decomposition**: Structure Term $\Delta_S = +0.0000$, Error-Model Term $\Delta_E = +0.0000$ (Robust: coordinate improvement carries the same sign under both error models)
- **Coordinate Recovery**: `ml_i` achieved `+50.6%` recovery vs `+50.6%` for `ml_f`.

### Noise Multiplier 1.5x
- **Held-Out NLL Gain ($\Delta$ Gain)**: `+0.0001` nats/reflection (Bootstrap SE: `0.0002`, 95% CI: `[-0.0004, +0.0003]`)
- **Pairwise Win Fraction**: `49.9%` (449 wins vs 451 losses)
- **Sign Tests**: McNemar $p = 9.73e-01$ | Wilcoxon signed-rank $p = 9.85e-01$
- **Two-Way Decomposition**: Structure Term $\Delta_S = +0.0001$, Error-Model Term $\Delta_E = -0.0000$ (Robust: coordinate improvement carries the same sign under both error models)
- **Coordinate Recovery**: `ml_i` achieved `+50.6%` recovery vs `+50.6%` for `ml_f`.

### Noise Multiplier 2x
- **Held-Out NLL Gain ($\Delta$ Gain)**: `+0.0004` nats/reflection (Bootstrap SE: `0.0003`, 95% CI: `[-0.0010, +0.0002]`)
- **Pairwise Win Fraction**: `48.6%` (437 wins vs 463 losses)
- **Sign Tests**: McNemar $p = 4.05e-01$ | Wilcoxon signed-rank $p = 3.50e-02$
- **Two-Way Decomposition**: Structure Term $\Delta_S = +0.0005$, Error-Model Term $\Delta_E = -0.0001$ (Robust: coordinate improvement carries the same sign under both error models)
- **Coordinate Recovery**: `ml_i` achieved `+50.6%` recovery vs `+50.6%` for `ml_f`.

### Noise Multiplier 3x
- **Held-Out NLL Gain ($\Delta$ Gain)**: `+0.0007` nats/reflection (Bootstrap SE: `0.0005`, 95% CI: `[-0.0017, +0.0002]`)
- **Pairwise Win Fraction**: `46.1%` (415 wins vs 485 losses)
- **Sign Tests**: McNemar $p = 2.14e-02$ | Wilcoxon signed-rank $p = 6.61e-03$
- **Two-Way Decomposition**: Structure Term $\Delta_S = +0.0007$, Error-Model Term $\Delta_E = +0.0000$ (Robust: coordinate improvement carries the same sign under both error models)
- **Coordinate Recovery**: `ml_i` achieved `+50.5%` recovery vs `+50.5%` for `ml_f`.

### Noise Multiplier 5x
- **Held-Out NLL Gain ($\Delta$ Gain)**: `+0.0007` nats/reflection (Bootstrap SE: `0.0007`, 95% CI: `[-0.0020, +0.0006]`)
- **Pairwise Win Fraction**: `46.4%` (418 wins vs 482 losses)
- **Sign Tests**: McNemar $p = 3.57e-02$ | Wilcoxon signed-rank $p = 2.37e-02$
- **Two-Way Decomposition**: Structure Term $\Delta_S = +0.0005$, Error-Model Term $\Delta_E = +0.0002$ (Robust: coordinate improvement carries the same sign under both error models)
- **Coordinate Recovery**: `ml_i` achieved `+50.4%` recovery vs `+50.4%` for `ml_f`.

### Noise Multiplier 10x
- **Held-Out NLL Gain ($\Delta$ Gain)**: `-0.0001` nats/reflection (Bootstrap SE: `0.0016`, 95% CI: `[-0.0031, +0.0031]`)
- **Pairwise Win Fraction**: `50.0%` (450 wins vs 450 losses)
- **Sign Tests**: McNemar $p = 1.00e+00$ | Wilcoxon signed-rank $p = 5.28e-01$
- **Two-Way Decomposition**: Structure Term $\Delta_S = -0.0001$, Error-Model Term $\Delta_E = -0.0000$ (Robust: coordinate improvement carries the same sign under both error models)
- **Coordinate Recovery**: `ml_i` achieved `+50.0%` recovery vs `+49.9%` for `ml_f`.
