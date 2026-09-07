# Shake-and-Recover Refinement Benchmark on Synthetic Student-t Intensities (0.10 Å Perturbation)

## Overview
This study evaluates the comparative recovery performance of **Intensity-based likelihood (`ml_i`)** versus **Amplitude-based likelihood (`ml_f`)** starting from a perturbed model under controlled heavy-tailed noise.

### Protocol Details
- **System**: 6CZG (high-resolution model)
- **Perturbation**: 0.10 Å Cartesian RMSD on positions; +/- 10% fractional Gaussian shift on B-factors.
- **Noise Distribution**: Heavy-tailed Student-t error model ($\nu = 7.0$) preserving per-reflection experimental $I/\sigma$.
- **Noise Multipliers**: `[1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 50.0, 100.0]` applied to $\sigma_I$.
- **Cross-Validation**: 3-state holdout scheme (`ReflectionSplit` with 2-fold cross-fitting over the 10% test set).
- **Hyperparameters & Regularization**: Physical unit weighting ($w_{\text{xray}}=N_{\text{work}}, w_{\text{geom}}=1.0$) and hierarchical Student-t ADP prior ($
u_{\text{ADP}}=4, \tau_{1-2} < \tau_{1-3} < \tau_{\text{sphere}}$) optimized via empirical Bayes.

---

## Master Performance Table

| Multiplier | Model | 1D RMSD (Å) | 3D RMSD (Å) | MC RMSD (Å) | Recov MC % | B RMSD (Å²) | $r(B, B_0)$ | $R_{\text{free}}$ (Noisy) | $R_{\text{true}}$ (Amp) | $CC_{\text{true}}$ | Bonds (Å) | Angles (°) | $\Delta\text{NLL}_{I - F}$ | Boot SE | Win % ($I > F$) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1x** | Shaken | `0.1438` | `0.2491` | `0.1741` | --- | `2.86` | `0.9231` | `14.43%` | `18.90%` | `0.9907` | `0.1426` | `8.93` | --- | --- | --- |
| | `ml_f` (Amp) | `0.0777` | `0.1345` | `0.0619` | `+64.4%` | `2.99` | `0.9736` | `7.09%` | `16.51%` | `0.9982` | `0.0046` | `0.84` | *reference* | --- | --- |
| | **`ml_i` (Int)** | **`0.0777`** | **`0.1346`** | **`0.0620`** | **`+64.4%`** | **`3.00`** | **`0.9736`** | `7.10%` | **`16.52%`** | **`0.9982`** | `0.0046` | `0.84` | **`+0.0006`** | `0.0003` | **`41.2%`** |
| **1.5x** | Shaken | `0.1438` | `0.2491` | `0.1741` | --- | `2.86` | `0.9231` | `14.51%` | `18.85%` | `0.9907` | `0.1426` | `8.93` | --- | --- | --- |
| | `ml_f` (Amp) | `0.0778` | `0.1347` | `0.0623` | `+64.2%` | `2.99` | `0.9733` | `7.28%` | `16.45%` | `0.9982` | `0.0046` | `0.84` | *reference* | --- | --- |
| | **`ml_i` (Int)** | **`0.0778`** | **`0.1347`** | **`0.0624`** | **`+64.1%`** | **`3.00`** | **`0.9733`** | `7.29%` | **`16.47%`** | **`0.9982`** | `0.0046` | `0.84` | **`+0.0006`** | `0.0003` | **`42.0%`** |
| **2x** | Shaken | `0.1438` | `0.2491` | `0.1741` | --- | `2.86` | `0.9231` | `14.69%` | `18.89%` | `0.9907` | `0.1426` | `8.93` | --- | --- | --- |
| | `ml_f` (Amp) | `0.0779` | `0.1349` | `0.0629` | `+63.9%` | `3.04` | `0.9724` | `7.56%` | `16.53%` | `0.9982` | `0.0045` | `0.84` | *reference* | --- | --- |
| | **`ml_i` (Int)** | **`0.0779`** | **`0.1350`** | **`0.0630`** | **`+63.8%`** | **`3.06`** | **`0.9724`** | `7.59%` | **`16.55%`** | **`0.9981`** | `0.0046` | `0.84` | **`+0.0008`** | `0.0004` | **`41.2%`** |
| **3x** | Shaken | `0.1438` | `0.2491` | `0.1741` | --- | `2.86` | `0.9231` | `14.75%` | `18.78%` | `0.9907` | `0.1426` | `8.93` | --- | --- | --- |
| | `ml_f` (Amp) | `0.0781` | `0.1353` | `0.0636` | `+63.5%` | `3.04` | `0.9718` | `8.10%` | `16.43%` | `0.9981` | `0.0045` | `0.84` | *reference* | --- | --- |
| | **`ml_i` (Int)** | **`0.0781`** | **`0.1353`** | **`0.0637`** | **`+63.4%`** | **`3.07`** | **`0.9719`** | `8.13%` | **`16.44%`** | **`0.9981`** | `0.0045` | `0.84` | **`+0.0013`** | `0.0006` | **`40.8%`** |
| **5x** | Shaken | `0.1438` | `0.2491` | `0.1741` | --- | `2.86` | `0.9231` | `16.01%` | `18.43%` | `0.9908` | `0.1426` | `8.93` | --- | --- | --- |
| | `ml_f` (Amp) | `0.0407` | `0.0705` | `0.0650` | `+62.7%` | `3.10` | `0.9686` | `9.60%` | `16.03%` | `0.9980` | `0.0046` | `0.84` | *reference* | --- | --- |
| | **`ml_i` (Int)** | **`0.0407`** | **`0.0704`** | **`0.0650`** | **`+62.7%`** | **`3.13`** | **`0.9695`** | `9.55%` | **`16.03%`** | **`0.9981`** | `0.0046` | `0.84` | **`-0.0001`** | `0.0010` | **`51.1%`** |
| **10x** | Shaken | `0.1438` | `0.2491` | `0.1741` | --- | `2.86` | `0.9231` | `18.31%` | `17.80%` | `0.9908` | `0.1426` | `8.93` | --- | --- | --- |
| | `ml_f` (Amp) | `0.0430` | `0.0745` | `0.0696` | `+60.0%` | `3.13` | `0.9594` | `13.51%` | `15.28%` | `0.9977` | `0.0046` | `0.84` | *reference* | --- | --- |
| | **`ml_i` (Int)** | **`0.0608`** | **`0.1053`** | **`0.0683`** | **`+60.8%`** | **`3.22`** | **`0.9643`** | `13.39%` | **`15.17%`** | **`0.9979`** | `0.0046` | `0.84` | **`-0.0008`** | `0.0027` | **`50.7%`** |
| **50x** | Shaken | `0.1438` | `0.2491` | `0.1741` | --- | `2.86` | `0.9231` | `45.11%` | `14.58%` | `0.9906` | `0.1426` | `8.93` | --- | --- | --- |
| | `ml_f` (Amp) | `0.0491` | `0.0851` | `0.0807` | `+53.7%` | `3.54` | `0.8518` | `43.86%` | `10.51%` | `0.9960` | `0.0042` | `0.84` | *reference* | --- | --- |
| | **`ml_i` (Int)** | **`0.0478`** | **`0.0829`** | **`0.0787`** | **`+54.8%`** | **`3.16`** | **`0.8954`** | `43.83%` | **`10.43%`** | **`0.9967`** | `0.0043` | `0.84` | **`+0.0064`** | `0.0043` | **`47.1%`** |
| **100x** | Shaken | `0.1438` | `0.2491` | `0.1741` | --- | `2.86` | `0.9231` | `63.82%` | `14.48%` | `0.9898` | `0.1426` | `8.93` | --- | --- | --- |
| | `ml_f` (Amp) | `0.0494` | `0.0856` | `0.0812` | `+53.4%` | `4.26` | `0.7794` | `61.95%` | `11.28%` | `0.9949` | `0.0042` | `0.85` | *reference* | --- | --- |
| | **`ml_i` (Int)** | **`0.0657`** | **`0.1138`** | **`0.0807`** | **`+53.7%`** | **`3.90`** | **`0.8175`** | `62.29%` | **`11.14%`** | **`0.9956`** | `0.0043` | `0.84` | **`+0.0053`** | `0.0018` | **`42.9%`** |

---

## Statistical Insights

### Noise Multiplier 1x
- **Held-Out NLL Gain ($\Delta$ Gain)**: `+0.0006` nats/reflection (Bootstrap SE: `0.0003`, 95% CI: `[-0.0013, +0.0000]`)
- **Pairwise Win Fraction**: `41.2%` (371 wins vs 529 losses)
- **Sign Tests**: McNemar $p = 1.55e-07$ | Wilcoxon signed-rank $p = 1.59e-03$
- **Two-Way Decomposition**: Structure Term $\Delta_S = +0.0006$, Error-Model Term $\Delta_E = -0.0000$ (Robust: coordinate improvement carries the same sign under both error models)
- **Coordinate Recovery (MC)**: `ml_i` recovered `+64.4%` vs `+64.4%` for `ml_f`.

### Noise Multiplier 1.5x
- **Held-Out NLL Gain ($\Delta$ Gain)**: `+0.0006` nats/reflection (Bootstrap SE: `0.0003`, 95% CI: `[-0.0011, -0.0000]`)
- **Pairwise Win Fraction**: `42.0%` (378 wins vs 522 losses)
- **Sign Tests**: McNemar $p = 1.79e-06$ | Wilcoxon signed-rank $p = 2.03e-05$
- **Two-Way Decomposition**: Structure Term $\Delta_S = +0.0006$, Error-Model Term $\Delta_E = -0.0001$ (Robust: coordinate improvement carries the same sign under both error models)
- **Coordinate Recovery (MC)**: `ml_i` recovered `+64.1%` vs `+64.2%` for `ml_f`.

### Noise Multiplier 2x
- **Held-Out NLL Gain ($\Delta$ Gain)**: `+0.0008` nats/reflection (Bootstrap SE: `0.0004`, 95% CI: `[-0.0015, -0.0001]`)
- **Pairwise Win Fraction**: `41.2%` (371 wins vs 529 losses)
- **Sign Tests**: McNemar $p = 1.55e-07$ | Wilcoxon signed-rank $p = 1.34e-04$
- **Two-Way Decomposition**: Structure Term $\Delta_S = +0.0008$, Error-Model Term $\Delta_E = -0.0000$ (Robust: coordinate improvement carries the same sign under both error models)
- **Coordinate Recovery (MC)**: `ml_i` recovered `+63.8%` vs `+63.9%` for `ml_f`.

### Noise Multiplier 3x
- **Held-Out NLL Gain ($\Delta$ Gain)**: `+0.0013` nats/reflection (Bootstrap SE: `0.0006`, 95% CI: `[-0.0027, -0.0001]`)
- **Pairwise Win Fraction**: `40.8%` (367 wins vs 533 losses)
- **Sign Tests**: McNemar $p = 3.49e-08$ | Wilcoxon signed-rank $p = 7.38e-04$
- **Two-Way Decomposition**: Structure Term $\Delta_S = +0.0014$, Error-Model Term $\Delta_E = -0.0001$ (Robust: coordinate improvement carries the same sign under both error models)
- **Coordinate Recovery (MC)**: `ml_i` recovered `+63.4%` vs `+63.5%` for `ml_f`.

### Noise Multiplier 5x
- **Held-Out NLL Gain ($\Delta$ Gain)**: `-0.0001` nats/reflection (Bootstrap SE: `0.0010`, 95% CI: `[-0.0019, +0.0020]`)
- **Pairwise Win Fraction**: `51.1%` (460 wins vs 440 losses)
- **Sign Tests**: McNemar $p = 5.27e-01$ | Wilcoxon signed-rank $p = 4.25e-01$
- **Two-Way Decomposition**: Structure Term $\Delta_S = -0.0001$, Error-Model Term $\Delta_E = +0.0000$ (Robust: coordinate improvement carries the same sign under both error models)
- **Coordinate Recovery (MC)**: `ml_i` recovered `+62.7%` vs `+62.7%` for `ml_f`.

### Noise Multiplier 10x
- **Held-Out NLL Gain ($\Delta$ Gain)**: `-0.0008` nats/reflection (Bootstrap SE: `0.0027`, 95% CI: `[-0.0045, +0.0058]`)
- **Pairwise Win Fraction**: `50.7%` (456 wins vs 444 losses)
- **Sign Tests**: McNemar $p = 7.14e-01$ | Wilcoxon signed-rank $p = 8.02e-01$
- **Two-Way Decomposition**: Structure Term $\Delta_S = -0.0011$, Error-Model Term $\Delta_E = +0.0004$ (Robust: coordinate improvement carries the same sign under both error models)
- **Coordinate Recovery (MC)**: `ml_i` recovered `+60.8%` vs `+60.0%` for `ml_f`.

### Noise Multiplier 50x
- **Held-Out NLL Gain ($\Delta$ Gain)**: `+0.0064` nats/reflection (Bootstrap SE: `0.0043`, 95% CI: `[-0.0157, +0.0008]`)
- **Pairwise Win Fraction**: `47.1%` (424 wins vs 476 losses)
- **Sign Tests**: McNemar $p = 8.91e-02$ | Wilcoxon signed-rank $p = 8.94e-02$
- **Two-Way Decomposition**: Structure Term $\Delta_S = +0.0064$, Error-Model Term $\Delta_E = +0.0000$ (Robust: coordinate improvement carries the same sign under both error models)
- **Coordinate Recovery (MC)**: `ml_i` recovered `+54.8%` vs `+53.7%` for `ml_f`.

### Noise Multiplier 100x
- **Held-Out NLL Gain ($\Delta$ Gain)**: `+0.0053` nats/reflection (Bootstrap SE: `0.0018`, 95% CI: `[-0.0088, -0.0019]`)
- **Pairwise Win Fraction**: `42.9%` (386 wins vs 514 losses)
- **Sign Tests**: McNemar $p = 2.24e-05$ | Wilcoxon signed-rank $p = 6.16e-04$
- **Two-Way Decomposition**: Structure Term $\Delta_S = +0.0047$, Error-Model Term $\Delta_E = +0.0006$ (Robust: coordinate improvement carries the same sign under both error models)
- **Coordinate Recovery (MC)**: `ml_i` recovered `+53.7%` vs `+53.4%` for `ml_f`.
