# Synthetic Student-t Noise Drift Study: Refining from the Ground Truth Model

## Overview
This study evaluates **parameter drift and noise overfitting** when refining the unperturbed ground-truth 6CZG model directly against synthetic Student-t noisy data across increasing noise multipliers (1.0x to 10.0x).

### Key Questions
1. **Coordinate Drift**: How far do atomic positions wander from the true structure as data quality degrades?
2. **B-Factor Drift**: How do atomic displacement parameters adapt to noise, and does the prior preserve true B-factor rank-order?
3. **Ground Truth Fit**: When refining against noisy data, how does the model's agreement with the true noise-free structure factors (`ITRUE`) evolve?
4. **Protocol Comparison**: Does the Intensity target (`ml_i`) resist noise drift better than Amplitude (`ml_f`), especially in weak and negative reflection regimes?

---

## Master Drift Summary Table

| Multiplier | Model | Pos Drift 1D (Å) | MC Drift 3D (Å) | Max Drift (Å) | B RMSD (Å²) | $\Delta B_{\text{mean}}$ (Å²) | $r(B, B_{\text{true}})$ | $R_{\text{free}}$ (Noisy) | $R_{\text{true}}$ (Amp) | $CC_{\text{true}}$ | Bonds (Å) | Angles (°) | $\Delta\text{NLL}_{I - F}$ | Win % ($I > F$) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1x** | True Model | `0.0000` | `0.0000` | `0.0000` | `0.00` | `+0.00` | `1.0000` | `1.22%` | `13.80%` | `0.9998` | `0.0023` | `0.71` | --- | --- |
| | `ml_f` (Amp) | `0.0694` | `0.0309` | `2.3954` | `2.88` | `+2.70` | `0.9889` | `4.53%` | `16.09%` | `0.9990` | `0.0027` | `0.68` | *reference* | --- |
| | **`ml_i` (Int)** | **`0.0694`** | **`0.0310`** | **`2.3956`** | **`2.91`** | **`+2.73`** | **`0.9888`** | `4.57%` | **`16.12%`** | **`0.9990`** | `0.0027` | `0.68` | **`+0.0004`** | **`48.2%`** |
| **1.5x** | True Model | `0.0000` | `0.0000` | `0.0000` | `0.00` | `+0.00` | `1.0000` | `1.79%` | `13.72%` | `0.9997` | `0.0023` | `0.71` | --- | --- |
| | `ml_f` (Amp) | `0.0694` | `0.0311` | `2.3936` | `2.92` | `+2.74` | `0.9884` | `4.84%` | `16.02%` | `0.9990` | `0.0027` | `0.67` | *reference* | --- |
| | **`ml_i` (Int)** | **`0.0694`** | **`0.0312`** | **`2.3936`** | **`2.97`** | **`+2.78`** | **`0.9882`** | `4.88%` | **`16.04%`** | **`0.9990`** | `0.0027` | `0.67` | **`+0.0008`** | **`45.8%`** |
| **2x** | True Model | `0.0000` | `0.0000` | `0.0000` | `0.00` | `+0.00` | `1.0000` | `2.33%` | `13.75%` | `0.9997` | `0.0023` | `0.71` | --- | --- |
| | `ml_f` (Amp) | `0.0695` | `0.0323` | `2.3953` | `2.97` | `+2.78` | `0.9877` | `5.12%` | `16.11%` | `0.9990` | `0.0027` | `0.67` | *reference* | --- |
| | **`ml_i` (Int)** | **`0.0695`** | **`0.0322`** | **`2.3945`** | **`3.02`** | **`+2.83`** | **`0.9875`** | `5.17%` | **`16.12%`** | **`0.9990`** | `0.0027` | `0.67` | **`+0.0001`** | **`51.6%`** |
| **3x** | True Model | `0.0000` | `0.0000` | `0.0000` | `0.00` | `+0.00` | `1.0000` | `3.27%` | `13.58%` | `0.9997` | `0.0023` | `0.71` | --- | --- |
| | `ml_f` (Amp) | `0.0696` | `0.0329` | `2.4030` | `2.98` | `+2.76` | `0.9864` | `6.00%` | `15.95%` | `0.9990` | `0.0026` | `0.68` | *reference* | --- |
| | **`ml_i` (Int)** | **`0.0697`** | **`0.0330`** | **`2.4005`** | **`3.06`** | **`+2.85`** | **`0.9865`** | `6.05%` | **`15.97%`** | **`0.9989`** | `0.0026` | `0.68` | **`+0.0009`** | **`48.2%`** |
| **5x** | True Model | `0.0000` | `0.0000` | `0.0000` | `0.00` | `+0.00` | `1.0000` | `5.52%` | `13.01%` | `0.9998` | `0.0023` | `0.71` | --- | --- |
| | `ml_f` (Amp) | `0.0699` | `0.0351` | `2.3969` | `3.15` | `+2.90` | `0.9829` | `7.64%` | `15.60%` | `0.9989` | `0.0028` | `0.67` | *reference* | --- |
| | **`ml_i` (Int)** | **`0.0697`** | **`0.0342`** | **`2.3925`** | **`3.21`** | **`+2.98`** | **`0.9839`** | `7.62%` | **`15.55%`** | **`0.9989`** | `0.0027` | `0.67` | **`+0.0003`** | **`47.8%`** |
| **10x** | True Model | `0.0000` | `0.0000` | `0.0000` | `0.00` | `+0.00` | `1.0000` | `10.54%` | `11.91%` | `0.9998` | `0.0023` | `0.71` | --- | --- |
| | `ml_f` (Amp) | `0.0708` | `0.0406` | `2.3954` | `3.33` | `+2.92` | `0.9717` | `12.16%` | `14.66%` | `0.9986` | `0.0028` | `0.67` | *reference* | --- |
| | **`ml_i` (Int)** | **`0.0705`** | **`0.0388`** | **`2.3994`** | **`3.37`** | **`+3.06`** | **`0.9777`** | `12.13%` | **`14.44%`** | **`0.9987`** | `0.0027` | `0.68` | **`-0.0002`** | **`46.0%`** |
| **50x** | True Model | `0.0000` | `0.0000` | `0.0000` | `0.00` | `+0.00` | `1.0000` | `43.03%` | `4.82%` | `0.9998` | `0.0023` | `0.71` | --- | --- |
| | `ml_f` (Amp) | `0.0708` | `0.0415` | `2.3980` | `3.58` | `+0.38` | `0.8507` | `43.54%` | `8.76%` | `0.9975` | `0.0026` | `0.67` | *reference* | --- |
| | **`ml_i` (Int)** | **`0.0710`** | **`0.0423`** | **`2.3982`** | **`3.22`** | **`+1.34`** | **`0.9003`** | `43.73%` | **`8.87%`** | **`0.9982`** | `0.0027` | `0.67` | **`+0.0093`** | **`46.6%`** |
| **100x** | True Model | `0.0000` | `0.0000` | `0.0000` | `0.00` | `+0.00` | `1.0000` | `61.84%` | `5.06%` | `0.9993` | `0.0023` | `0.71` | --- | --- |
| | `ml_f` (Amp) | `0.0244` | `0.0440` | `0.3554` | `4.35` | `+0.73` | `0.7733` | `61.89%` | `9.76%` | `0.9963` | `0.0031` | `0.67` | *reference* | --- |
| | **`ml_i` (Int)** | **`0.0245`** | **`0.0441`** | **`0.3564`** | **`3.88`** | **`+0.73`** | **`0.8269`** | `62.11%` | **`9.70%`** | **`0.9973`** | `0.0032` | `0.67` | **`+0.0040`** | **`43.9%`** |

---

## Detailed Noise Case Analysis

### Noise Multiplier 1x
- **Coordinate Drift**: `ml_i` drifted `0.0694 Å` (MC: `0.0310 Å`, Max: `2.3956 Å`) vs `ml_f` `0.0694 Å`.
- **B-Factor Drift**: B-factor RMSD = `2.91 Å²`, correlation with true B = `0.9888`.
- **Fit to Noise-Free Ground Truth (ITRUE)**: Amplitude R-factor vs truth = `16.12%` (true model = `13.80%`). Correlation $CC(I_\text{calc}, I_\text{true}) = 0.9990$.
- **Held-Out NLL Comparison ($I$ vs $F$)**: `Delta NLL = +0.0004 nats/refl` (Bootstrap SE: `0.0006`, Win Fraction: `48.2%`).

### Noise Multiplier 1.5x
- **Coordinate Drift**: `ml_i` drifted `0.0694 Å` (MC: `0.0312 Å`, Max: `2.3936 Å`) vs `ml_f` `0.0694 Å`.
- **B-Factor Drift**: B-factor RMSD = `2.97 Å²`, correlation with true B = `0.9882`.
- **Fit to Noise-Free Ground Truth (ITRUE)**: Amplitude R-factor vs truth = `16.04%` (true model = `13.72%`). Correlation $CC(I_\text{calc}, I_\text{true}) = 0.9990$.
- **Held-Out NLL Comparison ($I$ vs $F$)**: `Delta NLL = +0.0008 nats/refl` (Bootstrap SE: `0.0007`, Win Fraction: `45.8%`).

### Noise Multiplier 2x
- **Coordinate Drift**: `ml_i` drifted `0.0695 Å` (MC: `0.0322 Å`, Max: `2.3945 Å`) vs `ml_f` `0.0695 Å`.
- **B-Factor Drift**: B-factor RMSD = `3.02 Å²`, correlation with true B = `0.9875`.
- **Fit to Noise-Free Ground Truth (ITRUE)**: Amplitude R-factor vs truth = `16.12%` (true model = `13.75%`). Correlation $CC(I_\text{calc}, I_\text{true}) = 0.9990$.
- **Held-Out NLL Comparison ($I$ vs $F$)**: `Delta NLL = +0.0001 nats/refl` (Bootstrap SE: `0.0010`, Win Fraction: `51.6%`).

### Noise Multiplier 3x
- **Coordinate Drift**: `ml_i` drifted `0.0697 Å` (MC: `0.0330 Å`, Max: `2.4005 Å`) vs `ml_f` `0.0696 Å`.
- **B-Factor Drift**: B-factor RMSD = `3.06 Å²`, correlation with true B = `0.9865`.
- **Fit to Noise-Free Ground Truth (ITRUE)**: Amplitude R-factor vs truth = `15.97%` (true model = `13.58%`). Correlation $CC(I_\text{calc}, I_\text{true}) = 0.9989$.
- **Held-Out NLL Comparison ($I$ vs $F$)**: `Delta NLL = +0.0009 nats/refl` (Bootstrap SE: `0.0012`, Win Fraction: `48.2%`).

### Noise Multiplier 5x
- **Coordinate Drift**: `ml_i` drifted `0.0697 Å` (MC: `0.0342 Å`, Max: `2.3925 Å`) vs `ml_f` `0.0699 Å`.
- **B-Factor Drift**: B-factor RMSD = `3.21 Å²`, correlation with true B = `0.9839`.
- **Fit to Noise-Free Ground Truth (ITRUE)**: Amplitude R-factor vs truth = `15.55%` (true model = `13.01%`). Correlation $CC(I_\text{calc}, I_\text{true}) = 0.9989$.
- **Held-Out NLL Comparison ($I$ vs $F$)**: `Delta NLL = +0.0003 nats/refl` (Bootstrap SE: `0.0021`, Win Fraction: `47.8%`).

### Noise Multiplier 10x
- **Coordinate Drift**: `ml_i` drifted `0.0705 Å` (MC: `0.0388 Å`, Max: `2.3994 Å`) vs `ml_f` `0.0708 Å`.
- **B-Factor Drift**: B-factor RMSD = `3.37 Å²`, correlation with true B = `0.9777`.
- **Fit to Noise-Free Ground Truth (ITRUE)**: Amplitude R-factor vs truth = `14.44%` (true model = `11.91%`). Correlation $CC(I_\text{calc}, I_\text{true}) = 0.9987$.
- **Held-Out NLL Comparison ($I$ vs $F$)**: `Delta NLL = -0.0002 nats/refl` (Bootstrap SE: `0.0041`, Win Fraction: `46.0%`).

### Noise Multiplier 50x
- **Coordinate Drift**: `ml_i` drifted `0.0710 Å` (MC: `0.0423 Å`, Max: `2.3982 Å`) vs `ml_f` `0.0708 Å`.
- **B-Factor Drift**: B-factor RMSD = `3.22 Å²`, correlation with true B = `0.9003`.
- **Fit to Noise-Free Ground Truth (ITRUE)**: Amplitude R-factor vs truth = `8.87%` (true model = `4.82%`). Correlation $CC(I_\text{calc}, I_\text{true}) = 0.9982$.
- **Held-Out NLL Comparison ($I$ vs $F$)**: `Delta NLL = +0.0093 nats/refl` (Bootstrap SE: `0.0046`, Win Fraction: `46.6%`).

### Noise Multiplier 100x
- **Coordinate Drift**: `ml_i` drifted `0.0245 Å` (MC: `0.0441 Å`, Max: `0.3564 Å`) vs `ml_f` `0.0244 Å`.
- **B-Factor Drift**: B-factor RMSD = `3.88 Å²`, correlation with true B = `0.8269`.
- **Fit to Noise-Free Ground Truth (ITRUE)**: Amplitude R-factor vs truth = `9.70%` (true model = `5.06%`). Correlation $CC(I_\text{calc}, I_\text{true}) = 0.9973$.
- **Held-Out NLL Comparison ($I$ vs $F$)**: `Delta NLL = +0.0040 nats/refl` (Bootstrap SE: `0.0018`, Win Fraction: `43.9%`).
