# Case study ledger

Append new entries at the bottom. Leave earlier entries as they were written.

## 2026-10-27 — Pipeline in place

44 TDP1 entries in `pdb_ids.txt`. 1QZQ was dropped because RCSB has no experimental data. 8UVZ is B. subtilis DHFR, not a TDP1 reference. Deposited models and structure-factor mmCIFs are in `structures/{id}/`.

Amplitudes are scaled anisotropically onto 8V0C. Sigma-A weighted 2mFo−DFc maps (columns `FWT` / `PHWT`) are warped onto the median cell, grid (120, 250, 480), space group P2₁2₁2₁, in the pose of 6DHU chain A. Dimers stay intact, one PDB per crystal, in `svd_frame/`. Each map is zero-mean and unit-variance before the SVD.

- `fwt_with_water/` — waters kept. Component 1 fraction 0.112, component 2 fraction 0.080.
- `fwt_no_water/` — waters removed. Component 1 fraction 0.119, component 2 fraction 0.083.
- `svd_with_water/` and `svd_no_water/` — the earlier difference-map SVDs, columns `FOFCWT` / `PHFOFCWT`. Left in place.

UMAP of the singular-value-weighted loadings (`s_k * v_ik`), `n_neighbors=10`, `min_dist=0.15`, `random_state=0`. Panels are components 1–4, 1–5, and 1–10, with and without water. Figure: `fwt_weight_umap.png`. On the with-water 1–4 panel the early complexes form one island, and 6N0N sits with them rather than with the other 6-series structures.

## 2026-10-27 — Four clusters from with-water components 1–4

Split the with-water components 1–4 UMAP with k-means, k=4, `n_init=50`, `random_state=0`. Ward linkage gives the same four groups. The centroid of a cluster is the structure nearest the mean of its points. Distance is Euclidean in that UMAP plane.

Each cluster is a directory under `clusters/with_water_1-4/`. `structures/` holds the aligned dimers from `svd_frame/`. `maps/` holds one CCP4 and one MTZ per member. The MTZ keeps that structure's FWT amplitudes and replaces PHWT with the centroid's PHWT. The CCP4 is the sigma-scaled Fourier map of those coefficients. Columns are `FWT` and `PHWT`. Script: `cluster_phases.py`. Assignments: `clusters/with_water_1-4/assignments.txt`.

| Directory | Centroid | n | Distance to center | Members |
|---|---|---|---|---|
| `01_6dji` | 6DJI | 16 | 0.264 | 6DHU, 6DIE, 6DIH, 6DJF, 6DJG, 6DJI, 6DJJ, 6MZ0, 6N0D, 6N0R, 6N17, 6N19, 6W7K, 7UFY, 8V0B, 9B3B |
| `02_1rgu` | 1RGU | 11 | 0.056 | 1MU7, 1MU9, 1NOP, 1RFF, 1RFI, 1RG1, 1RG2, 1RGT, 1RGU, 1RH0, 6N0N |
| `03_7ufz` | 7UFZ | 10 | 0.228 | 6W4R, 6W7J, 6W7L, 7UFZ, 8CVQ, 8CW2, 8UV1, 8UZV, 8UZZ, 8V0C |
| `04_6dje` | 6DJE | 7 | 0.135 | 6DIM, 6DJD, 6DJE, 6DJH, 6MJ5, 6MYZ, 6N0O |

Checked one non-centroid map in each directory: amplitudes match the member, phases match the centroid, and the centroid's own phases are unchanged.

## 2026-10-27 — What the phases are

Every phase set in this calculation is φ from a deposited, refined model. No dataset was phased from an unrefined molecular-replacement start.

`fwt_with_water` gives each crystal its own refined-model phase. The cluster maps then pick one of those phase sets, the centroid's, and apply it to every member of the cluster. For the centroid that phase is its own. For the other members it is the centroid's.

The amplitudes are still 2mFo−DFc coefficients. The member's model remains inside the amplitude through the figure of merit, σA, and |Fc|.

## 2026-10-27 — Cluster SVDs, amplitudes scaled to the centroid

Each cluster's observed amplitudes were anisotropically scaled onto that cluster's centroid dataset, then the with-water FWT maps were recomputed and rephased with the centroid. `maps/` now holds those maps. `scale.txt` in each cluster records R before and after. French–Wilson sets and 6DJH moved onto the common scale (R from about 1.6–1.9 down to 0.12–0.45). A few amplitude sets changed only slightly, and for 6DIE, 6MZ0, 6W7K, 7UFY, 9B3B, 1RH0, 6W4R, 8CW2, and 8UV1 the R after scaling is a little higher than before. 6W7L's b axis is 1.2% longer than 7UFZ, outside the similarity cutoff, so that dataset was placed in the 7UFZ cell before scaling; its R fell from 0.428 to 0.258.

The SVD is in `svd/` under each cluster directory. Columns are `FWT` / `PHWT`. Each map was zero-mean and unit-variance before the factorization. Script: `cluster_svd.py`.

| Directory | Common set | Component 1 | Component 2 |
|---|---|---|---|
| `01_6dji` | 6DJI | 0.923 | 0.020 |
| `02_1rgu` | 1RGU | 0.839 | 0.038 |
| `03_7ufz` | 7UFZ | 0.852 | 0.057 |
| `04_6dje` | 6DJE | 0.910 | 0.035 |

## 2026-10-27 — SVD of the maps with the cluster average removed

The first component of each cluster SVD was the shared density. For each cluster the normalized maps in `maps/` were averaged, that average was subtracted, and the residuals were factored. Output is `residual_svd/` in each cluster directory: `average.ccp4` plus `.mtz`, `residuals/{id}.ccp4` plus `.mtz`, and `component_*.ccp4` plus `.mtz`. Columns are `FWT` / `PHWT`. Subtracting the mean leaves a null component, which is recorded in `loadings.txt` and not written as a map. Script: `cluster_residual_svd.py`.

| Directory | Residual component 1 | Residual component 2 |
|---|---|---|
| `01_6dji` | 0.262 | 0.172 |
| `02_1rgu` | 0.238 | 0.220 |
| `03_7ufz` | 0.390 | 0.152 |
| `04_6dje` | 0.394 | 0.203 |

## 2026-10-27 — His A263 is a rotamer the centroid phases remove

Chain A His 263 Cα moves 0.28 Å across the aligned dimers. NE2 moves 3.58 Å between two rotamers, χ1 near −80° and χ1 near −157°. Both occur inside the same clusters. 6DJI, 7UFZ, and 6DJE, the phase sources for three clusters, are the −80° rotamer.

6N0D is the −157° rotamer at full occupancy. In its own-phase FWT map, NE2 is +2.0σ and ND1 is +2.4σ. In the cluster map phased by 6DJI, those values are 0.0σ and −0.6σ. The residual SVD is of the rephased maps, so this rotamer is not a residual component. The 1-series models are the −157° rotamer and are phased by 1RGU, which has the same rotamer. Their own-phase maps are weak at the ring: 1MU7 NE2 is +0.5σ.

## 2026-10-27 — Why the rotamer vanished, and centroid-model 2mFo−DFc maps

The `maps/` files were not centroid-phased 2mFo−DFc maps. They put the centroid's phase on |FWT| of the member, and that |FWT| already came from the member's own model, m, σA, and Fc. That is not a map in which a feature missing from the phasing model comes back at partial height. For 6N0D the ring fell to −0.6σ at ND1 and 0.0σ at NE2.

The new maps take the member's Fo, scaled onto the centroid dataset, and the centroid model, waters kept, moved onto the member's chain A in the member's cell. Fc, m, and σA come from that model against the member's data. FWT = 2m|Fo| − D|Fc| with the centroid model phase, warped into the SVD frame. Script: `cluster_model_phases.py`. Output is `centroid_model/maps/` and `centroid_model/residual_svd/` in each cluster.

6N0D His A263 at its own rotamer, phased by the 6DJI model: CG +2.9σ, ND1 +1.6σ, CE1 +1.7σ, NE2 +1.8σ.

| Directory | Residual component 1 | Residual component 2 |
|---|---|---|
| `01_6dji` | 0.319 | 0.243 |
| `02_1rgu` | 0.301 | 0.272 |
| `03_7ufz` | 0.405 | 0.231 |
| `04_6dje` | 0.453 | 0.199 |

In `01_6dji`, components 1, 3, and 4 have opposite-sign weights for the χ1 −157° models (6MZ0, 6N0D, 6N0R, 6N19, 6W7K, 7UFY) and the χ1 −80° models. Component 3: −2.1σ at the −157° ring, mean weight −0.12 on those six and +0.07 on the rest.

## 2026-10-27 — Power maps

`power_map.py` writes `power_all.ccp4` and `power_1-4.ccp4` in each `centroid_model/residual_svd/`. Power is `sum_k s_k^2 u_k(x)^2` with unit-norm `u_k`, divided by its mean. Over all components this is the summed squared residual across the cluster at each voxel. `power_*_residues.txt` ranks residues by the mean power at the centroid model's atoms.

In `01_6dji`, `03_7ufz`, and `04_6dje`, power within 1.5 Å of the model is about 2–2.5 times the cell mean, and beyond 5 Å it is about 0.2–0.3. In `02_1rgu`, power is flat everywhere. That led to the placement problem below.

## 2026-10-27 — Placement error in every warped map

The native maps are correct: 1RGU's own FWT map is +2.3σ at its own Cα atoms in its own cell. The warp into the common frame is not. `warp_map` fills one common unit cell by pulling native density through the rigid fit of chain A onto 6DHU A. Points of the molecule that lie outside that box are reached by a common-cell lattice translation. That only matches the native crystal when the fitted rotation is a lattice operation of P2₁2₁2₁.

- 1MU7 through 1RH0, and 6N0N: chain A is the NCS copy of 6DHU A. The fitted rotation is 24–25° away from any lattice operation, and chain B is within 1°. The warped maps are +0.0σ on average at their own Cα atoms.
- The other structures are within about 1° of a lattice operation. A 1° error moves density about 3.4 Å along the 194 Å c axis. Mean density at their own Cα atoms ranges from +3.5σ (6DHU, the pose reference) to −0.1σ (8CVQ). Values near zero: 6W4R, 6W7L, 8CVQ, 6MYZ, 7UFZ.

Every map built with `warp_map` is affected: `fwt_*`, `svd_*`, the UMAP, the four clusters, the cluster SVDs, and the power maps. The 1-series island in the UMAP, and 6N0N joining it, follow from this. The aligned PDBs in `svd_frame/` are correct as coordinates.

## 2026-10-27 — Rerun on a molecule-centred box

Every map is now sampled on one Cartesian box around the 6DHU dimer, through each crystal's own superposition onto 6DHU. The fit uses the Cα atoms of both chains, matched by chain and residue number. The box is 76.8 × 76.8 × 128.0 Å at 0.4 Å spacing, grid (192, 192, 320), written as a P1 cell with its origin at the box corner. The models in `box/models/` and `box/clusters/*/structures/` are in the same coordinates and keep their waters. Maps are zero-mean and unit-variance inside a mask of voxels within 5 Å of a non-water 6DHU atom, which is 29.5% of the box. The SVDs use only the masked voxels. MTZ files are P1 `FWT` / `PHWT` to 1.5 Å. Scripts: `box_frame.py`, `box_pipeline.py`. Map vectors are cached as `structures/{id}/box_*.npy`.

Placement check, mean density at each structure's own Cα atoms: 6DHU +3.3σ, 1RGU +2.3σ (was +0.0σ), 8CVQ +3.2σ (was −0.1σ), 6W4R +2.9σ, 6N0N +3.4σ.

FWT SVD with each crystal's own model, amplitudes scaled onto 8V0C: with water, component 1 is 0.821 and component 2 is 0.070; no water, 0.815 and 0.073. Component 1 is the shared density. Output: `box/fwt_with_water/` and `box/fwt_no_water/`. UMAP with the same settings: `box/fwt_umap.json`, `box/fwt_weight_umap.png`. The 1-series is still a separate island. 6N0N now sits with the other 6-series structures.

Four clusters from with-water components 1–4. k-means and Ward agree. Assignments: `box/clusters/assignments.txt`.

| Directory | Centroid | n | Members |
|---|---|---|---|
| `01_6w7j` | 6W7J | 13 | 6DIE, 6DJJ, 6N0D, 6N0N, 6N0R, 6N19, 6W7J, 7UFY, 7UFZ, 8CVQ, 8V0B, 8V0C, 9B3B |
| `02_6dih` | 6DIH | 12 | 6DHU, 6DIH, 6DIM, 6DJD, 6DJE, 6DJF, 6DJH, 6DJI, 6MJ5, 6MYZ, 6N17, 6W7K |
| `03_1rg1` | 1RG1 | 10 | 1MU7, 1MU9, 1NOP, 1RFF, 1RFI, 1RG1, 1RG2, 1RGT, 1RGU, 1RH0 |
| `04_8uzz` | 8UZZ | 9 | 6DJG, 6MZ0, 6N0O, 6W4R, 6W7L, 8CW2, 8UV1, 8UZV, 8UZZ |

Each cluster has `maps/` (member Fo scaled onto the centroid, centroid model placed in the member crystal, 2mFo−DFc), and `residual_svd/` with `average`, `residuals/`, components, `loadings.txt`, `power_all.ccp4`, `power_1-4.ccp4`, and ranked residue lists. Power is divided by its mean inside the mask.

| Directory | Residual component 1 | Residual component 2 | Highest power |
|---|---|---|---|
| `01_6w7j` | 0.243 | 0.190 | ligand TGM 701 in both chains, then Met A266 |
| `02_6dih` | 0.266 | 0.192 | ligand GJS 701 in both chains, then Thr B261 |
| `03_1rg1` | 0.278 | 0.194 | vanadate 699 in both chains (182 and 163 times the mean), then DNA |
| `04_8uzz` | 0.269 | 0.213 | ligand YA5, then His A263 (8.4) and His B263 (7.0) |

His A263 ranks 3rd of 897 residues in `04_8uzz` and 9th of 879 in `01_6w7j`. Both clusters mix the χ1 −80° and −157° rotamers.

The earlier directories `fwt_*`, `svd_*`, `clusters/`, and `fwt_weight_umap.png` were built with the faulty placement. They are left on disk and should not be used.

## 2026-10-28 — Moorhen loader

Activate the repo venv first with `source .venv/bin/activate` from the repo root; the pipeline scripts need cctbx, torch, and umap from it, while `moorhen_session.py` itself uses only the standard library. `moorhen_session.py <box directory>` writes `moorhen_load.js` into that directory and serves `box/` on `http://localhost:8765/` with CORS headers. In Moorhen, paste the file into Calculate → Interactive scripting. Every map is loaded as a difference map, positive green and negative red, at 3 × its rmsd. MTZ files are read as `FWT` / `PHWT` with the difference flag set. Maps without an MTZ, the power maps, are read from CCP4. The average, component 1, and `power_all` are shown at load; the rest load hidden. Loaders exist for the four clusters and for `box/fwt_with_water` and `box/fwt_no_water`.

Without the script, Moorhen reads `FWT` / `PHWT` as a 2mFo−DFc map, so the MTZ files open as single-colour maps.

## 2026-10-28 — Moorhen loader runs from the browser console

The Interactive scripting box on moorhen.org cannot run any script. Its Run button calls the script API constructor with no arguments, which fails on `null.getState()` and logs only to the browser console. This Moorhen release also builds molecules and maps from the Moorhen instance rather than from `(commandCentre, glRef)`. The loader now finds that instance in the page and calls its public `files.loadFiles` and `files.loadMtz`. Every map is loaded with the difference flag, which `loadMtz` sets for MTZ files and a `_diff.ccp4` file name sets for CCP4 files. Colours are on a 0–255 scale. To run it, open moorhen.org, press F12, paste `moorhen_load.js` into the Console, and press Enter while `moorhen_session.py` is serving. Cluster 01 (13 models, 41 maps) loads in about 100 s; all 41 maps are difference maps, and the average, component 1, and `power_all` show in green and red.

## 2026-10-28 — Moorhen structures as ball and stick

The loader now draws every model as ball and stick (bonds 0.12 Å wide, atom spheres 2.5 × the bond width) with no ribbons. It does this by setting Moorhen's default style to bonds before loading. `--no-maps` writes a loader for the structures alone, and all six loaders are currently written that way. Cluster 01 loads its 13 models in about 12 s, superposed in the box frame.

## 2026-10-28 — Moorhen loaders with maps again

All six loaders were written again with their maps, and models are still drawn as ball and stick. In the browser console, run `fetch("http://localhost:8765/<set>/moorhen_load.js").then(r => r.text()).then(eval)`. Cluster 01 loads 13 models and 41 maps in about 110 s. All 41 are difference maps in green and red, and the average, component 1, and `power_all` are visible at load.

## 2026-10-28 — Own-model phases, average removed, residual SVD

Every map now uses each structure's own model phases. The earlier cluster maps were phased with the centroid model placed in each member crystal; they are kept under `box/clusters/*/centroid_model/`. The earlier whole-set SVDs in `box/fwt_*` removed only each map's mean over its voxels, not the average map, so their component 1 (82%) was essentially the average map. `box_pipeline.py --residuals` writes, for each whole set and each cluster, `residual_svd/` containing the average 2mFo−DFc map, each map minus that average, the SVD of those residuals, and power maps. Cluster member Fo is scaled onto the cluster centroid; the whole sets are scaled onto 8V0C.

| set | residual comp 1 | comp 2 | comp 3 | top power |
|---|---|---|---|---|
| fwt_with_water (44) | 0.397 | 0.067 | 0.063 | Met A533, EDO A702, Phe A548, Gly A523, GHM A701 |
| fwt_no_water (44) | 0.403 | 0.073 | 0.061 | Met A533, Phe A548, Gly A523, Gly A371 |
| 01_6w7j (13) | 0.213 | 0.180 | 0.132 | TGM B701/A701, Thr A600, Pro A461, Thr B261, Met A266 |
| 02_6dih (12) | 0.224 | 0.167 | 0.139 | GJS B701/A701, Thr B261, Met A267, Asp A322 |
| 03_1rg1 (10) | 0.252 | 0.175 | 0.126 | VO4 B699/A699, DNA DT F806/D806 |
| 04_8uzz (9) | 0.283 | 0.179 | 0.136 | YA5 A707/B703, His A263 (11.0), His B263 (8.3) |

The calculation itself was not what made the maps look worse. The mean density at each model's own non-water atoms is about 3σ in both the own-phase maps and the centroid-phase maps (2.5–3.5σ), and the His A263 ring is at 1.6–3.4σ in both. The cause was the Moorhen display: the 2mFo−DFc maps were drawn as difference maps at 3 × rmsd. The loader now draws each structure's 2mFo−DFc map and the average as blue density maps at 1σ inside the mask. Residuals and components remain green/red difference maps at 3 × rmsd. Power maps load hidden.

## 2026-10-28 — Moorhen contour levels

Normal 2mFo−DFc maps, one per structure plus the average, are contoured at 1.25σ. Difference and perturbation maps (residuals, SVD components, power) are contoured at ±4σ. σ is the standard deviation inside the mask: 1 by construction for the 2mFo−DFc maps, and computed from each CCP4 file for the difference maps. Moorhen's own rmsd covers the whole box, including the zero voxels outside the mask, so it was not used.

## 2026-10-28 — Maps on the median cell, one Fourier grid

All 44 crystals are P2₁2₁2₁, with cell edges within 0.8% of the median, 49.95 × 105.15 × 193.79 Å. Each model is placed on 8V0C by the space-group operator and allowed origin shift that best superpose its Cα atoms. The Cα rmsd after placement is 0.22–0.99 Å, and 1.25 Å for 8UV1. The 1-series and 6N0N have chains A and B swapped relative to 8V0C. That label swap is what looked like a 24° NCS rotation in the box fit; with it accounted for, these crystals sit at origin shift (0, ½, ½) or apply a 2₁ operator.

Each crystal's 2mFo−DFc is computed with its own model in that placement, in its own cell, with Fo scaled onto the reference. The coefficients are then indexed on the median cell (180,135 reflections to 1.45 Å) and FFT'd on one grid, (144, 300, 540). No map is interpolated. The mask is 5 Å around the non-water 8V0C dimer, taken across cell edges, and covers 4,909,656 grid points. Maps are normalized inside the mask. Average, residual, and component maps are written as MTZ coefficients built from the member coefficients, so they have no mask edge, and each is scaled to unit σ inside the mask. Power maps are written as CCP4. Outputs are in `cell/`, from `cell_pipeline.py` and `cell_frame.py`. `redo_svd.model_coefficients` is split out of `density_map`, whose output is unchanged. The first MTZs lacked the cell-volume factor that viewers expect, so they appeared 10⁶× too weak in Moorhen; all 396 were rescaled and the writer is fixed.

The UMAP and clusters now use the residual SVD (average removed), with-water components 1–4 weighted by their singular values. On the box, they used the SVD with only each map's mean removed.

| set | n | residual comp 1 | comp 2 | comp 3 | top power |
|---|---|---|---|---|---|
| fwt_with_water | 44 | 0.433 | 0.098 | 0.081 | Thr A281, Met A491, Met A267, Phe A548, Lys B344 |
| fwt_no_water | 44 | 0.438 | 0.102 | 0.084 | Phe A548, Thr A281, Met A267, Asn A203 |
| 01_6djj | 14 | 0.273 | 0.146 | 0.112 | GM7 A701/B701, Met A266, Arg B304 |
| 02_8v0c | 11 | 0.255 | 0.200 | 0.139 | YE3 A706, Met A266, Met A491, Cys A251 |
| 03_1rg1 | 10 | 0.263 | 0.238 | 0.113 | VO4 A699/B699, DNA DT F806/D806 |
| 04_6djd | 9 | 0.349 | 0.161 | 0.133 | GJP A701/B701, Gly A540, Met A267 |

Clusters: 01_6djj = 6DJJ 6DHU 6DIE 6DJF 6MZ0 6N0D 6N0N 6N0R 6N19 6W7K 7UFY 8CW2 8V0B 9B3B; 02_8v0c = 8V0C 6DJG 6MJ5 6W4R 6W7J 6W7L 7UFZ 8CVQ 8UV1 8UZV 8UZZ; 03_1rg1 = the ten 1-series; 04_6djd = 6DJD 6DIH 6DIM 6DJE 6DJH 6DJI 6MYZ 6N0O 6N17. His A263 power ranks: 9th in 04, 11th in 02, 16th in 01, 87th in 03, 57th in the 44-set. In Moorhen, the 6DHU model's atoms sit at 3.5σ in its own map and 1RG1's at 2.6σ. The loader now serves the whole case-study folder, so loader paths start with `box/` or `cell/`.
