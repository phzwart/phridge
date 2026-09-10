"""Windowed omit map coefficients for the ``ml_i`` likelihood.

After refinement, for each real-space window covering the ASU atom set, compute
reciprocal-space map coefficients *as if* that window's atoms were absent from
the model, with a closed-form residual-variance adjustment. The worker returns
per-window complex coefficients on the full hkl list; the Phenix-side client
stitches them in real space and writes one MTZ of composite Fourier coefficients.

Caveats (fixed across all windows)
----------------------------------
- Bulk solvent and scales (``k_iso``, ``k_aniso``, ``k_mask``) are those of the
  **full** refined model. Removing atoms does **not** recompute the solvent mask.
- Globally estimated ``σ_A`` and Wilson ``Σ`` are held fixed as the base error
  model; only the residual intensity variance is increased by the omitted
  scattering power (encoded as an effective ``σ_{A,w}`` for the existing Rice
  maps API — see below).
- Atoms were refined in the presence of the omitted region (phase-memory bias).
  Do not over-interpret marginal density at the omit site as fully unbiased.

Variance bookkeeping (repo conventions)
---------------------------------------
Maps use ``alpha`` = ``σ_A`` and ``beta`` = Wilson ``Σ`` (not cctbx residual β).
Residual β is ``Σ(1 − σ_A²)``. For window ``w``:

$$
\\beta^{\\mathrm{res}}_{h,w} = \\Sigma_h\\,(1-\\sigma_{A,h}^2) + \\sigma_{A,h}^2\\,\\Sigma^{\\mathrm{omit}}_{h,w}
$$

$$
\\sigma_{A,h,w} = \\sigma_{A,h}\\sqrt{\\max\\bigl(0,\\, 1 - \\Sigma^{\\mathrm{omit}}_{h,w}/\\Sigma_h\\bigr)}
$$

with ``E_C`` from ``|F^{omit}| / √(ε Σ)`` and the same ``Σ``. Omitting all
atomic scattering drives ``σ_{A,w} → 0`` (Wilson prior).

Structure factors use linearity: ``ΔF_w`` from an occupancy-masked FFT of the
window atoms; ``F^{omit,model}_w = F_{model} − k_{scale}·ΔF_w``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Sequence, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator

TWO_PI2 = 2.0 * math.pi**2
EIGHT_PI2 = 8.0 * math.pi**2

OMIT_OP_NAME = "ml_i_omit_windows"


class OmitWindowOptions(BaseModel):
    """JSON options for the ``ml_i_omit_windows`` op / env knobs."""

    model_config = {"extra": "forbid"}

    enabled: bool = False
    box_size: float = Field(default=10.0, gt=0.0)
    mode: Literal["boxes", "residue_blocks"] = "boxes"
    block_size: int = Field(default=5, ge=1)
    coefficient_types: Literal["both", "model", "difference"] = "both"
    when: Literal["end_of_refinement", "every_macrocycle"] = "end_of_refinement"
    output_prefix: Optional[str] = None
    chunk_size: Optional[int] = Field(default=None, ge=1)
    shift: Optional[list[float]] = None  # fractional offset for a second tiling

    @field_validator("shift")
    @classmethod
    def _shift3(cls, v: Optional[list[float]]) -> Optional[list[float]]:
        if v is None:
            return None
        if len(v) != 3:
            raise ValueError("shift must have 3 fractional components")
        return [float(x) for x in v]


@dataclass
class WindowSpec:
    """One omit window (possibly empty)."""

    window_id: int
    atom_indices: np.ndarray  # int64
    bounds_frac: Optional[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = None
    label: str = ""
    chain: Optional[str] = None
    resseq_lo: Optional[int] = None
    resseq_hi: Optional[int] = None

    @property
    def n_atoms(self) -> int:
        return int(self.atom_indices.size)

    @property
    def empty(self) -> bool:
        return self.n_atoms == 0


@dataclass
class WindowPartition:
    windows: list[WindowSpec]
    atom_to_window: np.ndarray  # (N_atoms,) int; -1 if unassigned
    mode: str
    empty_window_ids: list[int] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def _cell_edge_lengths(unit_cell: Sequence[float]) -> tuple[float, float, float]:
    a, b, c = float(unit_cell[0]), float(unit_cell[1]), float(unit_cell[2])
    return a, b, c


def assign_box_windows(
    sites_frac: np.ndarray,
    unit_cell: Sequence[float],
    box_size: float = 10.0,
    shift: Optional[Sequence[float]] = None,
) -> WindowPartition:
    """Rectilinear fractional boxes tiling the unit cell; each atom → one box."""
    sites = np.asarray(sites_frac, dtype=np.float64).reshape(-1, 3)
    n = sites.shape[0]
    a, b, c = _cell_edge_lengths(unit_cell)
    n_a = max(1, int(round(a / float(box_size))))
    n_b = max(1, int(round(b / float(box_size))))
    n_c = max(1, int(round(c / float(box_size))))
    sh = np.zeros(3, dtype=np.float64) if shift is None else np.asarray(shift, dtype=np.float64).reshape(3)
    # ASU fold then apply shift before binning
    xf = np.mod(sites + sh[None, :], 1.0)
    ia = np.clip((xf[:, 0] * n_a).astype(np.int64), 0, n_a - 1)
    ib = np.clip((xf[:, 1] * n_b).astype(np.int64), 0, n_b - 1)
    ic = np.clip((xf[:, 2] * n_c).astype(np.int64), 0, n_c - 1)
    wid = ia + n_a * (ib + n_b * ic)
    n_win = n_a * n_b * n_c
    windows: list[WindowSpec] = []
    empty: list[int] = []
    for w in range(n_win):
        sel = np.where(wid == w)[0]
        ia_w = w % n_a
        ib_w = (w // n_a) % n_b
        ic_w = w // (n_a * n_b)
        bounds = (
            (ia_w / n_a, (ia_w + 1) / n_a),
            (ib_w / n_b, (ib_w + 1) / n_b),
            (ic_w / n_c, (ic_w + 1) / n_c),
        )
        spec = WindowSpec(
            window_id=w,
            atom_indices=sel.astype(np.int64),
            bounds_frac=bounds,
            label=f"box_{ia_w}_{ib_w}_{ic_w}",
        )
        windows.append(spec)
        if spec.empty:
            empty.append(w)
    return WindowPartition(
        windows=windows,
        atom_to_window=wid.astype(np.int64),
        mode="boxes",
        empty_window_ids=empty,
        meta={
            "n_a": n_a,
            "n_b": n_b,
            "n_c": n_c,
            "box_size": float(box_size),
            "shift": sh.tolist(),
            "n_atoms": n,
        },
    )


def assign_residue_block_windows(
    residue_ids: Sequence[tuple[str, int]],
    block_size: int = 5,
) -> WindowPartition:
    """Contiguous residue ranges of length ``block_size`` per chain.

    ``residue_ids[i]`` = ``(chain_id, resseq)`` for atom ``i``.
    """
    n = len(residue_ids)
    if n == 0:
        return WindowPartition(windows=[], atom_to_window=np.zeros(0, dtype=np.int64), mode="residue_blocks")
    # group atom indices by (chain, resseq)
    from collections import defaultdict

    by_chain: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, (ch, rs) in enumerate(residue_ids):
        by_chain[str(ch)][int(rs)].append(i)

    windows: list[WindowSpec] = []
    atom_to_window = np.full(n, -1, dtype=np.int64)
    wid = 0
    for ch in sorted(by_chain.keys()):
        resseqs = sorted(by_chain[ch].keys())
        for start in range(0, len(resseqs), int(block_size)):
            block = resseqs[start : start + int(block_size)]
            atoms: list[int] = []
            for rs in block:
                atoms.extend(by_chain[ch][rs])
            idx = np.asarray(atoms, dtype=np.int64)
            atom_to_window[idx] = wid
            windows.append(
                WindowSpec(
                    window_id=wid,
                    atom_indices=idx,
                    label=f"{ch}:{block[0]}-{block[-1]}",
                    chain=ch,
                    resseq_lo=int(block[0]),
                    resseq_hi=int(block[-1]),
                )
            )
            wid += 1
    empty = [w.window_id for w in windows if w.empty]
    return WindowPartition(
        windows=windows,
        atom_to_window=atom_to_window,
        mode="residue_blocks",
        empty_window_ids=empty,
        meta={"block_size": int(block_size), "n_atoms": n},
    )


def sigma_omit_raw(
    *,
    hkl: np.ndarray,
    unit_cell: Sequence[float],
    sites_frac: np.ndarray,
    occupancy: np.ndarray,
    u_iso: np.ndarray,
    u_star: np.ndarray,
    anisotropic: np.ndarray,
    type_index: np.ndarray,
    gauss_a: np.ndarray,
    gauss_b: np.ndarray,
    gauss_c: np.ndarray,
    rot: np.ndarray,
    trans: np.ndarray,
    multiplicity: np.ndarray,
    atom_indices: np.ndarray,
    epsilon: np.ndarray,
    fp: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Raw omitted scattering power per reflection (Wilson-Σ units before κ).

    $$
    \\Sigma^{\\mathrm{raw}}_h = \\varepsilon_h^{-1}\\sum_{j\\in w}\\sum_s
    (o_j m_j / n_{\\mathrm{sym}})^2\\,|f_j|^2\\,e^{-B_j s^2}
    $$

    with Debye–Waller ``e^{-4π² U d*²}`` (``B = 8π²U`` ⇒ ``e^{-B s²/2}`` with ``s²=d*²``).
    """
    from phridge.sfcalc.engine.cell import reciprocal_cartesian

    h = np.asarray(hkl, dtype=np.float64).reshape(-1, 3)
    n_h = h.shape[0]
    eps = np.maximum(np.asarray(epsilon, dtype=np.float64).reshape(-1), 1e-12)
    dstar2 = np.sum(reciprocal_cartesian(unit_cell, h.astype(np.int64)) ** 2, axis=1)
    stol2 = dstar2 / 4.0
    n_sym = int(rot.shape[0])
    out = np.zeros(n_h, dtype=np.float64)
    atoms = np.asarray(atom_indices, dtype=np.int64).ravel()
    if atoms.size == 0:
        return out
    fp_arr = np.zeros(len(occupancy), dtype=np.float64) if fp is None else np.asarray(fp, dtype=np.float64)
    for j in atoms:
        j = int(j)
        ga = gauss_a[type_index[j]]
        gb = gauss_b[type_index[j]]
        gc = float(gauss_c[type_index[j]])
        ff = gc + float(fp_arr[j]) + np.sum(ga[:, None] * np.exp(-gb[:, None] * stol2[None, :]), axis=0)
        w0 = float(occupancy[j]) * float(multiplicity[j]) / float(n_sym)
        for si in range(n_sym):
            if anisotropic[j]:
                from phridge.sfcalc.engine.cell import sym6_to_mat

                u_mat = sym6_to_mat(u_star[j : j + 1])[0]
                u_sym = rot[si] @ u_mat @ rot[si].T
                dw = np.exp(-TWO_PI2 * np.einsum("hi,ij,hj->h", h, u_sym, h))
            else:
                dw = np.exp(-TWO_PI2 * float(u_iso[j]) * dstar2)
            out += (w0**2) * (ff**2) * (dw**2)
    return out / eps


def calibrate_kappa(sigma_wilson: np.ndarray, sigma_omit_all_raw: np.ndarray) -> float:
    """Global κ so that omitting all atoms ≈ Wilson Σ (median ratio)."""
    sw = np.asarray(sigma_wilson, dtype=np.float64)
    raw = np.asarray(sigma_omit_all_raw, dtype=np.float64)
    ok = (sw > 1e-12) & (raw > 1e-12) & np.isfinite(sw) & np.isfinite(raw)
    if not np.any(ok):
        return 1.0
    return float(np.median(sw[ok] / raw[ok]))


def effective_sigma_a(
    sigma_a: np.ndarray,
    sigma_wilson: np.ndarray,
    sigma_omit: np.ndarray,
) -> np.ndarray:
    """Encode residual-β increase as effective σ_A for the Rice maps API."""
    sa = np.asarray(sigma_a, dtype=np.float64)
    sw = np.maximum(np.asarray(sigma_wilson, dtype=np.float64), 1e-12)
    so = np.maximum(np.asarray(sigma_omit, dtype=np.float64), 0.0)
    ratio = np.clip(1.0 - so / sw, 0.0, 1.0)
    return np.clip(sa * np.sqrt(ratio), 1e-4, 0.9999)


def residual_beta_adjust(
    sigma_a: np.ndarray,
    sigma_wilson: np.ndarray,
    sigma_omit: np.ndarray,
) -> np.ndarray:
    """``σ_A² Σ_omit`` increment added to residual β = Σ(1−σ_A²)."""
    sa = np.asarray(sigma_a, dtype=np.float64)
    so = np.asarray(sigma_omit, dtype=np.float64)
    return (sa**2) * so


def delta_f_masked(
    eng: Any,
    occupancy: np.ndarray,
    atom_indices: np.ndarray,
    *,
    sites_frac: Any = None,
    u_iso: Any = None,
    u_star: Any = None,
    fp: Any = None,
    fdp: Any = None,
) -> np.ndarray:
    """FFT structure factors of atoms in ``atom_indices`` only (others occ=0)."""
    import torch

    occ = np.asarray(occupancy, dtype=np.float64).copy()
    mask = np.zeros_like(occ, dtype=bool)
    idx = np.asarray(atom_indices, dtype=np.int64).ravel()
    if idx.size:
        mask[idx] = True
    occ = np.where(mask, occ, 0.0)
    model = eng.model
    sf = sites_frac if sites_frac is not None else model.sites_frac
    ui = u_iso if u_iso is not None else model.u_iso
    us = u_star if u_star is not None else model.u_star
    fp_v = fp if fp is not None else model.fp
    fdp_v = fdp if fdp is not None else model.fdp
    with torch.no_grad():
        t_sf = torch.as_tensor(sf, dtype=eng.dtype, device=eng.device)
        t_occ = torch.as_tensor(occ, dtype=eng.dtype, device=eng.device)
        t_ui = torch.as_tensor(ui, dtype=eng.dtype, device=eng.device)
        t_us = torch.as_tensor(us, dtype=eng.dtype, device=eng.device)
        t_fp = torch.as_tensor(fp_v, dtype=eng.dtype, device=eng.device)
        t_fdp = torch.as_tensor(fdp_v, dtype=eng.dtype, device=eng.device)
        fc = eng.f_calc(t_sf, t_occ, t_ui, t_us, t_fp, t_fdp)
    return fc.detach().cpu().numpy().astype(np.complex128)


def compute_window_delta_fs(
    eng: Any,
    occupancy: np.ndarray,
    partition: WindowPartition,
    *,
    skip_empty: bool = True,
) -> tuple[np.ndarray, list[int]]:
    """Return ``(W_used, N_refl)`` deltaF and list of window ids used."""
    model = eng.model
    used_ids: list[int] = []
    blocks: list[np.ndarray] = []
    for w in partition.windows:
        if skip_empty and w.empty:
            continue
        used_ids.append(w.window_id)
        blocks.append(delta_f_masked(eng, occupancy, w.atom_indices))
    if not blocks:
        n_h = int(eng.hkl.shape[0])
        return np.zeros((0, n_h), dtype=np.complex128), []
    return np.stack(blocks, axis=0), used_ids


def intensity_map_coefficients_batch(
    target: Any,
    f_calc_batch: Any,
    obs: Any,
    sigma_a_batch: Any,
    *,
    coefficient_types: str = "both",
    chunk_size: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """Chunked map coeffs for omit models; ``f_calc_batch`` / ``sigma_a_batch`` shape ``(W, N)``.

    Computes only ``difference`` / ``model`` (no Newton/gradient — those need autograd
    through the full NLL and are not required for omit artifacts).
    """
    import torch

    from phridge.contrib.intensity_ll.maps import posterior_moments
    from phridge.contrib.intensity_ll.mli import normalize

    fo = obs.data
    device = fo.device
    # Match observation dtype/device (MPS → float32 / complex64; CPU often float64)
    real_dtype = fo.dtype
    cdtype = torch.complex64 if real_dtype == torch.float32 else torch.complex128

    fc_b = torch.as_tensor(f_calc_batch, dtype=cdtype, device=device)
    sa_b = torch.as_tensor(sigma_a_batch, dtype=real_dtype, device=device)
    if fc_b.ndim == 1:
        fc_b = fc_b.unsqueeze(0)
        sa_b = sa_b.unsqueeze(0)
    w_tot, n = fc_b.shape
    cs = int(chunk_size) if chunk_size is not None else max(1, min(8, w_tot))
    want_model = coefficient_types in ("both", "model")
    want_diff = coefficient_types in ("both", "difference")

    eps = obs.epsilon if obs.epsilon is not None else torch.ones_like(fo)
    centric = obs.centric if obs.centric is not None else torch.zeros_like(fo, dtype=torch.bool)
    sW = target._sigma_wilson(obs)
    sig = target._sigma(obs)
    nu = target._nu(obs)

    models: list[np.ndarray] = []
    diffs: list[np.ndarray] = []

    for start in range(0, w_tot, cs):
        stop = min(start + cs, w_tot)
        for wi in range(start, stop):
            f_calc = fc_b[wi]
            sA = sa_b[wi]
            fc = f_calc.abs()
            ok = (sA > 0) & (sA < 1.0 - 1e-6) & (sW > 0) & (sig > 0) & (eps > 0) & (fc > 0)
            sA_s = torch.where(ok, sA, torch.full_like(sA, 0.5))
            sW_s = torch.where(ok, sW, torch.ones_like(sW))
            sig_s = torch.where(ok, sig, torch.ones_like(sig))
            eps_s = torch.where(ok, eps, torch.ones_like(eps))
            fc_s = torch.where(ok, fc, torch.ones_like(fc))
            fo_s = torch.where(ok, fo, torch.zeros_like(fo))
            with torch.no_grad():
                Ec, sA_n, Zo, sZ = normalize(fc_s, fo_s, sig_s, eps_s, sW_s, sA_s)
                post = posterior_moments(
                    Ec,
                    sA_n,
                    Zo,
                    sZ,
                    centric,
                    nu,
                    n_u=target.n_u,
                    snr_strong=target.snr_strong,
                    n_hermite=target.n_hermite,
                    n_legendre=target.n_legendre,
                    k_window=target.k_window,
                )
                scale = torch.sqrt(eps_s * sW_s)
                phase = f_calc / fc.clamp(min=1e-300)
                diff_E = post.Em_mean - sA_n * Ec
                model_E = torch.where(centric, post.Em_mean, 2 * post.Em_mean - sA_n * Ec)

                def _c(real_amp: Any) -> np.ndarray:
                    z = torch.where(ok, real_amp, torch.zeros_like(real_amp)).to(phase.dtype) * phase
                    return z.detach().cpu().numpy().astype(np.complex64)

                if want_diff:
                    diffs.append(_c(scale * diff_E))
                if want_model:
                    models.append(_c(scale * model_E))

    result: dict[str, np.ndarray] = {}
    if want_model:
        result["coef_model"] = np.stack(models, axis=0)
    if want_diff:
        result["coef_difference"] = np.stack(diffs, axis=0)
    return result


@dataclass
class OmitWindowStore:
    """In-memory / on-disk omit-window coefficient artifact."""

    hkl: np.ndarray
    coef_model: Optional[np.ndarray]
    coef_difference: Optional[np.ndarray]
    beta_adjust: np.ndarray
    window_meta: dict[str, Any]
    window_ids: np.ndarray
    atom_to_window: np.ndarray

    def coefficients(self, window_id: int, kind: str = "difference") -> np.ndarray:
        ids = list(self.window_ids)
        if window_id not in ids:
            raise KeyError(f"window_id {window_id} not in artifact (computed windows only)")
        i = ids.index(window_id)
        key = "model" if kind in ("model", "2mFo-DFc", "2fofc", "coef_model") else "difference"
        arr = self.coef_model if key == "model" else self.coef_difference
        if arr is None:
            raise KeyError(f"coefficient kind {kind!r} not stored")
        return arr[i]

    def window_for_atom(self, i_seq: int) -> int:
        return int(self.atom_to_window[int(i_seq)])

    def window_for_residue(self, chain: str, resid: int) -> int:
        for w in self.window_meta.get("windows", []):
            if w.get("chain") == chain and w.get("resseq_lo") is not None:
                if int(w["resseq_lo"]) <= int(resid) <= int(w.get("resseq_hi", w["resseq_lo"])):
                    return int(w["window_id"])
        raise KeyError(f"no residue window for {chain}:{resid}")

    def save(self, path: str) -> None:
        payload = {
            "hkl": np.asarray(self.hkl, dtype=np.int32),
            "beta_adjust": np.asarray(self.beta_adjust, dtype=np.float32),
            "window_ids": np.asarray(self.window_ids, dtype=np.int32),
            "atom_to_window": np.asarray(self.atom_to_window, dtype=np.int32),
            "window_meta_json": np.asarray(json.dumps(self.window_meta)),
        }
        if self.coef_model is not None:
            payload["coef_model"] = np.asarray(self.coef_model, dtype=np.complex64)
        if self.coef_difference is not None:
            payload["coef_difference"] = np.asarray(self.coef_difference, dtype=np.complex64)
        np.savez_compressed(path, **payload)


def load_omit_windows(path: str) -> OmitWindowStore:
    with np.load(path, allow_pickle=False) as zf:
        meta = json.loads(str(zf["window_meta_json"]))
        return OmitWindowStore(
            hkl=np.asarray(zf["hkl"]),
            coef_model=np.asarray(zf["coef_model"]) if "coef_model" in zf.files else None,
            coef_difference=np.asarray(zf["coef_difference"]) if "coef_difference" in zf.files else None,
            beta_adjust=np.asarray(zf["beta_adjust"]),
            window_meta=meta,
            window_ids=np.asarray(zf["window_ids"]),
            atom_to_window=np.asarray(zf["atom_to_window"]),
        )


def run_omit_windows_core(
    *,
    eng: Any,
    f_model: np.ndarray,
    k_scale: np.ndarray,
    obs_target: Any,
    obs: Any,
    sigma_a: np.ndarray,
    sigma_wilson: np.ndarray,
    epsilon: np.ndarray,
    partition: WindowPartition,
    options: OmitWindowOptions,
    extra_meta: Optional[dict[str, Any]] = None,
) -> OmitWindowStore:
    """Core omit-window computation given a ready SF engine and observations."""
    import torch

    model = eng.model
    occupancy = np.asarray(model.occupancy, dtype=np.float64)
    hkl = np.asarray(eng.hkl, dtype=np.int64)
    n_h = hkl.shape[0]
    sa = np.asarray(sigma_a, dtype=np.float64).reshape(n_h)
    sw = np.asarray(sigma_wilson, dtype=np.float64).reshape(n_h)
    eps = np.asarray(epsilon, dtype=np.float64).reshape(n_h)
    k_sc = np.asarray(k_scale, dtype=np.float64).reshape(n_h)
    fm = np.asarray(f_model, dtype=np.complex128).reshape(n_h)

    # Full-model atomic SF (for partition checks / diagnostics)
    with torch.no_grad():
        f_calc_full = eng.f_calc_numpy().astype(np.complex128)

    raw_all = sigma_omit_raw(
        hkl=hkl,
        unit_cell=model.unit_cell,
        sites_frac=model.sites_frac,
        occupancy=occupancy,
        u_iso=model.u_iso,
        u_star=model.u_star,
        anisotropic=model.anisotropic,
        type_index=model.type_index,
        gauss_a=model.gauss_a,
        gauss_b=model.gauss_b,
        gauss_c=model.gauss_c,
        rot=model.rot,
        trans=model.trans,
        multiplicity=model.multiplicity,
        atom_indices=np.arange(len(occupancy), dtype=np.int64),
        epsilon=eps,
        fp=model.fp,
    )
    kappa = calibrate_kappa(sw, raw_all)

    used: list[WindowSpec] = [w for w in partition.windows if not w.empty]
    # Always include at least the math path for empty-window identity via a sentinel if needed
    delta_list: list[np.ndarray] = []
    sa_list: list[np.ndarray] = []
    beta_adj_list: list[np.ndarray] = []
    used_ids: list[int] = []
    omit_fracs: list[float] = []

    chunk = options.chunk_size
    for w in used:
        dF = delta_f_masked(eng, occupancy, w.atom_indices)
        f_omit = fm - k_sc * dF
        raw_w = sigma_omit_raw(
            hkl=hkl,
            unit_cell=model.unit_cell,
            sites_frac=model.sites_frac,
            occupancy=occupancy,
            u_iso=model.u_iso,
            u_star=model.u_star,
            anisotropic=model.anisotropic,
            type_index=model.type_index,
            gauss_a=model.gauss_a,
            gauss_b=model.gauss_b,
            gauss_c=model.gauss_c,
            rot=model.rot,
            trans=model.trans,
            multiplicity=model.multiplicity,
            atom_indices=w.atom_indices,
            epsilon=eps,
            fp=model.fp,
        )
        so = kappa * raw_w
        sa_w = effective_sigma_a(sa, sw, so)
        delta_list.append(f_omit)
        sa_list.append(sa_w)
        beta_adj_list.append(residual_beta_adjust(sa, sw, so))
        used_ids.append(w.window_id)
        omit_fracs.append(float(np.mean(so / np.maximum(sw, 1e-12))))

    if not delta_list:
        # No atoms — single empty result matching full maps
        delta_list = [fm.copy()]
        sa_list = [sa.copy()]
        beta_adj_list = [np.zeros(n_h, dtype=np.float64)]
        used_ids = [-1]
        omit_fracs = [0.0]

    fc_batch = np.stack(delta_list, axis=0)
    sa_batch = np.stack(sa_list, axis=0)
    beta_adj = np.stack(beta_adj_list, axis=0).astype(np.float32)

    coeffs = intensity_map_coefficients_batch(
        obs_target,
        fc_batch,
        obs,
        sa_batch,
        coefficient_types=options.coefficient_types,
        chunk_size=chunk,
    )

    win_meta_list = []
    for w in partition.windows:
        entry = {
            "window_id": w.window_id,
            "n_atoms": w.n_atoms,
            "empty": w.empty,
            "label": w.label,
            "atom_indices": w.atom_indices.tolist(),
            "bounds_frac": w.bounds_frac,
            "chain": w.chain,
            "resseq_lo": w.resseq_lo,
            "resseq_hi": w.resseq_hi,
        }
        if w.window_id in used_ids:
            j = used_ids.index(w.window_id)
            entry["omitted_scattering_fraction"] = omit_fracs[j]
        win_meta_list.append(entry)

    meta = {
        "windows": win_meta_list,
        "empty_window_ids": list(partition.empty_window_ids),
        "mode": partition.mode,
        "partition": partition.meta,
        "kappa": kappa,
        "coefficient_types": options.coefficient_types,
        "n_reflections": n_h,
        "computed_window_ids": used_ids,
        "fixed_solvent_and_scales": True,
        "sigma_a_encoding": "effective_sigma_a_from_residual_beta",
        "f_calc_full_norm": float(np.mean(np.abs(f_calc_full))),
        **(extra_meta or {}),
    }
    return OmitWindowStore(
        hkl=hkl.astype(np.int32),
        coef_model=coeffs.get("coef_model"),
        coef_difference=coeffs.get("coef_difference"),
        beta_adjust=beta_adj,
        window_meta=meta,
        window_ids=np.asarray(used_ids, dtype=np.int32),
        atom_to_window=partition.atom_to_window.astype(np.int32),
    )
