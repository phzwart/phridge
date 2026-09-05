"""Large-N structure-factor gradient benchmark helpers (phridge CUDA vs CCTBX).

Used by ``tests/test_sf_gradients_gpu.py`` and ``examples/sf_gradient_benchmark.py``.
Import cctbx before torch in the calling process.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

SPACE_GROUPS = ("P1", "P21", "P212121", "C2", "P4132", "R3:H")
DEFAULT_ELEMENTS = ("C", "N", "O", "S")
DEFAULT_N_ATOMS = 1000
DEFAULT_D_MIN = 2.0
DEFAULT_SIGMA = 0.05  # Å Cartesian Gaussian site perturbation
DEFAULT_QUALITY = 1000.0


@dataclass
class TimingRow:
    cctbx_f_s: float = 0.0
    cctbx_grad_s: float = 0.0
    phridge_f_s: float = 0.0
    phridge_grad_s: float = 0.0
    remote_f_s: float = 0.0
    remote_grad_s: float = 0.0


@dataclass
class SpaceGroupResult:
    space_group: str
    n_atoms: int
    n_refl: int
    f_rel: float
    f_r_factor: float
    cosine: float
    angle_deg: float
    length_ratio: float
    site_rel_max: float
    timings: TimingRow = field(default_factory=TimingRow)
    device: str = "cuda"


def time_call(fn: Callable[[], Any]) -> tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def rel_max(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.abs(a - b).max() / max(np.abs(a).max(), 1e-300))


def grad_vector_metrics(g_ref: np.ndarray, g_mine: np.ndarray) -> dict[str, float]:
    """Compare two flattened gradient vectors (length + direction)."""
    a = np.asarray(g_ref, dtype=np.float64).ravel()
    b = np.asarray(g_mine, dtype=np.float64).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-300 or nb < 1e-300:
        cos = 0.0 if na > 1e-300 or nb > 1e-300 else 1.0
    else:
        cos = float(np.dot(a, b) / (na * nb))
    cos = float(np.clip(cos, -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cos)))
    length_ratio = nb / na if na > 1e-300 else float("nan")
    return {
        "cosine": cos,
        "angle_deg": angle_deg,
        "length_ratio": length_ratio,
        "rel_max": rel_max(a, b),
        "norm_ref": na,
        "norm_mine": nb,
    }


def build_structure(
    space_group: str = "P21",
    n_atoms: int = DEFAULT_N_ATOMS,
    elements: tuple[str, ...] = DEFAULT_ELEMENTS,
    seed: int = 0,
):
    """Random x-ray structure with ~n_atoms ASU scatterers (site/occ/u_iso grads on)."""
    import random

    from cctbx import sgtbx
    from cctbx.array_family import flex
    from cctbx.development import random_structure

    random.seed(seed)
    flex.set_random_seed(seed)
    n_repeat = max(1, int(round(n_atoms / len(elements))))
    xs = random_structure.xray_structure(
        space_group_info=sgtbx.space_group_info(space_group),
        elements=list(elements) * n_repeat,
        volume_per_atom=50,
        random_u_iso=True,
        random_occupancy=True,
        use_u_aniso=False,
    )
    for sc in xs.scatterers():
        sc.flags.set_grad_site(True)
        sc.flags.set_grad_occupancy(True)
        sc.flags.set_grad_u_iso(True)
        sc.flags.set_grad_u_aniso(False)
        sc.flags.set_grad_fp(False)
        sc.flags.set_grad_fdp(False)
    return xs


def make_fobs(xs_true, d_min: float = DEFAULT_D_MIN):
    """F_obs amplitudes from CCTBX direct F_calc of the true model."""
    fc = xs_true.structure_factors(d_min=d_min, algorithm="direct").f_calc()
    return fc.amplitudes(), fc


def perturb_sites_cart(xs, sigma_angstrom: float = DEFAULT_SIGMA, rng: Optional[np.random.Generator] = None):
    """Deep-copy and add Gaussian noise to Cartesian sites (Å)."""
    from cctbx.array_family import flex

    if rng is None:
        rng = np.random.default_rng(0)
    xs2 = xs.deep_copy_scatterers()
    sites = np.asarray(xs2.sites_cart(), dtype=np.float64).reshape(-1, 3)
    sites = sites + rng.normal(scale=sigma_angstrom, size=sites.shape)
    xs2.set_sites_cart(flex.vec3_double(sites.tolist()))
    return xs2


def ls_dtdf(xs_pert, f_obs, miller_set=None):
    """CCTBX least-squares ``d_target_d_f_calc`` at the perturbed model.

    Returns ``(dtdf_numpy, fc_pert, ls_target)``.
    """
    from cctbx.array_family import flex
    from cctbx.xray import ext

    d_min = float(f_obs.d_min())
    fc_pert = xs_pert.structure_factors(d_min=d_min, algorithm="direct").f_calc()
    if miller_set is not None:
        fc_pert = fc_pert.common_set(miller_set)
    n = fc_pert.size()
    w = flex.double(n, 1.0)
    ls = ext.targets_least_squares_residual(f_obs.data(), w, fc_pert.data(), True, 0.0)
    dtdf = np.asarray(ls.derivatives(), dtype=np.complex128)
    return dtdf, fc_pert, float(ls.target())


def cctbx_site_grads(xs, miller_set, dtdf: np.ndarray) -> np.ndarray:
    """Site-frac gradients from CCTBX ``gradients_direct``, shape (N, 3)."""
    from cctbx import xray
    from cctbx.array_family import flex

    ref = xray.structure_factors.gradients_direct(
        xray_structure=xs,
        u_iso_refinable_params=None,
        miller_set=miller_set,
        d_target_d_f_calc=flex.complex_double(dtdf),
        n_parameters=0,
    )._results
    return np.asarray(ref.d_target_d_site_frac(), dtype=np.float64).reshape(-1, 3)


def _phridge_engine(xs, hkl: np.ndarray, d_min: float, device: str, quality_factor: float = DEFAULT_QUALITY):
    from phridge.client.convert_xtal import scattering_table_from_cctbx, xray_from_cctbx
    from phridge.worker.ops.xtal_ops import scattering_model
    from phridge.worker.xtal.engine import EngineParams, StructureFactorEngine

    model = scattering_model(xray_from_cctbx(xs), scattering_table_from_cctbx(xs))
    params = EngineParams(d_min=d_min, quality_factor=quality_factor)
    return StructureFactorEngine(model, hkl, params, device=device)


def phridge_f_and_site_grads(
    xs,
    hkl: np.ndarray,
    dtdf: np.ndarray,
    d_min: float,
    device: str = "cuda",
    quality_factor: float = DEFAULT_QUALITY,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Direct CUDA/CPU engine: returns ``(f_calc, site_frac_grads, t_f, t_grad)``."""
    eng = _phridge_engine(xs, hkl, d_min, device=device, quality_factor=quality_factor)
    fc, t_f = time_call(eng.f_calc_numpy)
    grads, t_g = time_call(lambda: eng.gradients(dtdf))
    return fc, np.asarray(grads["site_frac"], dtype=np.float64), t_f, t_g


def remote_f_and_site_grads(
    bridge,
    xs,
    miller_set,
    dtdf: np.ndarray,
    d_min: float,
    quality_factor: float = DEFAULT_QUALITY,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Bridge ``RemoteStructureFactors`` path: ``(f_calc, site_frac, t_f, t_grad)``."""
    from phridge.client.xtal_engine import RemoteStructureFactors
    from phridge.models import SfEngineParams

    engine = RemoteStructureFactors(
        bridge,
        xs,
        miller_set,
        params=SfEngineParams(d_min=d_min, quality_factor=quality_factor),
    )
    fc_arr, t_f = time_call(lambda: np.asarray(engine.f_calc().data(), dtype=np.complex128))
    grads, t_g = time_call(lambda: engine.gradients(dtdf))
    site = np.asarray(grads.raw.d_site_frac, dtype=np.float64).reshape(-1, 3)
    return fc_arr, site, t_f, t_g


def run_space_group(
    space_group: str,
    *,
    n_atoms: int = DEFAULT_N_ATOMS,
    d_min: float = DEFAULT_D_MIN,
    sigma: float = DEFAULT_SIGMA,
    device: str = "cuda",
    quality_factor: float = DEFAULT_QUALITY,
    seed: int = 0,
    bridge=None,
    time_remote: bool = True,
) -> SpaceGroupResult:
    """Full workflow for one space group: F_obs → perturb → compare grads."""
    xs_true = build_structure(space_group, n_atoms=n_atoms, seed=seed)
    f_obs, _fc_true = make_fobs(xs_true, d_min=d_min)

    rng = np.random.default_rng(seed + 17)
    xs_pert = perturb_sites_cart(xs_true, sigma_angstrom=sigma, rng=rng)

    def _cctbx_f():
        return xs_pert.structure_factors(d_min=d_min, algorithm="direct").f_calc()

    fc_pert, t_cctbx_f = time_call(_cctbx_f)
    f_obs = f_obs.common_set(fc_pert)
    n = fc_pert.size()
    from cctbx.array_family import flex
    from cctbx.xray import ext

    w = flex.double(n, 1.0)
    ls = ext.targets_least_squares_residual(f_obs.data(), w, fc_pert.data(), True, 0.0)
    dtdf = np.asarray(ls.derivatives(), dtype=np.complex128)

    def _cctbx_g():
        return cctbx_site_grads(xs_pert, fc_pert, dtdf)

    g_ref, t_cctbx_g = time_call(_cctbx_g)

    hkl = np.asarray(list(fc_pert.indices()), dtype=np.int32)
    fc_mine, g_mine, t_pf, t_pg = phridge_f_and_site_grads(
        xs_pert, hkl, dtdf, d_min=d_min, device=device, quality_factor=quality_factor
    )
    ref_fc = np.asarray(fc_pert.data(), dtype=np.complex128)
    f_rel = rel_max(ref_fc, fc_mine)
    f_r = float(np.abs(np.abs(fc_mine) - np.abs(ref_fc)).sum() / max(np.abs(ref_fc).sum(), 1e-300))
    metrics = grad_vector_metrics(g_ref, g_mine)

    timings = TimingRow(
        cctbx_f_s=t_cctbx_f,
        cctbx_grad_s=t_cctbx_g,
        phridge_f_s=t_pf,
        phridge_grad_s=t_pg,
    )

    if time_remote and bridge is not None:
        _, _, t_rf, t_rg = remote_f_and_site_grads(
            bridge, xs_pert, fc_pert, dtdf, d_min=d_min, quality_factor=quality_factor
        )
        timings.remote_f_s = t_rf
        timings.remote_grad_s = t_rg

    return SpaceGroupResult(
        space_group=space_group,
        n_atoms=xs_pert.scatterers().size(),
        n_refl=n,
        f_rel=f_rel,
        f_r_factor=f_r,
        cosine=metrics["cosine"],
        angle_deg=metrics["angle_deg"],
        length_ratio=metrics["length_ratio"],
        site_rel_max=metrics["rel_max"],
        timings=timings,
        device=device,
    )


def format_result_table(rows: list[SpaceGroupResult]) -> str:
    """Plain-text summary table for pytest -s / example logs."""
    hdr = (
        f"{'SG':<10} {'Nat':>5} {'Nrefl':>7} {'R(F)':>8} {'cos':>8} "
        f"{'ang°':>7} {'|g|/|g|':>8} {'cctbxG':>8} {'phrG':>8} {'remG':>8}"
    )
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        t = r.timings
        lines.append(
            f"{r.space_group:<10} {r.n_atoms:5d} {r.n_refl:7d} {r.f_r_factor:8.2e} "
            f"{r.cosine:8.5f} {r.angle_deg:7.3f} {r.length_ratio:8.4f} "
            f"{t.cctbx_grad_s:8.2f} {t.phridge_grad_s:8.2f} {t.remote_grad_s:8.2f}"
        )
    return "\n".join(lines)
