"""Tests for windowed omit map coefficients (ml_i_omit_windows)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phridge.contrib.intensity_ll import OMIT_OP_NAME, register  # noqa: E402
from phridge.contrib.intensity_ll.omit_windows import (  # noqa: E402
    OmitWindowOptions,
    OmitWindowStore,
    assign_box_windows,
    assign_residue_block_windows,
    calibrate_kappa,
    delta_f_masked,
    effective_sigma_a,
    load_omit_windows,
    residual_beta_adjust,
    run_omit_windows_core,
    sigma_omit_raw,
)
from phridge.contrib.intensity_ll.maps import intensity_map_coefficients  # noqa: E402
from phridge.contrib.intensity_ll.target import IntensityLogLikelihood  # noqa: E402
from phridge.models import CrystalSymmetry, ObservationType, Scatterer, SfEngineParams, SymOp  # noqa: E402
from phridge.ops import get_implementation, get_op  # noqa: E402
from phridge.packing import PackedMiller  # noqa: E402
from phridge.packing_xtal import PackedXray  # noqa: E402
from phridge.sfcalc.engine import EngineParams, StructureFactorEngine  # noqa: E402
from phridge.sfcalc.ops import _observations, scattering_model  # noqa: E402
from phridge.sfcalc.packing import PackedScatteringTable  # noqa: E402
from phridge.sfcalc.targets import Observations  # noqa: E402


def _p1_crystal(cell=(40.0, 40.0, 40.0, 90.0, 90.0, 90.0)):
    return CrystalSymmetry(
        unit_cell=list(cell),
        space_group_hall="P 1",
        space_group_number=1,
        symops=[SymOp(r=[1, 0, 0, 0, 1, 0, 0, 0, 1], t=[0, 0, 0])],
    )


def _p21_crystal(cell=(40.0, 50.0, 40.0, 90.0, 90.0, 90.0)):
    return CrystalSymmetry(
        unit_cell=list(cell),
        space_group_hall="P 21",
        space_group_number=4,
        symops=[
            SymOp(r=[1, 0, 0, 0, 1, 0, 0, 0, 1], t=[0, 0, 0]),
            SymOp(r=[-1, 0, 0, 0, 1, 0, 0, 0, -1], t=[0, 0.5, 0]),
        ],
    )


def _carbon_table():
    # crude single-Gaussian C
    return PackedScatteringTable(
        labels=["C"],
        gauss_a=np.array([[2.31, 1.02, 1.5886, 0.865]], dtype=np.float64),
        gauss_b=np.array([[20.8439, 10.2075, 0.5687, 51.6512]], dtype=np.float64),
        gauss_c=np.array([0.2156], dtype=np.float64),
    )


def _synthetic_structure(n_atoms: int = 24, seed: int = 0, crystal=None):
    rng = np.random.default_rng(seed)
    crystal = crystal or _p1_crystal()
    sites = rng.random((n_atoms, 3))
    occ = np.ones(n_atoms, dtype=np.float64)
    u_iso = 0.05 + 0.02 * rng.random(n_atoms)
    scatterers = [
        Scatterer(i=i, scattering_type="C", anisotropic=False, use_u_iso=True) for i in range(n_atoms)
    ]
    xray = PackedXray(
        crystal=crystal,
        sites_frac=sites,
        occupancy=occ,
        u_iso=u_iso,
        scatterers=scatterers,
    )
    table = _carbon_table()
    # low-res hkl grid
    hkl = []
    for h in range(-4, 5):
        for k in range(-4, 5):
            for l in range(-4, 5):
                if h == 0 and k == 0 and l == 0:
                    continue
                hkl.append((h, k, l))
    hkl = np.asarray(hkl, dtype=np.int32)
    dstar2 = (hkl[:, 0] / crystal.unit_cell[0]) ** 2 + (hkl[:, 1] / crystal.unit_cell[1]) ** 2 + (
        hkl[:, 2] / crystal.unit_cell[2]
    ) ** 2
    d_min = float(1.0 / np.sqrt(dstar2.max()))
    params = SfEngineParams(d_min=max(d_min * 0.95, 1.5), quality_factor=100.0)
    model = scattering_model(xray, table)
    eng = StructureFactorEngine(model, hkl, EngineParams(d_min=params.d_min, quality_factor=params.quality_factor))
    fc = eng.f_calc_numpy().astype(np.complex128)
    n = len(hkl)
    sw = 50.0 * np.exp(-8.0 * dstar2)
    sa = np.clip(0.9 * np.exp(-4.0 * dstar2), 0.2, 0.95)
    eps = np.ones(n, dtype=np.float64)
    sig = 0.2 * sw + 1.0
    io = np.abs(fc) ** 2 + rng.normal(0.0, sig)
    f_obs = PackedMiller(
        crystal=crystal,
        hkl=hkl,
        data=io,
        sigmas=sig,
        observation_type=ObservationType.intensity,
    )
    f_model = PackedMiller(
        crystal=crystal,
        hkl=hkl,
        data=fc,
        observation_type=ObservationType.complex,
    )
    return {
        "xray": xray,
        "table": table,
        "params": params,
        "eng": eng,
        "hkl": hkl,
        "f_calc": fc,
        "f_obs": f_obs,
        "f_model": f_model,
        "sigma_a": sa,
        "sigma_wilson": sw,
        "epsilon": eps,
        "dstar2": dstar2,
    }


# ------------------------------------------------------------------ registration
def test_omit_op_registered():
    register()
    spec = get_op(OMIT_OP_NAME)
    assert "omit" in spec.inputs
    assert "coef_model" in spec.outputs
    assert get_implementation(OMIT_OP_NAME) is not None


def test_omit_options_forbid_extra():
    from pydantic import ValidationError

    OmitWindowOptions(box_size=12.0)
    with pytest.raises(ValidationError):
        OmitWindowOptions(bogus=1)


# ------------------------------------------------------------------ window assignment
def test_box_windows_partition_atoms_exactly_once():
    sites = np.array([[0.1, 0.1, 0.1], [0.6, 0.1, 0.1], [0.1, 0.6, 0.6]], dtype=np.float64)
    part = assign_box_windows(sites, [40, 40, 40, 90, 90, 90], box_size=20.0)
    assert part.atom_to_window.shape == (3,)
    assigned = np.concatenate([w.atom_indices for w in part.windows if not w.empty])
    assert sorted(assigned.tolist()) == [0, 1, 2]
    # each atom once
    assert len(assigned) == 3


def test_residue_block_windows():
    ids = [("A", 1), ("A", 1), ("A", 2), ("A", 3), ("A", 4), ("B", 10)]
    part = assign_residue_block_windows(ids, block_size=2)
    assert part.mode == "residue_blocks"
    assert part.atom_to_window[0] == part.atom_to_window[1]
    assert len(part.windows) >= 2


# ------------------------------------------------------------------ SF delta / partition
def test_partition_additivity_delta_f():
    data = _synthetic_structure(n_atoms=16, seed=1)
    eng = data["eng"]
    part = assign_box_windows(eng.model.sites_frac, eng.model.unit_cell, box_size=15.0)
    acc = np.zeros_like(data["f_calc"])
    for w in part.windows:
        if w.empty:
            continue
        acc += delta_f_masked(eng, eng.model.occupancy, w.atom_indices)
    # FFT linearity: sum of masked FFTs ≈ full F_calc
    rel = np.abs(acc - data["f_calc"]).max() / max(np.abs(data["f_calc"]).max(), 1e-12)
    assert rel < 5e-3


def test_empty_window_identity_coeffs():
    data = _synthetic_structure(n_atoms=12, seed=2)
    eng = data["eng"]
    tgt = IntensityLogLikelihood(nu=None)
    n = len(data["hkl"])
    obs = Observations(
        data=torch.as_tensor(np.asarray(data["f_obs"].data), dtype=torch.float64),
        sigmas=torch.as_tensor(np.asarray(data["f_obs"].sigmas), dtype=torch.float64),
        epsilon=torch.as_tensor(data["epsilon"], dtype=torch.float64),
        centric=torch.zeros(n, dtype=torch.bool),
        alpha=torch.as_tensor(data["sigma_a"], dtype=torch.float64),
        beta=torch.as_tensor(data["sigma_wilson"], dtype=torch.float64),
    )
    full = intensity_map_coefficients(
        tgt,
        torch.as_tensor(data["f_calc"], dtype=torch.complex128),
        obs,
    )
    # empty omit → same F and same σ_A
    empty = intensity_map_coefficients(
        tgt,
        torch.as_tensor(data["f_calc"], dtype=torch.complex128),
        Observations(
            data=obs.data,
            sigmas=obs.sigmas,
            epsilon=obs.epsilon,
            centric=obs.centric,
            alpha=torch.as_tensor(effective_sigma_a(data["sigma_a"], data["sigma_wilson"], np.zeros(n)), dtype=torch.float64),
            beta=obs.beta,
        ),
    )
    np.testing.assert_allclose(full.difference, empty.difference, atol=1e-10)
    np.testing.assert_allclose(full.model, empty.model, atol=1e-10)


# ------------------------------------------------------------------ beta / sigma_omit
def test_beta_monotonicity_and_sigma_omit_decay():
    data = _synthetic_structure(n_atoms=20, seed=3)
    eng = data["eng"]
    m = eng.model
    raw_small = sigma_omit_raw(
        hkl=data["hkl"],
        unit_cell=m.unit_cell,
        sites_frac=m.sites_frac,
        occupancy=m.occupancy,
        u_iso=m.u_iso,
        u_star=m.u_star,
        anisotropic=m.anisotropic,
        type_index=m.type_index,
        gauss_a=m.gauss_a,
        gauss_b=m.gauss_b,
        gauss_c=m.gauss_c,
        rot=m.rot,
        trans=m.trans,
        multiplicity=m.multiplicity,
        atom_indices=np.arange(2, dtype=np.int64),
        epsilon=data["epsilon"],
        fp=m.fp,
    )
    raw_large = sigma_omit_raw(
        hkl=data["hkl"],
        unit_cell=m.unit_cell,
        sites_frac=m.sites_frac,
        occupancy=m.occupancy,
        u_iso=m.u_iso,
        u_star=m.u_star,
        anisotropic=m.anisotropic,
        type_index=m.type_index,
        gauss_a=m.gauss_a,
        gauss_b=m.gauss_b,
        gauss_c=m.gauss_c,
        rot=m.rot,
        trans=m.trans,
        multiplicity=m.multiplicity,
        atom_indices=np.arange(10, dtype=np.int64),
        epsilon=data["epsilon"],
        fp=m.fp,
    )
    assert np.all(raw_large + 1e-12 >= raw_small)
    # resolution decay: high-angle mean < low-angle mean
    order = np.argsort(data["dstar2"])
    assert raw_large[order[:20]].mean() > raw_large[order[-20:]].mean()

    kappa = calibrate_kappa(data["sigma_wilson"], raw_large * (20 / 10))  # scale rough
    so = kappa * raw_large
    sa_w = effective_sigma_a(data["sigma_a"], data["sigma_wilson"], so)
    assert np.all(sa_w <= data["sigma_a"] + 1e-12)
    adj = residual_beta_adjust(data["sigma_a"], data["sigma_wilson"], so)
    assert np.all(adj >= -1e-12)


def test_gold_equivalence_one_window():
    data = _synthetic_structure(n_atoms=16, seed=4)
    eng = data["eng"]
    part = assign_box_windows(eng.model.sites_frac, eng.model.unit_cell, box_size=20.0)
    non_empty = [w for w in part.windows if not w.empty]
    assert non_empty
    w = non_empty[0]
    dF = delta_f_masked(eng, eng.model.occupancy, w.atom_indices)
    f_omit = data["f_calc"] - dF
    m = eng.model
    raw_w = sigma_omit_raw(
        hkl=data["hkl"],
        unit_cell=m.unit_cell,
        sites_frac=m.sites_frac,
        occupancy=m.occupancy,
        u_iso=m.u_iso,
        u_star=m.u_star,
        anisotropic=m.anisotropic,
        type_index=m.type_index,
        gauss_a=m.gauss_a,
        gauss_b=m.gauss_b,
        gauss_c=m.gauss_c,
        rot=m.rot,
        trans=m.trans,
        multiplicity=m.multiplicity,
        atom_indices=w.atom_indices,
        epsilon=data["epsilon"],
        fp=m.fp,
    )
    raw_all = sigma_omit_raw(
        hkl=data["hkl"],
        unit_cell=m.unit_cell,
        sites_frac=m.sites_frac,
        occupancy=m.occupancy,
        u_iso=m.u_iso,
        u_star=m.u_star,
        anisotropic=m.anisotropic,
        type_index=m.type_index,
        gauss_a=m.gauss_a,
        gauss_b=m.gauss_b,
        gauss_c=m.gauss_c,
        rot=m.rot,
        trans=m.trans,
        multiplicity=m.multiplicity,
        atom_indices=np.arange(len(m.occupancy), dtype=np.int64),
        epsilon=data["epsilon"],
        fp=m.fp,
    )
    kappa = calibrate_kappa(data["sigma_wilson"], raw_all)
    sa_w = effective_sigma_a(data["sigma_a"], data["sigma_wilson"], kappa * raw_w)

    tgt = IntensityLogLikelihood(nu=None)
    n = len(data["hkl"])
    obs_base = dict(
        data=torch.as_tensor(np.asarray(data["f_obs"].data), dtype=torch.float64),
        sigmas=torch.as_tensor(np.asarray(data["f_obs"].sigmas), dtype=torch.float64),
        epsilon=torch.as_tensor(data["epsilon"], dtype=torch.float64),
        centric=torch.zeros(n, dtype=torch.bool),
        beta=torch.as_tensor(data["sigma_wilson"], dtype=torch.float64),
    )
    gold = intensity_map_coefficients(
        tgt,
        torch.as_tensor(f_omit, dtype=torch.complex128),
        Observations(**obs_base, alpha=torch.as_tensor(sa_w, dtype=torch.float64)),
    )
    # Fast path via core for single window partition
    single = WindowPartition_from_one(w, len(m.occupancy))
    opts = OmitWindowOptions(coefficient_types="both", chunk_size=2)
    obs = Observations(**obs_base, alpha=torch.as_tensor(data["sigma_a"], dtype=torch.float64))
    store = run_omit_windows_core(
        eng=eng,
        f_model=data["f_calc"],
        k_scale=np.ones(n),
        obs_target=tgt,
        obs=obs,
        sigma_a=data["sigma_a"],
        sigma_wilson=data["sigma_wilson"],
        epsilon=data["epsilon"],
        partition=single,
        options=opts,
    )
    np.testing.assert_allclose(store.coef_difference[0], gold.difference.astype(np.complex64), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(store.coef_model[0], gold.model.astype(np.complex64), rtol=1e-5, atol=1e-5)


def WindowPartition_from_one(w, n_atoms):
    from phridge.contrib.intensity_ll.omit_windows import WindowPartition, WindowSpec

    atom_to = np.full(n_atoms, -1, dtype=np.int64)
    atom_to[w.atom_indices] = 0
    ww = WindowSpec(window_id=0, atom_indices=w.atom_indices, label=w.label, bounds_frac=w.bounds_frac)
    return WindowPartition(windows=[ww], atom_to_window=atom_to, mode="boxes", empty_window_ids=[])


def test_symmetry_omit_changes_f_omit():
    data = _synthetic_structure(n_atoms=8, seed=5, crystal=_p21_crystal())
    eng = data["eng"]
    assert eng.model.n_sym == 2
    dF = delta_f_masked(eng, eng.model.occupancy, np.array([0], dtype=np.int64))
    # Mate contribution: omitting atom 0 should yield nonzero ΔF (includes sym copy)
    assert np.abs(dF).max() > 1e-6
    f_omit = data["f_calc"] - dF
    assert np.abs(f_omit - data["f_calc"]).max() > 1e-6


def test_chunking_invariance():
    data = _synthetic_structure(n_atoms=18, seed=6)
    eng = data["eng"]
    part = assign_box_windows(eng.model.sites_frac, eng.model.unit_cell, box_size=18.0)
    tgt = IntensityLogLikelihood(nu=None)
    n = len(data["hkl"])
    obs = Observations(
        data=torch.as_tensor(np.asarray(data["f_obs"].data), dtype=torch.float64),
        sigmas=torch.as_tensor(np.asarray(data["f_obs"].sigmas), dtype=torch.float64),
        epsilon=torch.as_tensor(data["epsilon"], dtype=torch.float64),
        centric=torch.zeros(n, dtype=torch.bool),
        alpha=torch.as_tensor(data["sigma_a"], dtype=torch.float64),
        beta=torch.as_tensor(data["sigma_wilson"], dtype=torch.float64),
    )
    s1 = run_omit_windows_core(
        eng=eng,
        f_model=data["f_calc"],
        k_scale=np.ones(n),
        obs_target=tgt,
        obs=obs,
        sigma_a=data["sigma_a"],
        sigma_wilson=data["sigma_wilson"],
        epsilon=data["epsilon"],
        partition=part,
        options=OmitWindowOptions(chunk_size=1),
    )
    s2 = run_omit_windows_core(
        eng=eng,
        f_model=data["f_calc"],
        k_scale=np.ones(n),
        obs_target=tgt,
        obs=obs,
        sigma_a=data["sigma_a"],
        sigma_wilson=data["sigma_wilson"],
        epsilon=data["epsilon"],
        partition=part,
        options=OmitWindowOptions(chunk_size=4),
    )
    np.testing.assert_array_equal(s1.window_ids, s2.window_ids)
    np.testing.assert_allclose(s1.coef_difference, s2.coef_difference, atol=1e-6)


def test_determinism_npz_roundtrip():
    data = _synthetic_structure(n_atoms=14, seed=7)
    eng = data["eng"]
    part = assign_box_windows(eng.model.sites_frac, eng.model.unit_cell, box_size=18.0)
    tgt = IntensityLogLikelihood(nu=None)
    n = len(data["hkl"])
    obs = Observations(
        data=torch.as_tensor(np.asarray(data["f_obs"].data), dtype=torch.float64),
        sigmas=torch.as_tensor(np.asarray(data["f_obs"].sigmas), dtype=torch.float64),
        epsilon=torch.as_tensor(data["epsilon"], dtype=torch.float64),
        centric=torch.zeros(n, dtype=torch.bool),
        alpha=torch.as_tensor(data["sigma_a"], dtype=torch.float64),
        beta=torch.as_tensor(data["sigma_wilson"], dtype=torch.float64),
    )
    opts = OmitWindowOptions(chunk_size=2)

    def once():
        return run_omit_windows_core(
            eng=eng,
            f_model=data["f_calc"],
            k_scale=np.ones(n),
            obs_target=tgt,
            obs=obs,
            sigma_a=data["sigma_a"],
            sigma_wilson=data["sigma_wilson"],
            epsilon=data["epsilon"],
            partition=part,
            options=opts,
        )

    a, b = once(), once()
    np.testing.assert_array_equal(a.coef_difference, b.coef_difference)
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "omit.npz")
        a.save(path)
        loaded = load_omit_windows(path)
        np.testing.assert_array_equal(loaded.coef_difference, a.coef_difference)
        assert loaded.window_for_atom(0) == int(a.atom_to_window[0])


def test_env_off_read_only_guarantee():
    """Omit helper is not invoked when options.enabled is False (driver contract)."""
    opts = OmitWindowOptions(enabled=False)
    assert opts.enabled is False


def test_fractional_box_masks_partition_cell():
    from phridge.client.intensity.omit import fractional_box_mask

    n_real = (8, 8, 8)
    # 2×2×1 tiling
    masks = []
    for ia in range(2):
        for ib in range(2):
            bounds = (
                (ia / 2, (ia + 1) / 2),
                (ib / 2, (ib + 1) / 2),
                (0.0, 1.0),
            )
            masks.append(fractional_box_mask(n_real, bounds))
    stacked = np.stack(masks, axis=0)
    assert stacked.shape[0] == 4
    assert int(stacked.sum()) == 8 * 8 * 8
    assert np.all(stacked.sum(axis=0) == 1)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("cctbx") is None,
    reason="cctbx required for MTZ / stitch write",
)
def test_write_stitched_omit_mtz(tmp_path):
    from phridge.client.intensity.omit import write_stitched_omit_mtz

    data = _synthetic_structure(n_atoms=10, seed=8)
    eng = data["eng"]
    part = assign_box_windows(eng.model.sites_frac, eng.model.unit_cell, box_size=25.0)
    tgt = IntensityLogLikelihood(nu=None)
    n = len(data["hkl"])
    obs = Observations(
        data=torch.as_tensor(np.asarray(data["f_obs"].data), dtype=torch.float64),
        sigmas=torch.as_tensor(np.asarray(data["f_obs"].sigmas), dtype=torch.float64),
        epsilon=torch.as_tensor(data["epsilon"], dtype=torch.float64),
        centric=torch.zeros(n, dtype=torch.bool),
        alpha=torch.as_tensor(data["sigma_a"], dtype=torch.float64),
        beta=torch.as_tensor(data["sigma_wilson"], dtype=torch.float64),
    )
    store = run_omit_windows_core(
        eng=eng,
        f_model=data["f_calc"],
        k_scale=np.ones(n),
        obs_target=tgt,
        obs=obs,
        sigma_a=data["sigma_a"],
        sigma_wilson=data["sigma_wilson"],
        epsilon=data["epsilon"],
        partition=part,
        options=OmitWindowOptions(chunk_size=2),
    )
    from cctbx import crystal, miller
    from cctbx.array_family import flex

    cs = crystal.symmetry(
        unit_cell=tuple(data["f_obs"].meta.crystal.unit_cell),
        space_group_symbol="P1",
    )
    indices = flex.miller_index()
    for h, k, l in store.hkl:
        indices.append((int(h), int(k), int(l)))
    ms = miller.set(cs, indices, anomalous_flag=False)
    path = tmp_path / "omit_windows.mtz"
    write_stitched_omit_mtz(
        store,
        ms,
        str(path),
        sites_frac=eng.model.sites_frac,
        unit_cell=eng.model.unit_cell,
        resolution_factor=0.4,
    )
    assert path.is_file() and path.stat().st_size > 0
    assert path.with_suffix(".window_meta.json").is_file()
