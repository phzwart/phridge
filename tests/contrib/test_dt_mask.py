"""Distance-transform F_mask: φ, periodic EDT, options, CLI flags."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from phridge.contrib.intensity_ll import dt_mask as DT


class _FakeMiller:
    def __init__(self, data: np.ndarray):
        self._data = data

    def data(self) -> np.ndarray:
        return self._data


def test_f_mask_compare_identical_and_scaled():
    rng = np.random.default_rng(0)
    a = rng.normal(size=20) + 1j * rng.normal(size=20)
    same = DT._f_mask_compare(_FakeMiller(a), _FakeMiller(a))
    assert "|corr|=1.000" in same
    assert "‖new‖/‖old‖=1.000" in same
    scaled = DT._f_mask_compare(_FakeMiller(a), _FakeMiller(2.0 * a))
    assert "|corr|=1.000" in scaled
    assert "‖new‖/‖old‖=2.000" in scaled


def test_real_space_stats_reports_modulation():
    mask = np.zeros((8, 8, 8), dtype=np.float64)
    mask[:, :, 4:] = 1.0
    distance = np.zeros_like(mask)
    distance[:, :, 4:] = 2.0
    opts = DT.DistanceMaskOptions(alpha=0.4, length_scale=2.0, phi="exp")
    rho = DT.modulate_mask(mask, distance, opts)
    built = DT.BuiltDistanceMask(
        mask=mask,
        distance=distance,
        density=rho,
        density_mod=DT.modulation_component(mask, distance, opts),
        f_mask=None,
        f_mask_mod=None,
        sampling=(1.0, 1.0, 1.0),
    )
    text = DT.real_space_stats(built)
    assert "sol=0.50" in text
    assert "⟨d⟩=2.00Å" in text
    assert "rms(ρ−M)/M=" in text
    assert DT.real_space_stats(
        DT.BuiltDistanceMask(
            mask=np.zeros((1, 1, 1)),
            distance=np.zeros((1, 1, 1)),
            density=np.zeros((1, 1, 1)),
            density_mod=np.zeros((1, 1, 1)),
            f_mask=None,
            f_mask_mod=None,
            sampling=(1.0, 1.0, 1.0),
        )
    ) == "no real-space map"


def test_resolution_factor_from_mmtbx_grid_step():
    """mmtbx default grid_step_factor=4 must become resolution_factor=0.25, not 4."""

    class _P:
        def __init__(self, grid_step_factor: float, n_real=None):
            self.grid_step_factor = grid_step_factor
            self.n_real = n_real

    assert DT.resolution_factor_from_mask_params(None) == pytest.approx(1.0 / 3.0)
    assert DT.resolution_factor_from_mask_params(_P(4.0)) == pytest.approx(0.25)
    assert DT.resolution_factor_from_mask_params(_P(10.0)) == pytest.approx(0.10)
    assert DT.resolution_factor_from_mask_params(_P(0.33)) == pytest.approx(0.33)
    assert DT.resolution_factor_from_mask_params(_P(0.0)) == pytest.approx(1.0 / 3.0)
    assert DT.resolution_factor_from_mask_params(_P(4.0)) <= 0.5


def test_pydantic_accept_reject():
    DT.DistanceMaskOptions()
    DT.DistanceMaskOptions(enabled=True, alpha=0.2, length_scale=1.5, phi="shell")
    DT.DistanceMaskOptions(phi="exponential")
    with pytest.raises(ValidationError):
        DT.DistanceMaskOptions(alpha=-0.1)
    with pytest.raises(ValidationError):
        DT.DistanceMaskOptions(length_scale=0.0)
    with pytest.raises(ValidationError):
        DT.DistanceMaskOptions(phi="nope")
    with pytest.raises(ValidationError):
        DT.DistanceMaskOptions(not_a_field=1)  # type: ignore[call-arg]


def test_options_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("PHRIDGE_DT_MASK", raising=False)
    off = DT.options_from_env()
    assert off.enabled is False
    monkeypatch.setenv("PHRIDGE_DT_MASK", "1")
    monkeypatch.setenv("PHRIDGE_DT_MASK_ALPHA", "0.5")
    monkeypatch.setenv("PHRIDGE_DT_MASK_LENGTH", "3")
    monkeypatch.setenv("PHRIDGE_DT_MASK_PHI", "shell")
    monkeypatch.setenv("PHRIDGE_DT_MASK_MODE", "two_component")
    on = DT.options_from_env()
    assert on.enabled is True
    assert on.alpha == pytest.approx(0.5)
    assert on.length_scale == pytest.approx(3.0)
    assert on.phi == "shell"
    assert on.mode == "two_component"


def test_phi_decays_to_zero():
    d = np.array([0.0, 2.0, 20.0])
    exp = DT.phi_of_distance(d, DT.DistanceMaskOptions(phi="exp", length_scale=2.0))
    assert exp[0] == pytest.approx(1.0)
    assert exp[1] == pytest.approx(np.e ** -1.0)
    assert exp[2] < 1e-4
    shell = DT.phi_of_distance(d, DT.DistanceMaskOptions(phi="shell", center=1.4, width=1.0))
    assert shell[2] < 1e-8
    logistic = DT.phi_of_distance(d, DT.DistanceMaskOptions(phi="logistic", center=2.0, width=0.5))
    assert logistic[2] < 1e-10


def test_modulate_keeps_protein_zero_and_bulk_at_one():
    mask = np.zeros((4, 4, 4), dtype=np.float64)
    mask[:, :, 2:] = 1.0
    distance = np.zeros_like(mask)
    distance[:, :, 2:] = np.arange(2)[None, None, :] * 10.0  # 0 Å at the face, 10 Å in bulk
    opts = DT.DistanceMaskOptions(alpha=0.4, length_scale=2.0, phi="exp")
    rho = DT.modulate_mask(mask, distance, opts)
    assert np.all(rho[:, :, :2] == 0.0)
    # At the interface (d=0) the modulation is full: 1+α
    assert rho[0, 0, 2] == pytest.approx(1.4)
    # Far into solvent φ→0 so ρ→1 (the mask constant)
    assert rho[0, 0, 3] == pytest.approx(1.0, abs=1e-2)


def test_periodic_distance_transform_box():
    mask = np.ones((16, 16, 16), dtype=bool)
    mask[6:10, 6:10, 6:10] = False  # protein cube
    sampling = (1.0, 1.0, 1.0)
    d = DT.periodic_distance_transform(mask, sampling)
    assert d[8, 8, 8] == pytest.approx(0.0)
    # Voxel just outside the cube along +x: protein occupies 6..9 inclusive
    assert d[10, 8, 8] == pytest.approx(1.0, abs=1e-6)
    assert d[5, 8, 8] == pytest.approx(1.0, abs=1e-6)
    # Protein stays 0
    assert np.all(d[~mask] == 0.0)
    assert np.all(d[mask] > 0.0)


def test_periodic_distance_wraps():
    mask = np.ones((12, 12, 12), dtype=bool)
    mask[0:2, :, :] = False  # protein slab on the low-x face
    d = DT.periodic_distance_transform(mask, (1.0, 1.0, 1.0))
    # High-x face is one voxel from the wrapped protein at x=0
    assert d[11, 6, 6] == pytest.approx(1.0, abs=1e-6)
    assert d[2, 6, 6] == pytest.approx(1.0, abs=1e-6)


def test_densities_from_mask_two_component_orthogonal():
    mask = np.ones((8, 8, 8), dtype=np.float64)
    mask[3:5, 3:5, 3:5] = 0.0
    opts = DT.DistanceMaskOptions(enabled=True, alpha=0.5, length_scale=2.0, mode="two_component")
    distance, baked, mod = DT.densities_from_mask(mask, (1.0, 1.0, 1.0), opts)
    np.testing.assert_allclose(baked, mask + float(opts.alpha) * mod, atol=1e-12)
    assert np.all(mod[mask < 0.5] == 0.0)
    assert float(np.max(mod)) <= 1.0 + 1e-12


def test_build_f_masks_baked_differs_from_flat():
    pytest.importorskip("cctbx")
    pytest.importorskip("mmtbx")
    from cctbx import sgtbx
    from cctbx.array_family import flex
    from cctbx.development import random_structure

    from phridge.client.intensity.model import IntensityModel

    flex.set_random_seed(0)
    xs = random_structure.xray_structure(
        space_group_info=sgtbx.space_group_info("P1"),
        elements=["C", "N", "O"] * 4,
        volume_per_atom=50.0,
        random_u_iso=True,
    )
    fc = xs.structure_factors(d_min=3.0).f_calc()
    i_obs = fc.customized_copy(
        data=flex.double(np.maximum(np.abs(np.asarray(fc.data())) ** 2, 0.1).tolist()),
        sigmas=flex.double(fc.size(), 1.0),
    )
    i_obs.set_observation_type_xray_intensity()
    flat = IntensityModel(xs, i_obs, use_bulk_solvent=True, n_bins=4)
    flat.compute_mask()
    baked = IntensityModel(
        xs,
        i_obs,
        use_bulk_solvent=True,
        n_bins=4,
        dt_mask=DT.DistanceMaskOptions(enabled=True, alpha=0.4, length_scale=2.0, mode="baked"),
    )
    baked.compute_mask()
    a = np.asarray(flat.f_mask.data(), dtype=np.complex128)
    b = np.asarray(baked.f_mask.data(), dtype=np.complex128)
    assert a.shape == b.shape
    assert np.linalg.norm(a - b) > 1e-6 * np.linalg.norm(a)
    split = IntensityModel(
        xs,
        i_obs,
        use_bulk_solvent=True,
        n_bins=4,
        dt_mask=DT.DistanceMaskOptions(enabled=True, alpha=0.4, mode="two_component"),
    )
    split.compute_mask()
    assert split.f_mask_mod is not None
    assert split.f_mask_mod.size() == i_obs.size()


def test_k_mask_is_negligible():
    from phridge.client.intensity.engine import _k_mask_is_negligible

    assert _k_mask_is_negligible(None)
    assert _k_mask_is_negligible(np.zeros(10))
    assert _k_mask_is_negligible(np.full(10, 0.01))
    assert not _k_mask_is_negligible(np.linspace(0.0, 0.3, 10))


def test_fit_nu_this_cycle_defaults_to_first_only(monkeypatch: pytest.MonkeyPatch):
    from phridge.client.intensity.engine import _fit_nu_this_cycle

    monkeypatch.delenv("PHRIDGE_FIT_NU_EVERY_CYCLE", raising=False)
    monkeypatch.delenv("PHRIDGE_FIT_NU_CYCLES", raising=False)
    assert _fit_nu_this_cycle(1) is True
    assert _fit_nu_this_cycle(2) is False
    monkeypatch.setenv("PHRIDGE_FIT_NU_EVERY_CYCLE", "1")
    assert _fit_nu_this_cycle(4) is True
    monkeypatch.setenv("PHRIDGE_FIT_NU_EVERY_CYCLE", "0")
    monkeypatch.setenv("PHRIDGE_FIT_NU_CYCLES", "2")
    assert _fit_nu_this_cycle(2) is True
    assert _fit_nu_this_cycle(3) is False


def test_cli_scripts_accept_dt_mask_flag():
    root = Path(__file__).resolve().parents[2]
    driver = (root / "scripts" / "phenix_refine_mli.py").read_text()
    wrapper = (root / "scripts" / "run_phenix_intensity.sh").read_text()
    cli = (root / "src" / "phridge" / "client" / "intensity" / "cli.py").read_text()
    assert "--dt-mask" in driver
    assert "PHRIDGE_DT_MASK" in driver
    assert "--dt-mask" in wrapper
    assert "PHRIDGE_DT_MASK" in wrapper
    assert "--dt-mask" in cli
    assert "PHRIDGE_DT_MASK" in cli
