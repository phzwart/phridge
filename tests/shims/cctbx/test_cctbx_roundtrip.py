"""Bitwise xray.structure ↔ RObject round trip and packed-gradient layout."""

from __future__ import annotations

import numpy as np
import pytest

cctbx = pytest.importorskip("cctbx")

from shims.cctbx.conftest import random_xs  # noqa: E402
from phridge.shims.cctbx import from_robject, packed_order, to_robject  # noqa: E402

SPACE_GROUPS = ("P1", "P21", "P212121", "C2", "P4132", "R3:H")


def _columns(xs):
    scatterers = list(xs.scatterers())
    return {
        "sites_frac": np.array([tuple(sc.site) for sc in scatterers], dtype=np.float64),
        "u_iso": np.array([float(sc.u_iso) for sc in scatterers], dtype=np.float64),
        "u_star": np.array([tuple(float(x) for x in sc.u_star) for sc in scatterers], dtype=np.float64),
        "occupancy": np.array([float(sc.occupancy) for sc in scatterers], dtype=np.float64),
        "fp": np.array([float(sc.fp) for sc in scatterers], dtype=np.float64),
        "fdp": np.array([float(sc.fdp) for sc in scatterers], dtype=np.float64),
        "scatterer_type": [str(sc.scattering_type) for sc in scatterers],
        "uij_defined": np.array([bool(sc.flags.use_u_aniso()) for sc in scatterers]),
    }


@pytest.mark.parametrize("space_group", SPACE_GROUPS)
@pytest.mark.parametrize("aniso", [False, True])
def test_xray_structure_robject_roundtrip(space_group: str, aniso: bool) -> None:
    xs = random_xs(space_group, n_atoms=6, seed=3)
    for sc in xs.scatterers():
        sc.fp = 0.4
        sc.fdp = 1.2
        if aniso:
            sc.flags.set_use_u_iso(False)
            sc.flags.set_use_u_aniso(True)
            sc.u_star = (0.02, 0.03, 0.04, 0.001, -0.002, 0.003)
        else:
            sc.flags.set_use_u_iso(True)
            sc.flags.set_use_u_aniso(False)
    restored = from_robject(xs, to_robject(xs))
    before = _columns(xs)
    after = _columns(restored)
    for key in ("sites_frac", "u_iso", "u_star", "occupancy", "fp", "fdp", "uij_defined"):
        assert np.array_equal(before[key], after[key]), key
    assert before["scatterer_type"] == after["scatterer_type"]


def test_packed_order_matches_cctbx() -> None:
    xs = random_xs("P212121", n_atoms=8, elements=("C", "O", "N", "S"), seed=2)
    patterns = (
        dict(site=True, u_iso=True, u_aniso=False, occ=True, fp=False, fdp=False),
        dict(site=True, u_iso=False, u_aniso=True, occ=False, fp=True, fdp=True),
        dict(site=False, u_iso=True, u_aniso=True, occ=True, fp=True, fdp=False),
        dict(site=True, u_iso=True, u_aniso=True, occ=True, fp=True, fdp=True),
    )
    for i, sc in enumerate(xs.scatterers()):
        choice = patterns[i % len(patterns)]
        sc.flags.set_grad_site(choice["site"])
        sc.flags.set_use_u_iso(choice["u_iso"] or not choice["u_aniso"])
        sc.flags.set_grad_u_iso(choice["u_iso"])
        sc.flags.set_use_u_aniso(choice["u_aniso"])
        sc.flags.set_grad_u_aniso(choice["u_aniso"])
        sc.flags.set_grad_occupancy(choice["occ"])
        sc.flags.set_grad_fp(choice["fp"])
        sc.flags.set_grad_fdp(choice["fdp"])
        if choice["u_aniso"]:
            sc.u_star = (0.01, 0.02, 0.03, 0.0, 0.0, 0.0)
    order = packed_order(xs)
    for i, item in enumerate(xs.parameter_map()):
        for field, width in (
            ("site", 3),
            ("u_iso", 1),
            ("u_aniso", 6),
            ("occupancy", 1),
            ("fp", 1),
            ("fdp", 1),
        ):
            start = int(getattr(item, field))
            key = f"{field}_{i}"
            if start < 0:
                assert key not in order
            else:
                assert order[key] == slice(start, start + width)
