"""Restraint-provider conformance."""

from __future__ import annotations

import numpy as np

from phridge.contracts.packing import PackedRestraints
from phridge.contracts.tensors import RObject
from phridge.r.restraints.torch_geometry.provider import GEOMETRY_MANIFEST, TorchGeometry


def _triangle() -> PackedRestraints:
    return PackedRestraints(
        n_sites=3,
        bond_i_seqs=np.array([[0, 1], [1, 2], [2, 0]], dtype=np.int32),
        bond_distance_ideal=np.array([1.5, 1.5, 1.5]),
        bond_weight=np.array([100.0, 100.0, 100.0]),
        bond_slack=np.array([0.0, 0.0, 0.0]),
        angle_i_seqs=np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]], dtype=np.int32),
        angle_ideal=np.array([60.0, 60.0, 60.0]),
        angle_weight=np.array([1.0, 1.0, 1.0]),
    )


def _record(sites):
    import torch

    n = sites.shape[0]
    z = torch.zeros(n, dtype=sites.dtype)
    return RObject(
        sites_frac=sites,
        u_iso=z,
        u_star=torch.zeros(n, 6, dtype=sites.dtype),
        occupancy=torch.ones(n, dtype=sites.dtype),
        fp=z,
        fdp=z,
        scatterer_type=torch.zeros(n, dtype=torch.int64),
        uij_defined=torch.zeros(n, dtype=torch.bool),
    )


def test_gradient_matches_finite_difference() -> None:
    import torch

    provider = TorchGeometry(_triangle())
    sites = torch.tensor([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [0.9, 1.2, 0.0]], dtype=torch.float64)
    _, grad = provider.energy(_record(sites))
    eps = 1e-6
    numeric = torch.zeros_like(sites)
    for i in range(sites.shape[0]):
        for k in range(3):
            plus = sites.clone()
            minus = sites.clone()
            plus[i, k] += eps
            minus[i, k] -= eps
            vp, _ = provider.energy(_record(plus))
            vm, _ = provider.energy(_record(minus))
            numeric[i, k] = (vp - vm) / (2.0 * eps)
    rel = torch.linalg.norm(grad.sites_frac - numeric) / torch.linalg.norm(numeric)
    assert float(rel) < 1e-6


def test_energy_nonnegative() -> None:
    import torch

    provider = TorchGeometry(_triangle())
    sites = torch.tensor([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [0.9, 1.2, 0.0]], dtype=torch.float64)
    value, _ = provider.energy(_record(sites))
    assert float(value) >= 0.0


def test_covers_declared() -> None:
    provider = TorchGeometry(_triangle())
    assert provider.covers() == set(GEOMETRY_MANIFEST.produces)
