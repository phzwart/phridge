"""Direct-summation structure factors. Reference implementation, not a fast path."""

from __future__ import annotations

import copy
import math
from typing import Any, Optional

import numpy as np

from phridge.contracts.manifest import PluginManifest, Slot, Space, Tier
from phridge.contracts.tensors import ComplexArray, RObject
from phridge.ops import register_manifest
from phridge.transforms.fourier.nufft.nufft_engine import TWO_PI2, _form_factors
from phridge.transforms.fourier.ten_eyck.cell import orthogonalization_matrix, reciprocal_cartesian, sym6_to_mat

MANIFEST = PluginManifest(
    name="direct",
    slot=Slot.fourier,
    space=Space.transform,
    from_space=Space.R,
    to_space=Space.Q,
    tier=Tier.native,
    dtype="float64",
    provides=["adjoint"],
)


def direct_f_calc(model: Any, hkl: np.ndarray, atom_indices: Optional[np.ndarray] = None) -> np.ndarray:
    """Exact expanded-atom summation for a subset of ASU atoms."""
    idx = np.arange(model.n_scatterers) if atom_indices is None else np.asarray(atom_indices, dtype=np.int64)
    hkl = np.asarray(hkl, dtype=np.float64).reshape(-1, 3)
    dstar = reciprocal_cartesian(model.unit_cell, hkl)
    dstar2 = np.sum(dstar**2, axis=1)
    stol2 = dstar2 / 4.0
    ff_type = _form_factors(model.gauss_a, model.gauss_b, model.gauss_c, stol2)
    rot = np.asarray(model.rot, dtype=np.float64)
    trans = np.asarray(model.trans, dtype=np.float64)
    o = orthogonalization_matrix(model.unit_cell)
    f = np.zeros(hkl.shape[0], dtype=np.complex128)
    n_sym = rot.shape[0]
    for j in idx:
        ga = ff_type[int(model.type_index[j])]
        fj = ga + float(model.fp[j]) + 1j * float(model.fdp[j])
        w = float(model.occupancy[j]) * float(model.multiplicity[j]) / float(n_sym)
        u_iso = float(model.u_iso[j])
        aniso = bool(model.anisotropic[j])
        if aniso:
            u_mat = sym6_to_mat(np.asarray(model.u_star[j : j + 1], dtype=np.float64))[0]
        for s in range(n_sym):
            x = rot[s] @ model.sites_frac[j] + trans[s]
            if aniso:
                u_sym = rot[s] @ u_mat @ rot[s].T
                u_cart = o @ u_sym @ o.T
                dw = np.exp(-TWO_PI2 * np.einsum("hi,ij,hj->h", dstar, u_cart, dstar))
            else:
                dw = np.exp(-TWO_PI2 * u_iso * dstar2)
            with np.errstate(all="ignore"):
                phase = np.exp(2j * math.pi * (hkl @ x))
            f += w * fj * dw * phase
    return f


class DirectFourier:
    """Reference operator. ``model`` is the scattering layout; ``forward`` reads ``r``."""

    def __init__(self, model: Any) -> None:
        self.model = copy.deepcopy(model)

    def forward(self, r: RObject, hkl: Any) -> ComplexArray:
        import torch

        model = self.model
        model.sites_frac = r.sites_frac.detach().cpu().numpy()
        model.occupancy = r.occupancy.detach().cpu().numpy()
        model.u_iso = r.u_iso.detach().cpu().numpy()
        model.u_star = r.u_star.detach().cpu().numpy()
        model.fp = r.fp.detach().cpu().numpy()
        model.fdp = r.fdp.detach().cpu().numpy()
        values = torch.as_tensor(direct_f_calc(model, np.asarray(hkl.detach().cpu().numpy())), dtype=torch.complex128)
        return ComplexArray(hkl=hkl, values=values)

    def adjoint(self, r: RObject, hkl: Any, d_values: Any) -> RObject:
        """Central difference of ``forward`` against the complex residual. Reference only."""
        import torch

        eps = 1e-6
        base = r.sites_frac.detach()
        grad = torch.zeros_like(base)
        g = d_values.detach()
        for i in range(base.shape[0]):
            for k in range(3):
                plus = base.clone()
                minus = base.clone()
                plus[i, k] += eps
                minus[i, k] -= eps
                fp = self.forward(_replace(r, plus), hkl).values
                fm = self.forward(_replace(r, minus), hkl).values
                dq = ((fp - fm) * g.conj()).real.sum() / (2.0 * eps)
                grad[i, k] = dq
        return RObject(
            sites_frac=grad,
            u_iso=torch.zeros_like(r.u_iso),
            u_star=torch.zeros_like(r.u_star),
            occupancy=torch.zeros_like(r.occupancy),
            fp=torch.zeros_like(r.fp),
            fdp=torch.zeros_like(r.fdp),
            scatterer_type=r.scatterer_type,
            uij_defined=r.uij_defined,
        )


def _replace(record: RObject, sites_frac: Any) -> RObject:
    return record._replace(sites_frac=sites_frac)


def register() -> None:
    register_manifest(MANIFEST, impl=DirectFourier)
