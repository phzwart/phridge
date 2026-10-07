"""xray.structure adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from phridge.contracts.models import Scatterer
from phridge.contracts.packing import PackedXray
from phridge.shims.cctbx._flex import xray_scatterer
from phridge.shims.cctbx.symmetry import crystal_from_cctbx, crystal_to_cctbx


from phridge.contracts.manifest import PluginManifest, Slot, Space, Tier

MANIFEST = PluginManifest(
    name='cctbx_xray_structure',
    slot=Slot.io_adapter,
    space=Space.D,
    tier=Tier.shim,
)

def xray_from_cctbx(structure: Any) -> PackedXray:
    scatterers = structure.scatterers()
    n = scatterers.size()
    labels: list[Scatterer] = []
    u_star = np.zeros((n, 6), dtype=np.float64)
    u_iso = np.asarray(list(structure.extract_u_iso_or_u_equiv()), dtype=np.float64)
    for i, sc in enumerate(scatterers):
        aniso = bool(sc.flags.use_u_aniso())
        if aniso:
            u_star[i] = sc.u_star
        fp = float(sc.fp) if sc.fp not in (None, 0) else None
        fdp = float(sc.fdp) if sc.fdp not in (None, 0) else None
        labels.append(
            Scatterer(
                i=i,
                scattering_type=str(sc.scattering_type),
                fp=fp,
                fdp=fdp,
                anisotropic=aniso,
                use_u_iso=bool(sc.flags.use_u_iso()),
            )
        )
    return PackedXray(
        crystal=crystal_from_cctbx(structure.crystal_symmetry()),
        sites_frac=np.asarray(list(structure.sites_frac()), dtype=np.float64),
        occupancy=np.asarray(list(scatterers.extract_occupancies()), dtype=np.float64),
        u_iso=u_iso,
        scatterers=labels,
        u_star=u_star,
    )


def xray_to_cctbx(packed: PackedXray) -> Any:
    from cctbx import xray
    from cctbx.xray import scatterer

    items = []
    for meta, site, occ, u_iso, u_star in zip(
        packed.meta.scatterers,
        packed.sites_frac,
        packed.occupancy,
        packed.u_iso,
        packed.u_star,
    ):
        if meta.anisotropic:
            sc = scatterer(
                label=f"{meta.scattering_type}{meta.i}",
                scattering_type=meta.scattering_type,
                site=tuple(float(x) for x in site),
                occupancy=float(occ),
                u=tuple(float(x) for x in u_star),
            )
        else:
            sc = scatterer(
                label=f"{meta.scattering_type}{meta.i}",
                scattering_type=meta.scattering_type,
                site=tuple(float(x) for x in site),
                occupancy=float(occ),
                u=float(u_iso),
            )
        if meta.fp is not None:
            sc.fp = float(meta.fp)
        if meta.fdp is not None:
            sc.fdp = float(meta.fdp)
        items.append(sc)
    return xray.structure(
        crystal_symmetry=crystal_to_cctbx(packed.meta.crystal),
        scatterers=xray_scatterer(items),
    )


def to_robject(xs: Any) -> Any:
    """Scatterer fields as an :class:`RObject`. Numpy arrays; this package does not import torch."""
    from phridge.contracts.tensors import RObject

    scatterers = list(xs.scatterers())
    labels = [str(sc.scattering_type) for sc in scatterers]
    vocab = sorted(set(labels))
    index = {name: i for i, name in enumerate(vocab)}
    return RObject(
        sites_frac=np.array([tuple(sc.site) for sc in scatterers], dtype=np.float64),
        u_iso=np.array([float(sc.u_iso) for sc in scatterers], dtype=np.float64),
        u_star=np.array([tuple(float(x) for x in sc.u_star) for sc in scatterers], dtype=np.float64),
        occupancy=np.array([float(sc.occupancy) for sc in scatterers], dtype=np.float64),
        fp=np.array([float(sc.fp) for sc in scatterers], dtype=np.float64),
        fdp=np.array([float(sc.fdp) for sc in scatterers], dtype=np.float64),
        scatterer_type=np.array([index[name] for name in labels], dtype=np.int64),
        uij_defined=np.array([bool(sc.flags.use_u_aniso()) for sc in scatterers], dtype=bool),
    )


def from_robject(xs_template: Any, record: Any) -> Any:
    """Write an :class:`RObject` back onto a copy of ``xs_template``."""
    xs = xs_template.deep_copy_scatterers()
    scatterers = list(xs.scatterers())
    vocab = sorted({str(sc.scattering_type) for sc in scatterers})
    for i, sc in enumerate(scatterers):
        sc.site = tuple(float(x) for x in record.sites_frac[i])
        sc.occupancy = float(record.occupancy[i])
        sc.fp = float(record.fp[i])
        sc.fdp = float(record.fdp[i])
        sc.scattering_type = vocab[int(record.scatterer_type[i])]
        if bool(record.uij_defined[i]):
            sc.flags.set_use_u_aniso(True)
            sc.flags.set_use_u_iso(False)
            sc.u_star = tuple(float(x) for x in record.u_star[i])
        else:
            sc.flags.set_use_u_iso(True)
            sc.flags.set_use_u_aniso(False)
            sc.u_iso = float(record.u_iso[i])
    return xs


def packed_order(xs: Any) -> dict[str, slice]:
    """cctbx ``parameter_map`` layout: site, u_iso, u_aniso, occupancy, fp, fdp."""
    order: dict[str, slice] = {}
    cursor = 0
    for i, sc in enumerate(xs.scatterers()):
        flags = sc.flags
        if flags.grad_site():
            order[f"site_{i}"] = slice(cursor, cursor + 3)
            cursor += 3
        if flags.use_u_iso() and flags.grad_u_iso():
            order[f"u_iso_{i}"] = slice(cursor, cursor + 1)
            cursor += 1
        if flags.use_u_aniso() and flags.grad_u_aniso():
            order[f"u_aniso_{i}"] = slice(cursor, cursor + 6)
            cursor += 6
        if flags.grad_occupancy():
            order[f"occupancy_{i}"] = slice(cursor, cursor + 1)
            cursor += 1
        if flags.grad_fp():
            order[f"fp_{i}"] = slice(cursor, cursor + 1)
            cursor += 1
        if flags.grad_fdp():
            order[f"fdp_{i}"] = slice(cursor, cursor + 1)
            cursor += 1
    return order
