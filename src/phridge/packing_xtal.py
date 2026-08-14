"""Packed npz helpers for reflections, maps, and coordinates."""

from __future__ import annotations

import io
from typing import Optional

import numpy as np

from phridge.models import (
    AnomalousLayout,
    Atom,
    CartesianSites,
    ComplexMap,
    CoordinateFrame,
    CrystalSymmetry,
    FractionalSites,
    HendricksonLattman,
    Hierarchy,
    MapCoefficients,
    MapSpace,
    ObservationType,
    ReflectionColumn,
    ReflectionFile,
    Scatterer,
    XrayStructure,
)
from phridge.packing import (
    PackedMiller,
    as_canonical_complex,
    as_canonical_float,
    as_canonical_int,
)


def _savez(**arrays: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def _load_npz(blob: bytes):
    return np.load(io.BytesIO(blob), allow_pickle=False)


class PackedHL:
    def __init__(
        self,
        crystal: CrystalSymmetry,
        hkl: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        d: np.ndarray,
        *,
        anomalous: bool = False,
        anomalous_layout: AnomalousLayout = AnomalousLayout.asu,
        label: Optional[str] = None,
    ) -> None:
        self.hkl = as_canonical_int(np.asarray(hkl))
        self.a = as_canonical_float(np.asarray(a))
        self.b = as_canonical_float(np.asarray(b))
        self.c = as_canonical_float(np.asarray(c))
        self.d = as_canonical_float(np.asarray(d))
        n = self.hkl.shape[0]
        if self.hkl.ndim != 2 or self.hkl.shape[1] != 3:
            raise ValueError("hkl must have shape (N, 3)")
        for name, arr in ("A", self.a), ("B", self.b), ("C", self.c), ("D", self.d):
            if arr.shape[0] != n:
                raise ValueError(f"{name} length mismatch")
        self.meta = HendricksonLattman(
            label=label,
            crystal=crystal,
            anomalous=anomalous,
            anomalous_layout=anomalous_layout,
            n_refl=int(n),
            index_dtype="int32",
        )

    def pack(self) -> bytes:
        return _savez(hkl=self.hkl, A=self.a, B=self.b, C=self.c, D=self.d)


def unpack_hl(blob: bytes, meta: HendricksonLattman) -> PackedHL:
    with _load_npz(blob) as zf:
        return PackedHL(
            crystal=meta.crystal,
            hkl=zf["hkl"],
            a=zf["A"],
            b=zf["B"],
            c=zf["C"],
            d=zf["D"],
            anomalous=meta.anomalous,
            anomalous_layout=meta.anomalous_layout,
            label=meta.label,
        )


class PackedReflections:
    def __init__(
        self,
        crystal: CrystalSymmetry,
        hkl: np.ndarray,
        columns: list[ReflectionColumn],
        data: dict[str, np.ndarray],
        sigmas: Optional[dict[str, np.ndarray]] = None,
        *,
        wavelength: Optional[float] = None,
        history: Optional[list[str]] = None,
        anomalous_layout: AnomalousLayout = AnomalousLayout.asu,
    ) -> None:
        self.hkl = as_canonical_int(np.asarray(hkl))
        if self.hkl.ndim != 2 or self.hkl.shape[1] != 3:
            raise ValueError("hkl must have shape (N, 3)")
        n = self.hkl.shape[0]
        self.data = {k: as_canonical_float(np.asarray(v)) for k, v in data.items()}
        self.sigmas = {
            k: as_canonical_float(np.asarray(v)) for k, v in (sigmas or {}).items()
        }
        for key, arr in self.data.items():
            if arr.shape[0] != n:
                raise ValueError(f"column {key} length mismatch")
        self.meta = ReflectionFile(
            crystal=crystal,
            wavelength=wavelength,
            history=history or [],
            n_refl=int(n),
            index_dtype="int32",
            anomalous_layout=anomalous_layout,
            columns=columns,
        )

    def pack(self) -> bytes:
        payload: dict[str, np.ndarray] = {"hkl": self.hkl, **self.data}
        for name, arr in self.sigmas.items():
            payload[f"{name}_sigmas"] = arr
        return _savez(**payload)


def unpack_reflections(blob: bytes, meta: ReflectionFile) -> PackedReflections:
    with _load_npz(blob) as zf:
        data = {col.npz_name: zf[col.npz_name] for col in meta.columns}
        sigmas = {}
        for col in meta.columns:
            key = f"{col.npz_name}_sigmas"
            if key in zf.files:
                sigmas[col.npz_name] = zf[key]
        return PackedReflections(
            crystal=meta.crystal,
            hkl=zf["hkl"],
            columns=meta.columns,
            data=data,
            sigmas=sigmas,
            wavelength=meta.wavelength,
            history=list(meta.history),
            anomalous_layout=meta.anomalous_layout,
        )


class PackedComplexMap:
    def __init__(
        self,
        crystal: CrystalSymmetry,
        data: np.ndarray,
        origin: Optional[list[int]] = None,
        n_real: Optional[list[int]] = None,
        *,
        space: MapSpace = MapSpace.p1,
        label: Optional[str] = None,
    ) -> None:
        self.data = as_canonical_complex(np.asarray(data))
        if self.data.ndim != 3:
            raise ValueError("complex map data must be 3-D")
        if n_real is None:
            n_real = [int(n) for n in self.data.shape]
        if list(self.data.shape) != list(n_real):
            raise ValueError("data.shape must equal n_real")
        self.meta = ComplexMap(
            label=label,
            crystal=crystal,
            origin=[0, 0, 0] if origin is None else [int(x) for x in origin],
            n_real=[int(x) for x in n_real],
            space=space,
            dtype="complex128",
        )

    def pack(self) -> bytes:
        return _savez(data=self.data)


def unpack_complex_map(blob: bytes, meta: ComplexMap) -> PackedComplexMap:
    with _load_npz(blob) as zf:
        return PackedComplexMap(
            crystal=meta.crystal,
            data=zf["data"],
            origin=meta.origin,
            n_real=meta.n_real,
            space=meta.space,
            label=meta.label,
        )


class PackedMapCoefficients:
    def __init__(self, miller: PackedMiller, label: Optional[str] = None) -> None:
        if miller.meta.observation_type != ObservationType.complex:
            raise ValueError("MapCoefficients require complex miller data")
        self.miller = miller
        self.meta = MapCoefficients(label=label or miller.meta.label, miller=miller.meta)

    def pack(self) -> bytes:
        return self.miller.pack()


def unpack_map_coefficients(blob: bytes, meta: MapCoefficients) -> PackedMapCoefficients:
    from phridge.packing import unpack_miller

    packed = unpack_miller(blob, meta.miller)
    return PackedMapCoefficients(packed, label=meta.label)


class PackedCartesian:
    def __init__(
        self,
        xyz: np.ndarray,
        crystal: Optional[CrystalSymmetry] = None,
    ) -> None:
        self.xyz = as_canonical_float(np.asarray(xyz))
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise ValueError("xyz must have shape (N, 3)")
        self.meta = CartesianSites(
            crystal=crystal,
            n_sites=int(self.xyz.shape[0]),
            dtype="float64",
        )

    def pack(self) -> bytes:
        return _savez(xyz=self.xyz)


def unpack_cartesian(blob: bytes, meta: CartesianSites) -> PackedCartesian:
    with _load_npz(blob) as zf:
        packed = PackedCartesian(zf["xyz"], crystal=meta.crystal)
    if packed.meta.n_sites != meta.n_sites:
        raise ValueError("n_sites mismatch")
    return packed


class PackedFractional:
    def __init__(self, xyz: np.ndarray, crystal: CrystalSymmetry) -> None:
        self.xyz = as_canonical_float(np.asarray(xyz))
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise ValueError("xyz must have shape (N, 3)")
        self.meta = FractionalSites(
            crystal=crystal,
            n_sites=int(self.xyz.shape[0]),
            dtype="float64",
        )

    def pack(self) -> bytes:
        return _savez(xyz=self.xyz)


def unpack_fractional(blob: bytes, meta: FractionalSites) -> PackedFractional:
    with _load_npz(blob) as zf:
        packed = PackedFractional(zf["xyz"], crystal=meta.crystal)
    if packed.meta.n_sites != meta.n_sites:
        raise ValueError("n_sites mismatch")
    return packed


class PackedHierarchy:
    def __init__(
        self,
        xyz: np.ndarray,
        occupancy: np.ndarray,
        b_iso: np.ndarray,
        atoms: list[Atom],
        crystal: Optional[CrystalSymmetry] = None,
        frame: CoordinateFrame = CoordinateFrame.cartesian,
        u_cart: Optional[np.ndarray] = None,
        uij_defined: Optional[np.ndarray] = None,
    ) -> None:
        self.xyz = as_canonical_float(np.asarray(xyz))
        self.occupancy = as_canonical_float(np.asarray(occupancy))
        self.b_iso = as_canonical_float(np.asarray(b_iso))
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise ValueError("xyz must have shape (N, 3)")
        n = self.xyz.shape[0]
        if self.occupancy.shape[0] != n or self.b_iso.shape[0] != n:
            raise ValueError("occupancy/b_iso length mismatch")
        if len(atoms) != n:
            raise ValueError("atoms length must equal n_atoms")
        self.u_cart = None
        self.uij_defined = None
        has_uij = False
        if u_cart is not None:
            self.u_cart = as_canonical_float(np.asarray(u_cart))
            if self.u_cart.shape != (n, 6):
                raise ValueError("u_cart must have shape (N, 6)")
            self.uij_defined = (
                np.ones(n, dtype=np.uint8)
                if uij_defined is None
                else np.ascontiguousarray(uij_defined, dtype=np.uint8)
            )
            has_uij = True
        self.meta = Hierarchy(
            crystal=crystal,
            n_atoms=int(n),
            frame=frame,
            has_uij=has_uij,
            atoms=atoms,
        )

    def pack(self) -> bytes:
        payload = {"xyz": self.xyz, "occupancy": self.occupancy, "b_iso": self.b_iso}
        if self.u_cart is not None:
            payload["u_cart"] = self.u_cart
            payload["uij_defined"] = self.uij_defined
        return _savez(**payload)


def unpack_hierarchy(blob: bytes, meta: Hierarchy) -> PackedHierarchy:
    with _load_npz(blob) as zf:
        u_cart = zf["u_cart"] if "u_cart" in zf.files else None
        uij_defined = zf["uij_defined"] if "uij_defined" in zf.files else None
        return PackedHierarchy(
            xyz=zf["xyz"],
            occupancy=zf["occupancy"],
            b_iso=zf["b_iso"],
            atoms=meta.atoms,
            crystal=meta.crystal,
            frame=meta.frame,
            u_cart=u_cart,
            uij_defined=uij_defined,
        )


class PackedXray:
    def __init__(
        self,
        crystal: CrystalSymmetry,
        sites_frac: np.ndarray,
        occupancy: np.ndarray,
        u_iso: np.ndarray,
        scatterers: list[Scatterer],
        u_star: Optional[np.ndarray] = None,
    ) -> None:
        self.sites_frac = as_canonical_float(np.asarray(sites_frac))
        self.occupancy = as_canonical_float(np.asarray(occupancy))
        self.u_iso = as_canonical_float(np.asarray(u_iso))
        if self.sites_frac.ndim != 2 or self.sites_frac.shape[1] != 3:
            raise ValueError("sites_frac must have shape (N, 3)")
        n = self.sites_frac.shape[0]
        if any(arr.shape[0] != n for arr in (self.occupancy, self.u_iso)):
            raise ValueError("scatterer array length mismatch")
        if len(scatterers) != n:
            raise ValueError("scatterers length must equal n_scatterers")
        if u_star is None:
            self.u_star = np.zeros((n, 6), dtype=np.float64)
        else:
            self.u_star = as_canonical_float(np.asarray(u_star))
            if self.u_star.shape != (n, 6):
                raise ValueError("u_star must have shape (N, 6)")
        self.meta = XrayStructure(
            crystal=crystal,
            n_scatterers=int(n),
            scatterers=scatterers,
        )

    def pack(self) -> bytes:
        return _savez(
            sites_frac=self.sites_frac,
            occupancy=self.occupancy,
            u_iso=self.u_iso,
            u_star=self.u_star,
        )


def unpack_xray(blob: bytes, meta: XrayStructure) -> PackedXray:
    with _load_npz(blob) as zf:
        u_star = zf["u_star"] if "u_star" in zf.files else None
        return PackedXray(
            crystal=meta.crystal,
            sites_frac=zf["sites_frac"],
            occupancy=zf["occupancy"],
            u_iso=zf["u_iso"],
            scatterers=meta.scatterers,
            u_star=u_star,
        )


class PackedEmMap:
    def __init__(
        self,
        crystal: CrystalSymmetry,
        data: np.ndarray,
        origin: Optional[list[int]] = None,
        n_real: Optional[list[int]] = None,
        *,
        experiment_type: str = "cryo_em",
        wrapping: bool = False,
        is_mask: bool = False,
        pixel_sizes: Optional[list[float]] = None,
        origin_cart: Optional[list[float]] = None,
        resolution: Optional[float] = None,
        label: Optional[str] = None,
        space: MapSpace = MapSpace.p1,
    ) -> None:
        from phridge.models import EmMap, ExperimentType

        self.data = as_canonical_float(np.asarray(data))
        if self.data.ndim != 3:
            raise ValueError("EM map data must be 3-D")
        if n_real is None:
            n_real = [int(n) for n in self.data.shape]
        if list(self.data.shape) != list(n_real):
            raise ValueError("data.shape must equal n_real")
        self.meta = EmMap(
            label=label,
            crystal=crystal,
            origin=[0, 0, 0] if origin is None else [int(x) for x in origin],
            n_real=[int(x) for x in n_real],
            space=space,
            dtype="float64",
            experiment_type=ExperimentType(experiment_type),
            wrapping=wrapping,
            is_mask=is_mask,
            pixel_sizes=pixel_sizes,
            origin_cart=origin_cart,
            resolution=resolution,
        )

    def pack(self) -> bytes:
        return _savez(data=self.data)


def unpack_em_map(blob: bytes, meta: Any) -> PackedEmMap:
    with _load_npz(blob) as zf:
        return PackedEmMap(
            crystal=meta.crystal,
            data=zf["data"],
            origin=meta.origin,
            n_real=meta.n_real,
            experiment_type=meta.experiment_type.value,
            wrapping=meta.wrapping,
            is_mask=meta.is_mask,
            pixel_sizes=meta.pixel_sizes,
            origin_cart=meta.origin_cart,
            resolution=meta.resolution,
            label=meta.label,
            space=meta.space,
        )

