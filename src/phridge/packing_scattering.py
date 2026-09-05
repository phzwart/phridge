"""Packed npz helpers for structure-factor engine and target types."""

from __future__ import annotations

from typing import Optional

import numpy as np

from phridge.models import ScatteringTable, SfCurvatures, SfGradients, TargetResult
from phridge.packing import as_canonical_complex, as_canonical_float
from phridge.packing_xtal import _load_npz, _savez


class PackedScatteringTable:
    def __init__(
        self,
        labels: list[str],
        gauss_a: np.ndarray,
        gauss_b: np.ndarray,
        gauss_c: np.ndarray,
        *,
        table: str = "wk1995",
    ) -> None:
        self.gauss_a = as_canonical_float(np.atleast_2d(np.asarray(gauss_a)))
        self.gauss_b = as_canonical_float(np.atleast_2d(np.asarray(gauss_b)))
        self.gauss_c = as_canonical_float(np.asarray(gauss_c).reshape(-1))
        t = len(labels)
        if self.gauss_a.shape[0] != t or self.gauss_b.shape != self.gauss_a.shape or self.gauss_c.shape[0] != t:
            raise ValueError("scattering table shape mismatch")
        self.meta = ScatteringTable(table=table, labels=[str(x) for x in labels], n_terms=int(self.gauss_a.shape[1]))

    def pack(self) -> bytes:
        return _savez(gauss_a=self.gauss_a, gauss_b=self.gauss_b, gauss_c=self.gauss_c)

    def row_index(self, scattering_types: list[str]) -> np.ndarray:
        lookup = {label: i for i, label in enumerate(self.meta.labels)}
        try:
            return np.asarray([lookup[t] for t in scattering_types], dtype=np.int64)
        except KeyError as exc:
            raise KeyError(f"scattering type {exc} missing from table") from exc


def unpack_scattering_table(blob: bytes, meta: ScatteringTable) -> PackedScatteringTable:
    with _load_npz(blob) as zf:
        return PackedScatteringTable(
            labels=list(meta.labels),
            gauss_a=zf["gauss_a"],
            gauss_b=zf["gauss_b"],
            gauss_c=zf["gauss_c"],
            table=meta.table,
        )


class PackedSfGradients:
    FIELDS = ("d_site_frac", "d_occupancy", "d_u_iso", "d_u_star", "d_fp", "d_fdp")

    def __init__(
        self,
        d_site_frac: np.ndarray,
        d_occupancy: np.ndarray,
        d_u_iso: np.ndarray,
        d_u_star: np.ndarray,
        d_fp: np.ndarray,
        d_fdp: np.ndarray,
        *,
        target: Optional[float] = None,
    ) -> None:
        self.d_site_frac = as_canonical_float(np.asarray(d_site_frac))
        n = self.d_site_frac.shape[0]
        if self.d_site_frac.shape != (n, 3):
            raise ValueError("d_site_frac must be (N, 3)")
        self.d_occupancy = as_canonical_float(np.asarray(d_occupancy))
        self.d_u_iso = as_canonical_float(np.asarray(d_u_iso))
        self.d_u_star = as_canonical_float(np.asarray(d_u_star))
        self.d_fp = as_canonical_float(np.asarray(d_fp))
        self.d_fdp = as_canonical_float(np.asarray(d_fdp))
        for name in ("d_occupancy", "d_u_iso", "d_fp", "d_fdp"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"{name} must be (N,)")
        if self.d_u_star.shape != (n, 6):
            raise ValueError("d_u_star must be (N, 6)")
        self.meta = SfGradients(n_scatterers=int(n), target=target)

    def pack(self) -> bytes:
        return _savez(**{k: getattr(self, k) for k in self.FIELDS})


def unpack_sf_gradients(blob: bytes, meta: SfGradients) -> PackedSfGradients:
    with _load_npz(blob) as zf:
        packed = PackedSfGradients(**{k: zf[k] for k in PackedSfGradients.FIELDS}, target=meta.target)
    if packed.meta.n_scatterers != meta.n_scatterers:
        raise ValueError("n_scatterers mismatch")
    return packed


class PackedSfCurvatures:
    FIELDS = ("site_frac", "occupancy", "u_iso", "u_star", "fp", "fdp")

    def __init__(
        self,
        site_frac: np.ndarray,
        occupancy: np.ndarray,
        u_iso: np.ndarray,
        u_star: np.ndarray,
        fp: np.ndarray,
        fdp: np.ndarray,
    ) -> None:
        self.site_frac = as_canonical_float(np.asarray(site_frac))
        n = self.site_frac.shape[0]
        if self.site_frac.shape != (n, 3, 3):
            raise ValueError("site_frac must be (N, 3, 3)")
        self.occupancy = as_canonical_float(np.asarray(occupancy))
        self.u_iso = as_canonical_float(np.asarray(u_iso))
        self.u_star = as_canonical_float(np.asarray(u_star))
        self.fp = as_canonical_float(np.asarray(fp))
        self.fdp = as_canonical_float(np.asarray(fdp))
        for name in ("occupancy", "u_iso", "fp", "fdp"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"{name} must be (N,)")
        if self.u_star.shape != (n, 6, 6):
            raise ValueError("u_star must be (N, 6, 6)")
        self.meta = SfCurvatures(n_scatterers=int(n))

    def pack(self) -> bytes:
        return _savez(**{k: getattr(self, k) for k in self.FIELDS})

    def diagonal_as_gradients(self) -> PackedSfGradients:
        """Extract the parameter-wise diagonal into SfGradients layout."""
        return PackedSfGradients(
            d_site_frac=np.diagonal(self.site_frac, axis1=1, axis2=2).copy(),
            d_occupancy=self.occupancy,
            d_u_iso=self.u_iso,
            d_u_star=np.diagonal(self.u_star, axis1=1, axis2=2).copy(),
            d_fp=self.fp,
            d_fdp=self.fdp,
        )


def unpack_sf_curvatures(blob: bytes, meta: SfCurvatures) -> PackedSfCurvatures:
    with _load_npz(blob) as zf:
        packed = PackedSfCurvatures(**{k: zf[k] for k in PackedSfCurvatures.FIELDS})
    if packed.meta.n_scatterers != meta.n_scatterers:
        raise ValueError("n_scatterers mismatch")
    return packed


class PackedTargetResult:
    def __init__(
        self,
        name: str,
        value: float,
        per_reflection: np.ndarray,
        d_target_d_f_calc: np.ndarray,
        *,
        curv_radial: Optional[np.ndarray] = None,
        curv_tangential: Optional[np.ndarray] = None,
        value_test: Optional[float] = None,
        scale_factor: Optional[float] = None,
    ) -> None:
        self.per_reflection = as_canonical_float(np.asarray(per_reflection))
        self.d_target_d_f_calc = as_canonical_complex(np.asarray(d_target_d_f_calc))
        n = self.per_reflection.shape[0]
        if self.d_target_d_f_calc.shape != (n,):
            raise ValueError("d_target_d_f_calc length mismatch")
        self.curv_radial = None if curv_radial is None else as_canonical_float(np.asarray(curv_radial))
        self.curv_tangential = None if curv_tangential is None else as_canonical_float(np.asarray(curv_tangential))
        for arr in (self.curv_radial, self.curv_tangential):
            if arr is not None and arr.shape != (n,):
                raise ValueError("curvature length mismatch")
        self.meta = TargetResult(
            name=name,
            value=float(value),
            value_test=None if value_test is None else float(value_test),
            n_refl=int(n),
            scale_factor=None if scale_factor is None else float(scale_factor),
            has_curvature=self.curv_radial is not None,
        )

    def pack(self) -> bytes:
        payload = {"per_reflection": self.per_reflection, "d_target_d_f_calc": self.d_target_d_f_calc}
        if self.curv_radial is not None:
            payload["curv_radial"] = self.curv_radial
            payload["curv_tangential"] = self.curv_tangential
        return _savez(**payload)


def unpack_target_result(blob: bytes, meta: TargetResult) -> PackedTargetResult:
    with _load_npz(blob) as zf:
        return PackedTargetResult(
            name=meta.name,
            value=meta.value,
            per_reflection=zf["per_reflection"],
            d_target_d_f_calc=zf["d_target_d_f_calc"],
            curv_radial=zf["curv_radial"] if "curv_radial" in zf.files else None,
            curv_tangential=zf["curv_tangential"] if "curv_tangential" in zf.files else None,
            value_test=meta.value_test,
            scale_factor=meta.scale_factor,
        )
