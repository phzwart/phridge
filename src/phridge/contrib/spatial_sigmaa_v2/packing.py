"""Packed npz for SpatialSigmaAV2 and SpatialSigmaAV2Result. Torch-free."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from phridge.models import SpatialSigmaAV2, SpatialSigmaAV2Result
from phridge.packing import as_canonical_float, as_canonical_int
from phridge.packing_xtal import _load_npz, _savez


def _to_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


class PackedSpatialSigmaAV2:
    """Wire form of the optional per-atom error-model block."""

    FIELDS = ("lambda_c", "kappa_c", "coeff_hkl", "D0", "Sigma_miss", "shell_s2_edges", "v_k")

    def __init__(
        self,
        lambda_c: np.ndarray,
        kappa_c: np.ndarray,
        coeff_hkl: np.ndarray,
        D0: np.ndarray,
        Sigma_miss: np.ndarray,
        shell_s2_edges: np.ndarray,
        v_k: np.ndarray,
        *,
        enabled: bool = True,
        field_cutoff: float,
        entropy_weight: float = 0.0,
        spectral_taper_scale: float = 1.0,
        fisher: bool = False,
        n_scatterers: Optional[int] = None,
    ) -> None:
        self.lambda_c = as_canonical_float(np.asarray(_to_numpy(lambda_c)).reshape(-1))
        self.kappa_c = as_canonical_float(np.asarray(_to_numpy(kappa_c)).reshape(-1))
        k = int(self.lambda_c.shape[0])
        if self.kappa_c.shape != (k,):
            raise ValueError("kappa_c length must match lambda_c")
        hkl = np.asarray(_to_numpy(coeff_hkl), dtype=np.int32)
        if hkl.size == 0:
            hkl = np.zeros((0, 3), dtype=np.int32)
        self.coeff_hkl = as_canonical_int(hkl.reshape(-1, 3))
        if self.coeff_hkl.shape[0] != k:
            raise ValueError("coeff_hkl rows must match n_coeff")
        self.D0 = as_canonical_float(np.asarray(_to_numpy(D0)).reshape(-1))
        self.Sigma_miss = as_canonical_float(np.asarray(_to_numpy(Sigma_miss)).reshape(-1))
        s = int(self.D0.shape[0])
        if self.Sigma_miss.shape != (s,):
            raise ValueError("Sigma_miss length must match D0")
        edges = np.asarray(_to_numpy(shell_s2_edges)).reshape(-1)
        if edges.size == 0 and s == 0:
            edges = np.zeros((0,), dtype=np.float64)
        self.shell_s2_edges = as_canonical_float(edges)
        if s > 0 and self.shell_s2_edges.shape != (s + 1,):
            raise ValueError("shell_s2_edges must have length n_shells + 1")
        self.v_k = as_canonical_float(np.asarray(_to_numpy(v_k)).reshape(-1))
        if self.v_k.shape != (k,):
            raise ValueError("v_k length must match n_coeff")
        self.meta = SpatialSigmaAV2(
            enabled=bool(enabled),
            field_cutoff=float(field_cutoff),
            n_coeff=k,
            n_shells=s,
            entropy_weight=float(entropy_weight),
            spectral_taper_scale=float(spectral_taper_scale),
            fisher=bool(fisher),
            n_scatterers=None if n_scatterers is None else int(n_scatterers),
        )

    @classmethod
    def empty(
        cls,
        *,
        field_cutoff: float,
        entropy_weight: float = 0.0,
        spectral_taper_scale: float = 1.0,
        fisher: bool = False,
        n_scatterers: Optional[int] = None,
    ) -> PackedSpatialSigmaAV2:
        """Unfitted block: zero coefficients and an empty shell table."""
        z = np.zeros((0,), dtype=np.float64)
        return cls(
            lambda_c=z,
            kappa_c=z,
            coeff_hkl=np.zeros((0, 3), dtype=np.int32),
            D0=z,
            Sigma_miss=z,
            shell_s2_edges=z,
            v_k=z,
            enabled=True,
            field_cutoff=field_cutoff,
            entropy_weight=entropy_weight,
            spectral_taper_scale=spectral_taper_scale,
            fisher=fisher,
            n_scatterers=n_scatterers,
        )

    def pack(self) -> bytes:
        return _savez(**{k: getattr(self, k) for k in self.FIELDS})


def unpack_spatial_sigma_a_v2(blob: bytes, meta: SpatialSigmaAV2) -> PackedSpatialSigmaAV2:
    with _load_npz(blob) as zf:
        packed = PackedSpatialSigmaAV2(
            lambda_c=zf["lambda_c"],
            kappa_c=zf["kappa_c"],
            coeff_hkl=zf["coeff_hkl"],
            D0=zf["D0"],
            Sigma_miss=zf["Sigma_miss"],
            shell_s2_edges=zf["shell_s2_edges"],
            v_k=zf["v_k"],
            enabled=meta.enabled,
            field_cutoff=meta.field_cutoff,
            entropy_weight=meta.entropy_weight,
            spectral_taper_scale=meta.spectral_taper_scale,
            fisher=meta.fisher,
            n_scatterers=meta.n_scatterers,
        )
    if packed.meta.n_coeff != meta.n_coeff or packed.meta.n_shells != meta.n_shells:
        raise ValueError("SpatialSigmaAV2 n_coeff / n_shells mismatch")
    return packed


class PackedSpatialSigmaAV2Result:
    """Wire form of field-coefficient gradients and per-atom diagnostics."""

    FIELDS = ("d_lambda_c", "d_kappa_c", "d_D0", "d_Sigma_miss", "w", "u_err_star", "tr_U")

    def __init__(
        self,
        d_lambda_c: np.ndarray,
        d_kappa_c: np.ndarray,
        d_D0: np.ndarray,
        d_Sigma_miss: np.ndarray,
        w: np.ndarray,
        u_err_star: np.ndarray,
        tr_U: np.ndarray,
    ) -> None:
        self.d_lambda_c = as_canonical_float(np.asarray(_to_numpy(d_lambda_c)).reshape(-1))
        self.d_kappa_c = as_canonical_float(np.asarray(_to_numpy(d_kappa_c)).reshape(-1))
        k = int(self.d_lambda_c.shape[0])
        if self.d_kappa_c.shape != (k,):
            raise ValueError("d_kappa_c length must match d_lambda_c")
        self.d_D0 = as_canonical_float(np.asarray(_to_numpy(d_D0)).reshape(-1))
        self.d_Sigma_miss = as_canonical_float(np.asarray(_to_numpy(d_Sigma_miss)).reshape(-1))
        s = int(self.d_D0.shape[0])
        if self.d_Sigma_miss.shape != (s,):
            raise ValueError("d_Sigma_miss length must match d_D0")
        self.w = as_canonical_float(np.asarray(_to_numpy(w)).reshape(-1))
        n = int(self.w.shape[0])
        u = np.asarray(_to_numpy(u_err_star), dtype=np.float64)
        if u.size == 0:
            u = np.zeros((n, 6), dtype=np.float64)
        self.u_err_star = as_canonical_float(u.reshape(n, 6))
        self.tr_U = as_canonical_float(np.asarray(_to_numpy(tr_U)).reshape(-1))
        if self.tr_U.shape != (n,):
            raise ValueError("tr_U length must match w")
        self.meta = SpatialSigmaAV2Result(n_scatterers=n, n_coeff=k, n_shells=s)

    def pack(self) -> bytes:
        return _savez(**{k: getattr(self, k) for k in self.FIELDS})


def unpack_spatial_sigma_a_v2_result(
    blob: bytes, meta: SpatialSigmaAV2Result
) -> PackedSpatialSigmaAV2Result:
    with _load_npz(blob) as zf:
        packed = PackedSpatialSigmaAV2Result(
            d_lambda_c=zf["d_lambda_c"],
            d_kappa_c=zf["d_kappa_c"],
            d_D0=zf["d_D0"],
            d_Sigma_miss=zf["d_Sigma_miss"],
            w=zf["w"],
            u_err_star=zf["u_err_star"],
            tr_U=zf["tr_U"],
        )
    if (
        packed.meta.n_scatterers != meta.n_scatterers
        or packed.meta.n_coeff != meta.n_coeff
        or packed.meta.n_shells != meta.n_shells
    ):
        raise ValueError("SpatialSigmaAV2Result shape mismatch")
    return packed
