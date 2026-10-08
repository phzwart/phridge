"""Water peaks on a difference map. Distances use van der Waals radii from the shim."""

from __future__ import annotations

from typing import Any

import torch

from phridge.contracts.tensors import RObject
from phridge.shims.cctbx.rotamers import vdw_radius
from phridge.transforms.build.base import RChange


class WaterBuilder:
    """Peak search, then a distance filter and a B filter. The map is not modified."""

    def propose(self, maps: dict[str, Any], stats: dict[str, Any], selection: Any) -> RChange:
        _ = selection
        difference = maps["difference"]
        threshold = float(stats.get("threshold", 3.0 * float(difference.std().clamp_min(1e-8))))
        peaks = _local_maxima(difference, threshold)
        peaks = _distance_filter(peaks, stats.get("sites"), float(stats.get("min_distance", 2.0 * vdw_radius("O"))))
        max_b = float(stats.get("max_b", 80.0))
        if float(stats.get("b", 20.0)) > max_b:
            peaks = peaks[:0]
        return RChange(_waters(peaks, float(stats.get("b", 20.0))), _remove(stats.get("sites")), {})


def _local_maxima(difference: Any, threshold: float) -> Any:
    padded = torch.nn.functional.pad(difference, (1, 1, 1, 1, 1, 1), value=-1e9)
    higher = torch.ones_like(difference, dtype=torch.bool)
    for shift in (-1, 1):
        for axis in (0, 1, 2):
            higher = higher & _above_neighbor(padded, difference, axis, shift)
    higher = higher & (difference >= threshold)
    return higher.nonzero().to(dtype=difference.dtype) / torch.tensor(difference.shape, dtype=difference.dtype)


def _above_neighbor(padded: Any, difference: Any, axis: int, shift: int) -> Any:
    start = [1, 1, 1]
    start[axis] = 1 + shift
    slices = tuple(slice(start[i], start[i] + difference.shape[i]) for i in range(3))
    return difference >= padded[slices]


def _distance_filter(peaks: Any, sites: Any, minimum: float) -> Any:
    if sites is None or peaks.numel() == 0:
        return peaks
    delta = peaks[:, None, :] - sites.to(dtype=peaks.dtype)[None, :, :]
    dist = torch.linalg.vector_norm(delta, dim=-1)
    keep = dist.min(dim=1).values >= minimum
    return peaks[keep]


def _waters(peaks: Any, b_iso: float) -> RObject | None:
    count = int(peaks.shape[0])
    if count == 0:
        return None
    u_iso = torch.full((count,), b_iso / (8 * torch.pi**2), dtype=peaks.dtype)
    return RObject(
        sites_frac=peaks,
        u_iso=u_iso,
        u_star=torch.zeros((count, 6), dtype=peaks.dtype),
        occupancy=torch.ones(count, dtype=peaks.dtype),
        fp=torch.zeros(count, dtype=peaks.dtype),
        fdp=torch.zeros(count, dtype=peaks.dtype),
        scatterer_type=torch.zeros(count, dtype=torch.int64),
        uij_defined=torch.zeros(count, dtype=torch.bool),
    )


def _remove(sites: Any) -> Any:
    if sites is None:
        return torch.zeros(0, dtype=torch.bool)
    return torch.zeros(sites.shape[0], dtype=torch.bool)
