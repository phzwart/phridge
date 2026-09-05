"""Space-group expansion from numeric symmetry operators (no sgtbx).

Operators arrive as ``rot`` (n_sym, 3, 3) and ``trans`` (n_sym, 3) in
fractional coordinates, i.e. ``x' = rot @ x + trans``. This must be the
full list (centering included), which is what the client exports from
``sgtbx.space_group``.
"""

from __future__ import annotations

import numpy as np

from phridge.sfcalc.engine.cell import orthogonalization_matrix


def site_multiplicities(
    sites_frac: np.ndarray,
    rot: np.ndarray,
    trans: np.ndarray,
    unit_cell,
    min_distance_sym_equiv: float = 0.5,
) -> np.ndarray:
    """Number of distinct symmetry images per site (cctbx ``multiplicity``).

    Two images are the same if their minimum-image cartesian distance is
    below ``min_distance_sym_equiv`` (cctbx default 0.5 A).
    """
    sites = np.asarray(sites_frac, dtype=np.float64)
    o = orthogonalization_matrix(unit_cell)
    images = np.einsum("sij,nj->nsi", rot, sites) + trans[None, :, :]  # (n, n_sym, 3)
    n, n_sym, _ = images.shape
    mult = np.zeros(n, dtype=np.int64)
    for i in range(n):
        diff = images[i][:, None, :] - images[i][None, :, :]
        diff -= np.round(diff)
        dist = np.linalg.norm(diff @ o.T, axis=-1)
        same = dist < min_distance_sym_equiv
        # number of equivalence classes = n_sym / class size (classes are cosets)
        class_size = same.sum(axis=1)
        mult[i] = int(round(n_sym / float(class_size[0])))
    return mult


def identity_ops():
    return np.eye(3, dtype=np.float64)[None], np.zeros((1, 3), dtype=np.float64)
