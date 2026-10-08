"""Read a reflection file into ``Observed``. cctbx stays in the shim."""

from __future__ import annotations

from typing import Literal

import numpy as np

from phridge.contracts.models import ObservationType
from phridge.contracts.tensors import Observed
from phridge.shims.cctbx.miller import read_mtz_file

Kind = Literal["amplitude", "intensity"]


def read_data(path: str, kind: Kind = "amplitude") -> Observed:
    """MTZ path to indexed observations, plus a free-set mask when the file has one."""
    packed = read_mtz_file(path)
    wanted = ObservationType.iobs if kind == "intensity" else ObservationType.fobs
    column = _column(packed, wanted)
    values = np.asarray(packed.data[column.npz_name], dtype=np.float64)
    flags = _free_flags(packed, values.shape[0])
    return Observed(
        index={"hkl": np.asarray(packed.hkl, dtype=np.int32)},
        data={kind: values},
        partitions={"free": flags},
    )


def _column(packed: object, wanted: ObservationType) -> object:
    columns = list(packed.meta.columns)
    for column in columns:
        if column.observation_type == wanted:
            return column
    return columns[0]


def _free_flags(packed: object, length: int) -> np.ndarray:
    for column in packed.meta.columns:
        label = column.label.lower().replace("_", "").replace("-", "")
        if "freer" in label or label == "free":
            values = np.asarray(packed.data[column.npz_name])
            return values == 0
    flags = np.zeros(length, dtype=bool)
    flags[::10] = True
    return flags
