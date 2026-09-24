"""Optional per-atom error model (spatial σ_A v2).

Torch-free at import. The block is constructed only when
``--spatial-sigmaA-v2`` / ``PHRIDGE_SPATIAL_SIGMA_A_V2`` is on; otherwise
helpers return ``None`` and no object is packed onto the job.

Entry point: ``phridge.ops`` → ``phridge.contrib.spatial_sigmaa_v2:register``.
"""

from __future__ import annotations

from phridge.contrib.spatial_sigmaa_v2.options import (
    DEFAULT_ENTROPY_WEIGHT,
    DEFAULT_FIELD_CUTOFF,
    DEFAULT_SPECTRAL_TAPER,
    SpatialSigmaAV2Options,
    block_from_env,
    job_input_from_env,
    job_input_from_state,
    options_from_env,
)
from phridge.contrib.spatial_sigmaa_v2.packing import (
    PackedSpatialSigmaAV2,
    PackedSpatialSigmaAV2Result,
    unpack_spatial_sigma_a_v2,
    unpack_spatial_sigma_a_v2_result,
)
from phridge.contrib.spatial_sigmaa_v2.fields import (
    FieldBasis,
    evaluate_fields,
    grid_n_real,
    sample_on_grid,
)
from phridge.contrib.spatial_sigmaa_v2.viz import atom_display_columns, field_volumes
from phridge.contrib.spatial_sigmaa_v2.per_atom import (
    effective_weight,
    error_u_iso,
    luzzati_d_iso,
    modified_model,
    presence_weight,
)
from phridge.contrib.spatial_sigmaa_v2.moments import f_eff_direct, sigma_delta, sigma_p
from phridge.contrib.spatial_sigmaa_v2.target import RiceResult, rice_nll_intensity
from phridge.contrib.spatial_sigmaa_v2.gradients import (
    AtomGrads,
    add_atom_grads,
    field_gradients,
    mean_gradients_direct,
    variance_gradients_aniso,
    variance_gradients_iso,
)
from phridge.contrib.spatial_sigmaa_v2.regulariser import RegulariserResult, regulariser, spectral_taper
from phridge.contrib.spatial_sigmaa_v2.alternate import (
    FrozenAtomErrors,
    freeze_atom_errors,
    run_macrocycle_step,
)
from phridge.contrib.spatial_sigmaa_v2.fisher import iterate_fisher_closure
from phridge.contrib.spatial_sigmaa_v2.op import OP_NAME as STEP_OP_NAME, register_ops

__all__ = [
    "DEFAULT_ENTROPY_WEIGHT",
    "DEFAULT_FIELD_CUTOFF",
    "DEFAULT_SPECTRAL_TAPER",
    "AtomGrads",
    "FieldBasis",
    "FrozenAtomErrors",
    "PackedSpatialSigmaAV2",
    "PackedSpatialSigmaAV2Result",
    "RegulariserResult",
    "RiceResult",
    "SpatialSigmaAV2Options",
    "add_atom_grads",
    "block_from_env",
    "effective_weight",
    "field_gradients",
    "error_u_iso",
    "evaluate_fields",
    "field_volumes",
    "atom_display_columns",
    "grid_n_real",
    "sample_on_grid",
    "f_eff_direct",
    "freeze_atom_errors",
    "iterate_fisher_closure",
    "job_input_from_env",
    "job_input_from_state",
    "luzzati_d_iso",
    "mean_gradients_direct",
    "modified_model",
    "options_from_env",
    "presence_weight",
    "register",
    "regulariser",
    "rice_nll_intensity",
    "run_macrocycle_step",
    "STEP_OP_NAME",
    "sigma_delta",
    "spectral_taper",
    "variance_gradients_aniso",
    "variance_gradients_iso",
    "sigma_p",
    "unpack_spatial_sigma_a_v2",
    "unpack_spatial_sigma_a_v2_result",
]


def register() -> None:
    """Entry-point hook; types via ``CCTBX_TYPES``, op via ``register_ops``."""
    register_ops()
