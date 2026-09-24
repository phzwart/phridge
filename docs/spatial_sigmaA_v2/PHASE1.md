# Spatial σ_A v2 — Phase 1

Schema and plumbing only. The per-atom model (spec §1–§4, §6, §10) is not
implemented yet. With `--spatial-sigmaA-v2` off, no new object is packed and
existing worker ops ignore the unused optional slot.

## What was done

- LinkML classes `SpatialSigmaAV2` and `SpatialSigmaAV2Result` in
  [`schema/cctbx_scattering.yaml`](../../schema/cctbx_scattering.yaml).
- Hand-aligned pydantic models in [`src/phridge/models.py`](../../src/phridge/models.py)
  and packed npz types in
  [`src/phridge/contrib/spatial_sigmaa_v2/`](../../src/phridge/contrib/spatial_sigmaa_v2/).
- Codec + `CCTBX_TYPES` registration so the block can cross Redis.
- Optional input `spatial_sigma_a_v2` on `ml_i_nuisance_fit` and
  `ml_i_target_and_gradients`. Phase 1 does not consume it.
- CLI / env: `--spatial-sigmaA-v2` → `PHRIDGE_SPATIAL_SIGMA_A_V2`,
  `--spatial-sigmaA-v2-fisher` → `PHRIDGE_SPATIAL_SIGMA_A_V2_FISHER`
  (Phenix driver, wrapper script, standalone `phridge-intensity`).
- Client helper `job_input_from_env()` returns `{}` when the flag is off,
  so the job payload is unchanged.

## Spec equations

None. This phase is the wire contract for later work that will implement
eqs 3–4, 7–9, 14–16, 25–29.

## Tests

[`tests/contrib/test_spatial_sigmaa_v2.py`](../../tests/contrib/test_spatial_sigmaa_v2.py):

- `test_flag_off_is_inert` — spec §11 item 9: no block constructed or sent.
- Pack/unpack, pydantic accept/reject, codec roundtrip, CLI flag presence,
  optional op slots, jsonschema when generated.

## Deviations

None. Worker functions accept the new input and leave the existing
likelihood / gradient path untouched (Phase 2 will use the block).
