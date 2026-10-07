"""The calibrate policy matches the frozen intensity-likelihood nuisance fit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from regression.conftest import mli_case  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "mli_reference.npz"

OPTIONS = {
    "wilson_model": "binned",
    "sigma_a_mode": "bins",
    "beta_mode": "free",
    "nu_mode": "grid",
    "nu_grid": "5:20:5",
    "fit_nu": True,
    "fit_scale": False,
    "smooth_sigma_a": 1.0,
    "smooth_beta": 1.0,
}


def _observed(case: dict) -> object:
    from phridge.contracts.tensors import Observed

    def tensor(value: object, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype)

    return Observed(
        index={"hkl": torch.as_tensor(case["hkl"], dtype=torch.int32)},
        data={
            "i_obs": tensor(case["i_obs"]),
            "sig_i": tensor(case["sig_i"]),
            "epsilon": tensor(case["epsilon"]),
            "s_sq": tensor(case["s_sq"]),
            "unit_cell": tensor(case["unit_cell"]),
            "f_calc": torch.as_tensor(case["f_calc"]),
            "centric": torch.as_tensor(case["centric"], dtype=torch.bool),
        },
        partitions={"tune": torch.as_tensor(~np.asarray(case["flags"], dtype=bool))},
    )


def test_calibrate_matches_frozen_nuisance_fit() -> None:
    from phridge.policies.calibrate_intensity import (
        CalibrateIntensity,
        IntensityCalibrationOptions,
        components_for,
    )

    torch.set_default_dtype(torch.float64)
    case = mli_case()
    options = IntensityCalibrationOptions.model_validate(OPTIONS)
    out = CalibrateIntensity(options).run(components_for(options), _observed(case), ~np.asarray(case["flags"]))
    with np.load(FIXTURE) as frozen:
        np.testing.assert_allclose(out.sigma_a.detach().cpu().numpy(), frozen["sigma_a"], atol=1e-8)
        np.testing.assert_allclose(out.beta.detach().cpu().numpy(), frozen["beta"], atol=1e-8)
        np.testing.assert_allclose(out.sigma_w.detach().cpu().numpy(), frozen["sigma_w"], atol=1e-8)
        assert float(out.nu) == float(frozen["nu"])
