"""The scoring rule and posterior maps match the frozen intensity likelihood."""

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


def test_score_and_posterior_maps_match_the_reference() -> None:
    from phridge.contracts.tensors import Predictive
    from phridge.d.maps import posterior_map_coefficients
    from phridge.d.rules.mli_quad.rule import MliQuad
    from phridge.policies.calibrate_intensity import (
        CalibrateIntensity,
        IntensityCalibrationOptions,
        components_for,
    )

    torch.set_default_dtype(torch.float64)
    case = mli_case()
    work = ~np.asarray(case["flags"], dtype=bool)
    options = IntensityCalibrationOptions.model_validate(OPTIONS)
    cal = CalibrateIntensity(options).run(components_for(options), _observed(case), work)
    nu = torch.full((case["hkl"].shape[0],), float(cal.nu), dtype=torch.float64)
    pred = Predictive(
        {
            "location": torch.as_tensor(case["f_calc"]),
            "sigma_a": cal.sigma_a_refl,
            "sigma_w": cal.sigma_w,
            "beta_residual": cal.beta_residual,
            "nu": nu,
        }
    )
    value, grads = MliQuad().score(pred, _observed(case), work)
    d_location = grads["location"].detach().cpu().numpy()
    maps = posterior_map_coefficients(pred, _observed(case), grads["location"])
    with np.load(FIXTURE) as frozen:
        assert value == pytest.approx(float(frozen["nll"]), abs=1e-8)
        np.testing.assert_allclose(d_location, frozen["d_target_d_f_calc"], atol=1e-8)
        np.testing.assert_allclose(maps.difference, frozen["map_difference"], atol=1e-8)
        np.testing.assert_allclose(maps.model, frozen["map_model"], atol=1e-8)
    scale = np.sqrt(np.asarray(case["epsilon"], dtype=np.float64) * cal.sigma_w.detach().cpu().numpy())
    ok = np.abs(d_location) > 1e-8
    relative = maps.difference[ok] / (d_location[ok] * scale[ok])
    assert np.max(np.abs(relative.imag)) < 1e-6
