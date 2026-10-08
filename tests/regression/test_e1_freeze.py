"""Freeze the legacy intensity-likelihood outputs before the split."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.legacy

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


def test_freeze_legacy_intensity_ll_outputs() -> None:
    """CPU float64 reference: nuisance shells, NLL, gradient, map coefficients."""
    from phridge.legacy.intensity_ll_ops import (
        ml_i_maps,
        ml_i_nuisance_fit,
        ml_i_target_and_gradients,
    )

    torch.set_default_dtype(torch.float64)
    case = mli_case()
    tune = ~np.asarray(case["flags"], dtype=bool)
    fit = ml_i_nuisance_fit(
        case["f_calc_packed"],
        case["f_obs_packed"],
        tune_mask=tune,
        s_sq=case["s_sq"],
        epsilon=case["epsilon"],
        centric=case["centric"],
        **OPTIONS,
    )
    params = fit["sigma_a_params"]
    sigma_a = np.asarray(params["bin_sigma_a"], dtype=np.float64)
    beta = np.asarray(params["bin_beta"], dtype=np.float64)
    nu_params = fit["nu_params"]
    if "bin_nu" in nu_params:
        nu = np.asarray(nu_params["bin_nu"], dtype=np.float64)
    else:
        nu = np.asarray(nu_params["nu"], dtype=np.float64)
    sigma_w = np.asarray(fit["sigma_wilson"], dtype=np.float64)

    scored = ml_i_target_and_gradients(
        xray=case["xray"],
        table=case["table"],
        f_obs=case["f_obs_packed"],
        params=case["params"],
        target={"name": "ml_i"},
        alpha=fit["sigma_a"],
        beta=fit["sigma_wilson"],
        epsilon=case["epsilon"],
        centric=case["centric"],
        nu=fit["nu_per_refl"],
        beta_residual=fit["beta_residual"],
        r_free=case["flags"],
        precondition=False,
    )
    nll = float(scored["target"].meta.value)
    d_f = np.asarray(scored["target"].d_target_d_f_calc, dtype=np.complex128)

    maps = ml_i_maps(
        case["f_calc_packed"],
        case["f_obs_packed"],
        target={"name": "ml_i"},
        alpha=fit["sigma_a"],
        beta=fit["sigma_wilson"],
        epsilon=case["epsilon"],
        centric=case["centric"],
        nu=fit["nu_per_refl"],
        beta_residual=fit["beta_residual"],
        r_free=case["flags"],
    )
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        FIXTURE,
        sigma_a=sigma_a,
        beta=beta,
        nu=nu,
        sigma_w=sigma_w,
        nll=np.asarray(nll, dtype=np.float64),
        d_target_d_f_calc=d_f,
        map_difference=np.asarray(maps["difference"].data, dtype=np.complex128),
        map_model=np.asarray(maps["model"].data, dtype=np.complex128),
    )
    assert FIXTURE.is_file()
    with np.load(FIXTURE) as frozen:
        for key in (
            "sigma_a",
            "beta",
            "nu",
            "sigma_w",
            "nll",
            "d_target_d_f_calc",
            "map_difference",
            "map_model",
        ):
            assert key in frozen.files
