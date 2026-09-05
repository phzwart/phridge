"""Large-N GPU structure-factor gradients: phridge CUDA vs CCTBX direct.

Requires CUDA. Run::

    pytest -s -m "gpu and slow" tests/test_sf_gradients_gpu.py
    # or: make test-sf-gpu
"""

from __future__ import annotations

import pytest

cctbx = pytest.importorskip("cctbx")
torch = pytest.importorskip("torch")

from phridge.client import Bridge  # noqa: E402

from sf_gradient_bench import (  # noqa: E402
    SPACE_GROUPS,
    format_result_table,
    run_space_group,
)

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@pytest.fixture(scope="module")
def cuda_bridge():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return Bridge(memory=True, device="cuda", timeout=600)


@pytest.mark.parametrize("space_group", SPACE_GROUPS)
def test_site_gradients_length_and_direction_gpu(space_group, cuda_bridge, capsys):
    """F_obs from true model → Gaussian site shake → compare site-grad vectors."""
    row = run_space_group(
        space_group,
        n_atoms=1000,
        d_min=2.0,
        sigma=0.05,
        device="cuda",
        seed=0,
        bridge=cuda_bridge,
        time_remote=True,
    )
    with capsys.disabled():
        print(f"\n{format_result_table([row])}\n", flush=True)

    assert row.cosine > 0.99, f"{space_group}: cosine={row.cosine}"
    assert 0.95 <= row.length_ratio <= 1.05, f"{space_group}: length_ratio={row.length_ratio}"
    assert row.site_rel_max < 3e-2, f"{space_group}: site_rel_max={row.site_rel_max}"
    assert row.f_r_factor < 3e-3, f"{space_group}: R(F)={row.f_r_factor}"
    assert row.f_rel < 5e-3, f"{space_group}: F rel={row.f_rel}"
