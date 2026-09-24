import numpy as np
import pytest

from phridge.worker.convert import from_torch, to_torch

torch = pytest.importorskip("torch")


def test_float32_compute_casts_back_to_float64():
    array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    tensor = to_torch(array, device="cpu", compute_dtype=torch.float32)
    assert tensor.dtype == torch.float32
    back = from_torch(tensor)
    assert back.dtype == np.float64
    np.testing.assert_allclose(back, array)


def test_to_torch_leaves_spatial_sigma_a_v2_on_host():
    from phridge.contrib.spatial_sigmaa_v2.packing import PackedSpatialSigmaAV2

    packed = PackedSpatialSigmaAV2.empty(field_cutoff=15.0)
    out = to_torch(packed, device="cpu", compute_dtype=torch.float32)
    assert out is packed
    assert isinstance(out.lambda_c, np.ndarray)


def test_from_torch_hosts_spatial_sigma_a_v2_result_arrays():
    from phridge.contrib.spatial_sigmaa_v2.packing import PackedSpatialSigmaAV2Result

    result = PackedSpatialSigmaAV2Result(
        d_lambda_c=np.zeros(0),
        d_kappa_c=np.zeros(0),
        d_D0=np.zeros(0),
        d_Sigma_miss=np.zeros(0),
        w=np.array([0.5, 1.0]),
        u_err_star=np.zeros((2, 6)),
        tr_U=np.array([0.03, 0.0]),
    )
    result.w = torch.as_tensor(result.w, dtype=torch.float32)
    result.tr_U = torch.as_tensor(result.tr_U, dtype=torch.float32)
    back = from_torch(result)
    assert isinstance(back.w, np.ndarray)
    assert back.w.dtype == np.float64
    np.testing.assert_allclose(back.w, [0.5, 1.0])


def test_to_like_never_lands_float64_on_mps():
    from phridge.sfcalc.ops import _to_like

    host = np.array([1.5, 2.5], dtype=np.float64)
    like = torch.ones(2, dtype=torch.float32)
    t = _to_like(host, like)
    assert t.dtype == torch.float32
    assert t.device == like.device
    if torch.backends.mps.is_available():
        like_mps = torch.ones(2, dtype=torch.float32, device="mps")
        t_mps = _to_like(host, like_mps)
        assert t_mps.device.type == "mps"
        assert t_mps.dtype == torch.float32
        # Widening an MPS tensor goes CPU-first.
        wide = _to_like(like_mps, torch.ones(2, dtype=torch.float64))
        assert wide.device.type == "cpu"
        assert wide.dtype == torch.float64
