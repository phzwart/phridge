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
