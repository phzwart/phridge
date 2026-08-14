import numpy as np
import pytest

from phridge.models import AnomalousLayout, CrystalSymmetry, ObservationType
from phridge.packing import PackedMap, PackedMiller, unpack_map, unpack_miller


def _p1() -> CrystalSymmetry:
    return CrystalSymmetry(unit_cell=[10.0, 10.0, 10.0, 90.0, 90.0, 90.0], space_group_hall="P 1")


def test_miller_npz_roundtrip():
    packed = PackedMiller(
        crystal=_p1(),
        hkl=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int32),
        data=np.array([1.5, 2.5, 3.5], dtype=np.float32),
        sigmas=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        observation_type=ObservationType.fobs,
        anomalous_layout=AnomalousLayout.asu,
    )
    assert packed.data.dtype == np.float64
    assert packed.hkl.dtype == np.int32
    again = unpack_miller(packed.pack(), packed.meta)
    np.testing.assert_allclose(again.data, [1.5, 2.5, 3.5])
    np.testing.assert_array_equal(again.hkl, packed.hkl)
    np.testing.assert_allclose(again.sigmas, [0.1, 0.2, 0.3])


def test_miller_rejects_bad_hkl():
    with pytest.raises(ValueError):
        PackedMiller(crystal=_p1(), hkl=np.array([1, 0, 0]), data=np.array([1.0]))


def test_real_map_npz_roundtrip():
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    packed = PackedMap(crystal=_p1(), data=data, origin=[1, 0, 0])
    assert packed.data.dtype == np.float64
    again = unpack_map(packed.pack(), packed.meta)
    np.testing.assert_allclose(again.data, data.astype(np.float64))
    assert again.meta.origin == [1, 0, 0]
    assert again.meta.n_real == [2, 3, 4]


def test_client_to_canonical_dict_crystal():
    from phridge.client.convert import to_canonical

    cs = to_canonical(
        {"unit_cell": [4, 5, 6, 90, 90, 90], "space_group_hall": "P 1", "space_group_number": 1}
    )
    assert isinstance(cs, CrystalSymmetry)
    assert cs.unit_cell[0] == 4.0
