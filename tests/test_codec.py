import numpy as np

from phridge.codec import decode_ref, encode_value
from phridge.models import CrystalSymmetry, ObjectKind
from phridge.packing import PackedMiller
from phridge.redis_store import RedisStore


def test_codec_miller_and_crystal_through_redis():
    fakeredis = __import__("pytest").importorskip("fakeredis")
    store = RedisStore(fakeredis.FakeRedis())
    crystal = CrystalSymmetry(unit_cell=[10, 10, 10, 90, 90, 90], space_group_hall="P 1")
    crystal_ref = encode_value(store, "j1", "crystal", crystal)
    assert crystal_ref.kind == ObjectKind.cctbx
    assert crystal_ref.key is None
    assert decode_ref(store, crystal_ref).space_group_hall == "P 1"

    packed = PackedMiller(
        crystal=crystal,
        hkl=np.array([[1, 0, 0]], dtype=np.int32),
        data=np.array([9.0]),
    )
    miller_ref = encode_value(store, "j1", "miller", packed)
    assert miller_ref.kind == ObjectKind.cctbx
    assert miller_ref.cctbx_type == "MillerArray"
    again = decode_ref(store, miller_ref)
    np.testing.assert_allclose(again.data, [9.0])
