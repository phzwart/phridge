"""Parameter registry: versions, packing, freeze."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from phridge.contracts.manifest import Space
from phridge.core.registry import Block, PackEntry, Registry


def _block(id: str, value: torch.Tensor, **kw: object) -> Block:
    opts = dict(
        component="toy",
        space=Space.R,
        shape=tuple(value.shape),
        bounds=None,
        transform="identity",
        frozen=False,
        value_version=0,
        structural_version=0,
        value=value,
    )
    opts.update(kw)
    return Block(id=id, **opts)  # type: ignore[arg-type]


def test_duplicate_id_is_refused() -> None:
    registry = Registry()
    registry.register(_block("a", torch.zeros(2)))
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(_block("a", torch.ones(2)))


def test_set_value_refuses_a_wrong_shape() -> None:
    registry = Registry()
    registry.register(_block("a", torch.zeros(2)))
    with pytest.raises(ValueError, match="shape"):
        registry.set_value("a", torch.zeros(3))


def test_identity_pack_unpack_is_bitwise() -> None:
    registry = Registry()
    original = torch.tensor([1.25, -3.5, 0.0], dtype=torch.float64)
    registry.register(_block("a", original.clone()))
    packed = registry.pack([PackEntry("a")])
    packed.vector.add_(1.0)
    registry.unpack(packed.vector - 1.0, [PackEntry("a")])
    assert torch.equal(registry.block("a").value, original)


def test_log_and_logit_round_trip() -> None:
    registry = Registry()
    registry.register(_block("positive", torch.tensor([0.5, 2.0, 4.0], dtype=torch.float64), transform="log"))
    lo = torch.tensor(0.0, dtype=torch.float64)
    hi = torch.tensor(1.0, dtype=torch.float64)
    registry.register(
        _block(
            "unit",
            torch.tensor([0.2, 0.8], dtype=torch.float64),
            transform="logit",
            bounds=(lo, hi),
        )
    )
    entries = [PackEntry("positive"), PackEntry("unit")]
    packed = registry.pack(entries)
    registry.unpack(packed.vector, entries)
    torch.testing.assert_close(
        registry.block("positive").value, torch.tensor([0.5, 2.0, 4.0], dtype=torch.float64), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(
        registry.block("unit").value, torch.tensor([0.2, 0.8], dtype=torch.float64), atol=1e-12, rtol=0
    )


def test_unpack_bumps_only_touched_blocks() -> None:
    registry = Registry()
    registry.register(_block("a", torch.ones(2, dtype=torch.float64)))
    registry.register(_block("b", torch.ones(2, dtype=torch.float64)))
    before_b = registry.block("b").value_version
    registry.unpack(registry.pack([PackEntry("a")]).vector, [PackEntry("a")])
    assert registry.block("a").value_version == 1
    assert registry.block("b").value_version == before_b
    assert registry.block("a").structural_version == 0
    assert registry.block("b").structural_version == 0


def test_reshape_bumps_structural_version_of_that_block_only() -> None:
    registry = Registry()
    registry.register(_block("a", torch.ones(2)))
    registry.register(_block("b", torch.ones(3)))
    registry.reshape("a", (4,), torch.arange(4, dtype=torch.float32))
    assert registry.block("a").structural_version == 1
    assert registry.block("a").shape == (4,)
    assert registry.block("b").structural_version == 0
    assert registry.block("b").value_version == 0


def test_freeze_excludes_a_block_from_pack() -> None:
    registry = Registry()
    registry.register(_block("free", torch.tensor([1.0, 2.0])))
    registry.register(_block("held", torch.tensor([3.0, 4.0, 5.0])))
    registry.freeze("held")
    packed = registry.pack([PackEntry("free"), PackEntry("held")])
    assert packed.vector.tolist() == [1.0, 2.0]
    assert "held" not in packed.slices
    assert "free" in packed.slices
