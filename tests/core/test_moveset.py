"""Move sets: views, scales, and the four refusal rules."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from phridge.contracts.manifest import Space
from phridge.core.moveset import Entry, MoveSet, MoveSetRefused, View, resolve, validate
from phridge.core.registry import Block, Registry


def _registry() -> Registry:
    registry = Registry()
    for name, size in (("a", 4), ("b", 4), ("c", 3), ("d", 2)):
        registry.register(
            Block(
                id=name,
                component="toy",
                space=Space.R,
                shape=(size,),
                bounds=None,
                transform="identity",
                frozen=False,
                value_version=0,
                structural_version=0,
                value=torch.arange(1, size + 1, dtype=torch.float64),
            )
        )
    return registry


def _selector(name: str) -> torch.Tensor:
    if name == "even":
        return torch.tensor([True, False, True, False])
    if name == "first":
        return torch.tensor([True, False, False, False])
    raise AssertionError(name)


def test_index_and_predicate_views() -> None:
    registry = _registry()
    move_set = MoveSet(
        name="views",
        entries=[
            Entry(block="a", view=View(block="a", index=[0, 2])),
            Entry(block="b", view=View(block="b", predicate="even")),
        ],
    )
    resolved = resolve(move_set, registry, _selector)
    packed = registry.pack(resolved)
    assert packed.vector.tolist() == [1.0, 3.0, 1.0, 3.0]
    assert resolved[0].view_mask.tolist() == [True, False, True, False]


def test_selection_is_passed_to_the_selector() -> None:
    registry = _registry()
    seen: list[str] = []

    def selector(name: str) -> torch.Tensor:
        seen.append(name)
        return _selector("first")

    move_set = MoveSet(
        name="sel",
        entries=[Entry(block="a", view=View(block="a", selection="resseq 1:10"))],
    )
    resolved = resolve(move_set, registry, selector)
    assert seen == ["resseq 1:10"]
    assert int(resolved[0].view_mask.sum()) == 1


def test_scale_appears_on_packed() -> None:
    registry = _registry()
    move_set = MoveSet(name="scaled", entries=[Entry(block="a", scale=2.5), Entry(block="d", scale=0.5)])
    packed = registry.pack(resolve(move_set, registry, _selector))
    assert packed.scales.tolist() == [2.5, 2.5, 2.5, 2.5, 0.5, 0.5]


def test_frozen_block_is_refused() -> None:
    registry = _registry()
    registry.freeze("a")
    move_set = MoveSet(name="held", entries=[Entry(block="a")])
    with pytest.raises(MoveSetRefused, match="frozen") as caught:
        validate(move_set, registry, {})
    assert caught.value.entry == "a"


def test_frozen_entry_is_refused() -> None:
    registry = _registry()
    move_set = MoveSet(name="held", entries=[Entry(block="b", frozen=True)])
    with pytest.raises(MoveSetRefused, match="frozen") as caught:
        validate(move_set, registry, {})
    assert caught.value.entry == "b"


def test_structural_epoch_after_ready_epoch_is_refused() -> None:
    registry = _registry()
    registry.bump_epoch()
    registry.reshape("c", (3,), torch.ones(3, dtype=torch.float64))
    move_set = MoveSet(name="stale", entries=[Entry(block="c")], ready_epoch=0)
    with pytest.raises(MoveSetRefused, match="ready_epoch") as caught:
        validate(move_set, registry, {}, ready_epoch=0)
    assert caught.value.entry == "c"
    validate(move_set, registry, {}, ready_epoch=registry.epoch())


def test_generator_output_overlap_is_refused() -> None:
    registry = _registry()
    move_set = MoveSet(name="both", entries=[Entry(block="a"), Entry(block="b")])
    with pytest.raises(MoveSetRefused, match="overlaps") as caught:
        validate(move_set, registry, {"a": {"adjoint"}}, generator_outputs={"a": {"b"}})
    assert caught.value.entry == "a"


def test_missing_adjoint_is_refused() -> None:
    registry = _registry()
    move_set = MoveSet(name="gen", entries=[Entry(block="a")])
    with pytest.raises(MoveSetRefused, match="adjoint") as caught:
        validate(move_set, registry, {}, generator_outputs={"a": {"b"}})
    assert caught.value.entry == "a"
    validate(move_set, registry, {"a": {"adjoint"}}, generator_outputs={"a": {"b"}})
