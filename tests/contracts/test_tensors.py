"""Tensor contract records."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from phridge.contracts.tensors import (  # noqa: E402
    ComplexArray,
    DensityObject,
    Observed,
    Predictive,
    RObject,
)


def _n(shape: tuple[int, ...]) -> torch.Tensor:
    return torch.randn(*shape)


def test_robject_validate_and_mismatch() -> None:
    good = RObject(
        sites_frac=_n((4, 3)),
        u_iso=_n((4,)),
        u_star=_n((4, 6)),
        occupancy=_n((4,)),
        fp=_n((4,)),
        fdp=_n((4,)),
        scatterer_type=torch.zeros(4, dtype=torch.long),
        uij_defined=torch.zeros(4, dtype=torch.bool),
    )
    good.validate()
    bad = good._replace(occupancy=_n((3,)))
    with pytest.raises(ValueError, match="leading axes"):
        bad.validate()


def test_density_and_complex_validate() -> None:
    density = DensityObject(values=_n((5, 2)), origin=_n((5, 3)), spacing=_n((5, 3)), cell=_n((5, 6)))
    density.validate()
    with pytest.raises(ValueError, match="leading axes"):
        density._replace(cell=_n((4, 6))).validate()
    coeffs = ComplexArray(hkl=torch.zeros((6, 3), dtype=torch.long), values=torch.randn(6, dtype=torch.complex64))
    coeffs.validate()
    with pytest.raises(ValueError, match="leading axes"):
        coeffs._replace(values=torch.randn(5, dtype=torch.complex64)).validate()


def test_predictive_require_and_observed() -> None:
    pred = Predictive(fields={"sigma_w": _n((3,)), "f_calc": _n((3, 2))})
    pred.validate()
    assert pred.has("sigma_w")
    assert not pred.has("missing")
    with pytest.raises(KeyError, match="missing"):
        pred.require("sigma_w", "missing")
    with pytest.raises(ValueError, match="leading axes"):
        Predictive(fields={"a": _n((2,)), "b": _n((3,))}).validate()
    obs = Observed(
        index={"hkl": torch.zeros((4, 3), dtype=torch.long)},
        data={"i": _n((4,))},
        partitions={"free": torch.zeros(4, dtype=torch.bool)},
    )
    obs.validate()
    with pytest.raises(ValueError, match="leading axes"):
        obs._replace(data={"i": _n((2,))}).validate()
