"""Example configurations build, and the three refusal rules fire on the right path."""

from __future__ import annotations

from pathlib import Path

import pytest

from phridge.core.config import ConfigRefused, build, load

_ROOT = Path(__file__).resolve().parents[2]
_AMPLITUDE = _ROOT / "examples" / "configs" / "amplitude_ml.yaml"
_INTENSITY = _ROOT / "examples" / "configs" / "intensity_mli.yaml"


def test_amplitude_and_intensity_examples_build() -> None:
    amplitude = build(load(str(_AMPLITUDE)))
    _registry, graph, partitions, components = amplitude
    assert set(partitions.masks) == {"tune", "free"}
    assert callable(components["french_wilson"])
    assert "ml_f" in graph.nodes
    assert "partition.audit" not in {read for node in graph.nodes.values() for read in node.reads}

    _registry, graph, partitions, components = build(load(str(_INTENSITY)))
    assert set(partitions.masks) == {"calibration", "tune", "free", "audit"}
    assert graph.readers("partition.audit") == {"statistics"}
    assert isinstance(components["ten_eyck"], type)
    assert components["ten_eyck"].__name__ == "TenEyckFourier"
    assert "nu_shell" in _registry.block("nu_shell").id


def test_audit_wired_into_a_fit_is_refused() -> None:
    config = load(str(_INTENSITY))
    components = []
    for spec in config.components:
        if spec.name == "nu_screen":
            components.append(spec.model_copy(update={"reads": ["partition.audit"]}))
        else:
            components.append(spec)
    with pytest.raises(ConfigRefused, match="partition.audit") as caught:
        build(config.model_copy(update={"components": components}))
    assert caught.value.path == "partition.audit"


def test_needs_that_are_not_produced_are_refused() -> None:
    config = load(str(_AMPLITUDE))
    components = [spec for spec in config.components if spec.name == "ml_f"]
    components[0] = components[0].model_copy(update={"name": "mli_quad"})
    refused = config.model_copy(update={"components": components})
    with pytest.raises(ConfigRefused, match="needs not produced") as caught:
        build(refused)
    assert caught.value.path == "mli_quad"
    assert "location" in caught.value.reason


def test_q_component_reading_an_r_node_directly_is_refused() -> None:
    config = load(str(_INTENSITY))
    kept = []
    for spec in config.components:
        if spec.name == "distance_mask":
            kept.append(spec)
        elif spec.name == "wilson_aniso":
            kept.append(spec.model_copy(update={"reads": ["mask_density"]}))
    two = config.partitions.model_copy(update={"scheme": "two"})
    refused = config.model_copy(update={"components": kept, "partitions": two})
    with pytest.raises(ConfigRefused, match="directly") as caught:
        build(refused)
    assert caught.value.path == "wilson_aniso"
