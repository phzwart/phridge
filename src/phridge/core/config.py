"""Load a configuration and refuse one that does not close."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict

from phridge.contracts.manifest import BlockDecl, PluginManifest, Slot, Space
from phridge.contracts.tensors import Observed
from phridge.core.graph import Graph
from phridge.core.moveset import MoveSet
from phridge.core.partitions import SCHEME_NAMES, AuditRefused, Partitions, audit_rule
from phridge.core.registry import Block, Registry

Unit = Literal["reflection", "particle", "crystal_consistent"]
_PLUGINS_LOADED = False


class ConfigRefused(Exception):
    """``build`` rejected a configuration. ``path`` is the node or field at fault."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


class ComponentSpec(BaseModel):
    """One component: a registered plugin, or a local impl with reads and produces."""

    model_config = ConfigDict(extra="forbid")

    name: str
    plugin: Optional[str] = None
    impl: Optional[str] = None
    reads: Optional[list[str]] = None
    produces: Optional[list[str]] = None
    needs: Optional[list[str]] = None
    space: Optional[str] = None
    slot: Optional[str] = None
    from_space: Optional[str] = None
    to_space: Optional[str] = None


class PartitionSpec(BaseModel):
    """Which partition scheme the objective may consult."""

    model_config = ConfigDict(extra="forbid")

    scheme: Literal["two", "four", "halfmap"]
    fractions: Optional[dict[str, float]] = None
    seed: int = 0
    unit: Unit = "reflection"


class ObservedSpec(BaseModel):
    """How many rows the toy observed set has. Real data is not loaded here."""

    model_config = ConfigDict(extra="forbid")

    n: int = 1
    n_shell: Optional[int] = None
    pdb: Optional[str] = None
    mtz: Optional[str] = None
    kind: Literal["amplitude", "intensity"] = "amplitude"


class Config(BaseModel):
    """The document ``load`` returns."""

    model_config = ConfigDict(extra="forbid")

    components: list[ComponentSpec]
    granularity: str
    partitions: PartitionSpec
    controller: str
    observed: ObservedSpec
    move_sets: list[MoveSet] = []


def load(path: str) -> Config:
    """Read one YAML document and validate it."""
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    return Config.model_validate(_parse_yaml(text))


def build(config: Config) -> tuple[Registry, Graph, Partitions, dict[str, Any]]:
    """Load plugins, check closure, and return the registry, graph, partitions, and bound impls."""
    _load_plugins()
    manifests = [_resolve(spec) for spec in config.components]
    _unique(manifests)
    known = _known(manifests, config.partitions.scheme)
    _check_reads(manifests, known)
    _check_wall(manifests)
    _check_needs(manifests)
    registry = _registry(manifests, config.observed)
    graph = Graph.from_manifests(manifests, registry)
    partitions = Partitions(
        config.partitions.scheme,
        config.partitions.fractions,
        config.partitions.seed,
        config.partitions.unit,
        _observed(config.observed),
    )
    try:
        audit_rule(graph)
    except AuditRefused as exc:
        raise ConfigRefused(exc.path, exc.reason) from exc
    return registry, graph, partitions, _bind(config.components)


def _unique(manifests: list[PluginManifest]) -> None:
    seen: set[str] = set()
    for manifest in manifests:
        if manifest.name in seen:
            raise ConfigRefused(manifest.name, "duplicate component")
        seen.add(manifest.name)


def _resolve(spec: ComponentSpec) -> PluginManifest:
    from phridge.ops import get_manifest

    if spec.impl is not None and spec.plugin is None:
        return _local_manifest(spec)
    plugin = spec.plugin or spec.name
    try:
        base = get_manifest(plugin)
    except KeyError as exc:
        raise ConfigRefused(spec.name, f"unknown component {plugin}") from exc
    update: dict[str, Any] = {"name": spec.name}
    if spec.reads is not None:
        update["reads"] = list(spec.reads)
    if spec.produces is not None:
        update["produces"] = list(spec.produces)
    if spec.needs is not None:
        update["needs"] = list(spec.needs)
    return base.model_copy(update=update)


def _local_manifest(spec: ComponentSpec) -> PluginManifest:
    if spec.space is None or spec.slot is None:
        raise ConfigRefused(spec.name, "a local component needs space and slot")
    space = Space(spec.space)
    kwargs: dict[str, Any] = {
        "name": spec.name,
        "slot": Slot(spec.slot),
        "space": space,
        "reads": list(spec.reads or []),
        "produces": list(spec.produces or []),
        "needs": list(spec.needs or []),
    }
    if space == Space.transform:
        if spec.from_space is None or spec.to_space is None:
            raise ConfigRefused(spec.name, "a transform needs from and to")
        kwargs["from_space"] = Space(spec.from_space)
        kwargs["to_space"] = Space(spec.to_space)
    return PluginManifest(**kwargs)


def _fields(manifest: PluginManifest) -> set[str]:
    """Produced predictive fields. A Fourier transform into Q also produces ``location``."""
    fields = set(manifest.produces)
    if manifest.slot == Slot.fourier and manifest.to_space == Space.Q:
        fields.add("location")
    return fields


def _known(manifests: list[PluginManifest], scheme: str) -> set[str]:
    names = {manifest.name for manifest in manifests}
    for manifest in manifests:
        names.update(_fields(manifest))
    names.update(f"partition.{name}" for name in SCHEME_NAMES[scheme])
    return names


def _check_reads(manifests: list[PluginManifest], known: set[str]) -> None:
    for manifest in manifests:
        for read in manifest.reads:
            if read not in known:
                raise ConfigRefused(read, "not produced")


def _origin(read: str, manifests: list[PluginManifest]) -> Optional[Space]:
    if read.startswith("partition."):
        return None
    for manifest in manifests:
        if read == manifest.name or read in _fields(manifest):
            if manifest.space == Space.transform:
                return manifest.to_space
            return manifest.space
    return None


def _check_wall(manifests: list[PluginManifest]) -> None:
    """A component may read another space only through a transform's from/to."""
    for manifest in manifests:
        for read in manifest.reads:
            origin = _origin(read, manifests)
            if origin is None:
                continue
            if manifest.space == Space.transform:
                if origin != manifest.from_space:
                    raise ConfigRefused(manifest.name, f"transform reads {read} from {origin} directly")
                continue
            if origin != manifest.space:
                raise ConfigRefused(manifest.name, f"reads {read} in {origin} directly")


def _check_needs(manifests: list[PluginManifest]) -> None:
    produced: set[str] = set()
    for manifest in manifests:
        produced.update(_fields(manifest))
    for manifest in manifests:
        missing = [need for need in manifest.needs if need not in produced]
        if missing:
            raise ConfigRefused(manifest.name, "needs not produced: " + ", ".join(missing))


def _registry(manifests: list[PluginManifest], observed: ObservedSpec) -> Registry:
    import torch

    registry = Registry()
    for manifest in manifests:
        space = manifest.space if manifest.space != Space.transform else Space.Q
        for decl in manifest.blocks:
            shape = _shape(decl, observed)
            registry.register(
                Block(
                    id=decl.id,
                    component=manifest.name,
                    space=space,
                    shape=shape,
                    bounds=_bounds(decl, torch.float64),
                    transform=decl.transform,
                    frozen=decl.frozen,
                    value_version=0,
                    structural_version=0,
                    value=torch.zeros(shape, dtype=torch.float64),
                )
            )
    return registry


def _shape(decl: BlockDecl, observed: ObservedSpec) -> tuple[int, ...]:
    shell = observed.n_shell if observed.n_shell is not None else observed.n
    resolved: list[int] = []
    for dim in decl.shape:
        if isinstance(dim, int):
            resolved.append(dim)
        elif dim == "n_shell":
            resolved.append(int(shell))
        else:
            raise ConfigRefused(decl.id, f"unresolved shape {dim}")
    return tuple(resolved)


def _bounds(decl: BlockDecl, dtype: Any) -> Optional[tuple[Any, Any]]:
    import torch

    if decl.bounds is None:
        return None
    lo, hi = decl.bounds
    return torch.tensor(lo, dtype=dtype), torch.tensor(hi, dtype=dtype)


def _observed(spec: ObservedSpec) -> Observed:
    import torch

    return Observed(
        index={"h": torch.arange(spec.n)},
        data={"F": torch.ones(spec.n, dtype=torch.float64)},
        partitions={},
    )


def _bind(specs: list[ComponentSpec]) -> dict[str, Any]:
    from phridge.ops import get_implementation

    bound: dict[str, Any] = {}
    for spec in specs:
        if spec.impl is not None:
            bound[spec.name] = _import_impl(spec.impl)
            continue
        plugin = spec.plugin or spec.name
        try:
            bound[spec.name] = get_implementation(plugin)
        except Exception as exc:
            raise ConfigRefused(spec.name, f"no implementation: {exc}") from exc
    return bound


def _import_impl(path: str) -> Any:
    import importlib

    module_name, sep, attr = path.partition(":")
    if not sep or not attr:
        raise ConfigRefused(path, "impl path must be module:attr")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _load_plugins() -> None:
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    for _name, target in _plugin_targets():
        module_name, sep, attr = target.partition(":")
        if not sep:
            continue
        try:
            import importlib

            register = getattr(importlib.import_module(module_name), attr)
            register()
        except Exception:
            continue
    _PLUGINS_LOADED = True


def _plugin_targets() -> list[tuple[str, str]]:
    """Installed entry points, or the table in ``pyproject.toml`` when none are installed."""
    from importlib.metadata import entry_points

    discovered = entry_points()
    group = discovered.select(group="phridge.plugins") if hasattr(discovered, "select") else discovered.get("phridge.plugins", [])
    found = [(entry.name, entry.value) for entry in group]
    if found:
        return found
    return _plugins_from_pyproject()


def _plugins_from_pyproject() -> list[tuple[str, str]]:
    import tomllib
    from pathlib import Path

    for parent in Path(__file__).resolve().parents:
        path = parent / "pyproject.toml"
        if not path.is_file():
            continue
        with path.open("rb") as handle:
            data = tomllib.load(handle)
        table = data.get("project", {}).get("entry-points", {}).get("phridge.plugins", {})
        return [(str(name), str(target)) for name, target in table.items()]
    return []


def _parse_yaml(text: str) -> Any:
    lines = _yaml_lines(text)
    value, index = _parse_block(lines, 0, 0)
    if index != len(lines):
        raise ValueError(f"yaml trailing content at {lines[index][1]!r}")
    return value


def _yaml_lines(text: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        stripped = raw.split("#", 1)[0].rstrip()
        if not stripped.strip():
            continue
        lines.append((len(stripped) - len(stripped.lstrip(" ")), stripped.strip()))
    return lines


def _parse_block(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[Any, int]:
    if index >= len(lines) or lines[index][0] < indent:
        return {}, index
    if lines[index][1].startswith("- "):
        return _parse_list(lines, index, indent)
    mapping: dict[str, Any] = {}
    while index < len(lines) and lines[index][0] == indent and not lines[index][1].startswith("- "):
        key, value, index = _parse_pair(lines, index, indent)
        mapping[key] = value
    return mapping, index


def _parse_pair(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[str, Any, int]:
    key, _, rest = lines[index][1].partition(":")
    rest = rest.strip()
    index += 1
    if rest == "":
        value, index = _parse_block(lines, index, indent + 2)
        return key.strip(), value, index
    if rest.startswith("["):
        return key.strip(), _inline_list(rest), index
    return key.strip(), _scalar(rest), index


def _parse_list(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[list[Any], int]:
    items: list[Any] = []
    while index < len(lines) and lines[index][0] == indent and lines[index][1].startswith("- "):
        body = lines[index][1][2:].strip()
        index += 1
        if body == "":
            value, index = _parse_block(lines, index, indent + 2)
            items.append(value)
            continue
        if ":" in body:
            key, _, rest = body.partition(":")
            item: dict[str, Any] = {key.strip(): _typed(rest.strip()) if rest.strip() else {}}
            if rest.strip() == "":
                item[key.strip()], index = _parse_block(lines, index, indent + 4)
            while index < len(lines) and lines[index][0] >= indent + 2 and not lines[index][1].startswith("- "):
                child, value, index = _parse_pair(lines, index, indent + 2)
                item[child] = value
            items.append(item)
            continue
        items.append(_scalar(body))
    return items, index


def _typed(text: str) -> Any:
    if text.startswith("["):
        return _inline_list(text)
    return _scalar(text)


def _inline_list(text: str) -> list[Any]:
    inner = text.strip()[1:-1].strip()
    if not inner:
        return []
    return [_scalar(part.strip()) for part in inner.split(",")]


def _scalar(text: str) -> Any:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1]
    if text in {"null", "~"}:
        return None
    if text == "true":
        return True
    if text == "false":
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text
