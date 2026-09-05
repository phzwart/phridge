# Agent directive: `phridge.contrib`

Non-negotiable rules for anything under `src/phridge/contrib/`.
Also mirrored by `.cursor/rules/phridge-contrib.mdc`.

## Typing

- Use `from __future__ import annotations` in every module.
- Public callables, `register()` hooks, target/op option models, and I/O helpers must have complete type annotations.
- Do not expose untyped `**kwargs` at the package boundary; validate options with pydantic instead.

## Pydantic

- Any JSON / options / target-spec dict that crosses the contrib boundary must be validated with a pydantic `BaseModel` (or an existing LinkML-generated model from `phridge.models` via `model_validate`).
- Prefer `model_config = {"extra": "forbid"}` on contrib option models.
- Do not read ad-hoc keys from raw `dict`s in public constructors.

## LinkML / JSON Schema

- Prefer existing packed types (`array`, `json`, `MillerArray`, …) so most contrib targets need **no** new schema.
- New science object *kinds* or envelope fields require LinkML in `schema/` + regenerated JSON Schema (`make schema-docs` / `scripts/generate_models.py`).
- When schema surface changes, add fixtures that `jsonschema.validate` against `schema/generated/*.schema.json` (see `tests/test_linkml_validate.py`).
- When registering new **ops**, cover `JobEnvelope` / op JSON in tests.
- Contract path details: [`docs/extending.md`](../../../docs/extending.md) (LinkML checklist). Prefer that path; do not invent a second validation story.

## Tests required per contrib package

1. Registration smoke (`register()` / entry point → name in `list_targets()` or op catalog).
2. Pydantic accept/reject for option models.
3. LinkML/jsonschema coverage whenever schema surface changes.
4. A small numeric / evaluate smoke for targets.

## Layout

```
src/phridge/contrib/<name>/
  __init__.py    # register() zero-arg; safe if called twice
  …              # typed modules; pydantic options; no torch on client helpers
```

Wire discovery in `pyproject.toml`:

```toml
[project.entry-points."phridge.targets"]
my_target = "phridge.contrib.<name>:register"
# or phridge.ops for new ops
```

Put shared FFT / SF-server core in `phridge.sfcalc`, not here.
