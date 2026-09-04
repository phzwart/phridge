# phridge documentation

- [Types, packing, and i_seq correspondence](types.md)
- [Client converters, geometry, and EM maps](client.md)
- [Torch structure-factor engine, targets, and the Phenix drop-in](engine.md)
- [Generated LinkML class catalog](schema/index.md) (`make schema-docs`)

JSON Schema for metadata is under [`schema/generated/`](../schema/generated/).
Instance checks (`linkml-validate` / `linkml.validator.validate`) cover JSON
only; packed npz and i_seq identity stay in Python.

The job protocol lives in [`schema/phridge.yaml`](../schema/phridge.yaml).
Science types live in `schema/cctbx*.yaml`. Redis is the only shared
process boundary: the Phenix client never imports torch; the worker
never imports cctbx.
