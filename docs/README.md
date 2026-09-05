# phridge documentation

- [Types, packing, and i_seq correspondence](types.md)
- [Client converters, geometry, and EM maps](client.md)
- [**Structure-factor server** (cctbx-feel API → remote GPU)](sf_server.md)
- [Torch structure-factor engine, targets, and the Phenix drop-in](engine.md) (`phridge.sfcalc`)
- [**Tutorial: FFT targets / gradients + custom likelihood**](tutorial_sf_targets.md)
- [Redis store, remote jobs, and `Bridge(memory=True)`](redis.md)
- [External ops, `phridge.contrib`, and custom workers](extending.md) (worked example: agentsg Niggli)
- [Generated LinkML class catalog](schema/index.md) (`make schema-docs`)

JSON Schema for metadata is under [`schema/generated/`](../schema/generated/).
Instance checks (`linkml-validate` / `linkml.validator.validate`) cover JSON
only; packed npz and i_seq identity stay in Python.

The job protocol lives in [`schema/phridge.yaml`](../schema/phridge.yaml).
Science types live in `schema/cctbx*.yaml`. Redis is mandatory for
cross-process jobs; use `Bridge(memory=True)` when you do not want a
server. Drivers and workers are peer processes on separate streams:
Phenix never imports torch; the torch worker never imports cctbx; the
CCTBX worker never imports torch. See [redis.md](redis.md).
