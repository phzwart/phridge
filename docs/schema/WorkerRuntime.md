---
search:
  boost: 2.0
---


# Enum: WorkerRuntime 



<div data-search-exclude markdown="1">

URI: [phridge:WorkerRuntime](https://github.com/phzwart/phridge/schema/phridge/WorkerRuntime)

## Permissible Values
| Value | Meaning | Description |
| --- | --- | --- |
| torch | None | PyTorch worker (GPU/CPU science kernels; never imports cctbx) |
| cctbx | None | CCTBX/mmtbx worker (packing, restraints build; never imports torch) |




## Slots

| Name | Description |
| ---  | --- |
| [runtime](runtime.md) | Which worker stream consumes this op (torch vs cctbx) |










## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge






## LinkML Source

<details>
```yaml
name: WorkerRuntime
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
permissible_values:
  torch:
    text: torch
    description: PyTorch worker (GPU/CPU science kernels; never imports cctbx)
  cctbx:
    text: cctbx
    description: CCTBX/mmtbx worker (packing, restraints build; never imports torch)

```
</details>

</div>