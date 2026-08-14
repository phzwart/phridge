---
search:
  boost: 2.0
---


# Enum: ObjectKind 



<div data-search-exclude markdown="1">

URI: [phridge:ObjectKind](https://github.com/phzwart/phridge/schema/phridge/ObjectKind)

## Permissible Values
| Value | Meaning | Description |
| --- | --- | --- |
| array | None | Contiguous numeric buffer (raw or npy) |
| json | None | UTF-8 JSON document |
| blob | None | Opaque bytes plus content_type |
| cctbx | None | Instance of a phridge_cctbx class; see cctbx_type |




## Slots

| Name | Description |
| ---  | --- |
| [kind](kind.md) |  |










## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge






## LinkML Source

<details>
```yaml
name: ObjectKind
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
permissible_values:
  array:
    text: array
    description: Contiguous numeric buffer (raw or npy)
  json:
    text: json
    description: UTF-8 JSON document
  blob:
    text: blob
    description: Opaque bytes plus content_type
  cctbx:
    text: cctbx
    description: Instance of a phridge_cctbx class; see cctbx_type

```
</details>

</div>