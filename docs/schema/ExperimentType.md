---
search:
  boost: 2.0
---


# Enum: ExperimentType 



<div data-search-exclude markdown="1">

URI: [phridge:ExperimentType](https://github.com/phzwart/phridge/schema/phridge/ExperimentType)

## Permissible Values
| Value | Meaning | Description |
| --- | --- | --- |
| xray | None | MX / wrapping crystal map |
| cryo_em | None | Cryo-EM / MRC map (typically non-wrapping) |
| neutron | None |  |
| other | None |  |




## Slots

| Name | Description |
| ---  | --- |
| [experiment_type](experiment_type.md) |  |










## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge






## LinkML Source

<details>
```yaml
name: ExperimentType
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
permissible_values:
  xray:
    text: xray
    description: MX / wrapping crystal map
  cryo_em:
    text: cryo_em
    description: Cryo-EM / MRC map (typically non-wrapping)
  neutron:
    text: neutron
  other:
    text: other

```
</details>

</div>