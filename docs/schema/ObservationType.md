---
search:
  boost: 2.0
---


# Enum: ObservationType 



<div data-search-exclude markdown="1">

URI: [phridge:ObservationType](https://github.com/phzwart/phridge/schema/phridge/ObservationType)

## Permissible Values
| Value | Meaning | Description |
| --- | --- | --- |
| fobs | None | X-ray amplitude observations (F) |
| iobs | None | X-ray intensity observations (I) |
| amplitude | None | Amplitudes that are not tagged as F-obs |
| intensity | None | Intensities that are not tagged as I-obs |
| complex | None | Complex values (e |
| hl | None | Hendrickson–Lattman coefficients |
| other | None | Unclassified or custom data |




## Slots

| Name | Description |
| ---  | --- |
| [observation_type](observation_type.md) |  |










## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge






## LinkML Source

<details>
```yaml
name: ObservationType
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
permissible_values:
  fobs:
    text: fobs
    description: X-ray amplitude observations (F)
  iobs:
    text: iobs
    description: X-ray intensity observations (I)
  amplitude:
    text: amplitude
    description: Amplitudes that are not tagged as F-obs
  intensity:
    text: intensity
    description: Intensities that are not tagged as I-obs
  complex:
    text: complex
    description: Complex values (e.g. Fcalc, map coefficients)
  hl:
    text: hl
    description: Hendrickson–Lattman coefficients
  other:
    text: other
    description: Unclassified or custom data

```
</details>

</div>