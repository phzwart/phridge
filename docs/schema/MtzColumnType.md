---
search:
  boost: 2.0
---


# Enum: MtzColumnType 




_CCP4 MTZ column type character (iotbx.mtz column.type()). H/K/L are indices and are not stored as ReflectionColumn rows._

__



<div data-search-exclude markdown="1">

URI: [phridge:MtzColumnType](https://github.com/phzwart/phridge/schema/phridge/MtzColumnType)

## Permissible Values
| Value | Meaning | Description |
| --- | --- | --- |
| F | None | Amplitude |
| J | None | Intensity |
| D | None | Anomalous difference |
| Q | None | Standard deviation |
| G | None | F(+) |
| L | None | I(+) |
| K | None | F(-) |
| M | None | I(-) |
| P | None | Phase (degrees) |
| W | None | Weight |
| A | None | Hendrickson–Lattman A (or HL group) |
| B | None | Hendrickson–Lattman B |
| C | None | Hendrickson–Lattman C |
| I | None | Integer flag / batch |
| R | None | Real (e |
| other | None | Anything else; see mtz_type_raw |




## Slots

| Name | Description |
| ---  | --- |
| [mtz_type](mtz_type.md) |  |










## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge






## LinkML Source

<details>
```yaml
name: MtzColumnType
description: 'CCP4 MTZ column type character (iotbx.mtz column.type()). H/K/L are
  indices and are not stored as ReflectionColumn rows.

  '
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
permissible_values:
  F:
    text: F
    description: Amplitude
  J:
    text: J
    description: Intensity
  D:
    text: D
    description: Anomalous difference
  Q:
    text: Q
    description: Standard deviation
  G:
    text: G
    description: F(+)
  L:
    text: L
    description: I(+)
  K:
    text: K
    description: F(-)
  M:
    text: M
    description: I(-)
  P:
    text: P
    description: Phase (degrees)
  W:
    text: W
    description: Weight
  A:
    text: A
    description: Hendrickson–Lattman A (or HL group)
  B:
    text: B
    description: Hendrickson–Lattman B
  C:
    text: C
    description: Hendrickson–Lattman C
  I:
    text: I
    description: Integer flag / batch
  R:
    text: R
    description: Real (e.g. FOM)
  other:
    text: other
    description: Anything else; see mtz_type_raw

```
</details>

</div>