---
search:
  boost: 2.0
---


# Enum: CoordinateFrame 



<div data-search-exclude markdown="1">

URI: [phridge:CoordinateFrame](https://github.com/phzwart/phridge/schema/phridge/CoordinateFrame)

## Permissible Values
| Value | Meaning | Description |
| --- | --- | --- |
| cartesian | None | Ångström orthogonal coordinates (cctbx sites_cart) |
| fractional | None | Fractional crystal coordinates (cctbx sites_frac) |




## Slots

| Name | Description |
| ---  | --- |
| [frame](frame.md) |  |










## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge






## LinkML Source

<details>
```yaml
name: CoordinateFrame
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
permissible_values:
  cartesian:
    text: cartesian
    description: Ångström orthogonal coordinates (cctbx sites_cart)
  fractional:
    text: fractional
    description: Fractional crystal coordinates (cctbx sites_frac)

```
</details>

</div>