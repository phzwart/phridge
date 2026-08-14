---
search:
  boost: 2.0
---


# Enum: AnomalousLayout 



<div data-search-exclude markdown="1">

URI: [phridge:AnomalousLayout](https://github.com/phzwart/phridge/schema/phridge/AnomalousLayout)

## Permissible Values
| Value | Meaning | Description |
| --- | --- | --- |
| asu | None | Unique ASU indices only |
| both_hemispheres | None | Friedel mates both present |




## Slots

| Name | Description |
| ---  | --- |
| [anomalous_layout](anomalous_layout.md) | ASU vs both hemispheres; never inferred |










## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge






## LinkML Source

<details>
```yaml
name: AnomalousLayout
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
permissible_values:
  asu:
    text: asu
    description: Unique ASU indices only
  both_hemispheres:
    text: both_hemispheres
    description: Friedel mates both present

```
</details>

</div>