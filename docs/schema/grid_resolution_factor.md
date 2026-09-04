---
search:
  boost: 5.0
---

# Slot: grid_resolution_factor 


_Grid spacing = d_min * factor (default 1/3)_



<div data-search-exclude markdown="1">



URI: [phridge:grid_resolution_factor](https://github.com/phzwart/phridge/schema/phridge/grid_resolution_factor)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [SfEngineParams](SfEngineParams.md) | Gridding / accuracy controls for the FFT structure-factor engine |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Float](Float.md) |
| Domain Of | [SfEngineParams](SfEngineParams.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
### Slot Characteristics

| Property | Value |
| --- | --- |
| Owner | [SfEngineParams](SfEngineParams.md) |












## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:grid_resolution_factor |
| native | phridge:grid_resolution_factor |




## LinkML Source

<details>
```yaml
name: grid_resolution_factor
description: Grid spacing = d_min * factor (default 1/3)
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
owner: SfEngineParams
domain_of:
- SfEngineParams
range: float

```
</details></div>