---
search:
  boost: 5.0
---

# Slot: wing_cutoff 


_Relative density at the sampling cutoff radius (default 1e-4)_



<div data-search-exclude markdown="1">



URI: [phridge:wing_cutoff](https://github.com/phzwart/phridge/schema/phridge/wing_cutoff)
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
| self | phridge:wing_cutoff |
| native | phridge:wing_cutoff |




## LinkML Source

<details>
```yaml
name: wing_cutoff
description: Relative density at the sampling cutoff radius (default 1e-4)
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
owner: SfEngineParams
domain_of:
- SfEngineParams
range: float

```
</details></div>