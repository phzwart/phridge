---
search:
  boost: 5.0
---

# Slot: field_cutoff 


_Band-limit of λ(x) and κ(x) in Å_



<div data-search-exclude markdown="1">



URI: [phridge:field_cutoff](https://github.com/phzwart/phridge/schema/phridge/field_cutoff)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [SpatialSigmaAV2](SpatialSigmaAV2.md) | Optional per-atom error-model block (spatial σ_A v2) |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Float](Float.md) |
| Domain Of | [SpatialSigmaAV2](SpatialSigmaAV2.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |
### Slot Characteristics

| Property | Value |
| --- | --- |
| Owner | [SpatialSigmaAV2](SpatialSigmaAV2.md) |












## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:field_cutoff |
| native | phridge:field_cutoff |




## LinkML Source

<details>
```yaml
name: field_cutoff
description: Band-limit of λ(x) and κ(x) in Å
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
owner: SpatialSigmaAV2
domain_of:
- SpatialSigmaAV2
range: float
required: true

```
</details></div>