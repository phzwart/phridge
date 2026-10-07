---
search:
  boost: 5.0
---

# Slot: n_coeff 

<div data-search-exclude markdown="1">



URI: [phridge:n_coeff](https://github.com/phzwart/phridge/schema/phridge/n_coeff)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [SpatialSigmaAV2](SpatialSigmaAV2.md) | Optional per-atom error-model block (spatial σ_A v2) |  no  |
| [SpatialSigmaAV2Result](SpatialSigmaAV2Result.md) | Field-coefficient gradients and per-atom diagnostics returned after a field s... |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [String](String.md) |
| Domain Of | [SpatialSigmaAV2](SpatialSigmaAV2.md), [SpatialSigmaAV2Result](SpatialSigmaAV2Result.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |










## Identifier and Mapping Information






## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:n_coeff |
| native | phridge:n_coeff |




## LinkML Source

<details>
```yaml
name: n_coeff
domain_of:
- SpatialSigmaAV2
- SpatialSigmaAV2Result
range: string

```
</details></div>