---
search:
  boost: 5.0
---

# Slot: n_shells 

<div data-search-exclude markdown="1">



URI: [phridge:n_shells](https://github.com/phzwart/phridge/schema/phridge/n_shells)
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
| self | phridge:n_shells |
| native | phridge:n_shells |




## LinkML Source

<details>
```yaml
name: n_shells
domain_of:
- SpatialSigmaAV2
- SpatialSigmaAV2Result
range: string

```
</details></div>