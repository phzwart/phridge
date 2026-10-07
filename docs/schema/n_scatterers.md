---
search:
  boost: 5.0
---

# Slot: n_scatterers 

<div data-search-exclude markdown="1">



URI: [phridge:n_scatterers](https://github.com/phzwart/phridge/schema/phridge/n_scatterers)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [XrayStructure](XrayStructure.md) | Canonical cctbx |  no  |
| [SfGradients](SfGradients.md) | d(target)/d(scatterer parameters), index-aligned with the XrayStructure the g... |  no  |
| [SfCurvatures](SfCurvatures.md) | Per-atom Gauss-Newton curvature blocks diag/blocks of J^T H_F J, index-aligne... |  no  |
| [SpatialSigmaAV2](SpatialSigmaAV2.md) | Optional per-atom error-model block (spatial σ_A v2) |  no  |
| [SpatialSigmaAV2Result](SpatialSigmaAV2Result.md) | Field-coefficient gradients and per-atom diagnostics returned after a field s... |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [String](String.md) |
| Domain Of | [XrayStructure](XrayStructure.md), [SfGradients](SfGradients.md), [SfCurvatures](SfCurvatures.md), [SpatialSigmaAV2](SpatialSigmaAV2.md), [SpatialSigmaAV2Result](SpatialSigmaAV2Result.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |










## Identifier and Mapping Information






## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:n_scatterers |
| native | phridge:n_scatterers |




## LinkML Source

<details>
```yaml
name: n_scatterers
domain_of:
- XrayStructure
- SfGradients
- SfCurvatures
- SpatialSigmaAV2
- SpatialSigmaAV2Result
range: string

```
</details></div>