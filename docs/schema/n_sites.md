---
search:
  boost: 5.0
---

# Slot: n_sites 

<div data-search-exclude markdown="1">



URI: [phridge:n_sites](https://github.com/phzwart/phridge/schema/phridge/n_sites)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [CartesianSites](CartesianSites.md) | cctbx xrs |  no  |
| [FractionalSites](FractionalSites.md) | cctbx xrs |  no  |
| [GeometryRestraints](GeometryRestraints.md) | Canonical cctbx |  no  |
| [ModelGeometry](ModelGeometry.md) | Correspondence header asserting that hierarchy, xray scatterers, and restrain... |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [String](String.md) |
| Domain Of | [CartesianSites](CartesianSites.md), [FractionalSites](FractionalSites.md), [GeometryRestraints](GeometryRestraints.md), [ModelGeometry](ModelGeometry.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |










## Identifier and Mapping Information






## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:n_sites |
| native | phridge:n_sites |




## LinkML Source

<details>
```yaml
name: n_sites
domain_of:
- CartesianSites
- FractionalSites
- GeometryRestraints
- ModelGeometry
range: string

```
</details></div>