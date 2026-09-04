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






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [String](String.md) |
| Domain Of | [XrayStructure](XrayStructure.md), [SfGradients](SfGradients.md) |

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
range: string

```
</details></div>