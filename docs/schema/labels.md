---
search:
  boost: 5.0
---

# Slot: labels 


_Scattering type label per row (C, N, O, S, Se, water, ...)_



<div data-search-exclude markdown="1">



URI: [phridge:labels](https://github.com/phzwart/phridge/schema/phridge/labels)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [ScatteringTable](ScatteringTable.md) | Gaussian form-factor coefficients per scattering type (cctbx scattering_type_... |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [String](String.md) |
| Domain Of | [ScatteringTable](ScatteringTable.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |
| Multivalued | Yes |
### Slot Characteristics

| Property | Value |
| --- | --- |
| Owner | [ScatteringTable](ScatteringTable.md) |












## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:labels |
| native | phridge:labels |




## LinkML Source

<details>
```yaml
name: labels
description: Scattering type label per row (C, N, O, S, Se, water, ...)
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
owner: ScatteringTable
domain_of:
- ScatteringTable
range: string
required: true
multivalued: true

```
</details></div>