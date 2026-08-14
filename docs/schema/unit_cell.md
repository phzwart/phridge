---
search:
  boost: 5.0
---

# Slot: unit_cell 


_a, b, c (Å), alpha, beta, gamma (degrees)_



<div data-search-exclude markdown="1">



URI: [phridge:unit_cell](https://github.com/phzwart/phridge/schema/phridge/unit_cell)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [CrystalSymmetry](CrystalSymmetry.md) | Canonical form of cctbx |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Float](Float.md) |
| Domain Of | [CrystalSymmetry](CrystalSymmetry.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |
| Multivalued | Yes |
| Minimum Cardinality | 6 |
| Maximum Cardinality | 6 |
### Slot Characteristics

| Property | Value |
| --- | --- |
| Owner | [CrystalSymmetry](CrystalSymmetry.md) |












## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:unit_cell |
| native | phridge:unit_cell |




## LinkML Source

<details>
```yaml
name: unit_cell
description: a, b, c (Å), alpha, beta, gamma (degrees)
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
owner: CrystalSymmetry
domain_of:
- CrystalSymmetry
range: float
required: true
multivalued: true
minimum_cardinality: 6
maximum_cardinality: 6

```
</details></div>