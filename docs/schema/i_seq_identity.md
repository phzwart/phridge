---
search:
  boost: 5.0
---

# Slot: i_seq_identity 


_True when Hierarchy.atoms[i].i == XrayStructure.scatterers[i].i == i for all i in 0..n_sites-1, and every restraint i_seq is in that range._

__



<div data-search-exclude markdown="1">



URI: [phridge:i_seq_identity](https://github.com/phzwart/phridge/schema/phridge/i_seq_identity)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [ModelGeometry](ModelGeometry.md) | Correspondence header asserting that hierarchy, xray scatterers, and restrain... |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Boolean](Boolean.md) |
| Domain Of | [ModelGeometry](ModelGeometry.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |
### Slot Characteristics

| Property | Value |
| --- | --- |
| Owner | [ModelGeometry](ModelGeometry.md) |












## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:i_seq_identity |
| native | phridge:i_seq_identity |




## LinkML Source

<details>
```yaml
name: i_seq_identity
description: 'True when Hierarchy.atoms[i].i == XrayStructure.scatterers[i].i == i
  for all i in 0..n_sites-1, and every restraint i_seq is in that range.

  '
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
owner: ModelGeometry
domain_of:
- ModelGeometry
range: boolean
required: true

```
</details></div>