---
search:
  boost: 5.0
---

# Slot: table 


_Registry table name (wk1995, it1992, n_gaussian, ...)_



<div data-search-exclude markdown="1">



URI: [phridge:table](https://github.com/phzwart/phridge/schema/phridge/table)
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
| self | phridge:table |
| native | phridge:table |




## LinkML Source

<details>
```yaml
name: table
description: Registry table name (wk1995, it1992, n_gaussian, ...)
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
owner: ScatteringTable
domain_of:
- ScatteringTable
range: string
required: true

```
</details></div>