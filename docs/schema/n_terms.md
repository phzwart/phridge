---
search:
  boost: 5.0
---

# Slot: n_terms 


_K, number of Gaussian terms per row_



<div data-search-exclude markdown="1">



URI: [phridge:n_terms](https://github.com/phzwart/phridge/schema/phridge/n_terms)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [ScatteringTable](ScatteringTable.md) | Gaussian form-factor coefficients per scattering type (cctbx scattering_type_... |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Integer](Integer.md) |
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
| self | phridge:n_terms |
| native | phridge:n_terms |




## LinkML Source

<details>
```yaml
name: n_terms
description: K, number of Gaussian terms per row
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
owner: ScatteringTable
domain_of:
- ScatteringTable
range: integer
required: true

```
</details></div>