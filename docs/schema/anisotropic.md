---
search:
  boost: 5.0
---

# Slot: anisotropic 


_If true, npz u_star[i] is live (cctbx u_star, fractional)_



<div data-search-exclude markdown="1">



URI: [phridge:anisotropic](https://github.com/phzwart/phridge/schema/phridge/anisotropic)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [Scatterer](Scatterer.md) | One cctbx |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Boolean](Boolean.md) |
| Domain Of | [Scatterer](Scatterer.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |
### Slot Characteristics

| Property | Value |
| --- | --- |
| Owner | [Scatterer](Scatterer.md) |












## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:anisotropic |
| native | phridge:anisotropic |




## LinkML Source

<details>
```yaml
name: anisotropic
description: If true, npz u_star[i] is live (cctbx u_star, fractional)
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
owner: Scatterer
domain_of:
- Scatterer
range: boolean
required: true

```
</details></div>