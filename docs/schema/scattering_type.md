---
search:
  boost: 5.0
---

# Slot: scattering_type 


_cctbx scattering type (C, N, S, AU, water, …)_



<div data-search-exclude markdown="1">



URI: [phridge:scattering_type](https://github.com/phzwart/phridge/schema/phridge/scattering_type)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [Scatterer](Scatterer.md) | One cctbx |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [String](String.md) |
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
| self | phridge:scattering_type |
| native | phridge:scattering_type |




## LinkML Source

<details>
```yaml
name: scattering_type
description: cctbx scattering type (C, N, S, AU, water, …)
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
owner: Scatterer
domain_of:
- Scatterer
range: string
required: true

```
</details></div>