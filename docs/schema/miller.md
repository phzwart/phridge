---
search:
  boost: 5.0
---

# Slot: miller 


_Must have observation_type complex_



<div data-search-exclude markdown="1">



URI: [phridge:miller](https://github.com/phzwart/phridge/schema/phridge/miller)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [MapCoefficients](MapCoefficients.md) | Complex miller array used as FFT map coefficients (cctbx miller array of Fcal... |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [MillerArray](MillerArray.md) |
| Domain Of | [MapCoefficients](MapCoefficients.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |
### Slot Characteristics

| Property | Value |
| --- | --- |
| Owner | [MapCoefficients](MapCoefficients.md) |












## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:miller |
| native | phridge:miller |




## LinkML Source

<details>
```yaml
name: miller
description: Must have observation_type complex
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
owner: MapCoefficients
domain_of:
- MapCoefficients
range: MillerArray
required: true
inlined: true

```
</details></div>