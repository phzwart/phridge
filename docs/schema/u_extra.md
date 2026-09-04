---
search:
  boost: 5.0
---

# Slot: u_extra 


_Override the extra isotropic U added before sampling_



<div data-search-exclude markdown="1">



URI: [phridge:u_extra](https://github.com/phzwart/phridge/schema/phridge/u_extra)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [SfEngineParams](SfEngineParams.md) | Gridding / accuracy controls for the FFT structure-factor engine |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Float](Float.md) |
| Domain Of | [SfEngineParams](SfEngineParams.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
### Slot Characteristics

| Property | Value |
| --- | --- |
| Owner | [SfEngineParams](SfEngineParams.md) |












## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:u_extra |
| native | phridge:u_extra |




## LinkML Source

<details>
```yaml
name: u_extra
description: Override the extra isotropic U added before sampling
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
owner: SfEngineParams
domain_of:
- SfEngineParams
range: float

```
</details></div>