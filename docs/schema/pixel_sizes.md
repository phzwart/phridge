---
search:
  boost: 5.0
---

# Slot: pixel_sizes 


_Voxel size (Å) along a, b, c_



<div data-search-exclude markdown="1">



URI: [phridge:pixel_sizes](https://github.com/phzwart/phridge/schema/phridge/pixel_sizes)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [EmMap](EmMap.md) | Canonical iotbx |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Float](Float.md) |
| Domain Of | [EmMap](EmMap.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Multivalued | Yes |
| Minimum Cardinality | 3 |
| Maximum Cardinality | 3 |
### Slot Characteristics

| Property | Value |
| --- | --- |
| Owner | [EmMap](EmMap.md) |












## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:pixel_sizes |
| native | phridge:pixel_sizes |




## LinkML Source

<details>
```yaml
name: pixel_sizes
description: Voxel size (Å) along a, b, c
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
owner: EmMap
domain_of:
- EmMap
range: float
multivalued: true
minimum_cardinality: 3
maximum_cardinality: 3

```
</details></div>