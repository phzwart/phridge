---
search:
  boost: 5.0
---

# Slot: origin_cart 


_Cartesian origin shift (Å), map_manager.shift_cart_



<div data-search-exclude markdown="1">



URI: [phridge:origin_cart](https://github.com/phzwart/phridge/schema/phridge/origin_cart)
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
| self | phridge:origin_cart |
| native | phridge:origin_cart |




## LinkML Source

<details>
```yaml
name: origin_cart
description: Cartesian origin shift (Å), map_manager.shift_cart
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