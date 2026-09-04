---
search:
  boost: 5.0
---

# Slot: r 

<div data-search-exclude markdown="1">



URI: [phridge:r](https://github.com/phzwart/phridge/schema/phridge/r)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [SymOp](SymOp.md) | One symmetry operator in fractional coordinates |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Float](Float.md) |
| Domain Of | [SymOp](SymOp.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |
| Multivalued | Yes |
| Minimum Cardinality | 9 |
| Maximum Cardinality | 9 |
### Slot Characteristics

| Property | Value |
| --- | --- |
| Owner | [SymOp](SymOp.md) |












## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:r |
| native | phridge:r |




## LinkML Source

<details>
```yaml
name: r
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
owner: SymOp
domain_of:
- SymOp
range: float
required: true
multivalued: true
minimum_cardinality: 9
maximum_cardinality: 9

```
</details></div>