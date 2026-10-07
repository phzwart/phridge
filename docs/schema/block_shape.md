---
search:
  boost: 5.0
---

# Slot: block_shape 


_Axis extents; a string names a symbolic dimension_



<div data-search-exclude markdown="1">



URI: [phridge:block_shape](https://github.com/phzwart/phridge/schema/phridge/block_shape)
Alias: shape

<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [BlockDecl](BlockDecl.md) |  |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [String](String.md)&nbsp;or&nbsp;<br />[Integer](Integer.md) |
| Domain Of | [BlockDecl](BlockDecl.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |
| Multivalued | Yes |
### Slot Characteristics

| Property | Value |
| --- | --- |
| Owner | [BlockDecl](BlockDecl.md) |


<details>
<summary>Expressions & Logic</summary>
#### Any Of

Value must satisfy at least one of:
- AnonymousSlotExpression({'range': 'integer'})
- AnonymousSlotExpression({'range': 'string'})

</details>











## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:block_shape |
| native | phridge:block_shape |




## LinkML Source

<details>
```yaml
name: block_shape
description: Axis extents; a string names a symbolic dimension
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
alias: shape
owner: BlockDecl
domain_of:
- BlockDecl
range: string
required: true
multivalued: true
any_of:
- range: integer
- range: string

```
</details></div>