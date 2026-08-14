---
search:
  boost: 5.0
---

# Slot: npz_name 


_Array name inside the packed npz (usually the label)_



<div data-search-exclude markdown="1">



URI: [phridge:npz_name](https://github.com/phzwart/phridge/schema/phridge/npz_name)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [ReflectionColumn](ReflectionColumn.md) | One data column in a ReflectionFile (not H/K/L) |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [String](String.md) |
| Domain Of | [ReflectionColumn](ReflectionColumn.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |
### Slot Characteristics

| Property | Value |
| --- | --- |
| Owner | [ReflectionColumn](ReflectionColumn.md) |












## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:npz_name |
| native | phridge:npz_name |




## LinkML Source

<details>
```yaml
name: npz_name
description: Array name inside the packed npz (usually the label)
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
owner: ReflectionColumn
domain_of:
- ReflectionColumn
range: string
required: true

```
</details></div>