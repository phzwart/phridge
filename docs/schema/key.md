---
search:
  boost: 5.0
---

# Slot: key 


_Redis key for bytes; omitted for JSON-only CrystalSymmetry_



<div data-search-exclude markdown="1">



URI: [phridge:key](https://github.com/phzwart/phridge/schema/phridge/key)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [ObjectRef](ObjectRef.md) | Pointer at bytes in Redis plus enough metadata to decode them |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [String](String.md) |
| Domain Of | [ObjectRef](ObjectRef.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
### Slot Characteristics

| Property | Value |
| --- | --- |
| Owner | [ObjectRef](ObjectRef.md) |












## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:key |
| native | phridge:key |




## LinkML Source

<details>
```yaml
name: key
description: Redis key for bytes; omitted for JSON-only CrystalSymmetry
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
owner: ObjectRef
domain_of:
- ObjectRef
range: string

```
</details></div>