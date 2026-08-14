---
search:
  boost: 5.0
---

# Slot: meta 


_Kind-specific JSON (ArrayMeta, BlobMeta, or a CctbxObject)_



<div data-search-exclude markdown="1">



URI: [phridge:meta](https://github.com/phzwart/phridge/schema/phridge/meta)
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
| self | phridge:meta |
| native | phridge:meta |




## LinkML Source

<details>
```yaml
name: meta
description: Kind-specific JSON (ArrayMeta, BlobMeta, or a CctbxObject)
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
owner: ObjectRef
domain_of:
- ObjectRef
range: string

```
</details></div>