---
search:
  boost: 5.0
---

# Slot: kind 

<div data-search-exclude markdown="1">



URI: [phridge:kind](https://github.com/phzwart/phridge/schema/phridge/kind)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [ObjectRef](ObjectRef.md) | Pointer at bytes in Redis plus enough metadata to decode them |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [ObjectKind](ObjectKind.md) |
| Domain Of | [ObjectRef](ObjectRef.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |
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
| self | phridge:kind |
| native | phridge:kind |




## LinkML Source

<details>
```yaml
name: kind
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
owner: ObjectRef
domain_of:
- ObjectRef
range: ObjectKind
required: true

```
</details></div>