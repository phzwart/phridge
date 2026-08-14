---
search:
  boost: 5.0
---

# Slot: byte_order 

<div data-search-exclude markdown="1">



URI: [phridge:byte_order](https://github.com/phzwart/phridge/schema/phridge/byte_order)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [ObjectRef](ObjectRef.md) | Pointer at bytes in Redis plus enough metadata to decode them |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [ByteOrder](ByteOrder.md) |
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
| self | phridge:byte_order |
| native | phridge:byte_order |




## LinkML Source

<details>
```yaml
name: byte_order
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
owner: ObjectRef
domain_of:
- ObjectRef
range: ByteOrder

```
</details></div>