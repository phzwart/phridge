---
search:
  boost: 5.0
---

# Slot: compression 

<div data-search-exclude markdown="1">



URI: [phridge:compression](https://github.com/phzwart/phridge/schema/phridge/compression)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [ObjectRef](ObjectRef.md) | Pointer at bytes in Redis plus enough metadata to decode them |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Compression](Compression.md) |
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
| self | phridge:compression |
| native | phridge:compression |




## LinkML Source

<details>
```yaml
name: compression
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
owner: ObjectRef
domain_of:
- ObjectRef
range: Compression

```
</details></div>