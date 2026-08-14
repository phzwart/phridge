---
search:
  boost: 5.0
---

# Slot: shape 

<div data-search-exclude markdown="1">



URI: [phridge:shape](https://github.com/phzwart/phridge/schema/phridge/shape)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [ArrayMeta](ArrayMeta.md) |  |  no  |
| [ObjectRef](ObjectRef.md) | Pointer at bytes in Redis plus enough metadata to decode them |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [String](String.md) |
| Domain Of | [ArrayMeta](ArrayMeta.md), [ObjectRef](ObjectRef.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |










## Identifier and Mapping Information






## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:shape |
| native | phridge:shape |




## LinkML Source

<details>
```yaml
name: shape
domain_of:
- ArrayMeta
- ObjectRef
range: string

```
</details></div>