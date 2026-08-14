---
search:
  boost: 5.0
---

# Slot: data_dtype 

<div data-search-exclude markdown="1">



URI: [phridge:data_dtype](https://github.com/phzwart/phridge/schema/phridge/data_dtype)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [MillerArray](MillerArray.md) | Canonical form of cctbx |  no  |
| [ReflectionColumn](ReflectionColumn.md) | One data column in a ReflectionFile (not H/K/L) |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [String](String.md) |
| Domain Of | [MillerArray](MillerArray.md), [ReflectionColumn](ReflectionColumn.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |










## Identifier and Mapping Information






## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:data_dtype |
| native | phridge:data_dtype |




## LinkML Source

<details>
```yaml
name: data_dtype
domain_of:
- MillerArray
- ReflectionColumn
range: string

```
</details></div>