---
search:
  boost: 5.0
---

# Slot: observation_type 

<div data-search-exclude markdown="1">



URI: [phridge:observation_type](https://github.com/phzwart/phridge/schema/phridge/observation_type)
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
| self | phridge:observation_type |
| native | phridge:observation_type |




## LinkML Source

<details>
```yaml
name: observation_type
domain_of:
- MillerArray
- ReflectionColumn
range: string

```
</details></div>