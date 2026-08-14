---
search:
  boost: 5.0
---

# Slot: index_dtype 

<div data-search-exclude markdown="1">



URI: [phridge:index_dtype](https://github.com/phzwart/phridge/schema/phridge/index_dtype)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [MillerArray](MillerArray.md) | Canonical form of cctbx |  no  |
| [HendricksonLattman](HendricksonLattman.md) | Canonical HL coefficients (cctbx miller array of hendrickson_lattman) |  no  |
| [ReflectionFile](ReflectionFile.md) | Canonical iotbx |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [String](String.md) |
| Domain Of | [MillerArray](MillerArray.md), [HendricksonLattman](HendricksonLattman.md), [ReflectionFile](ReflectionFile.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |










## Identifier and Mapping Information






## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:index_dtype |
| native | phridge:index_dtype |




## LinkML Source

<details>
```yaml
name: index_dtype
domain_of:
- MillerArray
- HendricksonLattman
- ReflectionFile
range: string

```
</details></div>