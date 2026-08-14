---
search:
  boost: 5.0
---

# Slot: anomalous 

<div data-search-exclude markdown="1">



URI: [phridge:anomalous](https://github.com/phzwart/phridge/schema/phridge/anomalous)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [MillerArray](MillerArray.md) | Canonical form of cctbx |  no  |
| [HendricksonLattman](HendricksonLattman.md) | Canonical HL coefficients (cctbx miller array of hendrickson_lattman) |  no  |
| [ReflectionColumn](ReflectionColumn.md) | One data column in a ReflectionFile (not H/K/L) |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [String](String.md) |
| Domain Of | [MillerArray](MillerArray.md), [HendricksonLattman](HendricksonLattman.md), [ReflectionColumn](ReflectionColumn.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |










## Identifier and Mapping Information






## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:anomalous |
| native | phridge:anomalous |




## LinkML Source

<details>
```yaml
name: anomalous
domain_of:
- MillerArray
- HendricksonLattman
- ReflectionColumn
range: string

```
</details></div>