---
search:
  boost: 5.0
---

# Slot: n_refl 

<div data-search-exclude markdown="1">



URI: [phridge:n_refl](https://github.com/phzwart/phridge/schema/phridge/n_refl)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [MillerArray](MillerArray.md) | Canonical form of cctbx |  no  |
| [HendricksonLattman](HendricksonLattman.md) | Canonical HL coefficients (cctbx miller array of hendrickson_lattman) |  no  |
| [ReflectionFile](ReflectionFile.md) | Canonical iotbx |  no  |
| [TargetResult](TargetResult.md) | Evaluation of a reciprocal-space target on a reflection list |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [String](String.md) |
| Domain Of | [MillerArray](MillerArray.md), [HendricksonLattman](HendricksonLattman.md), [ReflectionFile](ReflectionFile.md), [TargetResult](TargetResult.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |










## Identifier and Mapping Information






## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:n_refl |
| native | phridge:n_refl |




## LinkML Source

<details>
```yaml
name: n_refl
domain_of:
- MillerArray
- HendricksonLattman
- ReflectionFile
- TargetResult
range: string

```
</details></div>