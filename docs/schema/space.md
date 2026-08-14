---
search:
  boost: 5.0
---

# Slot: space 

<div data-search-exclude markdown="1">



URI: [phridge:space](https://github.com/phzwart/phridge/schema/phridge/space)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [CrystalGridding](CrystalGridding.md) | Canonical maptbx |  no  |
| [RealMap](RealMap.md) | Canonical real-space map (maptbx / iotbx |  no  |
| [ComplexMap](ComplexMap.md) | Complex grid (e |  no  |
| [EmMap](EmMap.md) | Canonical iotbx |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [String](String.md) |
| Domain Of | [CrystalGridding](CrystalGridding.md), [RealMap](RealMap.md), [ComplexMap](ComplexMap.md), [EmMap](EmMap.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |










## Identifier and Mapping Information






## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:space |
| native | phridge:space |




## LinkML Source

<details>
```yaml
name: space
domain_of:
- CrystalGridding
- RealMap
- ComplexMap
- EmMap
range: string

```
</details></div>