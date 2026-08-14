---
search:
  boost: 5.0
---

# Slot: label 

<div data-search-exclude markdown="1">



URI: [phridge:label](https://github.com/phzwart/phridge/schema/phridge/label)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [MillerArray](MillerArray.md) | Canonical form of cctbx |  no  |
| [HendricksonLattman](HendricksonLattman.md) | Canonical HL coefficients (cctbx miller array of hendrickson_lattman) |  no  |
| [ReflectionColumn](ReflectionColumn.md) | One data column in a ReflectionFile (not H/K/L) |  no  |
| [RealMap](RealMap.md) | Canonical real-space map (maptbx / iotbx |  no  |
| [ComplexMap](ComplexMap.md) | Complex grid (e |  no  |
| [MapCoefficients](MapCoefficients.md) | Complex miller array used as FFT map coefficients (cctbx miller array of Fcal... |  no  |
| [EmMap](EmMap.md) | Canonical iotbx |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [String](String.md) |
| Domain Of | [MillerArray](MillerArray.md), [HendricksonLattman](HendricksonLattman.md), [ReflectionColumn](ReflectionColumn.md), [RealMap](RealMap.md), [ComplexMap](ComplexMap.md), [MapCoefficients](MapCoefficients.md), [EmMap](EmMap.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |










## Identifier and Mapping Information






## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:label |
| native | phridge:label |




## LinkML Source

<details>
```yaml
name: label
domain_of:
- MillerArray
- HendricksonLattman
- ReflectionColumn
- RealMap
- ComplexMap
- MapCoefficients
- EmMap
range: string

```
</details></div>