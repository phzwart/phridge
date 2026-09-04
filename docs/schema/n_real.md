---
search:
  boost: 5.0
---

# Slot: n_real 

<div data-search-exclude markdown="1">



URI: [phridge:n_real](https://github.com/phzwart/phridge/schema/phridge/n_real)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [CrystalGridding](CrystalGridding.md) | Canonical maptbx |  no  |
| [RealMap](RealMap.md) | Canonical real-space map (maptbx / iotbx |  no  |
| [ComplexMap](ComplexMap.md) | Complex grid (e |  no  |
| [EmMap](EmMap.md) | Canonical iotbx |  no  |
| [SfEngineParams](SfEngineParams.md) | Gridding / accuracy controls for the FFT structure-factor engine |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [String](String.md) |
| Domain Of | [CrystalGridding](CrystalGridding.md), [RealMap](RealMap.md), [ComplexMap](ComplexMap.md), [EmMap](EmMap.md), [SfEngineParams](SfEngineParams.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |










## Identifier and Mapping Information






## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:n_real |
| native | phridge:n_real |




## LinkML Source

<details>
```yaml
name: n_real
domain_of:
- CrystalGridding
- RealMap
- ComplexMap
- EmMap
- SfEngineParams
range: string

```
</details></div>