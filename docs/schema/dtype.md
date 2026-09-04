---
search:
  boost: 5.0
---

# Slot: dtype 

<div data-search-exclude markdown="1">



URI: [phridge:dtype](https://github.com/phzwart/phridge/schema/phridge/dtype)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [ArrayMeta](ArrayMeta.md) |  |  no  |
| [ObjectRef](ObjectRef.md) | Pointer at bytes in Redis plus enough metadata to decode them |  no  |
| [RealMap](RealMap.md) | Canonical real-space map (maptbx / iotbx |  no  |
| [ComplexMap](ComplexMap.md) | Complex grid (e |  no  |
| [EmMap](EmMap.md) | Canonical iotbx |  no  |
| [CartesianSites](CartesianSites.md) | cctbx xrs |  no  |
| [FractionalSites](FractionalSites.md) | cctbx xrs |  no  |
| [SfEngineParams](SfEngineParams.md) | Gridding / accuracy controls for the FFT structure-factor engine |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [String](String.md) |
| Domain Of | [ArrayMeta](ArrayMeta.md), [ObjectRef](ObjectRef.md), [RealMap](RealMap.md), [ComplexMap](ComplexMap.md), [EmMap](EmMap.md), [CartesianSites](CartesianSites.md), [FractionalSites](FractionalSites.md), [SfEngineParams](SfEngineParams.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |










## Identifier and Mapping Information






## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:dtype |
| native | phridge:dtype |




## LinkML Source

<details>
```yaml
name: dtype
domain_of:
- ArrayMeta
- ObjectRef
- RealMap
- ComplexMap
- EmMap
- CartesianSites
- FractionalSites
- SfEngineParams
range: string

```
</details></div>