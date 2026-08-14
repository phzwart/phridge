---
search:
  boost: 5.0
---

# Slot: origin 

<div data-search-exclude markdown="1">



URI: [phridge:origin](https://github.com/phzwart/phridge/schema/phridge/origin)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [RealMap](RealMap.md) | Canonical real-space map (maptbx / iotbx |  no  |
| [ComplexMap](ComplexMap.md) | Complex grid (e |  no  |
| [EmMap](EmMap.md) | Canonical iotbx |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [String](String.md) |
| Domain Of | [RealMap](RealMap.md), [ComplexMap](ComplexMap.md), [EmMap](EmMap.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |










## Identifier and Mapping Information






## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:origin |
| native | phridge:origin |




## LinkML Source

<details>
```yaml
name: origin
domain_of:
- RealMap
- ComplexMap
- EmMap
range: string

```
</details></div>