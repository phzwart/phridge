---
search:
  boost: 5.0
---

# Slot: target 


_Target value when the gradients came from a target op_



<div data-search-exclude markdown="1">



URI: [phridge:target](https://github.com/phzwart/phridge/schema/phridge/target)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [SfGradients](SfGradients.md) | d(target)/d(scatterer parameters), index-aligned with the XrayStructure the g... |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Float](Float.md) |
| Domain Of | [SfGradients](SfGradients.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
### Slot Characteristics

| Property | Value |
| --- | --- |
| Owner | [SfGradients](SfGradients.md) |












## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:target |
| native | phridge:target |




## LinkML Source

<details>
```yaml
name: target
description: Target value when the gradients came from a target op
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
owner: SfGradients
domain_of:
- SfGradients
range: float

```
</details></div>