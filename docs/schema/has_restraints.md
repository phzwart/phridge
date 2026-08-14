---
search:
  boost: 5.0
---

# Slot: has_restraints 

<div data-search-exclude markdown="1">



URI: [phridge:has_restraints](https://github.com/phzwart/phridge/schema/phridge/has_restraints)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [ModelGeometry](ModelGeometry.md) | Correspondence header asserting that hierarchy, xray scatterers, and restrain... |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Boolean](Boolean.md) |
| Domain Of | [ModelGeometry](ModelGeometry.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |
### Slot Characteristics

| Property | Value |
| --- | --- |
| Owner | [ModelGeometry](ModelGeometry.md) |












## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:has_restraints |
| native | phridge:has_restraints |




## LinkML Source

<details>
```yaml
name: has_restraints
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
owner: ModelGeometry
domain_of:
- ModelGeometry
range: boolean
required: true

```
</details></div>