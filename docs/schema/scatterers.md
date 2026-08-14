---
search:
  boost: 5.0
---

# Slot: scatterers 

<div data-search-exclude markdown="1">



URI: [phridge:scatterers](https://github.com/phzwart/phridge/schema/phridge/scatterers)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [XrayStructure](XrayStructure.md) | Canonical cctbx |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Scatterer](Scatterer.md) |
| Domain Of | [XrayStructure](XrayStructure.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |
| Multivalued | Yes |
### Slot Characteristics

| Property | Value |
| --- | --- |
| Owner | [XrayStructure](XrayStructure.md) |












## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:scatterers |
| native | phridge:scatterers |




## LinkML Source

<details>
```yaml
name: scatterers
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
owner: XrayStructure
domain_of:
- XrayStructure
range: Scatterer
required: true
multivalued: true
inlined: true

```
</details></div>