---
search:
  boost: 5.0
---

# Slot: transform 

<div data-search-exclude markdown="1">



URI: [phridge:transform](https://github.com/phzwart/phridge/schema/phridge/transform)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [BlockDecl](BlockDecl.md) |  |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [BlockTransform](BlockTransform.md) |
| Domain Of | [BlockDecl](BlockDecl.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
### Slot Characteristics

| Property | Value |
| --- | --- |
| If Absent | `string(identity)` |
| Owner | [BlockDecl](BlockDecl.md) |












## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:transform |
| native | phridge:transform |




## LinkML Source

<details>
```yaml
name: transform
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
ifabsent: string(identity)
owner: BlockDecl
domain_of:
- BlockDecl
range: BlockTransform

```
</details></div>