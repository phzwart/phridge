---
search:
  boost: 5.0
---

# Slot: mtz_type 

<div data-search-exclude markdown="1">



URI: [phridge:mtz_type](https://github.com/phzwart/phridge/schema/phridge/mtz_type)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [ReflectionColumn](ReflectionColumn.md) | One data column in a ReflectionFile (not H/K/L) |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [MtzColumnType](MtzColumnType.md) |
| Domain Of | [ReflectionColumn](ReflectionColumn.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |
### Slot Characteristics

| Property | Value |
| --- | --- |
| Owner | [ReflectionColumn](ReflectionColumn.md) |












## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:mtz_type |
| native | phridge:mtz_type |




## LinkML Source

<details>
```yaml
name: mtz_type
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
owner: ReflectionColumn
domain_of:
- ReflectionColumn
range: MtzColumnType
required: true

```
</details></div>