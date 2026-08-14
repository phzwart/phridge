---
search:
  boost: 5.0
---

# Slot: columns 

<div data-search-exclude markdown="1">



URI: [phridge:columns](https://github.com/phzwart/phridge/schema/phridge/columns)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [ReflectionFile](ReflectionFile.md) | Canonical iotbx |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [ReflectionColumn](ReflectionColumn.md) |
| Domain Of | [ReflectionFile](ReflectionFile.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |
| Multivalued | Yes |
### Slot Characteristics

| Property | Value |
| --- | --- |
| Owner | [ReflectionFile](ReflectionFile.md) |












## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:columns |
| native | phridge:columns |




## LinkML Source

<details>
```yaml
name: columns
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
owner: ReflectionFile
domain_of:
- ReflectionFile
range: ReflectionColumn
required: true
multivalued: true
inlined: true

```
</details></div>