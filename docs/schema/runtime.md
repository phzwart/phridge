---
search:
  boost: 5.0
---

# Slot: runtime 


_Which worker stream consumes this op (torch vs cctbx)_



<div data-search-exclude markdown="1">



URI: [phridge:runtime](https://github.com/phzwart/phridge/schema/phridge/runtime)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [OpSpec](OpSpec.md) |  |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [WorkerRuntime](WorkerRuntime.md) |
| Domain Of | [OpSpec](OpSpec.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |
### Slot Characteristics

| Property | Value |
| --- | --- |
| If Absent | `string(torch)` |
| Owner | [OpSpec](OpSpec.md) |












## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:runtime |
| native | phridge:runtime |




## LinkML Source

<details>
```yaml
name: runtime
description: Which worker stream consumes this op (torch vs cctbx)
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
ifabsent: string(torch)
owner: OpSpec
domain_of:
- OpSpec
range: WorkerRuntime
required: true

```
</details></div>