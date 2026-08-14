---
search:
  boost: 5.0
---

# Slot: status 

<div data-search-exclude markdown="1">



URI: [phridge:status](https://github.com/phzwart/phridge/schema/phridge/status)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [JobEnvelope](JobEnvelope.md) |  |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [JobStatus](JobStatus.md) |
| Domain Of | [JobEnvelope](JobEnvelope.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |
### Slot Characteristics

| Property | Value |
| --- | --- |
| Owner | [JobEnvelope](JobEnvelope.md) |












## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:status |
| native | phridge:status |




## LinkML Source

<details>
```yaml
name: status
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
owner: JobEnvelope
domain_of:
- JobEnvelope
range: JobStatus
required: true

```
</details></div>