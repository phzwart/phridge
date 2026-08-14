---
search:
  boost: 5.0
---

# Slot: job_id 

<div data-search-exclude markdown="1">



URI: [phridge:job_id](https://github.com/phzwart/phridge/schema/phridge/job_id)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [JobEnvelope](JobEnvelope.md) |  |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [String](String.md) |
| Domain Of | [JobEnvelope](JobEnvelope.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Required | Yes |
### Slot Characteristics

| Property | Value |
| --- | --- |
| Identifier | Yes |
| Owner | [JobEnvelope](JobEnvelope.md) |












## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:job_id |
| native | phridge:job_id |




## LinkML Source

<details>
```yaml
name: job_id
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
identifier: true
owner: JobEnvelope
domain_of:
- JobEnvelope
range: string
required: true

```
</details></div>