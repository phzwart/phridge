---
search:
  boost: 5.0
---

# Slot: created_at 

<div data-search-exclude markdown="1">



URI: [phridge:created_at](https://github.com/phzwart/phridge/schema/phridge/created_at)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [JobEnvelope](JobEnvelope.md) |  |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Datetime](Datetime.md) |
| Domain Of | [JobEnvelope](JobEnvelope.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
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
| self | phridge:created_at |
| native | phridge:created_at |




## LinkML Source

<details>
```yaml
name: created_at
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
owner: JobEnvelope
domain_of:
- JobEnvelope
range: datetime

```
</details></div>