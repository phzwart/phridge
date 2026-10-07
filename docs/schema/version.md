---
search:
  boost: 5.0
---

# Slot: version 

<div data-search-exclude markdown="1">



URI: [phridge:version](https://github.com/phzwart/phridge/schema/phridge/version)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [PluginManifest](PluginManifest.md) |  |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [String](String.md) |
| Domain Of | [PluginManifest](PluginManifest.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
### Slot Characteristics

| Property | Value |
| --- | --- |
| If Absent | `string(0)` |
| Owner | [PluginManifest](PluginManifest.md) |












## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:version |
| native | phridge:version |




## LinkML Source

<details>
```yaml
name: version
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
ifabsent: string(0)
owner: PluginManifest
domain_of:
- PluginManifest
range: string

```
</details></div>