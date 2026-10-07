---
search:
  boost: 5.0
---

# Slot: tier 

<div data-search-exclude markdown="1">



URI: [phridge:tier](https://github.com/phzwart/phridge/schema/phridge/tier)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [PluginManifest](PluginManifest.md) |  |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [Tier](Tier.md) |
| Domain Of | [PluginManifest](PluginManifest.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
### Slot Characteristics

| Property | Value |
| --- | --- |
| If Absent | `string(native)` |
| Owner | [PluginManifest](PluginManifest.md) |












## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:tier |
| native | phridge:tier |




## LinkML Source

<details>
```yaml
name: tier
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
ifabsent: string(native)
owner: PluginManifest
domain_of:
- PluginManifest
range: Tier

```
</details></div>