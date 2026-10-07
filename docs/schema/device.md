---
search:
  boost: 5.0
---

# Slot: device 

<div data-search-exclude markdown="1">



URI: [phridge:device](https://github.com/phzwart/phridge/schema/phridge/device)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [PluginManifest](PluginManifest.md) |  |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [ManifestDevice](ManifestDevice.md) |
| Domain Of | [PluginManifest](PluginManifest.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
### Slot Characteristics

| Property | Value |
| --- | --- |
| If Absent | `string(any)` |
| Owner | [PluginManifest](PluginManifest.md) |












## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:device |
| native | phridge:device |




## LinkML Source

<details>
```yaml
name: device
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
ifabsent: string(any)
owner: PluginManifest
domain_of:
- PluginManifest
range: ManifestDevice

```
</details></div>