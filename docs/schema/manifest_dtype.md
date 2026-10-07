---
search:
  boost: 5.0
---

# Slot: manifest_dtype 

<div data-search-exclude markdown="1">



URI: [phridge:manifest_dtype](https://github.com/phzwart/phridge/schema/phridge/manifest_dtype)
Alias: dtype

<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [PluginManifest](PluginManifest.md) |  |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [ManifestDtype](ManifestDtype.md) |
| Domain Of | [PluginManifest](PluginManifest.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
### Slot Characteristics

| Property | Value |
| --- | --- |
| If Absent | `string(float64)` |
| Owner | [PluginManifest](PluginManifest.md) |












## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:manifest_dtype |
| native | phridge:manifest_dtype |




## LinkML Source

<details>
```yaml
name: manifest_dtype
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
ifabsent: string(float64)
alias: dtype
owner: PluginManifest
domain_of:
- PluginManifest
range: ManifestDtype

```
</details></div>