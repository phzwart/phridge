---
search:
  boost: 5.0
---

# Slot: symops 


_Optional full list of symmetry operators (centering included), x' = r @ x + t in fractional coordinates. Exported by the client from sgtbx so a worker without cctbx can expand to P1._

__



<div data-search-exclude markdown="1">



URI: [phridge:symops](https://github.com/phzwart/phridge/schema/phridge/symops)
<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [CrystalSymmetry](CrystalSymmetry.md) | Canonical form of cctbx |  no  |






## Properties

### Type and Range

| Property | Value |
| --- | --- |
| Range | [SymOp](SymOp.md) |
| Domain Of | [CrystalSymmetry](CrystalSymmetry.md) |

### Cardinality and Requirements

| Property | Value |
| --- | --- |
| Multivalued | Yes |
### Slot Characteristics

| Property | Value |
| --- | --- |
| Owner | [CrystalSymmetry](CrystalSymmetry.md) |












## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:symops |
| native | phridge:symops |




## LinkML Source

<details>
```yaml
name: symops
description: 'Optional full list of symmetry operators (centering included), x'' =
  r @ x + t in fractional coordinates. Exported by the client from sgtbx so a worker
  without cctbx can expand to P1.

  '
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
owner: CrystalSymmetry
domain_of:
- CrystalSymmetry
range: SymOp
multivalued: true
inlined: true
inlined_as_list: true

```
</details></div>