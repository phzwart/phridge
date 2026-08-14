---
search:
  boost: 10.0
---

# Class: CrystalGridding 


_Canonical maptbx.crystal_gridding. JSON only._

__



<div data-search-exclude markdown="1">



URI: [phridge:CrystalGridding](https://github.com/phzwart/phridge/schema/phridge/CrystalGridding)





```mermaid
 classDiagram
    class CrystalGridding
    click CrystalGridding href "../CrystalGridding/"
      CctbxObject <|-- CrystalGridding
        click CctbxObject href "../CctbxObject/"
      
      CrystalGridding : crystal
        
          
    
        
        
        CrystalGridding --> "1" CrystalSymmetry : crystal
        click CrystalSymmetry href "../CrystalSymmetry/"
    

        
      CrystalGridding : d_min
        
      CrystalGridding : n_real
        
      CrystalGridding : resolution_factor
        
      CrystalGridding : space
        
          
    
        
        
        CrystalGridding --> "1" MapSpace : space
        click MapSpace href "../MapSpace/"
    

        
      
```





## Inheritance
* [CctbxObject](CctbxObject.md)
    * **CrystalGridding**


## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [crystal](crystal.md) | 1 <br/> [CrystalSymmetry](CrystalSymmetry.md) |  | direct |
| [n_real](n_real.md) | 1..* <br/> [Integer](Integer.md) |  | direct |
| [resolution_factor](resolution_factor.md) | 0..1 <br/> [Float](Float.md) | Typical FFT factor (e | direct |
| [d_min](d_min.md) | 0..1 <br/> [Float](Float.md) | Resolution (Å) used to choose the grid, when known | direct |
| [space](space.md) | 1 <br/> [MapSpace](MapSpace.md) |  | direct |















## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:CrystalGridding |
| native | phridge:CrystalGridding |






## LinkML Source

<!-- TODO: investigate https://stackoverflow.com/questions/37606292/how-to-create-tabbed-code-blocks-in-mkdocs-or-sphinx -->

### Direct

<details>
```yaml
name: CrystalGridding
description: 'Canonical maptbx.crystal_gridding. JSON only.

  '
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
is_a: CctbxObject
attributes:
  crystal:
    name: crystal
    from_schema: https://github.com/phzwart/phridge/schema/cctbx_maps
    domain_of:
    - MillerArray
    - HendricksonLattman
    - ReflectionFile
    - CrystalGridding
    - RealMap
    - ComplexMap
    - EmMap
    - CartesianSites
    - FractionalSites
    - Hierarchy
    - XrayStructure
    - GeometryRestraints
    range: CrystalSymmetry
    required: true
    inlined: true
  n_real:
    name: n_real
    from_schema: https://github.com/phzwart/phridge/schema/cctbx_maps
    rank: 1000
    domain_of:
    - CrystalGridding
    - RealMap
    - ComplexMap
    - EmMap
    range: integer
    required: true
    multivalued: true
    minimum_cardinality: 3
    maximum_cardinality: 3
  resolution_factor:
    name: resolution_factor
    description: Typical FFT factor (e.g. 1/3) when known
    from_schema: https://github.com/phzwart/phridge/schema/cctbx_maps
    rank: 1000
    domain_of:
    - CrystalGridding
    range: float
  d_min:
    name: d_min
    description: Resolution (Å) used to choose the grid, when known
    from_schema: https://github.com/phzwart/phridge/schema/cctbx_maps
    rank: 1000
    domain_of:
    - CrystalGridding
    range: float
  space:
    name: space
    from_schema: https://github.com/phzwart/phridge/schema/cctbx_maps
    rank: 1000
    domain_of:
    - CrystalGridding
    - RealMap
    - ComplexMap
    - EmMap
    range: MapSpace
    required: true

```
</details>

### Induced

<details>
```yaml
name: CrystalGridding
description: 'Canonical maptbx.crystal_gridding. JSON only.

  '
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
is_a: CctbxObject
attributes:
  crystal:
    name: crystal
    from_schema: https://github.com/phzwart/phridge/schema/cctbx_maps
    owner: CrystalGridding
    domain_of:
    - MillerArray
    - HendricksonLattman
    - ReflectionFile
    - CrystalGridding
    - RealMap
    - ComplexMap
    - EmMap
    - CartesianSites
    - FractionalSites
    - Hierarchy
    - XrayStructure
    - GeometryRestraints
    range: CrystalSymmetry
    required: true
    inlined: true
  n_real:
    name: n_real
    from_schema: https://github.com/phzwart/phridge/schema/cctbx_maps
    rank: 1000
    owner: CrystalGridding
    domain_of:
    - CrystalGridding
    - RealMap
    - ComplexMap
    - EmMap
    range: integer
    required: true
    multivalued: true
    minimum_cardinality: 3
    maximum_cardinality: 3
  resolution_factor:
    name: resolution_factor
    description: Typical FFT factor (e.g. 1/3) when known
    from_schema: https://github.com/phzwart/phridge/schema/cctbx_maps
    rank: 1000
    owner: CrystalGridding
    domain_of:
    - CrystalGridding
    range: float
  d_min:
    name: d_min
    description: Resolution (Å) used to choose the grid, when known
    from_schema: https://github.com/phzwart/phridge/schema/cctbx_maps
    rank: 1000
    owner: CrystalGridding
    domain_of:
    - CrystalGridding
    range: float
  space:
    name: space
    from_schema: https://github.com/phzwart/phridge/schema/cctbx_maps
    rank: 1000
    owner: CrystalGridding
    domain_of:
    - CrystalGridding
    - RealMap
    - ComplexMap
    - EmMap
    range: MapSpace
    required: true

```
</details></div>