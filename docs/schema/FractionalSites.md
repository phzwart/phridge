---
search:
  boost: 10.0
---

# Class: FractionalSites 


_cctbx xrs.sites_frac(). npz array xyz float64[N,3]._

__



<div data-search-exclude markdown="1">



URI: [phridge:FractionalSites](https://github.com/phzwart/phridge/schema/phridge/FractionalSites)





```mermaid
 classDiagram
    class FractionalSites
    click FractionalSites href "../FractionalSites/"
      CctbxObject <|-- FractionalSites
        click CctbxObject href "../CctbxObject/"
      
      FractionalSites : crystal
        
          
    
        
        
        FractionalSites --> "1" CrystalSymmetry : crystal
        click CrystalSymmetry href "../CrystalSymmetry/"
    

        
      FractionalSites : dtype
        
      FractionalSites : n_sites
        
      
```





## Inheritance
* [CctbxObject](CctbxObject.md)
    * **FractionalSites**


## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [crystal](crystal.md) | 1 <br/> [CrystalSymmetry](CrystalSymmetry.md) |  | direct |
| [n_sites](n_sites.md) | 1 <br/> [Integer](Integer.md) |  | direct |
| [dtype](dtype.md) | 1 <br/> [String](String.md) |  | direct |















## Identifier and Mapping Information





### Schema Source


* from schema: https://github.com/phzwart/phridge/schema/phridge




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | phridge:FractionalSites |
| native | phridge:FractionalSites |






## LinkML Source

<!-- TODO: investigate https://stackoverflow.com/questions/37606292/how-to-create-tabbed-code-blocks-in-mkdocs-or-sphinx -->

### Direct

<details>
```yaml
name: FractionalSites
description: 'cctbx xrs.sites_frac(). npz array xyz float64[N,3].

  '
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
is_a: CctbxObject
attributes:
  crystal:
    name: crystal
    from_schema: https://github.com/phzwart/phridge/schema/cctbx_coordinates
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
  n_sites:
    name: n_sites
    from_schema: https://github.com/phzwart/phridge/schema/cctbx_coordinates
    domain_of:
    - CartesianSites
    - FractionalSites
    - GeometryRestraints
    - ModelGeometry
    range: integer
    required: true
  dtype:
    name: dtype
    from_schema: https://github.com/phzwart/phridge/schema/cctbx_coordinates
    domain_of:
    - ArrayMeta
    - ObjectRef
    - RealMap
    - ComplexMap
    - EmMap
    - CartesianSites
    - FractionalSites
    required: true

```
</details>

### Induced

<details>
```yaml
name: FractionalSites
description: 'cctbx xrs.sites_frac(). npz array xyz float64[N,3].

  '
from_schema: https://github.com/phzwart/phridge/schema/phridge
rank: 1000
is_a: CctbxObject
attributes:
  crystal:
    name: crystal
    from_schema: https://github.com/phzwart/phridge/schema/cctbx_coordinates
    owner: FractionalSites
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
  n_sites:
    name: n_sites
    from_schema: https://github.com/phzwart/phridge/schema/cctbx_coordinates
    owner: FractionalSites
    domain_of:
    - CartesianSites
    - FractionalSites
    - GeometryRestraints
    - ModelGeometry
    range: integer
    required: true
  dtype:
    name: dtype
    from_schema: https://github.com/phzwart/phridge/schema/cctbx_coordinates
    owner: FractionalSites
    domain_of:
    - ArrayMeta
    - ObjectRef
    - RealMap
    - ComplexMap
    - EmMap
    - CartesianSites
    - FractionalSites
    range: string
    required: true

```
</details></div>