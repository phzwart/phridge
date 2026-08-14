# phridge job and Redis protocol

Job envelopes and object references for the Redis object store.
Science types live in schema/cctbx*.yaml. Redis points at them with
kind=cctbx and cctbx_type set to the class name.


URI: https://github.com/phzwart/phridge/schema/phridge

Name: phridge



## Classes

| Class | Description |
| --- | --- |
| [ArrayMeta](ArrayMeta.md) |  |
| [Atom](Atom.md) | One iotbx |
| [BlobMeta](BlobMeta.md) |  |
| [CartesianSites](CartesianSites.md) | cctbx xrs |
| [CctbxObject](CctbxObject.md) | Base for canonical phenix/cctbx types |
| [ComplexMap](ComplexMap.md) | Complex grid (e |
| [CrystalGridding](CrystalGridding.md) | Canonical maptbx |
| [CrystalSymmetry](CrystalSymmetry.md) | Canonical form of cctbx |
| [EmMap](EmMap.md) | Canonical iotbx |
| [FractionalSites](FractionalSites.md) | cctbx xrs |
| [GeometryRestraints](GeometryRestraints.md) | Canonical cctbx |
| [HendricksonLattman](HendricksonLattman.md) | Canonical HL coefficients (cctbx miller array of hendrickson_lattman) |
| [Hierarchy](Hierarchy.md) | Canonical iotbx |
| [JobEnvelope](JobEnvelope.md) |  |
| [JobError](JobError.md) |  |
| [MapCoefficients](MapCoefficients.md) | Complex miller array used as FFT map coefficients (cctbx miller array of Fcal... |
| [MillerArray](MillerArray.md) | Canonical form of cctbx |
| [ModelGeometry](ModelGeometry.md) | Correspondence header asserting that hierarchy, xray scatterers, and restrain... |
| [ObjectRef](ObjectRef.md) | Pointer at bytes in Redis plus enough metadata to decode them |
| [OpSpec](OpSpec.md) |  |
| [RealMap](RealMap.md) | Canonical real-space map (maptbx / iotbx |
| [ReflectionColumn](ReflectionColumn.md) | One data column in a ReflectionFile (not H/K/L) |
| [ReflectionFile](ReflectionFile.md) | Canonical iotbx |
| [Scatterer](Scatterer.md) | One cctbx |
| [SlotBinding](SlotBinding.md) |  |
| [XrayStructure](XrayStructure.md) | Canonical cctbx |



## Slots

| Slot | Description |
| --- | --- |
| [altloc](altloc.md) | Alternate conformer; empty if none |
| [anisotropic](anisotropic.md) | If true, npz u_star[i] is live (cctbx u_star, fractional) |
| [anomalous](anomalous.md) |  |
| [anomalous_layout](anomalous_layout.md) | ASU vs both hemispheres; never inferred |
| [atoms](atoms.md) |  |
| [bond_asu_rt_mx](bond_asu_rt_mx.md) | sgtbx |
| [byte_order](byte_order.md) |  |
| [cctbx_type](cctbx_type.md) | Cctbx class name when kind=cctbx (MillerArray, RealMap, CrystalSymmetry) |
| [chain_id](chain_id.md) |  |
| [columns](columns.md) |  |
| [compression](compression.md) |  |
| [content_type](content_type.md) |  |
| [created_at](created_at.md) |  |
| [crystal](crystal.md) |  |
| [d_min](d_min.md) | Resolution (Å) used to choose the grid, when known |
| [data_dtype](data_dtype.md) | Canonical store dtype for data; v1 is float64 or complex128 |
| [dtype](dtype.md) | numpy dtype name (float64, int32, complex128, …) |
| [element](element.md) |  |
| [error](error.md) |  |
| [experiment_type](experiment_type.md) |  |
| [fdp](fdp.md) | f'' when set |
| [fp](fp.md) | f' when set |
| [frame](frame.md) |  |
| [has_hierarchy](has_hierarchy.md) |  |
| [has_restraints](has_restraints.md) |  |
| [has_sigmas](has_sigmas.md) |  |
| [has_uij](has_uij.md) | True if u_cart is present in the npz |
| [has_xray](has_xray.md) |  |
| [hetero](hetero.md) |  |
| [history](history.md) | MTZ history lines |
| [i](i.md) | Row index into packed xyz; same i_seq as scatterers and restraints |
| [i_seq_identity](i_seq_identity.md) | True when Hierarchy |
| [icode](icode.md) | Insertion code; empty if none |
| [index_dtype](index_dtype.md) | Canonical store dtype for hkl; v1 is int32 |
| [inputs](inputs.md) | Map of input name to ObjectRef |
| [is_mask](is_mask.md) |  |
| [job_id](job_id.md) |  |
| [key](key.md) | Redis key for bytes; omitted for JSON-only CrystalSymmetry |
| [kind](kind.md) |  |
| [label](label.md) | MTZ / miller array id (e |
| [message](message.md) |  |
| [meta](meta.md) | Kind-specific JSON (ArrayMeta, BlobMeta, or a CctbxObject) |
| [miller](miller.md) | Must have observation_type complex |
| [model_id](model_id.md) | Model id string (usually "1") |
| [mtz_type](mtz_type.md) |  |
| [mtz_type_raw](mtz_type_raw.md) | Original iotbx |
| [n_angles](n_angles.md) |  |
| [n_atoms](n_atoms.md) |  |
| [n_bond_asu](n_bond_asu.md) |  |
| [n_bond_similarities](n_bond_similarities.md) |  |
| [n_bonds](n_bonds.md) |  |
| [n_chiralities](n_chiralities.md) |  |
| [n_dihedrals](n_dihedrals.md) |  |
| [n_nonbonded](n_nonbonded.md) |  |
| [n_parallelities](n_parallelities.md) |  |
| [n_planarities](n_planarities.md) |  |
| [n_real](n_real.md) |  |
| [n_reference_coords](n_reference_coords.md) |  |
| [n_refl](n_refl.md) |  |
| [n_scatterers](n_scatterers.md) |  |
| [n_sites](n_sites.md) |  |
| [name](name.md) |  |
| [npz_name](npz_name.md) | Array name inside the packed npz (usually the label) |
| [observation_type](observation_type.md) |  |
| [op](op.md) |  |
| [order](order.md) | Memory order; v1 is C |
| [origin](origin.md) | Grid origin (grid units) |
| [origin_cart](origin_cart.md) | Cartesian origin shift (Å), map_manager |
| [outputs](outputs.md) | Map of output name to ObjectRef (filled by worker) |
| [pixel_sizes](pixel_sizes.md) | Voxel size (Å) along a, b, c |
| [resname](resname.md) |  |
| [resolution](resolution.md) | Nominal high resolution (Å) when known |
| [resolution_factor](resolution_factor.md) | Typical FFT factor (e |
| [resseq](resseq.md) |  |
| [scatterers](scatterers.md) |  |
| [scattering_type](scattering_type.md) | cctbx scattering type (C, N, S, AU, water, …) |
| [schema_version](schema_version.md) | Must match this schema file version (1) |
| [shape](shape.md) |  |
| [space](space.md) |  |
| [space_group_hall](space_group_hall.md) | Hall symbol as understood by cctbx |
| [space_group_number](space_group_number.md) | International Tables space-group number when known |
| [status](status.md) |  |
| [traceback](traceback.md) |  |
| [type](type.md) |  |
| [unit_cell](unit_cell.md) | a, b, c (Å), alpha, beta, gamma (degrees) |
| [updated_at](updated_at.md) |  |
| [use_u_iso](use_u_iso.md) | cctbx flags |
| [wavelength](wavelength.md) | Dataset wavelength (Å) when known |
| [wrapping](wrapping.md) |  |


## Enumerations

| Enumeration | Description |
| --- | --- |
| [AnomalousLayout](AnomalousLayout.md) |  |
| [ByteOrder](ByteOrder.md) |  |
| [Compression](Compression.md) |  |
| [CoordinateFrame](CoordinateFrame.md) |  |
| [ExperimentType](ExperimentType.md) |  |
| [JobStatus](JobStatus.md) |  |
| [MapSpace](MapSpace.md) |  |
| [MtzColumnType](MtzColumnType.md) | CCP4 MTZ column type character (iotbx |
| [ObjectKind](ObjectKind.md) |  |
| [ObservationType](ObservationType.md) |  |


## Types

| Type | Description |
| --- | --- |
| [Boolean](Boolean.md) | A binary (true or false) value |
| [Curie](Curie.md) | a compact URI |
| [Date](Date.md) | a date (year, month and day) in an idealized calendar |
| [DateOrDatetime](DateOrDatetime.md) | Either a date or a datetime |
| [Datetime](Datetime.md) | The combination of a date and time |
| [Decimal](Decimal.md) | A real number with arbitrary precision that conforms to the xsd:decimal speci... |
| [Double](Double.md) | A real number that conforms to the xsd:double specification |
| [Float](Float.md) | A real number that conforms to the xsd:float specification |
| [Integer](Integer.md) | An integer |
| [Jsonpath](Jsonpath.md) | A string encoding a JSON Path |
| [Jsonpointer](Jsonpointer.md) | A string encoding a JSON Pointer |
| [Ncname](Ncname.md) | Prefix part of CURIE |
| [Nodeidentifier](Nodeidentifier.md) | A URI, CURIE or BNODE that represents a node in a model |
| [Objectidentifier](Objectidentifier.md) | A URI or CURIE that represents an object in the model |
| [Sparqlpath](Sparqlpath.md) | A string encoding a SPARQL Property Path |
| [String](String.md) | A character string |
| [Time](Time.md) | A time object represents a (local) time of day, independent of any particular... |
| [Uri](Uri.md) | a complete URI |
| [Uriorcurie](Uriorcurie.md) | a URI or a CURIE |


## Subsets

| Subset | Description |
| --- | --- |
