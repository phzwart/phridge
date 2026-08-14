"""cctbx converters for MTZ, maps, hierarchy, and xray.structure."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from phridge.client.convert import crystal_from_cctbx, crystal_to_cctbx, miller_from_cctbx, miller_to_cctbx
from phridge.models import (
    AnomalousLayout,
    Atom,
    CoordinateFrame,
    CrystalGridding,
    CrystalSymmetry,
    MapSpace,
    MtzColumnType,
    ObservationType,
    ReflectionColumn,
    Scatterer,
)
from phridge.packing import PackedMiller
from phridge.packing_xtal import (
    PackedCartesian,
    PackedComplexMap,
    PackedFractional,
    PackedHL,
    PackedHierarchy,
    PackedMapCoefficients,
    PackedReflections,
    PackedXray,
)

_MTZ_TYPE = {item.value: item for item in MtzColumnType}

_MTZ_OBS = {
    "F": ObservationType.fobs,
    "G": ObservationType.fobs,
    "K": ObservationType.fobs,
    "J": ObservationType.iobs,
    "L": ObservationType.iobs,
    "M": ObservationType.iobs,
    "A": ObservationType.hl,
    "B": ObservationType.hl,
    "C": ObservationType.hl,
}


def _label_of(array: Any) -> Optional[str]:
    info = array.info()
    if info is None or not info.labels:
        return None
    return str(info.labels[0])


def miller_or_hl_from_cctbx(array: Any, **kwargs: Any) -> PackedMiller | PackedHL:
    if array.is_hendrickson_lattman_array():
        return hl_from_cctbx(array, **kwargs)
    packed = miller_from_cctbx(array, **kwargs)
    return packed


def hl_from_cctbx(array: Any, *, anomalous_layout: Optional[AnomalousLayout] = None) -> PackedHL:
    anomalous = bool(array.anomalous_flag())
    if anomalous_layout is None:
        anomalous_layout = (
            AnomalousLayout.both_hemispheres if anomalous else AnomalousLayout.asu
        )
    rows = [tuple(float(x) for x in row) for row in array.data()]
    a, b, c, d = zip(*rows) if rows else ([], [], [], [])
    return PackedHL(
        crystal=crystal_from_cctbx(array.crystal_symmetry()),
        hkl=np.asarray(list(array.indices()), dtype=np.int32),
        a=np.asarray(a),
        b=np.asarray(b),
        c=np.asarray(c),
        d=np.asarray(d),
        anomalous=anomalous,
        anomalous_layout=anomalous_layout,
        label=_label_of(array),
    )


def hl_to_cctbx(packed: PackedHL) -> Any:
    from cctbx import miller
    from cctbx.array_family import flex as cctbx_flex

    crystal = crystal_to_cctbx(packed.meta.crystal)
    indices = cctbx_flex.miller_index([tuple(int(x) for x in row) for row in packed.hkl])
    miller_set = miller.set(crystal, indices, anomalous_flag=packed.meta.anomalous)
    hl = cctbx_flex.hendrickson_lattman(
        list(zip(packed.a.tolist(), packed.b.tolist(), packed.c.tolist(), packed.d.tolist()))
    )
    return miller.array(miller_set=miller_set, data=hl)


def reflections_from_mtz(obj: Any) -> PackedReflections:
    labels = list(obj.column_labels())
    types = list(obj.column_types())
    hkl = np.asarray(list(obj.extract_miller_indices()), dtype=np.int32)
    crystals = list(obj.crystals())
    crystal = crystal_from_cctbx(crystals[-1].crystal_symmetry())
    wavelength = None
    for cryst in crystals:
        for dataset in cryst.datasets():
            wave = float(dataset.wavelength())
            if wave > 0:
                wavelength = wave
    history = [str(line) for line in obj.history()]
    columns: list[ReflectionColumn] = []
    data: dict[str, np.ndarray] = {}
    sigmas: dict[str, np.ndarray] = {}
    last_data_name: Optional[str] = None
    anomalous_layout = AnomalousLayout.asu
    for label, raw in zip(labels, types):
        raw = str(raw)
        if raw == "H":
            continue
        values = np.asarray(list(obj.get_column(label).extract_values()), dtype=np.float64)
        if raw == "Q" and last_data_name is not None:
            sigmas[last_data_name] = values
            columns[-1].has_sigmas = True
            continue
        mtz_type = _MTZ_TYPE.get(raw, MtzColumnType.other)
        if raw in {"G", "K", "L", "M"}:
            anomalous_layout = AnomalousLayout.both_hemispheres
        col = ReflectionColumn(
            label=str(label),
            mtz_type=mtz_type,
            mtz_type_raw=None if mtz_type is not MtzColumnType.other else raw,
            observation_type=_MTZ_OBS.get(raw, ObservationType.other),
            anomalous=raw in {"G", "K", "L", "M", "D"},
            has_sigmas=False,
            npz_name=str(label),
            data_dtype="float64",
        )
        columns.append(col)
        data[col.npz_name] = values
        last_data_name = col.npz_name
    if not columns:
        raise ValueError("mtz object has no data columns")
    return PackedReflections(
        crystal=crystal,
        hkl=hkl,
        columns=columns,
        data=data,
        sigmas=sigmas,
        wavelength=wavelength,
        history=history,
        anomalous_layout=anomalous_layout,
    )


def reflections_to_mtz(packed: PackedReflections) -> Any:
    arrays = []
    for col in packed.meta.columns:
        miller = PackedMiller(
            crystal=packed.meta.crystal,
            hkl=packed.hkl,
            data=packed.data[col.npz_name],
            sigmas=packed.sigmas.get(col.npz_name),
            anomalous=col.anomalous,
            observation_type=col.observation_type,
            anomalous_layout=packed.meta.anomalous_layout,
            label=col.label,
        )
        array = miller_to_cctbx(miller)
        if col.observation_type is ObservationType.fobs:
            array = array.set_observation_type_xray_amplitude()
        elif col.observation_type is ObservationType.iobs:
            array = array.set_observation_type_xray_intensity()
        arrays.append((col.label, array))
    root, first = arrays[0]
    dataset = first.as_mtz_dataset(column_root_label=root)
    for label, array in arrays[1:]:
        dataset.add_miller_array(array, column_root_label=label)
    return dataset.mtz_object()


def map_coefficients_from_cctbx(array: Any) -> PackedMapCoefficients:
    if not array.is_complex_array():
        raise TypeError("map coefficients require a complex miller array")
    packed = miller_from_cctbx(array)
    packed.meta.label = _label_of(array)
    packed.meta.observation_type = ObservationType.complex
    return PackedMapCoefficients(packed, label=packed.meta.label)


def map_coefficients_to_cctbx(packed: PackedMapCoefficients) -> Any:
    return miller_to_cctbx(packed.miller)


def complex_map_from_cctbx(
    data: Any,
    crystal: Any,
    origin: Optional[list[int]] = None,
) -> PackedComplexMap:
    array = np.asarray(data if isinstance(data, np.ndarray) else list(data))
    if array.ndim != 3:
        raise ValueError(f"expected 3-D complex map, got shape {array.shape}")
    if hasattr(crystal, "unit_cell"):
        crystal = crystal_from_cctbx(crystal)
    return PackedComplexMap(crystal=crystal, data=array, origin=origin, space=MapSpace.p1)


def complex_map_to_numpy(packed: PackedComplexMap) -> np.ndarray:
    return packed.data


def gridding_from_cctbx(grid: Any) -> CrystalGridding:
    crystal = crystal_from_cctbx(grid.crystal_symmetry())
    n_real = [int(x) for x in grid.n_real()]
    d_min = float(grid.d_min()) if grid.d_min() is not None else None
    resolution_factor = float(grid.resolution_factor()) if grid.resolution_factor() is not None else None
    return CrystalGridding(
        crystal=crystal,
        n_real=n_real,
        d_min=d_min,
        resolution_factor=resolution_factor,
        space=MapSpace.p1,
    )


def sites_cart_from_cctbx(sites: Any, crystal: Optional[Any] = None) -> PackedCartesian:
    xyz = np.asarray(list(sites), dtype=np.float64)
    cs = crystal_from_cctbx(crystal) if crystal is not None and hasattr(crystal, "unit_cell") else crystal
    return PackedCartesian(xyz, crystal=cs)


def sites_frac_from_cctbx(sites: Any, crystal: Any) -> PackedFractional:
    xyz = np.asarray(list(sites), dtype=np.float64)
    cs = crystal_from_cctbx(crystal) if hasattr(crystal, "unit_cell") else crystal
    return PackedFractional(xyz, crystal=cs)


def sites_cart_to_cctbx(packed: PackedCartesian) -> Any:
    from scitbx.array_family import flex

    return flex.vec3_double([tuple(row) for row in packed.xyz])


def sites_frac_to_cctbx(packed: PackedFractional) -> Any:
    from scitbx.array_family import flex

    return flex.vec3_double([tuple(row) for row in packed.xyz])


def hierarchy_from_cctbx(root: Any, crystal: Optional[Any] = None) -> PackedHierarchy:
    atoms_flex = root.atoms()
    n = atoms_flex.size()
    xyz = np.zeros((n, 3), dtype=np.float64)
    occ = np.zeros(n, dtype=np.float64)
    b_iso = np.zeros(n, dtype=np.float64)
    u_cart = None
    uij_defined = None
    labels: list[Atom] = []
    for i, atom in enumerate(atoms_flex):
        xyz[i] = atom.xyz
        occ[i] = atom.occ
        b_iso[i] = atom.b
        if atom.uij_is_defined():
            if u_cart is None:
                u_cart = np.zeros((n, 6), dtype=np.float64)
                uij_defined = np.zeros(n, dtype=np.uint8)
            u_cart[i] = atom.uij
            uij_defined[i] = 1
        ag = atom.parent()
        rg = ag.parent()
        chain = rg.parent()
        model = chain.parent()
        icode = str(rg.icode).strip() or None
        altloc = str(ag.altloc).strip() or None
        labels.append(
            Atom(
                i=i,
                name=str(atom.name),
                element=str(atom.element).strip(),
                model_id=str(model.id) or None,
                chain_id=str(chain.id),
                resseq=int(rg.resseq_as_int()),
                icode=icode,
                resname=str(ag.resname).strip(),
                altloc=altloc,
                hetero=bool(atom.hetero),
            )
        )
    cs = None
    if crystal is not None:
        cs = crystal_from_cctbx(crystal) if hasattr(crystal, "unit_cell") else crystal
    return PackedHierarchy(
        xyz=xyz,
        occupancy=occ,
        b_iso=b_iso,
        atoms=labels,
        crystal=cs,
        frame=CoordinateFrame.cartesian,
        u_cart=u_cart,
        uij_defined=uij_defined,
    )


def hierarchy_to_cctbx(packed: PackedHierarchy) -> Any:
    from iotbx.pdb import hierarchy

    root = hierarchy.root()
    models: dict[str, Any] = {}
    chains: dict[tuple, Any] = {}
    groups: dict[tuple, Any] = {}
    atom_groups: dict[tuple, Any] = {}
    for atom_meta, xyz, occ, b_iso in zip(
        packed.meta.atoms, packed.xyz, packed.occupancy, packed.b_iso
    ):
        model_id = atom_meta.model_id or ""
        if model_id not in models:
            model = hierarchy.model(id=model_id)
            models[model_id] = model
            root.append_model(model)
        model = models[model_id]
        chain_key = (model_id, atom_meta.chain_id)
        if chain_key not in chains:
            chain = hierarchy.chain(id=atom_meta.chain_id)
            chains[chain_key] = chain
            model.append_chain(chain)
        chain = chains[chain_key]
        icode = atom_meta.icode or " "
        rg_key = (model_id, atom_meta.chain_id, atom_meta.resseq, icode)
        if rg_key not in groups:
            rg = hierarchy.residue_group(resseq="%4d" % atom_meta.resseq, icode=icode)
            groups[rg_key] = rg
            chain.append_residue_group(rg)
        rg = groups[rg_key]
        altloc = atom_meta.altloc or ""
        ag_key = rg_key + (altloc, atom_meta.resname)
        if ag_key not in atom_groups:
            ag = hierarchy.atom_group(altloc=altloc, resname=atom_meta.resname)
            atom_groups[ag_key] = ag
            rg.append_atom_group(ag)
        ag = atom_groups[ag_key]
        atom = hierarchy.atom()
        name = atom_meta.name if len(atom_meta.name) >= 4 else atom_meta.name.center(4)
        atom.set_name(name)
        atom.set_element(atom_meta.element)
        atom.set_xyz(tuple(float(x) for x in xyz))
        atom.set_occ(float(occ))
        atom.set_b(float(b_iso))
        atom.set_hetero(bool(atom_meta.hetero))
        if packed.uij_defined is not None and packed.u_cart is not None and packed.uij_defined[atom_meta.i]:
            atom.set_uij(tuple(float(x) for x in packed.u_cart[atom_meta.i]))
        ag.append_atom(atom)
    return root


def xray_from_cctbx(structure: Any) -> PackedXray:
    scatterers = structure.scatterers()
    n = scatterers.size()
    labels: list[Scatterer] = []
    u_star = np.zeros((n, 6), dtype=np.float64)
    u_iso = np.asarray(list(structure.extract_u_iso_or_u_equiv()), dtype=np.float64)
    for i, sc in enumerate(scatterers):
        aniso = bool(sc.flags.use_u_aniso())
        if aniso:
            u_star[i] = sc.u_star
        fp = float(sc.fp) if sc.fp not in (None, 0) else None
        fdp = float(sc.fdp) if sc.fdp not in (None, 0) else None
        labels.append(
            Scatterer(
                i=i,
                scattering_type=str(sc.scattering_type),
                fp=fp,
                fdp=fdp,
                anisotropic=aniso,
                use_u_iso=bool(sc.flags.use_u_iso()),
            )
        )
    return PackedXray(
        crystal=crystal_from_cctbx(structure.crystal_symmetry()),
        sites_frac=np.asarray(list(structure.sites_frac()), dtype=np.float64),
        occupancy=np.asarray(list(scatterers.extract_occupancies()), dtype=np.float64),
        u_iso=u_iso,
        scatterers=labels,
        u_star=u_star,
    )


def xray_to_cctbx(packed: PackedXray) -> Any:
    from cctbx import xray
    from cctbx.array_family import flex as cctbx_flex
    from cctbx.xray import scatterer

    items = []
    for meta, site, occ, u_iso, u_star in zip(
        packed.meta.scatterers,
        packed.sites_frac,
        packed.occupancy,
        packed.u_iso,
        packed.u_star,
    ):
        if meta.anisotropic:
            sc = scatterer(
                label=f"{meta.scattering_type}{meta.i}",
                scattering_type=meta.scattering_type,
                site=tuple(float(x) for x in site),
                occupancy=float(occ),
                u=tuple(float(x) for x in u_star),
            )
        else:
            sc = scatterer(
                label=f"{meta.scattering_type}{meta.i}",
                scattering_type=meta.scattering_type,
                site=tuple(float(x) for x in site),
                occupancy=float(occ),
                u=float(u_iso),
            )
        if meta.fp is not None:
            sc.fp = float(meta.fp)
        if meta.fdp is not None:
            sc.fdp = float(meta.fdp)
        items.append(sc)
    return xray.structure(
        crystal_symmetry=crystal_to_cctbx(packed.meta.crystal),
        scatterers=cctbx_flex.xray_scatterer(items),
    )


def em_map_from_map_manager(mm: Any, label: Optional[str] = None) -> Any:
    from phridge.packing_xtal import PackedEmMap

    data = mm.map_data()
    array = np.asarray(data.as_numpy_array()) if hasattr(data, "as_numpy_array") else np.asarray(list(data)).reshape(tuple(data.all()))
    origin = [int(x) for x in data.origin()] if hasattr(data, "origin") else [0, 0, 0]
    n_real = [int(x) for x in (mm.map_data().all() if hasattr(mm.map_data(), "all") else array.shape)]
    crystal = crystal_from_cctbx(mm.crystal_symmetry())
    pixels = None
    if hasattr(mm, "pixel_sizes"):
        pixels = [float(x) for x in mm.pixel_sizes()]
    origin_cart = None
    if hasattr(mm, "shift_cart"):
        origin_cart = [float(x) for x in mm.shift_cart()]
    exp = mm.experiment_type() if hasattr(mm, "experiment_type") else "cryo_em"
    if exp in (None, ""):
        exp = "other"
    wrapping = bool(mm.wrapping()) if hasattr(mm, "wrapping") else False
    resolution = None
    if hasattr(mm, "resolution") and mm.resolution() not in (None, 0, -1):
        try:
            resolution = float(mm.resolution())
        except Exception:
            resolution = None
    is_mask = bool(mm.is_mask()) if hasattr(mm, "is_mask") else False
    return PackedEmMap(
        crystal=crystal,
        data=array,
        origin=origin,
        n_real=list(array.shape),
        experiment_type=str(exp),
        wrapping=wrapping,
        is_mask=is_mask,
        pixel_sizes=pixels,
        origin_cart=origin_cart,
        resolution=resolution,
        label=label,
    )


def em_map_to_map_manager(packed: Any) -> Any:
    from iotbx.map_manager import map_manager
    from scitbx.array_family import flex

    grid = flex.grid(tuple(int(n) for n in packed.meta.n_real))
    data = flex.double(packed.data.reshape(-1).tolist())
    data.reshape(grid)
    mm = map_manager(
        map_data=data,
        unit_cell_grid=tuple(int(n) for n in packed.meta.n_real),
        unit_cell_crystal_symmetry=crystal_to_cctbx(packed.meta.crystal),
        wrapping=packed.meta.wrapping,
    )
    if packed.meta.experiment_type:
        try:
            mm.set_experiment_type(packed.meta.experiment_type.value)
        except Exception:
            pass
    if packed.meta.resolution is not None:
        try:
            mm.set_resolution(packed.meta.resolution)
        except Exception:
            pass
    return mm


def density_at_sites(packed_em: Any, sites_cart: np.ndarray) -> np.ndarray:
    """Sample an EM/real map at Cartesian coordinates (Å)."""
    from scitbx.array_family import flex

    mm = em_map_to_map_manager(packed_em)
    sites = flex.vec3_double([tuple(float(x) for x in row) for row in np.asarray(sites_cart)])
    return np.asarray(list(mm.density_at_sites_cart(sites)), dtype=np.float64)
