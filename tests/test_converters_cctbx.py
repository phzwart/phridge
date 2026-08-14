import numpy as np
import pytest

cctbx = pytest.importorskip("cctbx")

from phridge.client.convert import (  # noqa: E402
    crystal_from_cctbx,
    crystal_to_cctbx,
    map_from_cctbx,
    map_to_cctbx,
    miller_from_cctbx,
    miller_to_cctbx,
    to_canonical,
)
from phridge.client.convert_xtal import (  # noqa: E402
    gridding_from_cctbx,
    hierarchy_from_cctbx,
    hierarchy_to_cctbx,
    hl_from_cctbx,
    hl_to_cctbx,
    map_coefficients_from_cctbx,
    reflections_from_mtz,
    reflections_to_mtz,
    sites_cart_from_cctbx,
    sites_frac_from_cctbx,
    xray_from_cctbx,
    xray_to_cctbx,
)
from phridge.models import CrystalSymmetry  # noqa: E402


def test_crystal_symmetry_cctbx_roundtrip():
    from cctbx import crystal

    original = crystal.symmetry(unit_cell=(78.1, 78.1, 37.0, 90, 90, 90), space_group_symbol="P 43 21 2")
    canonical = crystal_from_cctbx(original)
    assert isinstance(canonical, CrystalSymmetry)
    assert canonical.space_group_number == 96
    back = crystal_to_cctbx(canonical)
    assert back.unit_cell().parameters()[0] == pytest.approx(78.1, rel=1e-6)
    assert back.space_group_info().type().number() == 96


def test_miller_array_cctbx_roundtrip():
    from cctbx import crystal, miller
    from cctbx.array_family import flex as cctbx_flex
    from scitbx.array_family import flex

    cs = crystal.symmetry(unit_cell=(10, 10, 10, 90, 90, 90), space_group_symbol="P 1")
    indices = cctbx_flex.miller_index([(1, 0, 0), (0, 2, 0), (1, 1, 1)])
    data = flex.double([10.0, 20.0, 30.0])
    sigmas = flex.double([0.5, 0.6, 0.7])
    array = miller.array(
        miller_set=miller.set(cs, indices, anomalous_flag=False),
        data=data,
    ).customized_copy(sigmas=sigmas)
    packed = miller_from_cctbx(array)
    np.testing.assert_allclose(packed.data, [10.0, 20.0, 30.0])
    back = miller_to_cctbx(packed)
    np.testing.assert_allclose(list(back.data()), [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(list(back.indices()), [(1, 0, 0), (0, 2, 0), (1, 1, 1)])


def test_real_map_cctbx_roundtrip():
    from cctbx import crystal

    cs = crystal.symmetry(unit_cell=(10, 10, 10, 90, 90, 90), space_group_symbol="P 1")
    data = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    packed = map_from_cctbx(data, cs, origin=[0, 0, 0])
    flex_map = map_to_cctbx(packed)
    np.testing.assert_allclose(list(flex_map), data.reshape(-1))


def test_mtz_reflection_file_roundtrip():
    from cctbx import crystal, miller
    from cctbx.array_family import flex as cctbx_flex
    from scitbx.array_family import flex

    cs = crystal.symmetry(unit_cell=(10, 10, 10, 90, 90, 90), space_group_symbol="P 1")
    indices = cctbx_flex.miller_index([(1, 0, 0), (0, 1, 0)])
    array = miller.array(
        miller_set=miller.set(cs, indices, anomalous_flag=False),
        data=flex.double([12.0, 8.0]),
    ).set_observation_type_xray_amplitude()
    array = array.customized_copy(sigmas=flex.double([0.4, 0.2]))
    mtz_obj = array.as_mtz_dataset(column_root_label="F").mtz_object()
    packed = reflections_from_mtz(mtz_obj)
    assert packed.meta.columns[0].label == "F"
    assert packed.meta.columns[0].has_sigmas
    np.testing.assert_allclose(packed.data["F"], [12.0, 8.0])
    back = reflections_to_mtz(packed)
    arrays = back.as_miller_arrays()
    np.testing.assert_allclose(list(arrays[0].data()), [12.0, 8.0])
    via = to_canonical(mtz_obj)
    assert via.meta.n_refl == 2


def test_hl_roundtrip():
    from cctbx import crystal, miller
    from cctbx.array_family import flex as cctbx_flex

    cs = crystal.symmetry(unit_cell=(10, 10, 10, 90, 90, 90), space_group_symbol="P 1")
    indices = cctbx_flex.miller_index([(1, 0, 0), (2, 0, 0)])
    hl = cctbx_flex.hendrickson_lattman([(1.0, 2.0, 3.0, 4.0), (0.1, 0.2, 0.3, 0.4)])
    array = miller.array(miller.set(cs, indices, anomalous_flag=False), data=hl)
    packed = hl_from_cctbx(array)
    np.testing.assert_allclose(packed.a, [1.0, 0.1])
    back = hl_to_cctbx(packed)
    assert back.is_hendrickson_lattman_array()
    np.testing.assert_allclose(tuple(back.data()[0]), (1.0, 2.0, 3.0, 4.0))


def test_map_coefficients_and_fft_map():
    from cctbx import crystal, miller
    from cctbx.array_family import flex as cctbx_flex
    from scitbx.array_family import flex

    cs = crystal.symmetry(unit_cell=(10, 10, 10, 90, 90, 90), space_group_symbol="P 1")
    indices = cctbx_flex.miller_index([(1, 0, 0), (0, 1, 0)])
    array = miller.array(
        miller_set=miller.set(cs, indices, anomalous_flag=False),
        data=flex.complex_double([1 + 0j, 0.5 + 0.25j]),
    )
    coef = map_coefficients_from_cctbx(array)
    assert coef.meta.miller.observation_type.value == "complex"
    back = miller_to_cctbx(coef.miller)
    np.testing.assert_allclose(np.asarray(list(back.data())), np.asarray(list(array.data())))
    fft = array.fft_map(resolution_factor=1 / 3)
    packed_map = to_canonical(fft)
    assert packed_map.data.ndim == 3


def test_hierarchy_and_xray_roundtrip():
    from cctbx import crystal
    from iotbx import pdb

    src = """\
CRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1
ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 20.00           C
ATOM      2  N   ALA A   1       0.000   1.000   2.000  1.00 15.00           N
END
"""
    root = pdb.input(source_info=None, lines=src.splitlines()).construct_hierarchy()
    packed = hierarchy_from_cctbx(root)
    assert packed.meta.n_atoms == 2
    back = hierarchy_to_cctbx(packed)
    assert back.atoms_size() == 2
    np.testing.assert_allclose(back.atoms()[0].xyz, packed.xyz[0], atol=1e-4)

    cs = crystal.symmetry(unit_cell=(10, 10, 10, 90, 90, 90), space_group_symbol="P 1")
    xrs = root.extract_xray_structure(crystal_symmetry=cs)
    px = xray_from_cctbx(xrs)
    assert px.meta.n_scatterers == 2
    assert not px.meta.scatterers[0].anisotropic
    xback = xray_to_cctbx(px)
    np.testing.assert_allclose(
        np.asarray(list(xback.sites_frac())), px.sites_frac, atol=1e-6
    )


def test_sites_and_gridding():
    from cctbx import crystal, maptbx

    cs = crystal.symmetry(unit_cell=(10, 10, 10, 90, 90, 90), space_group_symbol="P 1")
    xrs_sites = sites_frac_from_cctbx([(0.1, 0.2, 0.3)], cs)
    assert xrs_sites.meta.n_sites == 1
    cart = sites_cart_from_cctbx([(1.0, 2.0, 3.0)], cs)
    assert cart.xyz[0, 2] == 3.0
    grid = maptbx.crystal_gridding(
        unit_cell=cs.unit_cell(),
        space_group_info=cs.space_group_info(),
        d_min=2.0,
        resolution_factor=1 / 3,
    )
    meta = gridding_from_cctbx(grid)
    assert len(meta.n_real) == 3
def test_anisotropic_xray_and_hierarchy_uij():
    from cctbx import adptbx, crystal, xray
    from cctbx.array_family import flex as cctbx_flex
    from cctbx.xray import scatterer
    from iotbx.pdb import hierarchy

    cs = crystal.symmetry(unit_cell=(10, 10, 10, 90, 90, 90), space_group_symbol="P 1")
    u_cart = (0.05, 0.04, 0.06, 0.01, 0.0, 0.0)
    u_star = adptbx.u_cart_as_u_star(cs.unit_cell(), u_cart)
    sc = scatterer(label="C", site=(0.1, 0.2, 0.3), occupancy=1.0, scattering_type="C", u=u_star)
    xrs = xray.structure(crystal_symmetry=cs, scatterers=cctbx_flex.xray_scatterer([sc]))
    packed = xray_from_cctbx(xrs)
    assert packed.meta.scatterers[0].anisotropic
    np.testing.assert_allclose(packed.u_star[0], u_star, atol=1e-8)
    back = xray_to_cctbx(packed)
    assert back.scatterers()[0].flags.use_u_aniso()
    np.testing.assert_allclose(back.scatterers()[0].u_star, u_star, atol=1e-8)

    root = hierarchy.root()
    model = hierarchy.model(id="")
    chain = hierarchy.chain(id="A")
    rg = hierarchy.residue_group(resseq="   1", icode=" ")
    ag = hierarchy.atom_group(altloc="", resname="ALA")
    atom = hierarchy.atom()
    atom.set_name(" CA ")
    atom.set_element("C")
    atom.set_xyz((1.0, 2.0, 3.0))
    atom.set_occ(1.0)
    atom.set_b(20.0)
    atom.set_uij(u_cart)
    ag.append_atom(atom)
    rg.append_atom_group(ag)
    chain.append_residue_group(rg)
    model.append_chain(chain)
    root.append_model(model)
    ph = hierarchy_from_cctbx(root)
    assert ph.meta.has_uij
    np.testing.assert_allclose(ph.u_cart[0], u_cart, atol=1e-6)
    hb = hierarchy_to_cctbx(ph)
    assert hb.atoms()[0].uij_is_defined()


def test_geometry_restraints_i_seq_tables():
    from phridge.client.convert_geometry import restraints_to_proxies
    from phridge.packing_geometry import PackedRestraints, unpack_restraints

    packed = PackedRestraints(
        n_sites=3,
        bond_i_seqs=np.array([[0, 1]], dtype=np.int32),
        bond_distance_ideal=np.array([1.5]),
        bond_weight=np.array([1.0]),
        bond_slack=np.array([0.0]),
        bond_origin_id=np.array([0], dtype=np.int32),
        angle_i_seqs=np.array([[0, 1, 2]], dtype=np.int32),
        angle_ideal=np.array([120.0]),
        angle_weight=np.array([1.0]),
        angle_origin_id=np.array([0], dtype=np.int32),
    )
    assert packed.meta.n_bonds == 1
    assert packed.meta.n_angles == 1
    proxies = restraints_to_proxies(packed)
    assert proxies["bonds"][0].i_seqs == (0, 1)
    assert proxies["angles"][0].i_seqs == (0, 1, 2)
    restored = unpack_restraints(packed.pack(), packed.meta)
    np.testing.assert_array_equal(restored.bond_i_seqs, [[0, 1]])


def test_em_map_and_density_at_sites():
    from cctbx import crystal, maptbx
    from iotbx.map_manager import map_manager
    from scitbx.array_family import flex

    from phridge.client.em import density_at_sites, em_map_from_map_manager

    cs = crystal.symmetry(unit_cell=(10, 10, 10, 90, 90, 90), space_group_symbol="P 1")
    n = (8, 8, 8)
    data = flex.double(flex.grid(n), 0)
    data[4, 4, 4] = 5.0
    mm = map_manager(map_data=data, unit_cell_grid=n, unit_cell_crystal_symmetry=cs, wrapping=False)
    packed = em_map_from_map_manager(mm, label="half1")
    assert packed.meta.experiment_type.value in {"cryo_em", "xray", "other"}
    assert packed.data.shape == (8, 8, 8)
    samples = density_at_sites(packed, np.array([[5.0, 5.0, 5.0]]))
    assert samples.shape == (1,)

    from phridge.client.em import density_at_hierarchy, em_map_as_real_map, em_map_from_numpy
    from phridge.client.geometry import model_geometry
    from phridge.packing_geometry import PackedRestraints
    from phridge.packing_xtal import PackedHierarchy
    from phridge.models import Atom, CoordinateFrame

    packed2 = em_map_from_numpy(packed.data, packed.meta.crystal, wrapping=False, label="half1")
    real = em_map_as_real_map(packed2)
    assert real.data.shape == packed.data.shape
    atoms = [
        Atom(i=0, name=" CA ", element="C", chain_id="A", resseq=1, resname="ALA"),
        Atom(i=1, name=" C  ", element="C", chain_id="A", resseq=1, resname="ALA"),
    ]
    hier = PackedHierarchy(
        xyz=np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]]),
        occupancy=np.array([1.0, 1.0]),
        b_iso=np.array([20.0, 20.0]),
        atoms=atoms,
        frame=CoordinateFrame.cartesian,
    )
    dens = density_at_hierarchy(packed, hier)
    assert dens.shape == (2,)
    header = model_geometry(hierarchy=hier, restraints=PackedRestraints(n_sites=2))
    assert header.n_sites == 2 and header.i_seq_identity
    try:
        model_geometry(hierarchy=hier, restraints=PackedRestraints(n_sites=3))
        raise AssertionError("expected n_sites mismatch")
    except ValueError:
        pass


