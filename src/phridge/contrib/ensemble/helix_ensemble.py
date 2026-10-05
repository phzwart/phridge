"""Poly-alanine α-helix, three soft normal modes, refined along each axis.

The helix is the ideal α geometry from ``mmtbx`` secondary-structure
idealization, placed on the campaign cell with its axis along c.

Modes are the three softest internal modes of an elastic network. Along each
mode, ``n_steps`` amplitudes run from ``-amplitude`` to ``+amplitude`` in
ångström RMSD. Every frame is refined in the subspace orthogonal to that mode
and to rigid motion, so the requested amplitude stays put while local geometry
relaxes:

- 1–2 distances (bonds) toward the ideal helix
- 1–4 distances toward the ideal helix
- helical secondary structure: O(i)…N(i+4) and CA(i)…CA(i+4)

A one-sided steric floor keeps non-local pairs from occupying the same point.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

CELL = (30.0, 30.0, 40.0, 90.0, 90.0, 90.0)
N_RESIDUES = 16
N_MODES = 3
N_STEPS = 25
AMPLITUDE = 1.5  # Å RMSD at the end of each axis
BOND_SIGMA = 0.02
PAIR14_SIGMA = 0.10
HBOND_SIGMA = 0.20
CA4_SIGMA = 0.25
CLASH_SIGMA = 0.15
CLASH_FLOOR = 2.4


@dataclass
class Helix:
    """Ideal poly-Ala coordinates and the distance restraints taken from them."""

    name: np.ndarray
    resseq: np.ndarray
    element: np.ndarray
    xyz: np.ndarray
    bonds: np.ndarray
    pairs14: np.ndarray
    hbonds: np.ndarray
    ca4: np.ndarray
    clashes: np.ndarray
    bond_ideal: np.ndarray
    pair14_ideal: np.ndarray
    hbond_ideal: np.ndarray
    ca4_ideal: np.ndarray


@dataclass
class ModeScan:
    """Refined frames along one normal mode."""

    vector: np.ndarray
    amplitudes: np.ndarray
    frames: np.ndarray
    bond_rms: np.ndarray
    pair14_rms: np.ndarray
    hbond_rms: np.ndarray
    ca_rmsd: np.ndarray


def _rotation_to_z(source: np.ndarray) -> np.ndarray:
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(source, z)
    c = float(np.dot(source, z))
    s = np.linalg.norm(v)
    if s < 1e-10:
        return np.eye(3) if c > 0.0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def ideal_helix(
    n_res: int = N_RESIDUES,
    cell: Sequence[float] = CELL,
) -> Helix:
    """Build an ideal poly-Ala α-helix and center it on ``cell`` along c."""
    import iotbx.pdb
    from mmtbx.secondary_structure.build import ss_idealization as ssb

    built = ssb.secondary_structure_from_sequence(ssb.alpha_helix_str, "A" * int(n_res))
    raw = iotbx.pdb.input(source_info=None, lines=built.as_pdb_string().splitlines())
    hierarchy = raw.construct_hierarchy()
    atoms = list(hierarchy.atoms())
    xyz = np.array(list(hierarchy.atoms().extract_xyz()), dtype=np.float64)
    name = np.array([atom.name for atom in atoms])
    resseq = np.array([atom.parent().parent().resseq_as_int() for atom in atoms], dtype=np.int32)
    element = np.array([atom.element.strip() or atom.name.strip()[:1] for atom in atoms])

    ca = np.array([atom.name.strip() == "CA" for atom in atoms])
    centered = xyz[ca] - xyz[ca].mean(axis=0)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    if axis[2] < 0.0:
        axis = -axis
    origin = np.array([float(cell[0]) / 2.0, float(cell[1]) / 2.0, float(cell[2]) / 2.0])
    placed = (xyz - xyz.mean(axis=0)) @ _rotation_to_z(axis).T + origin
    return _helix_from_sites(name, resseq, element, placed)


def _helix_from_sites(
    name: np.ndarray,
    resseq: np.ndarray,
    element: np.ndarray,
    xyz: np.ndarray,
) -> Helix:
    bonds = _alanine_bonds(name, resseq)
    graph = _bond_distances(bonds, xyz.shape[0])
    upper = np.triu_indices(xyz.shape[0], k=1)
    kind = graph[upper]
    pairs14 = np.stack(upper, axis=1)[kind == 3]
    far = np.stack(upper, axis=1)[kind >= 5]
    ref = np.linalg.norm(xyz[far[:, 0]] - xyz[far[:, 1]], axis=1)
    clashes = far[ref > CLASH_FLOOR]
    hbonds = _named_pairs(name, resseq, "O", "N", 4)
    ca4 = _named_pairs(name, resseq, "CA", "CA", 4)
    return Helix(
        name=name,
        resseq=resseq,
        element=element,
        xyz=xyz.astype(np.float64),
        bonds=bonds,
        pairs14=pairs14.astype(np.int32),
        hbonds=hbonds,
        ca4=ca4,
        clashes=clashes.astype(np.int32),
        bond_ideal=_lengths(xyz, bonds),
        pair14_ideal=_lengths(xyz, pairs14),
        hbond_ideal=_lengths(xyz, hbonds),
        ca4_ideal=_lengths(xyz, ca4),
    )


def _alanine_bonds(name: np.ndarray, resseq: np.ndarray) -> np.ndarray:
    index = {(int(r), str(n).strip()): i for i, (r, n) in enumerate(zip(resseq, name))}
    intra = (("N", "CA"), ("CA", "C"), ("C", "O"), ("CA", "CB"))
    pairs: list[tuple[int, int]] = []
    residues = sorted(set(int(r) for r in resseq))
    for res in residues:
        for a, b in intra:
            ia = index.get((res, a))
            ib = index.get((res, b))
            if ia is not None and ib is not None:
                pairs.append((ia, ib))
        nxt = index.get((res + 1, "N"))
        carbon = index.get((res, "C"))
        if nxt is not None and carbon is not None:
            pairs.append((carbon, nxt))
    return np.asarray(pairs, dtype=np.int32)


def _named_pairs(
    name: np.ndarray,
    resseq: np.ndarray,
    left: str,
    right: str,
    separation: int,
) -> np.ndarray:
    index = {(int(r), str(n).strip()): i for i, (r, n) in enumerate(zip(resseq, name))}
    residues = sorted(set(int(r) for r in resseq))
    pairs = []
    for res in residues:
        ia = index.get((res, left))
        ib = index.get((res + separation, right))
        if ia is not None and ib is not None:
            pairs.append((ia, ib))
    return np.asarray(pairs, dtype=np.int32)


def _bond_distances(bonds: np.ndarray, n: int) -> np.ndarray:
    adj: list[list[int]] = [[] for _ in range(n)]
    for i, j in bonds:
        adj[int(i)].append(int(j))
        adj[int(j)].append(int(i))
    dist = np.full((n, n), 999, dtype=np.int16)
    for start in range(n):
        dist[start, start] = 0
        queue = [start]
        for atom in queue:
            for nxt in adj[atom]:
                step = int(dist[start, atom]) + 1
                if dist[start, nxt] > step:
                    dist[start, nxt] = step
                    queue.append(nxt)
    return dist


def _lengths(xyz: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    if len(pairs) == 0:
        return np.zeros(0, dtype=np.float64)
    return np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1)


def _pair_energy(
    xyz: np.ndarray,
    pairs: np.ndarray,
    ideal: np.ndarray,
    sigma: float,
    *,
    floor: bool = False,
) -> tuple[float, np.ndarray]:
    grad = np.zeros_like(xyz)
    if len(pairs) == 0:
        return 0.0, grad
    delta = xyz[pairs[:, 0]] - xyz[pairs[:, 1]]
    dist = np.linalg.norm(delta, axis=1)
    dist = np.maximum(dist, 1e-8)
    if floor:
        excess = np.maximum(ideal - dist, 0.0)
        residual = excess
    else:
        residual = dist - ideal
    weight = 1.0 / float(sigma) ** 2
    energy = float(np.sum(weight * residual**2))
    coef = (2.0 * weight * residual / dist)[:, None]
    if floor:
        coef = np.where((ideal - dist)[:, None] > 0.0, -coef, 0.0)
    force = coef * delta
    np.add.at(grad, pairs[:, 0], force)
    np.add.at(grad, pairs[:, 1], -force)
    return energy, grad


def geometry_energy(helix: Helix, xyz: np.ndarray) -> tuple[float, np.ndarray]:
    """1–2, 1–4, helical hydrogen bonds, CA(i)…CA(i+4), and a steric floor."""
    total = 0.0
    grad = np.zeros_like(xyz)
    for pairs, ideal, sigma, floor in (
        (helix.bonds, helix.bond_ideal, BOND_SIGMA, False),
        (helix.pairs14, helix.pair14_ideal, PAIR14_SIGMA, False),
        (helix.hbonds, helix.hbond_ideal, HBOND_SIGMA, False),
        (helix.ca4, helix.ca4_ideal, CA4_SIGMA, False),
        (helix.clashes, np.full(len(helix.clashes), CLASH_FLOOR), CLASH_SIGMA, True),
    ):
        energy, g = _pair_energy(xyz, pairs, ideal, sigma, floor=floor)
        total += energy
        grad += g
    return total, grad


def elastic_modes(xyz: np.ndarray, n_modes: int = N_MODES, cutoff: float = 12.0) -> np.ndarray:
    """Softest internal modes of a uniform elastic network. Each has RMSD 1 Å."""
    n = xyz.shape[0]
    hessian = np.zeros((3 * n, 3 * n), dtype=np.float64)
    for i in range(n):
        delta = xyz[i + 1 :] - xyz[i]
        dist = np.linalg.norm(delta, axis=1)
        keep = (dist < cutoff) & (dist > 1e-8)
        for offset, vec, length in zip(np.flatnonzero(keep), delta[keep], dist[keep]):
            j = i + 1 + int(offset)
            outer = np.outer(vec, vec) / length**2
            sl_i = slice(3 * i, 3 * i + 3)
            sl_j = slice(3 * j, 3 * j + 3)
            hessian[sl_i, sl_j] -= outer
            hessian[sl_j, sl_i] -= outer
            hessian[sl_i, sl_i] += outer
            hessian[sl_j, sl_j] += outer
    eigval, eigvec = np.linalg.eigh(hessian)
    order = np.argsort(eigval)
    internal = [eigvec[:, k].reshape(n, 3) for k in order if eigval[k] > 1e-6]
    if len(internal) < n_modes:
        raise RuntimeError(f"elastic network produced {len(internal)} internal modes, need {n_modes}")
    modes = []
    for vec in internal[:n_modes]:
        rmsd = float(np.sqrt(np.mean(np.sum(vec**2, axis=1))))
        modes.append(vec / rmsd)
    return np.stack(modes)


def _constraint_nullspace(xyz: np.ndarray, mode: np.ndarray) -> np.ndarray:
    """Columns span motions orthogonal to ``mode`` and to rigid motion."""
    n = xyz.shape[0]
    center = xyz.mean(axis=0)
    rel = xyz - center
    columns = [mode.reshape(-1)]
    for axis in range(3):
        shift = np.zeros((n, 3), dtype=np.float64)
        shift[:, axis] = 1.0
        columns.append(shift.reshape(-1))
    rotations = (
        np.stack([np.zeros(n), -rel[:, 2], rel[:, 1]], axis=1),
        np.stack([rel[:, 2], np.zeros(n), -rel[:, 0]], axis=1),
        np.stack([-rel[:, 1], rel[:, 0], np.zeros(n)], axis=1),
    )
    columns.extend(rot.reshape(-1) for rot in rotations)
    basis = np.column_stack(columns)
    q, _r = np.linalg.qr(basis, mode="complete")
    return q[:, basis.shape[1] :]


def refine_along_mode(
    helix: Helix,
    mode: np.ndarray,
    amplitude: float,
    *,
    maxiter: int = 60,
) -> np.ndarray:
    """Relax local geometry at a fixed projection onto ``mode``.

    ``mode`` is an RMSD-1 displacement. The returned coordinates differ from
    the ideal helix by ``amplitude`` ångström along that mode, plus whatever
    orthogonal shift the restraints ask for.
    """
    from scipy.optimize import minimize

    null = _constraint_nullspace(helix.xyz, mode)
    origin = helix.xyz + float(amplitude) * mode

    def objective(coords: np.ndarray) -> tuple[float, np.ndarray]:
        xyz = origin + (null @ coords).reshape(helix.xyz.shape)
        energy, grad = geometry_energy(helix, xyz)
        return energy, null.T @ grad.reshape(-1)

    start = np.zeros(null.shape[1], dtype=np.float64)
    result = minimize(objective, start, method="L-BFGS-B", jac=True, options={"maxiter": int(maxiter), "ftol": 1e-12})
    if not result.success and result.nit < 1:
        raise RuntimeError(result.message)
    return origin + (null @ result.x).reshape(helix.xyz.shape)


def scan_mode(
    helix: Helix,
    mode: np.ndarray,
    *,
    n_steps: int = N_STEPS,
    amplitude: float = AMPLITUDE,
    maxiter: int = 60,
) -> ModeScan:
    """``n_steps`` amplitudes from ``-amplitude`` to ``+amplitude``, each refined."""
    amplitudes = np.linspace(-float(amplitude), float(amplitude), int(n_steps))
    frames = []
    bond_rms = []
    pair_rms = []
    hbond_rms = []
    ca_rmsd = []
    ca = np.array([str(n).strip() == "CA" for n in helix.name])
    for value in amplitudes:
        xyz = refine_along_mode(helix, mode, float(value), maxiter=maxiter)
        frames.append(xyz)
        bond_rms.append(_rms(xyz, helix.bonds, helix.bond_ideal))
        pair_rms.append(_rms(xyz, helix.pairs14, helix.pair14_ideal))
        hbond_rms.append(_rms(xyz, helix.hbonds, helix.hbond_ideal))
        ca_rmsd.append(float(np.sqrt(np.mean((xyz[ca] - helix.xyz[ca]) ** 2))))
    return ModeScan(
        vector=mode,
        amplitudes=amplitudes,
        frames=np.stack(frames),
        bond_rms=np.asarray(bond_rms),
        pair14_rms=np.asarray(pair_rms),
        hbond_rms=np.asarray(hbond_rms),
        ca_rmsd=np.asarray(ca_rmsd),
    )


def _rms(xyz: np.ndarray, pairs: np.ndarray, ideal: np.ndarray) -> float:
    if len(pairs) == 0:
        return 0.0
    dist = _lengths(xyz, pairs)
    return float(np.sqrt(np.mean((dist - ideal) ** 2)))


def unrefined_bond_rms(helix: Helix, mode: np.ndarray, amplitude: float) -> float:
    return _rms(helix.xyz + float(amplitude) * mode, helix.bonds, helix.bond_ideal)


def write_pdb(path: Path, helix: Helix, frames: Iterable[np.ndarray], cell: Sequence[float] = CELL) -> None:
    """One PDB. Several frames become MODEL records Moorhen can step through."""
    frames = list(frames)
    a, b, c, alpha, beta, gamma = (float(x) for x in cell)
    lines = [f"CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{alpha:7.2f}{beta:7.2f}{gamma:7.2f} P 1\n"]
    multi = len(frames) > 1
    for model, xyz in enumerate(frames, start=1):
        if multi:
            lines.append(f"MODEL     {model:4d}\n")
        for serial, (atom, res, element, coord) in enumerate(
            zip(helix.name, helix.resseq, helix.element, xyz),
            start=1,
        ):
            label = f"{str(atom):>4s}"[:4]
            el = f"{str(element).strip():>2s}"[:2]
            lines.append(
                f"ATOM  {serial:5d} {label} ALA A{int(res):4d}    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00 15.00          {el}\n"
            )
        if multi:
            lines.append("ENDMDL\n")
    lines.append("END\n")
    path.write_text("".join(lines))


def write_reference_map(path: Path, helix: Helix, cell: Sequence[float] = CELL, d_min: float = 2.0) -> None:
    """Sigma-scaled Fc of the ideal helix, for the same cell as the PDBs."""
    from cctbx import crystal, miller
    import iotbx.pdb

    buf = Path(path).with_name("_helix_ref_tmp.pdb")
    write_pdb(buf, helix, [helix.xyz], cell)
    try:
        inp = iotbx.pdb.input(file_name=str(buf))
        cs = crystal.symmetry(unit_cell=tuple(float(x) for x in cell), space_group_symbol="P 1")
        xrs = inp.xray_structure_simple(crystal_symmetry=cs)
        xrs.set_b_iso(value=15.0)
        miller_set = miller.build_set(crystal_symmetry=cs, anomalous_flag=False, d_min=float(d_min))
        fc = miller_set.structure_factors_from_scatterers(xray_structure=xrs, algorithm="direct").f_calc()
        fft = fc.fft_map(resolution_factor=0.33)
        fft.apply_sigma_scaling()
        fft.as_ccp4_map(file_name=str(path), labels=["poly-Ala alpha helix Fc; sigma = 1"])
    finally:
        buf.unlink(missing_ok=True)


def write_ensemble(
    directory: Path,
    *,
    n_res: int = N_RESIDUES,
    n_steps: int = N_STEPS,
    amplitude: float = AMPLITUDE,
    maxiter: int = 60,
    write_map: bool = True,
) -> list[ModeScan]:
    """Write the reference model, three 25-frame trajectories, and a score table."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    helix = ideal_helix(n_res)
    modes = elastic_modes(helix.xyz, N_MODES)
    write_pdb(directory / "helix.pdb", helix, [helix.xyz])
    scans = []
    lines = [
        "Poly-alanine alpha helix. Three softest elastic-network modes.",
        f"{n_res} residues, cell {CELL[0]:.0f} {CELL[1]:.0f} {CELL[2]:.0f} P 1, axis along c.",
        f"{n_steps} amplitudes on each mode, from {-amplitude:.2f} to {amplitude:.2f} A RMSD.",
        "Each frame is refined orthogonal to its mode: bonds (1-2), 1-4 distances,",
        "and helical O(i)...N(i+4) plus CA(i)...CA(i+4).",
        "helix.pdb is the ideal model. helix_fc.ccp4 is its density (sigma = 1, contour ~1.5).",
        "mode_01.pdb, mode_02.pdb, mode_03.pdb are MODEL trajectories. Step them in Moorhen.",
        "",
        "mode  step  amplitude  bond_rms  pair14_rms  hbond_rms  ca_rmsd",
    ]
    for index, mode in enumerate(modes, start=1):
        scan = scan_mode(helix, mode, n_steps=n_steps, amplitude=amplitude, maxiter=maxiter)
        scans.append(scan)
        write_pdb(directory / f"mode_{index:02d}.pdb", helix, scan.frames)
        for step, (amp, bond, pair, hbond, rmsd) in enumerate(
            zip(scan.amplitudes, scan.bond_rms, scan.pair14_rms, scan.hbond_rms, scan.ca_rmsd),
            start=1,
        ):
            lines.append(
                f"{index:4d}  {step:4d}  {amp:9.3f}  {bond:8.4f}  {pair:10.4f}  {hbond:9.4f}  {rmsd:7.3f}"
            )
    if write_map:
        write_reference_map(directory / "helix_fc.ccp4", helix)
    (directory / "scores.txt").write_text("\n".join(lines) + "\n")
    return scans


def load_frames(directory: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coordinates of every refined MODEL, with the mode index and amplitude.

    Reads ``mode_01.pdb``, ``mode_02.pdb``, ``mode_03.pdb`` and the amplitude
    column of ``scores.txt``.
    """
    import iotbx.pdb

    directory = Path(directory)
    amplitudes: list[float] = []
    modes: list[int] = []
    for line in (directory / "scores.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
            modes.append(int(parts[0]))
            amplitudes.append(float(parts[2]))
    frames: list[np.ndarray] = []
    for index in (1, 2, 3):
        hierarchy = iotbx.pdb.input(file_name=str(directory / f"mode_{index:02d}.pdb")).construct_hierarchy()
        for model in hierarchy.models():
            frames.append(np.array(list(model.atoms().extract_xyz()), dtype=np.float64))
    xyz = np.stack(frames)
    if xyz.shape[0] != len(amplitudes):
        raise RuntimeError(f"{xyz.shape[0]} models but {len(amplitudes)} score rows")
    return xyz, np.asarray(modes, dtype=np.int32), np.asarray(amplitudes, dtype=np.float64)


def _f_calc_many(
    reference_pdb: Path,
    frames: np.ndarray,
    *,
    cell: Sequence[float] = CELL,
    d_min: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Direct-summation structure factors for each frame. Returns hkl, F, centric, epsilon."""
    from cctbx import crystal, miller
    from cctbx.array_family import flex
    import iotbx.pdb

    cs = crystal.symmetry(unit_cell=tuple(float(x) for x in cell), space_group_symbol="P 1")
    inp = iotbx.pdb.input(file_name=str(reference_pdb))
    xrs = inp.xray_structure_simple(crystal_symmetry=cs)
    xrs.scattering_type_registry(table="wk1995")
    miller_set = miller.build_set(crystal_symmetry=cs, anomalous_flag=False, d_min=float(d_min))
    if xrs.scatterers().size() != frames.shape[1]:
        raise RuntimeError("reference PDB atom count does not match the trajectory")
    columns = []
    for xyz in frames:
        moved = xrs.deep_copy_scatterers()
        moved.set_sites_cart(flex.vec3_double(np.asarray(xyz, dtype=np.float64)))
        fc = miller_set.structure_factors_from_scatterers(xray_structure=moved, algorithm="direct").f_calc()
        columns.append(np.array(fc.data(), dtype=np.complex128))
    hkl = np.array([tuple(h) for h in fc.indices()], dtype=np.int32)
    centric = np.array(fc.centric_flags().data(), dtype=bool)
    epsilon = np.array(fc.epsilons().data().as_double(), dtype=np.float64)
    return hkl, np.stack(columns), centric, epsilon


def intensities_from_frames(
    f_calc: np.ndarray,
    *,
    noise: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """``I = |F|²`` plus Gaussian noise ``σ_I = noise |F|² + 0.5``."""
    rng = np.random.default_rng(int(seed))
    intensity = np.abs(np.asarray(f_calc, dtype=np.complex128)) ** 2
    sig_i = float(noise) * intensity + 0.5
    i_obs = intensity + rng.normal(scale=sig_i)
    return i_obs.astype(np.float64), sig_i.astype(np.float64)


def _wilson_from_mean(f_mu: np.ndarray, shell_index: np.ndarray) -> np.ndarray:
    power = np.abs(f_mu) ** 2
    wilson = np.empty_like(power)
    for shell in np.unique(shell_index):
        sel = shell_index == shell
        wilson[sel] = max(float(np.mean(power[sel])), 1e-3)
    return wilson


def _leading_modes(delta_f: np.ndarray, rank: int) -> np.ndarray:
    embedded = np.concatenate([delta_f.real, delta_f.imag], axis=1)
    _u, _s, vh = np.linalg.svd(embedded, full_matrices=False)
    n_h = delta_f.shape[1]
    return (vh[:rank, :n_h] + 1j * vh[:rank, n_h:]).astype(np.complex128)


def fit_from_mean_model(
    directory: Path,
    *,
    d_min: float = 2.5,
    rank: int = 3,
    n_sweeps: int = 8,
    noise: float = 0.05,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Intensities of the refined frames, then EM started at the mean structure.

    ``F_μ`` is the structure factor of the average coordinates. The first
    E-step is at ``z = 0`` on that mean model; the sweeps then estimate
    ``Δ0``, the modes, and the scores.
    """
    from phridge.contrib.ensemble.fit import em_fit
    from phridge.contrib.ensemble.gauge import principal_angles
    from phridge.contrib.ensemble.synthetic import equal_count_shells

    directory = Path(directory)
    frames, mode_id, amplitude = load_frames(directory)
    mean_xyz = frames.mean(axis=0)
    hkl, f_frames, centric, epsilon = _f_calc_many(directory / "helix.pdb", np.vstack([mean_xyz[None], frames]), d_min=d_min)
    f_mu = f_frames[0]
    f_calc = f_frames[1:]
    i_obs, sig_i = intensities_from_frames(f_calc, noise=noise, seed=seed)
    shells = equal_count_shells(1.0 / np.sqrt(np.maximum(_s2(hkl), 1e-12)), bin_size=max(32, hkl.shape[0] // 8))
    wilson_1 = _wilson_from_mean(f_mu, shells)
    n_data = i_obs.shape[0]
    wilson = np.broadcast_to(wilson_1, (n_data, wilson_1.shape[0])).copy()
    result = em_fit(
        i_obs,
        sig_i,
        np.ones_like(i_obs, dtype=bool),
        f_mu,
        epsilon,
        centric,
        wilson,
        rank=rank,
        n_sweeps=n_sweeps,
        dof_correct=False,
        shell_index=shells,
        sigma_a0=0.9,
        hkl=hkl,
        cell=CELL,
    )
    planted = _leading_modes(f_calc - f_mu, rank)
    phase = f_mu / np.maximum(np.abs(f_mu), 1e-12)
    visible = np.real(np.conj(phase) * (f_calc - f_mu)) * phase
    angles = principal_angles(planted, result.modes)
    visible_angles = principal_angles(_leading_modes(visible, rank), result.modes)
    correlations = []
    for which in range(1, rank + 1):
        sel = mode_id == which
        amps = amplitude[sel] - float(np.mean(amplitude[sel]))
        centered = result.scores[sel] - result.scores[sel].mean(axis=0)
        if float(np.dot(amps, amps)) < 1e-12:
            correlations.append(float("nan"))
            continue
        direction = centered.T @ amps / float(np.dot(amps, amps))
        predicted = centered @ direction
        correlations.append(float(np.corrcoef(amps, predicted)[0, 1]))
    _write_mean_pdb(directory / "mean_model.pdb", directory / "helix.pdb", mean_xyz)
    _write_mode_maps(directory, hkl, result.modes, f_mu)
    summary = _em_summary(result, angles, visible_angles, correlations, hkl.shape[0], d_min, n_sweeps)
    (directory / "em_summary.txt").write_text(summary)
    np.savez(
        directory / "ensemble_data.npz",
        hkl=hkl,
        i_obs=i_obs,
        sig_i=sig_i,
        f_mu=f_mu,
        f_calc=f_calc,
        mode_id=mode_id,
        amplitude=amplitude,
        centric=centric,
        epsilon=epsilon,
        shell_index=shells,
        delta0=result.delta0,
        modes=result.modes,
        scores=result.scores,
        loglik_history=result.loglik_history,
    )
    return {
        "angles": angles,
        "score_correlation": np.asarray(correlations),
        "loglik_history": result.loglik_history,
    }


def _s2(hkl: np.ndarray, cell: Sequence[float] = CELL) -> np.ndarray:
    h = np.asarray(hkl, dtype=np.float64)
    return (h[:, 0] / cell[0]) ** 2 + (h[:, 1] / cell[1]) ** 2 + (h[:, 2] / cell[2]) ** 2


def _write_mean_pdb(path: Path, reference_pdb: Path, xyz: np.ndarray) -> None:
    text = Path(reference_pdb).read_text().splitlines()
    atoms = [line for line in text if line.startswith("ATOM")]
    if len(atoms) != xyz.shape[0]:
        raise RuntimeError("mean coordinates do not match helix.pdb")
    lines = [line + "\n" for line in text if line.startswith("CRYST1")]
    for line, coord in zip(atoms, xyz):
        lines.append(f"{line[:30]}{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}{line[54:]}\n")
    lines.append("END\n")
    path.write_text("".join(lines))


def _write_mode_maps(directory: Path, hkl: np.ndarray, modes: np.ndarray, f_mu: np.ndarray) -> None:
    from cctbx import crystal, miller
    from cctbx.array_family import flex

    cs = crystal.symmetry(unit_cell=CELL, space_group_symbol="P 1")
    indices = flex.miller_index([tuple(int(x) for x in row) for row in hkl])
    miller_set = miller.set(crystal_symmetry=cs, indices=indices, anomalous_flag=False)
    base = miller.array(miller_set=miller_set, data=flex.complex_double(np.asarray(f_mu, dtype=np.complex128)))
    fft = base.fft_map(resolution_factor=0.5)
    fft.apply_sigma_scaling()
    fft.as_ccp4_map(file_name=str(directory / "mean_fc.ccp4"), labels=["mean model Fc; sigma = 1"])
    grids = []
    for coeff in modes:
        arr = base.customized_copy(data=flex.complex_double(np.asarray(coeff, dtype=np.complex128)))
        mode_fft = arr.fft_map(resolution_factor=0.5)
        mode_fft.apply_volume_scaling()
        rm = mode_fft.real_map_unpadded()
        grids.append(np.array(rm, dtype=np.float64).reshape(tuple(int(v) for v in rm.focus())))
    scale = float(np.sqrt(np.mean(np.stack(grids) ** 2)))
    for index, coeff in enumerate(modes, start=1):
        arr = base.customized_copy(data=flex.complex_double(np.asarray(coeff / scale, dtype=np.complex128)))
        mode_fft = arr.fft_map(resolution_factor=0.5)
        mode_fft.apply_volume_scaling()
        mode_fft.as_ccp4_map(
            file_name=str(directory / f"em_mode_{index:02d}.ccp4"),
            labels=[f"EM mode {index}; pooled mode RMS = 1"],
        )


def _em_summary(
    result: Any,
    angles: np.ndarray,
    visible_angles: np.ndarray,
    correlations: list[float],
    n_h: int,
    d_min: float,
    n_sweeps: int,
) -> str:
    history = result.loglik_history
    lines = [
        "EM started at the mean model (z = 0, F_mu = F of the average coordinates).",
        f"{result.scores.shape[0]} datasets, {n_h} reflections to {d_min:.2f} A, rank {result.modes.shape[0]}, {n_sweeps} sweeps.",
        f"Training log-likelihood {history[0]:.1f} -> {history[-1]:.1f}.",
        "Principal angles (degrees) to the leading singular vectors of F_n - F_mu:",
        "  " + "  ".join(f"{float(a):.2f}" for a in angles),
        "The same angles against only the part of F_n - F_mu parallel to the mean-model phase:",
        "  " + "  ".join(f"{float(a):.2f}" for a in visible_angles),
        "Multiple correlation of each planted amplitude with the EM scores of that mode's frames:",
    ]
    for index, value in enumerate(correlations, start=1):
        lines.append(f"  mode {index}: {value:.3f}")
    lines.append("mean_model.pdb and mean_fc.ccp4 are the kickstart. em_mode_XX.ccp4 are the fitted modes.")
    lines.append("Contour mean_fc near +1.5 and the EM modes near +/- 2.")
    return "\n".join(lines) + "\n"
