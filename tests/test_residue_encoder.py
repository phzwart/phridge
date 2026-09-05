"""Round-trip tests for the fixed invertible residue encoder."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))


def test_residue_encoder_round_trip():
    from residue_encoder import ResidueEncoder, residue_slices_from_atoms

    class _Atom:
        def __init__(self, chain_id: str, resseq: int) -> None:
            self.chain_id = chain_id
            self.resseq = resseq

    # 3 residues × 5 atoms
    atoms = [_Atom("A", r) for r in (1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3)]
    slices = residue_slices_from_atoms(atoms)
    assert slices == [slice(0, 5), slice(5, 10), slice(10, 15)]

    enc = ResidueEncoder(in_dim=15, latent_dim=64, seed=11)
    enc.eval()
    xyz = torch.randn(15, 3, dtype=torch.float64)
    with torch.no_grad():
        z = enc.encode_sites(xyz, slices)
        back = enc.decode_sites(z, n_atoms=15, residue_slices=slices, template_xyz=xyz)
    assert z.shape == (3, 64)
    assert torch.allclose(back, xyz, atol=1e-10)


def test_residue_encoder_vjp_shape():
    from residue_encoder import ResidueEncoder

    enc = ResidueEncoder(in_dim=15, latent_dim=64, seed=3)
    slices = [slice(0, 5), slice(5, 10)]
    z = torch.randn(2, 64, dtype=torch.float64, requires_grad=True)
    xyz = enc.decode_sites(z, n_atoms=10, residue_slices=slices)
    g = torch.ones_like(xyz)
    (xyz * g).sum().backward()
    assert z.grad is not None
    assert z.grad.shape == z.shape
    assert torch.isfinite(z.grad).all()
