"""Fixed invertible per-residue encoder (15-D ALA xyz ↔ 64-D latent).

Seeded weights; never trained. Used by ``examples/latent_restraints_adam.py``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn


class InvertibleAffine(nn.Module):
    """Square affine y = Wx + b with explicit inverse (LU-stable init)."""

    def __init__(self, dim: int, *, seed: int) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(dim, dim, generator=g, dtype=torch.float64)
        # Orthogonalize for a well-conditioned invertible map.
        q, _ = torch.linalg.qr(w)
        self.weight = nn.Parameter(q, requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(dim, dtype=torch.float64), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T + self.bias

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        # Solve W^T x^T = (y - b)^T  →  x = (y - b) @ W^{-T} = (y - b) @ W  (W orthogonal)
        return (y - self.bias) @ self.weight


class InvertibleLeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.2) -> None:
        super().__init__()
        self.negative_slope = float(negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, x, self.negative_slope * x)

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        inv = 1.0 / self.negative_slope
        return torch.where(y >= 0, y, inv * y)


class ResidueEncoder(nn.Module):
    """Per-residue: pad/embed atom xyz → 64-D, then invertible 2-layer MLP.

    For ALA (5 atoms → 15 coords), coords occupy the first ``in_dim`` slots;
    remaining latent dims are zeros at encode time and discarded at decode.
    """

    def __init__(
        self,
        *,
        in_dim: int = 15,
        latent_dim: int = 64,
        seed: int = 0,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        if in_dim > latent_dim:
            raise ValueError("in_dim must be <= latent_dim")
        self.in_dim = int(in_dim)
        self.latent_dim = int(latent_dim)
        self.fc1 = InvertibleAffine(latent_dim, seed=seed)
        self.act = InvertibleLeakyReLU(negative_slope)
        self.fc2 = InvertibleAffine(latent_dim, seed=seed + 1)

    def embed(self, flat: torch.Tensor) -> torch.Tensor:
        """(R, in_dim) → (R, latent_dim) by zero-pad."""
        if flat.shape[-1] != self.in_dim:
            raise ValueError(f"expected last dim {self.in_dim}, got {flat.shape[-1]}")
        out = flat.new_zeros(flat.shape[:-1] + (self.latent_dim,))
        out[..., : self.in_dim] = flat
        return out

    def unembed(self, padded: torch.Tensor) -> torch.Tensor:
        return padded[..., : self.in_dim]

    def encode_flat(self, flat: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.embed(flat))))

    def decode_flat(self, z: torch.Tensor) -> torch.Tensor:
        return self.unembed(self.fc1.inverse(self.act.inverse(self.fc2.inverse(z))))

    def encode_sites(
        self,
        xyz: torch.Tensor,
        residue_slices: Sequence[slice],
    ) -> torch.Tensor:
        """xyz (N,3) → z (n_res, latent_dim)."""
        flats = []
        for sl in residue_slices:
            block = xyz[sl].reshape(-1)
            if block.numel() != self.in_dim:
                raise ValueError(f"residue slice {sl} has {block.numel()} coords, need {self.in_dim}")
            flats.append(block)
        return self.encode_flat(torch.stack(flats, dim=0))

    def decode_sites(
        self,
        z: torch.Tensor,
        *,
        n_atoms: int,
        residue_slices: Sequence[slice],
        template_xyz: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """z (n_res, latent_dim) → xyz (N,3). Unmentioned atoms kept from template."""
        flats = self.decode_flat(z)
        if template_xyz is None:
            xyz = z.new_zeros((n_atoms, 3))
        else:
            xyz = template_xyz.clone()
        for i, sl in enumerate(residue_slices):
            xyz[sl] = flats[i].reshape(-1, 3)
        return xyz


def residue_slices_from_atoms(atoms: Sequence[Any]) -> list[slice]:
    """Group packed hierarchy atoms by (chain_id, resseq) into contiguous slices."""
    if not atoms:
        return []
    slices: list[slice] = []
    start = 0
    key0 = (atoms[0].chain_id, int(atoms[0].resseq))
    for i in range(1, len(atoms)):
        key = (atoms[i].chain_id, int(atoms[i].resseq))
        if key != key0:
            slices.append(slice(start, i))
            start = i
            key0 = key
    slices.append(slice(start, len(atoms)))
    return slices
