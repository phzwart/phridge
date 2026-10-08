"""Rotamer fitting. The library is loaded once in the shim; the score is torch."""

from __future__ import annotations

from typing import Any

import torch

from phridge.shims.cctbx.rotamers import rotamer_library
from phridge.transforms.build.base import RChange


class RotamerBuilder:
    """Pick the library rotamer with the best torch score on ``selection``."""

    def propose(self, maps: dict[str, Any], stats: dict[str, Any], selection: Any) -> RChange:
        _ = maps
        _ = selection
        residue = str(stats.get("residue", "val")).lower()
        table = rotamer_library()[residue]
        chi = torch.tensor(table["chi"], dtype=torch.float64)
        frequency = torch.tensor(table["frequency"], dtype=torch.float64)
        current = stats.get("chi")
        if current is None:
            best = int(frequency.argmax())
        else:
            delta = _wrap(chi - torch.as_tensor(current, dtype=torch.float64))
            score = (delta**2).sum(dim=-1) - torch.log(frequency.clamp_min(1e-8))
            best = int(score.argmin())
        chosen = chi[best]
        return RChange(None, torch.zeros(0, dtype=torch.bool), {"chi": chosen})


def _wrap(delta: Any) -> Any:
    return (delta + 180.0) % 360.0 - 180.0
