"""Crystal symmetry adapter."""

from __future__ import annotations

from typing import Any

from phridge.contracts.models import CrystalSymmetry, SymOp


from phridge.contracts.manifest import PluginManifest, Slot, Space, Tier

MANIFEST = PluginManifest(
    name='cctbx_symmetry',
    slot=Slot.io_adapter,
    space=Space.D,
    tier=Tier.hybrid,
)

def crystal_from_cctbx(sym: Any, *, with_symops: bool = True) -> CrystalSymmetry:
    info = sym.space_group_info()
    hall = str(info.type().hall_symbol())
    symops = None
    if with_symops:
        symops = [
            SymOp(r=[float(x) for x in op.r().as_double()], t=[float(x) for x in op.t().as_double()])
            for op in sym.space_group()
        ]
    return CrystalSymmetry(
        unit_cell=[float(x) for x in sym.unit_cell().parameters()],
        space_group_hall=hall,
        space_group_number=int(info.type().number()),
        symops=symops,
    )


def crystal_to_cctbx(cs: CrystalSymmetry) -> Any:
    from cctbx import crystal, sgtbx

    try:
        symbols = sgtbx.space_group_symbols(cs.space_group_hall)
        group = sgtbx.space_group(symbols.hall())
    except Exception:
        if cs.space_group_number is None:
            raise ValueError(
                f"cannot reconstruct crystal.symmetry from Hall {cs.space_group_hall!r}"
            ) from None
        try:
            symbols = sgtbx.space_group_symbols(cs.space_group_number)
            group = sgtbx.space_group(symbols.hall())
        except Exception as exc:
            raise ValueError(
                f"cannot reconstruct crystal.symmetry from Hall {cs.space_group_hall!r} "
                f"or number {cs.space_group_number}"
            ) from exc
    return crystal.symmetry(unit_cell=tuple(cs.unit_cell), space_group=group)
