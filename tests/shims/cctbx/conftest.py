"""Import cctbx before torch: importing torch first segfaults cctbx's Boost.Python extensions."""

from __future__ import annotations

try:
    import cctbx  # noqa: F401
    from cctbx import xray  # noqa: F401
except ImportError:
    pass


def random_xs(space_group: str, n_atoms: int = 8, *, elements: tuple[str, ...] = ("C", "N", "O", "S"), seed: int = 0):
    """Random x-ray structure with about ``n_atoms`` ASU scatterers.

    Site, occupancy, and u_iso gradients are enabled. Shared with
    ``tests/sf_gradient_bench.py`` (P1, P21, P212121, C2, P4132, R3:H).
    """
    import random

    from cctbx import sgtbx
    from cctbx.array_family import flex
    from cctbx.development import random_structure

    random.seed(seed)
    flex.set_random_seed(seed)
    n_repeat = max(1, int(round(n_atoms / len(elements))))
    xs = random_structure.xray_structure(
        space_group_info=sgtbx.space_group_info(space_group),
        elements=list(elements) * n_repeat,
        volume_per_atom=50,
        random_u_iso=True,
        random_occupancy=True,
        use_u_aniso=False,
    )
    for sc in xs.scatterers():
        sc.flags.set_grad_site(True)
        sc.flags.set_grad_occupancy(True)
        sc.flags.set_grad_u_iso(True)
        sc.flags.set_grad_u_aniso(False)
        sc.flags.set_grad_fp(False)
        sc.flags.set_grad_fdp(False)
    return xs
