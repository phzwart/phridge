"""Normal-mode scan of an ideal poly-Ala helix."""

from __future__ import annotations

import pytest

mmtbx = pytest.importorskip("mmtbx")
del mmtbx


def test_refinement_repairs_bonds_and_holds_the_mode():
    import numpy as np

    from phridge.tools.ensemble.helix_ensemble import (
        elastic_modes,
        ideal_helix,
        scan_mode,
        unrefined_bond_rms,
    )

    helix = ideal_helix(8)
    mode = elastic_modes(helix.xyz, n_modes=1)[0]
    raw = unrefined_bond_rms(helix, mode, 1.0)
    scan = scan_mode(helix, mode, n_steps=3, amplitude=1.0, maxiter=40)
    assert raw > 0.01
    assert scan.bond_rms[-1] < raw
    assert scan.bond_rms[-1] < 0.03
    assert scan.pair14_rms[-1] < 0.08
    displacement = scan.frames[-1] - helix.xyz
    amplitude = float(np.sum(displacement * mode) / np.sum(mode**2))
    assert abs(amplitude - 1.0) < 1e-6
    assert scan.frames.shape[0] == 3
