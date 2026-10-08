"""Stock macrocycle and the calibration policy's controller hook."""

from __future__ import annotations

from phridge.policies.base import Policy
from phridge.policies.calibrate_intensity import CalibrateIntensity
from phridge.policies.stock_macrocycle import StockMacrocycle


def test_stock_macrocycle_is_calibrate_step_report() -> None:
    policy = StockMacrocycle(cycles=2)
    history: list[Policy] = []
    names: list[str] = []
    while True:
        action = policy.next(history)
        if action is None:
            break
        names.append(action.name)
        history.append(action)  # type: ignore[arg-type]
    assert names.count("calibrate") == 2
    assert names.count("step") == 4
    assert [action.target for action in history if action.name == "step"] == ["sites", "u_iso", "sites", "u_iso"]  # type: ignore[attr-defined]
    assert names.count("report") == 2
    assert names[-1] == "finalize"
    assert policy.weight_by_grad_norm is True


def test_calibrate_intensity_adopts_next() -> None:
    policy = CalibrateIntensity()
    first = policy.next([])
    assert first is not None and first.name == "calibrate"
    assert policy.next([first]) is None
