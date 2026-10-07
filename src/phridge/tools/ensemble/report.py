"""Text report: K scan, half-split correlation, latent cloud, S_post.

The integrated residual, when printed, is ``S_post`` or ``S_prior``. A bare
``S`` is the small-molecule goodness-of-fit and is not used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def format_report(
    result: Mapping[str, Any],
    *,
    k_scan: Optional[Mapping[str, Any]] = None,
    half_split: Optional[Mapping[str, Any]] = None,
    cloud: Optional[Mapping[str, Any]] = None,
    s_post: Optional[float] = None,
    s_prior: Optional[float] = None,
) -> str:
    lines = ["Ensemble perturbation modes", ""]
    stats = result.get("stats") or {}
    lines.append(f"likelihood: {stats.get('likelihood', 'intensity')}")
    lines.append(f"k: {stats.get('k', '')}")
    history = result.get("loglik_history")
    if history is not None:
        hist = np.asarray(history, dtype=np.float64).reshape(-1)
        if hist.size:
            lines.append(f"training_loglik_final: {float(hist[-1]):.6f}")
    if result.get("loglik_unprojected") is not None or result.get("loglik_projected") is not None:
        lines.append(
            "Projected and unprojected log-likelihoods are different objectives and are not comparable."
        )
        lines.append(f"loglik_unprojected: {result.get('loglik_unprojected')}")
        lines.append(f"loglik_projected: {result.get('loglik_projected')}")
    if stats.get("hl_weight") not in (None, 0, 0.0):
        lines.append(
            "Training likelihoods at different hl_weight values are not comparable."
        )
    lines.append("")
    lines.append("K scan")
    if not k_scan:
        lines.append("(not run)")
    else:
        lines.append("k  heldout_loglik  unexplained_fraction")
        for row in k_scan.get("rows", []):
            lines.append(
                f"{row['k']}  {row.get('loglik_mean', float('nan')):.6f}  {row.get('unexplained_fraction_mean', float('nan')):.6f}"
            )
    lines.append("")
    lines.append("Half-split correlation")
    if not half_split:
        lines.append("(not run)")
    else:
        lines.append("raw correlation by mode and shell:")
        for k, rows in enumerate(half_split.get("correlation") or []):
            for row in rows:
                lines.append(f"mode {k} shell {row['shell']}: {row['correlation']:.4f}")
        lines.append(
            "correlation_optimistic_note_s6 is optimistic (sqrt(2C/(1+C)); note section 6)."
        )
    lines.append("")
    lines.append("Latent cloud")
    if not cloud:
        lines.append("(not run)")
    else:
        r2 = cloud.get("quadratic_r2") or []
        lines.append(f"quadratic_r2: {['%.4f' % float(v) for v in r2]}")
        if "positivity_margin" in cloud:
            lines.append(f"positivity_margin: {np.asarray(cloud['positivity_margin']).tolist()}")
    lines.append("")
    lines.append("Integrated residual")
    if s_post is None and s_prior is None:
        lines.append("S_post: not computed")
        lines.append("S_prior: not computed")
    else:
        if s_post is not None:
            lines.append(f"S_post: {float(s_post):.6f}")
        if s_prior is not None:
            lines.append(f"S_prior: {float(s_prior):.6f}")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_report(path: str | Path, text: str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return out
