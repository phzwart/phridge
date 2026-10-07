"""Intensity likelihood quadrature and the ml_i scoring rule."""

from __future__ import annotations

from phridge.d.rules.mli_quad.mli_likelihood import (
    log_likelihood_normal,
    log_likelihood_t,
    normalize,
    quadrature_terms_t,
)
from phridge.d.rules.mli_quad.mli_quadrature import (
    _legendre,
    _ratio,
    acen_E,
    acen_E_log,
    cen_E,
    posterior_mode_E,
    quadrature_terms_normal,
)

__all__ = [
    "_legendre",
    "_ratio",
    "acen_E",
    "acen_E_log",
    "cen_E",
    "log_likelihood_normal",
    "log_likelihood_t",
    "normalize",
    "posterior_mode_E",
    "quadrature_terms_normal",
    "quadrature_terms_t",
]


def register() -> None:
    from phridge.contrib.intensity_ll import register as register_ops
    from phridge.d.rules.manifests import register as register_manifests

    register_ops()
    register_manifests()
