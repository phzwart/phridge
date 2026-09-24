"""Fisher information shares, Cook's distances, consistency.

Three classical complementary quantities, kept separate. No composite score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from phridge.contrib.multixtal.factors import leave_one_out_factors


@dataclass
class InfluenceResult:
    shares: np.ndarray
    effective_n: float
    cooks: np.ndarray
    principal_angles: np.ndarray
    cooks_loadings: np.ndarray
    consistency: np.ndarray
    quantity: str


class ClassicalInfluence:
    """Phase-1 ``InfluenceAnalyzer``."""

    def information_shares(
        self,
        weights: np.ndarray,
        design: Optional[np.ndarray] = None,
        group_id: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Group shares of Fisher information. Rows sum to 1 per quantity.

        ``weights`` is ``(N, H)`` or observation-level ``(O,)``. ``group_id``
        (length ``N`` or ``O``) aggregates an arbitrary grouping — crystal,
        dose bin, pass. If ``design`` is None this is the share of ``μ``;
        if ``design`` is ``(N, p)`` or ``(O, p)`` the share is the diagonal
        contribution to ``x xᵀ``.
        """
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim == 1:
            return self._shares_1d(w, design=design, group_id=group_id)
        n_data, n_h = w.shape
        if group_id is not None:
            g = np.asarray(group_id).reshape(-1)
            if g.size != n_data:
                raise ValueError("group_id must have one entry per weight row")
            ids, inv = np.unique(g, return_inverse=True)
            collapsed = np.zeros((ids.size, n_h), dtype=np.float64)
            for i in range(ids.size):
                collapsed[i] = w[inv == i].sum(axis=0)
            x = None
            if design is not None:
                xd = np.asarray(design, dtype=np.float64)
                x = np.zeros((ids.size, xd.shape[1]), dtype=np.float64)
                for i in range(ids.size):
                    sel = inv == i
                    x[i] = xd[sel].mean(axis=0) if np.any(sel) else 0.0
            return self.information_shares(collapsed, design=x, group_id=None)
        if design is None:
            tot = np.sum(w, axis=0, keepdims=True)
            share_h = np.divide(w, tot, out=np.zeros_like(w), where=tot > 0)
            col = share_h.sum(axis=1)
            s = col / col.sum() if col.sum() > 0 else np.full(n_data, 1.0 / max(n_data, 1))
            return s.reshape(n_data, 1)
        x = np.asarray(design, dtype=np.float64)
        p = x.shape[1]
        contrib = np.zeros((n_data, p), dtype=np.float64)
        total = np.zeros(p, dtype=np.float64)
        for h in range(n_h):
            for d in range(n_data):
                if w[d, h] <= 0:
                    continue
                diag = w[d, h] * (x[d] ** 2)
                contrib[d] += diag
                total += diag
        shares = np.divide(contrib, total[None, :], out=np.zeros_like(contrib), where=total[None, :] > 0)
        return shares

    def _shares_1d(
        self,
        weights: np.ndarray,
        design: Optional[np.ndarray] = None,
        group_id: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if group_id is None:
            g = np.arange(w.size)
        else:
            g = np.asarray(group_id).reshape(-1)
            if g.size != w.size:
                raise ValueError("group_id must align with observation weights")
        ids, inv = np.unique(g, return_inverse=True)
        n_g = int(ids.size)
        if design is None:
            acc = np.zeros(n_g, dtype=np.float64)
            for i in range(n_g):
                acc[i] = float(np.sum(w[inv == i]))
            tot = float(acc.sum())
            s = acc / tot if tot > 0 else np.full(n_g, 1.0 / max(n_g, 1))
            return s.reshape(n_g, 1)
        x = np.asarray(design, dtype=np.float64)
        if x.shape[0] != w.size:
            raise ValueError("design must have one row per observation")
        p = x.shape[1]
        contrib = np.zeros((n_g, p), dtype=np.float64)
        total = np.zeros(p, dtype=np.float64)
        for i in range(w.size):
            if w[i] <= 0:
                continue
            diag = w[i] * (x[i] ** 2)
            contrib[inv[i]] += diag
            total += diag
        return np.divide(contrib, total[None, :], out=np.zeros_like(contrib), where=total[None, :] > 0)

    def cooks_distance(self, theta_full: np.ndarray, theta_loo: np.ndarray, information: np.ndarray, n_params: int) -> float:
        """``(θ − θ_{-d})ᵀ I_full (θ − θ_{-d}) / p``."""
        delta = np.asarray(theta_full, dtype=np.float64).reshape(-1) - np.asarray(theta_loo, dtype=np.float64).reshape(-1)
        info = np.asarray(information, dtype=np.float64)
        p = max(int(n_params), 1)
        if info.ndim == 1:
            val = float(np.sum(delta * info * delta))
        else:
            val = float(delta @ info @ delta)
        return val / p

    def consistency(self, chi2_loo: np.ndarray, dof: np.ndarray) -> np.ndarray:
        return np.asarray(chi2_loo, dtype=np.float64) / np.maximum(np.asarray(dof, dtype=np.float64), 1.0)


def effective_n(shares: np.ndarray) -> float:
    s = np.asarray(shares, dtype=np.float64).reshape(-1)
    num = float(np.sum(s) ** 2)
    den = float(np.sum(s**2))
    return num / den if den > 0 else 0.0


def principal_angles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Principal angles (radians) between column spaces of ``a`` and ``b`` (each p×r or r×p)."""
    if a.size == 0 or b.size == 0:
        return np.zeros(0, dtype=np.float64)
    ua, _ = np.linalg.qr(np.asarray(a, dtype=np.float64).T if a.shape[0] < a.shape[1] else np.asarray(a, dtype=np.float64), mode="reduced")
    ub, _ = np.linalg.qr(np.asarray(b, dtype=np.float64).T if b.shape[0] < b.shape[1] else np.asarray(b, dtype=np.float64), mode="reduced")
    # Loadings are r×H — use H×r columns.
    if ua.shape[0] < ua.shape[1]:
        ua = ua.T
    if ub.shape[0] < ub.shape[1]:
        ub = ub.T
    k = min(ua.shape[1], ub.shape[1])
    if k == 0:
        return np.zeros(0, dtype=np.float64)
    s = np.linalg.svd(ua[:, :k].T @ ub[:, :k], compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return np.arccos(s)


def procrustes_align(ref: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Orthogonal Procrustes: rotate ``other`` (r×H) onto ``ref``."""
    if ref.size == 0:
        return other
    u, _, vt = np.linalg.svd(other @ ref.T, full_matrices=False)
    r = u @ vt
    return r.T @ other


def analyze_influence(
    y: np.ndarray,
    w_model: np.ndarray,
    w_meas: np.ndarray,
    mu: np.ndarray,
    scores: np.ndarray,
    loadings: np.ndarray,
    chi2_loo: np.ndarray,
    ridge: float = 1e-6,
    anom_intercept: Optional[np.ndarray] = None,
    anom_weights: Optional[np.ndarray] = None,
    site_z: Optional[np.ndarray] = None,
    site_z_loo: Optional[np.ndarray] = None,
) -> list[InfluenceResult]:
    """Per-dataset influence on the shared quantities (full-data vs leave-one-out)."""
    analyzer = ClassicalInfluence()
    n_data = y.shape[0]
    rank = loadings.shape[0]
    results: list[InfluenceResult] = []

    share_mu = analyzer.information_shares(w_model)
    cooks_mu = np.zeros(n_data, dtype=np.float64)
    info_mu = np.sum(w_model, axis=0)
    angles = np.zeros((n_data, max(rank, 0)), dtype=np.float64)
    cooks_l = np.zeros(n_data, dtype=np.float64)
    for d in range(n_data):
        if n_data < 2:
            break
        mu_d, l_d, _z_d = leave_one_out_factors(y, w_model, d, rank, ridge=ridge)
        cooks_mu[d] = analyzer.cooks_distance(mu, mu_d, info_mu, n_params=int(np.sum(info_mu > 0)))
        if rank > 0:
            aligned = procrustes_align(loadings, l_d)
            ang = principal_angles(loadings.T, aligned.T)
            if ang.size:
                angles[d, : ang.size] = ang
            info_l = np.sum(w_model, axis=0)
            cooks_l[d] = analyzer.cooks_distance(loadings.reshape(-1), aligned.reshape(-1), np.tile(info_l, rank), n_params=int(loadings.size))

    results.append(
        InfluenceResult(
            shares=share_mu,
            effective_n=effective_n(share_mu),
            cooks=cooks_mu,
            principal_angles=angles,
            cooks_loadings=cooks_l,
            consistency=chi2_loo,
            quantity="population_mean",
        )
    )
    if rank > 0:
        design = np.concatenate([np.ones((n_data, 1)), scores], axis=1)
        share_l = analyzer.information_shares(w_model, design=design)
        results.append(
            InfluenceResult(
                shares=share_l,
                effective_n=effective_n(share_l.mean(axis=1) if share_l.ndim == 2 else share_l),
                cooks=cooks_l,
                principal_angles=angles,
                cooks_loadings=cooks_l,
                consistency=chi2_loo,
                quantity="loadings",
            )
        )
    if anom_intercept is not None and anom_weights is not None:
        share_a = analyzer.information_shares(anom_weights)
        results.append(
            InfluenceResult(
                shares=share_a,
                effective_n=effective_n(share_a),
                cooks=np.zeros(n_data),
                principal_angles=np.zeros((n_data, 0)),
                cooks_loadings=np.zeros(n_data),
                consistency=chi2_loo,
                quantity="anomalous_intercept",
            )
        )
    if site_z is not None and site_z_loo is not None:
        results.append(
            InfluenceResult(
                shares=np.full((n_data, 1), 1.0 / max(n_data, 1)),
                effective_n=float(n_data),
                cooks=np.abs(np.asarray(site_z) - np.asarray(site_z_loo)),
                principal_angles=np.zeros((n_data, 0)),
                cooks_loadings=np.zeros(n_data),
                consistency=chi2_loo,
                quantity="anomalous_peak_z",
            )
        )
    return results


def analyze_grouped_influence(
    y: np.ndarray,
    weights: np.ndarray,
    group_id: np.ndarray,
    ridge: float = 1e-6,
) -> InfluenceResult:
    """Observation- or row-level shares / Cook's / consistency by ``group_id``.

    Cook's is leave-one-group-out of a weighted intercept. Does not auto-reject.
    """
    analyzer = ClassicalInfluence()
    w = np.asarray(weights, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    g = np.asarray(group_id).reshape(-1)
    shares = analyzer.information_shares(w if w.ndim == 1 else w, group_id=g if w.ndim == 1 else g)
    ids = np.unique(g)
    n_g = int(ids.size)
    cooks = np.zeros(n_g, dtype=np.float64)
    cons = np.zeros(n_g, dtype=np.float64)
    if w.ndim == 1:
        yy = yy.reshape(-1)
        w = w.reshape(-1)
        tot_w = float(np.sum(w))
        mu = float(np.sum(w * yy) / tot_w) if tot_w > 0 else 0.0
        info = tot_w
        for i, gid in enumerate(ids):
            keep = g != gid
            tw = float(np.sum(w[keep]))
            mu_loo = float(np.sum(w[keep] * yy[keep]) / tw) if tw > 0 else mu
            cooks[i] = analyzer.cooks_distance(np.array([mu]), np.array([mu_loo]), np.array([info]), n_params=1)
            sel = g == gid
            dof = max(int(sel.sum()) - 1, 1)
            cons[i] = float(np.sum(w[sel] * (yy[sel] - mu_loo) ** 2) / dof) if np.any(sel) else 0.0
    else:
        # Treat rows as groups already if group_id matches rows.
        n_row = w.shape[0]
        if g.size != n_row:
            raise ValueError("group_id must have one entry per row")
        mu = np.sum(w * yy, axis=0) / np.maximum(np.sum(w, axis=0), 1e-12)
        info = np.sum(w, axis=0)
        for i, gid in enumerate(ids):
            keep = g != gid
            w_k = w[keep]
            y_k = yy[keep]
            mu_loo = np.sum(w_k * y_k, axis=0) / np.maximum(np.sum(w_k, axis=0), 1e-12)
            cooks[i] = analyzer.cooks_distance(mu, mu_loo, info, n_params=int(np.sum(info > 0)))
            sel = g == gid
            dof = max(int(np.sum(w[sel] > 0)) - 1, 1)
            cons[i] = float(np.sum(w[sel] * (yy[sel] - mu_loo) ** 2) / dof)
    return InfluenceResult(
        shares=shares,
        effective_n=effective_n(shares),
        cooks=cooks,
        principal_angles=np.zeros((n_g, 0)),
        cooks_loadings=np.zeros(n_g),
        consistency=cons,
        quantity="dose_bin",
    )
