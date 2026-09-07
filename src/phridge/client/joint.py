"""Joint x-ray + geometry site refinement with a geometry-aware preconditioner.

Target and Gauss-Newton model (cartesian sites x):

    Q(x) = Q_xray(x) + w E_geom(x)
    H    = F^T H_xray^GN F + w 2 J^T W J        (F: fractionalization matrix, x_frac = F x_cart)

Each Newton step solves (H + mu D) p = -grad Q by preconditioned conjugate gradients where

  * H v comes from two worker ops per CG iteration: ``gauss_newton_hvp`` (x-ray, FFT engine,
    curvatures from the target) and ``geometry_hvp`` (restraints, double vjp);
  * the preconditioner is the *factorised* geometry Gauss-Newton matrix with the x-ray
    per-atom block diagonal added, ``geometry_gn_solve`` — this captures the bonded
    coupling that a diagonal cannot (see docs/geometry_curvature.md).

Only cartesian site parameters are refined here (grad flags on the structure are honoured
for the x-ray gradient but the step touches sites only); ADPs / occupancies stay fixed.
Everything is expressed through Bridge calls so it runs against a Redis worker unchanged.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    from cctbx import adptbx
except ImportError:
    adptbx = None

from phridge.client.geometry import RemoteGeometry
from phridge.packing_scattering import PackedSfGradients
from phridge.sfcalc.client import RemoteGradients, RemoteRefinementTarget, _miller_template, _packed_xray, psd_target


def _frac_matrix(xray_structure: Any) -> np.ndarray:
    return np.asarray(xray_structure.unit_cell().fractionalization_matrix(), dtype=np.float64).reshape(3, 3)


def _extract_adp_vec(xs: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract flat adp_vec and is_aniso flags from xray_structure scatterers."""
    from cctbx import adptbx

    uc = xs.unit_cell()
    scatterers = xs.scatterers()
    is_aniso = np.array([bool(sc.flags.use_u_aniso()) for sc in scatterers], dtype=bool)
    vals: list[float] = []
    for sc in scatterers:
        if sc.flags.use_u_aniso():
            u_cart = np.asarray(adptbx.u_star_as_u_cart(uc, tuple(sc.u_star)), dtype=np.float64)
            b_cart = (8.0 * np.pi**2) * u_cart
            m00, m11, m22, m01, m02, m12 = b_cart
            M = np.array([[m00, m01, m02], [m01, m11, m12], [m02, m12, m22]], dtype=np.float64)
            eigvals, eigvecs = np.linalg.eigh(M)
            log_vals = np.log(np.maximum(eigvals, 1e-6))
            S_mat = eigvecs @ np.diag(log_vals) @ eigvecs.T
            vals.extend([S_mat[0, 0], S_mat[1, 1], S_mat[2, 2], S_mat[0, 1], S_mat[0, 2], S_mat[1, 2]])
        else:
            b_iso = float(sc.u_iso * 8.0 * np.pi**2)
            vals.append(float(np.log(max(b_iso, 1e-4))))
    return np.asarray(vals, dtype=np.float64), is_aniso


def _apply_adp_vec(xs: Any, adp_vec: np.ndarray, is_aniso: np.ndarray) -> None:
    """Update scatterers in xray_structure in-place from flat adp_vec."""
    from cctbx import adptbx

    uc = xs.unit_cell()
    scatterers = xs.scatterers()
    offset = 0
    for i, sc in enumerate(scatterers):
        if is_aniso[i]:
            s_vec = adp_vec[offset : offset + 6]
            offset += 6
            s00, s11, s22, s01, s02, s12 = s_vec
            S_mat = np.array([[s00, s01, s02], [s01, s11, s12], [s02, s12, s22]], dtype=np.float64)
            eigvals, eigvecs = np.linalg.eigh(S_mat)
            b_mat = eigvecs @ np.diag(np.exp(eigvals)) @ eigvecs.T
            u_cart = (1.0 / (8.0 * np.pi**2)) * np.array(
                [b_mat[0, 0], b_mat[1, 1], b_mat[2, 2], b_mat[0, 1], b_mat[0, 2], b_mat[1, 2]]
            )
            u_star = adptbx.u_cart_as_u_star(uc, tuple(u_cart))
            sc.u_star = tuple(u_star)
        else:
            beta = float(adp_vec[offset])
            offset += 1
            b_iso = float(np.exp(beta))
            sc.u_iso = b_iso / (8.0 * np.pi**2)


class JointSiteRefinement:
    """Newton-CG on sites and/or ADPs with the geometry/ADP GN factorisation as preconditioner."""

    def __init__(
        self,
        refiner: RemoteRefinementTarget,
        geometry: RemoteGeometry,
        weight: float,
        *,
        weight_adp: Optional[float] = None,
        refine: tuple[str, ...] = ("sites",),
        hessian_geom: str = "gn",
        psd: bool = True,
    ) -> None:
        self.refiner = refiner
        self.geometry = geometry
        self.weight = float(weight)
        self.weight_adp = float(weight_adp if weight_adp is not None else weight)
        self.refine = tuple(refine)
        self.hessian_geom = str(hessian_geom)
        self.psd = bool(psd)
        self.history: list[dict[str, float]] = []

    # -- pieces -----------------------------------------------------------------------
    def _xray(self, xs: Any) -> tuple[float, np.ndarray, dict, np.ndarray]:
        """(Q_xray, grad (flat parameters for refine blocks), raw refine_gradients output, F)."""
        out = self.refiner.compute(xs)
        F = _frac_matrix(xs)
        raw = RemoteGradients(xs, out["gradients"]).raw
        value = float(out["target"].meta.value)
        if self.psd:
            out = dict(out, target=psd_target(out["target"]))  # PSD Gauss-Newton surrogate for HVP / blocks

        g_parts: list[np.ndarray] = []
        if "sites" in self.refine:
            g_frac = np.asarray(raw.d_site_frac, dtype=np.float64)
            g_cart = g_frac @ F
            g_parts.append(g_cart.reshape(-1))

        if "adp" in self.refine:
            scatterers = xs.scatterers()
            adp_grads: list[float] = []
            for i, sc in enumerate(scatterers):
                if sc.flags.use_u_aniso():
                    # For aniso, approximate chain rule scale via mean u_iso
                    u_iso_mean = float(np.mean(adptbx.u_star_as_u_cart(xs.unit_cell(), tuple(sc.u_star))[:3]))
                    d_star = np.asarray(raw.d_u_star[i], dtype=np.float64)
                    adp_grads.extend((u_iso_mean * d_star).tolist())
                else:
                    u_i = float(sc.u_iso)
                    d_u = float(raw.d_u_iso[i])
                    adp_grads.append(u_i * d_u)
            g_parts.append(np.asarray(adp_grads, dtype=np.float64))

        g_flat = np.concatenate(g_parts) if g_parts else np.zeros(0, dtype=np.float64)
        return value, g_flat, out, F

    def _xray_cart_diagonal(self, xs: Any, out: dict, F: np.ndarray) -> np.ndarray:
        xray, table = _packed_xray(xs, self.refiner.table)
        curv = self.refiner.bridge.call(
            "gauss_newton_blocks",
            xray=xray,
            table=table,
            params=self.refiner.params,
            target=out["target"],
            hkl=_miller_template(self.refiner.functor.f_obs),
        )
        diag_parts: list[np.ndarray] = []
        if "sites" in self.refine:
            blocks_frac = np.asarray(curv.site_frac, dtype=np.float64)  # (N, 3, 3)
            blocks_cart = np.einsum("ab,nbc,cd->nad", F.T, blocks_frac, F)
            diag_sites = np.maximum(np.diagonal(blocks_cart, axis1=1, axis2=2), 0.0).copy().reshape(-1)
            diag_parts.append(diag_sites)

        if "adp" in self.refine:
            scatterers = xs.scatterers()
            adp_diag: list[float] = []
            for i, sc in enumerate(scatterers):
                if sc.flags.use_u_aniso():
                    u_iso_mean = float(np.mean(adptbx.u_star_as_u_cart(xs.unit_cell(), tuple(sc.u_star))[:3]))
                    h_star = np.diag(np.asarray(curv.u_star[i], dtype=np.float64))
                    adp_diag.extend(np.maximum((u_iso_mean**2) * h_star, 0.0).tolist())
                else:
                    u_i = float(sc.u_iso)
                    h_u = float(curv.u_iso[i])
                    adp_diag.append(max((u_i**2) * h_u, 0.0))
            diag_parts.append(np.asarray(adp_diag, dtype=np.float64))

        return np.concatenate(diag_parts) if diag_parts else np.zeros(0, dtype=np.float64)

    def _xray_hvp(self, xs: Any, out: dict, F: np.ndarray, v_flat: np.ndarray) -> np.ndarray:
        n = len(xs.scatterers())
        v_idx = 0
        v_frac = np.zeros((n, 3), dtype=np.float64)
        if "sites" in self.refine:
            v_cart = v_flat[v_idx : v_idx + 3 * n].reshape(-1, 3)
            v_frac = v_cart @ F.T
            v_idx += 3 * n

        d_u_iso = np.zeros(n, dtype=np.float64)
        d_u_star = np.zeros((n, 6), dtype=np.float64)
        if "adp" in self.refine:
            scatterers = xs.scatterers()
            for i, sc in enumerate(scatterers):
                if sc.flags.use_u_aniso():
                    u_iso_mean = float(np.mean(adptbx.u_star_as_u_cart(xs.unit_cell(), tuple(sc.u_star))[:3]))
                    d_u_star[i] = u_iso_mean * v_flat[v_idx : v_idx + 6]
                    v_idx += 6
                else:
                    d_u_iso[i] = float(sc.u_iso) * float(v_flat[v_idx])
                    v_idx += 1

        v = PackedSfGradients(
            d_site_frac=v_frac,
            d_occupancy=np.zeros(n),
            d_u_iso=d_u_iso,
            d_u_star=d_u_star,
            d_fp=np.zeros(n),
            d_fdp=np.zeros(n),
        )
        xray, table = _packed_xray(xs, self.refiner.table)
        hv = self.refiner.bridge.call(
            "gauss_newton_hvp",
            xray=xray,
            table=table,
            params=self.refiner.params,
            target=out["target"],
            hkl=_miller_template(self.refiner.functor.f_obs),
            v=v,
        )
        out_parts: list[np.ndarray] = []
        if "sites" in self.refine:
            hv_cart = np.asarray(hv.d_site_frac, dtype=np.float64) @ F
            out_parts.append(hv_cart.reshape(-1))

        if "adp" in self.refine:
            adp_hv: list[float] = []
            scatterers = xs.scatterers()
            for i, sc in enumerate(scatterers):
                if sc.flags.use_u_aniso():
                    u_iso_mean = float(np.mean(adptbx.u_star_as_u_cart(xs.unit_cell(), tuple(sc.u_star))[:3]))
                    adp_hv.extend((u_iso_mean * np.asarray(hv.d_u_star[i], dtype=np.float64)).tolist())
                else:
                    adp_hv.append(float(sc.u_iso) * float(hv.d_u_iso[i]))
            out_parts.append(np.asarray(adp_hv, dtype=np.float64))

        return np.concatenate(out_parts) if out_parts else np.zeros(0, dtype=np.float64)

    def _geom(self, sites_cart: np.ndarray, adp_vec: Optional[np.ndarray] = None) -> tuple[float, np.ndarray]:
        e_tot = 0.0
        g_parts: list[np.ndarray] = []
        if "sites" in self.refine:
            c = self.geometry.curvature(sites_cart, sparse=False)
            e_tot += self.weight * float(c["energy"])
            g_parts.append(self.weight * np.asarray(c["gradient"], dtype=np.float64).reshape(-1))

        if "adp" in self.refine and adp_vec is not None:
            c_adp = self.geometry.adp_eval(adp_vec, sites_cart=sites_cart)
            e_tot += self.weight_adp * float(c_adp["energy"])
            g_parts.append(self.weight_adp * np.asarray(c_adp["gradient"], dtype=np.float64).reshape(-1))

        g_flat = np.concatenate(g_parts) if g_parts else np.zeros(0, dtype=np.float64)
        return e_tot, g_flat

    def total(self, xs: Any) -> float:
        q, _, _, _ = self._xray(xs)
        x_cart = np.asarray(xs.sites_cart(), dtype=np.float64)
        adp_vec = _extract_adp_vec(xs)[0] if "adp" in self.refine else None
        e, _ = self._geom(x_cart, adp_vec=adp_vec)
        return q + e

    # -- solver -----------------------------------------------------------------------
    def newton_cg(
        self,
        xray_structure: Optional[Any] = None,
        *,
        max_iterations: int = 10,
        cg_max_iter: int = 30,
        cg_tol: float = 1e-4,
        damping: float = 1e-3,
        step_max: float = 0.3,
        precondition: bool = True,
        method: str = "sparse",
        groups: Optional[list[Any]] = None,
    ) -> dict[str, Any]:
        """Damped Newton-CG; returns diagnostics and a refined deep copy as ``xray_structure``."""
        if method == "tridiagonal" and groups is None:
            groups = self.geometry.residue_groups()
        from cctbx.array_family import flex

        xs = (xray_structure if xray_structure is not None else self.refiner.xray_structure).deep_copy_scatterers()
        for sc in xs.scatterers():
            if "sites" in self.refine:
                sc.flags.set_grad_site(True)
            if "adp" in self.refine:
                if sc.flags.use_u_aniso():
                    sc.flags.set_grad_u_aniso(True)
                else:
                    sc.flags.set_grad_u_iso(True)

        n_hvp = 0
        n_sites = len(xs.scatterers())
        n3 = 3 * n_sites

        for it in range(int(max_iterations)):
            x_cart = np.asarray(xs.sites_cart(), dtype=np.float64)
            adp_vec, is_aniso = _extract_adp_vec(xs) if "adp" in self.refine else (None, None)

            q, g_x, out, F = self._xray(xs)
            e_g, g_g = self._geom(x_cart, adp_vec=adp_vec)
            f0 = q + e_g
            g_flat = g_x + g_g
            d_x = self._xray_cart_diagonal(xs, out, F)

            def hvp(v_flat: np.ndarray) -> np.ndarray:
                nonlocal n_hvp
                n_hvp += 1
                hv_xray = self._xray_hvp(xs, out, F, v_flat)
                hv_geom_parts: list[np.ndarray] = []
                v_idx = 0
                if "sites" in self.refine:
                    v_s = v_flat[v_idx : v_idx + n3].reshape(-1, 3)
                    hv_s = self.weight * self.geometry.hvp(v_s, x_cart, hessian=self.hessian_geom)
                    hv_geom_parts.append(hv_s.reshape(-1))
                    v_idx += n3
                if "adp" in self.refine and adp_vec is not None:
                    v_a = v_flat[v_idx:]
                    hv_a = self.weight_adp * self.geometry.adp_hvp(adp_vec, v_a, sites_cart=x_cart)
                    hv_geom_parts.append(hv_a.reshape(-1))

                hv_geom = np.concatenate(hv_geom_parts) if hv_geom_parts else np.zeros_like(hv_xray)
                return hv_xray + hv_geom

            # Configure blocks for gn_solve
            blocks_arg: list[str] = []
            if "sites" in self.refine:
                blocks_arg.append("sites")
            if "adp" in self.refine:
                blocks_arg.append("adp")

            if precondition:
                def msolve(r_flat: np.ndarray) -> np.ndarray:
                    return self.geometry.gn_solve(
                        r_flat,
                        x_cart,
                        extra_diag=d_x,
                        weight=self.weight,
                        damping=damping,
                        method=method,
                        groups=groups,
                        blocks=blocks_arg,
                        adp_params=adp_vec,
                    ).reshape(-1)
            else:
                d_tot = np.maximum(d_x, damping * np.median(d_x[d_x > 0]) if np.any(d_x > 0) else 1.0)

                def msolve(r_flat: np.ndarray) -> np.ndarray:
                    return r_flat / d_tot

            mu = damping * float(np.median(d_x[d_x > 0])) if np.any(d_x > 0) else damping
            p_flat, cg_iters = _pcg(lambda v: hvp(v) + mu * v, -g_flat, msolve, max_iter=int(cg_max_iter), tol=cg_tol)

            # Enforce max step on sites
            p_step = p_flat.copy()
            if "sites" in self.refine:
                p_s = p_step[:n3].reshape(-1, 3)
                step_s = float(np.abs(p_s).max())
                if step_s > step_max:
                    p_step[:n3] *= step_max / step_s

            alpha, gp = 1.0, float(g_flat @ p_step)
            accepted = False
            for _ in range(12):
                xs_try = xs.deep_copy_scatterers()
                p_cur = alpha * p_step
                c_idx = 0
                if "sites" in self.refine:
                    p_s = p_cur[c_idx : c_idx + n3].reshape(-1, 3)
                    xs_try.set_sites_cart(flex.vec3_double((x_cart + p_s).tolist()))
                    c_idx += n3
                if "adp" in self.refine and adp_vec is not None and is_aniso is not None:
                    p_a = p_cur[c_idx:]
                    _apply_adp_vec(xs_try, adp_vec + p_a, is_aniso)

                f1 = self.total(xs_try)
                if f1 <= f0 + 1e-4 * alpha * gp:
                    accepted = True
                    break
                alpha *= 0.5

            self.history.append(
                {
                    "iter": it,
                    "target": f0,
                    "xray": q,
                    "geom": e_g,
                    "grad_norm": float(np.linalg.norm(g_flat)),
                    "cg_iters": cg_iters,
                    "alpha": alpha,
                    "accepted": float(accepted),
                }
            )
            if not accepted:
                break
            xs = xs_try

        return {"xray_structure": xs, "history": self.history, "n_hvp": n_hvp, "final_target": self.total(xs)}


def _pcg(hvp, b, msolve, max_iter=30, tol=1e-4):
    """Preconditioned CG; stops on negative curvature (returns the best descent direction so far)."""
    x = np.zeros_like(b)
    r = b.copy()
    z = msolve(r)
    p = z.copy()
    rz = float(r @ z)
    b_norm = float(np.linalg.norm(b)) or 1.0
    k = 0
    for k in range(1, max_iter + 1):
        Ap = hvp(p)
        pAp = float(p @ Ap)
        if pAp <= 0.0:
            if k == 1:
                x = z
            break
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        if np.linalg.norm(r) <= tol * b_norm:
            break
        z = msolve(r)
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, k
