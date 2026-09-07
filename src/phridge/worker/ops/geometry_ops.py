"""Worker op: torch geometry-restraint minimization (LBFGS / Adam / AdamW / SGD)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from phridge.packing_geometry import PackedRestraints
from phridge.packing_xtal import PackedCartesian
from phridge.sfcalc import ops as sfcalc_ops
from phridge.worker.geometry.energy import energy_and_sites

_DEVICE = sfcalc_ops._DEVICE


def geometry_minimize(
    sites: PackedCartesian,
    restraints: PackedRestraints,
    params: Optional[Any] = None,
) -> dict[str, Any]:
    """Minimize cartesian sites under packed bond/angle/dihedral restraints.

    ``params`` JSON:
      - ``max_iterations`` (default 100; 0 = energy only)
      - ``optimizer``: ``"lbfgs"`` | ``"adam"`` | ``"adamw"`` | ``"sgd"``
      - ``lr`` / ``lr_min``: peak and floor learning rates
      - ``schedule``: ``"none"`` | ``"cosine"`` | ``"triangular"``
      - ``momentum``: SGD momentum (default 0)
    """
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise TypeError("params must be a dict")
    max_iterations = int(params.get("max_iterations", 100))
    optimizer = str(params.get("optimizer", "lbfgs"))
    lr = params.get("lr", None)
    if lr is not None:
        lr = float(lr)
    lr_min = params.get("lr_min", None)
    if lr_min is not None:
        lr_min = float(lr_min)
    schedule = params.get("schedule", None)
    momentum = float(params.get("momentum", 0.0))
    preconditioner = params.get("preconditioner", None)
    precond_refresh = int(params.get("precond_refresh", 0))
    xyz = sites.xyz
    if hasattr(xyz, "detach"):
        xyz = xyz.detach().cpu().numpy()
    xyz = np.asarray(xyz, dtype=np.float64)
    out_xyz, target = energy_and_sites(
        xyz,
        restraints,
        max_iterations=max_iterations,
        optimizer=optimizer,
        lr=lr,
        lr_min=lr_min,
        schedule=schedule,
        momentum=momentum,
        preconditioner=preconditioner,
        precond_refresh=precond_refresh,
        device=_DEVICE["device"],
    )
    extra = {k: target[k] for k in ("preconditioner", "precond_refresh", "n_precond_refresh", "curvature",
                                    "gn_final_damping", "gn_rejected_steps") if k in target}
    return {
        "sites": PackedCartesian(out_xyz, crystal=sites.meta.crystal),
        "target": {
            "before": float(target["before"]),
            "after": float(target["after"]),
            "n_iter": int(target["n_calls"]),
            "n_steps": int(target["n_steps"]),
            "n_calls": int(target["n_calls"]),
            "optimizer": str(target["optimizer"]),
            "schedule": str(target.get("schedule") or "none"),
            "momentum": float(target.get("momentum") or 0.0),
            "lr": target.get("lr"),
            "lr_min": target.get("lr_min"),
            "rss_before_mb": float(target.get("rss_before_mb") or 0.0),
            "rss_after_mb": float(target.get("rss_after_mb") or 0.0),
            "rss_delta_mb": float(target.get("rss_delta_mb") or 0.0),
            "optimizer_state_mb": float(target.get("optimizer_state_mb") or 0.0),
            "cuda_peak_mb": float(target.get("cuda_peak_mb") or 0.0),
            **extra,
        },
    }


geometry_minimize.compute_dtype = "float64"


def _sites_tensor(sites: Any, restraints: PackedRestraints):
    import torch

    if hasattr(sites, "xyz"):
        xyz = sites.xyz
    else:
        xyz = sites
    if hasattr(xyz, "detach"):
        xyz = xyz.detach().cpu().numpy()
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] != restraints.n_sites:
        raise ValueError(f"sites must be (n_sites, 3) with n_sites={restraints.n_sites}, got {xyz.shape}")
    return torch.as_tensor(xyz, dtype=torch.float64, device=_DEVICE["device"])


def _curvature(restraints: PackedRestraints):
    import torch

    from phridge.worker.geometry.energy import _RestraintTables
    from phridge.worker.geometry.curvature import RestraintCurvature

    return RestraintCurvature(_RestraintTables.from_packed(restraints, device=_DEVICE["device"], dtype=torch.float64))


def geometry_curvature(
    sites: PackedCartesian,
    restraints: PackedRestraints,
    params: Optional[Any] = None,
) -> dict[str, Any]:
    """Gauss-Newton second derivatives of the restraint energy at ``sites``.

    Returns ``diagonal`` (N, 3), ``blocks`` (N, 3, 3), ``gradient`` dE/dx (N, 3), the sparse matrix 2 J^T W J as COO
    triplets ``rows`` / ``cols`` / ``vals`` over the 3N cartesian coordinates (duplicates to
    be summed), ``energy`` and summary ``stats``. ``params`` JSON: ``{"sparse": bool}``
    (default True).
    """
    from phridge.worker.geometry.curvature import as_numpy, curvature_summary, gn_sparse_coo

    params = dict(params or {})
    curv = _curvature(restraints)
    xyz = _sites_tensor(sites, restraints)
    import torch

    blocks = curv.gn_blocks(xyz)
    diag = blocks.diagonal(dim1=1, dim2=2).contiguous()
    xr = xyz.clone().requires_grad_(True)
    e = curv.energy_from_residuals(xr)
    (g,) = torch.autograd.grad(e, xr)
    out: dict[str, Any] = {
        "diagonal": as_numpy(diag),
        "blocks": as_numpy(blocks),
        "gradient": as_numpy(g),
        "energy": float(e.detach().cpu()),
        "stats": curvature_summary(diag),
    }
    if bool(params.get("sparse", True)):
        rows, cols, vals, n3 = gn_sparse_coo(curv, xyz)
        out.update(rows=rows.astype(np.int64), cols=cols.astype(np.int64), vals=vals)
        out["stats"]["nnz_coo"] = int(vals.size)
        out["stats"]["n3"] = int(n3)
    else:
        out.update(rows=np.zeros(0, np.int64), cols=np.zeros(0, np.int64), vals=np.zeros(0))
    return out


geometry_curvature.compute_dtype = "float64"


def geometry_hvp(
    sites: PackedCartesian,
    restraints: PackedRestraints,
    v: Any,
    params: Optional[Any] = None,
) -> dict[str, Any]:
    """Hessian-vector product of the restraint energy: ``params.hessian`` = "gn" (default) | "full"."""
    import torch

    from phridge.worker.geometry.curvature import as_numpy

    params = dict(params or {})
    curv = _curvature(restraints)
    xyz = _sites_tensor(sites, restraints)
    vt = torch.as_tensor(np.asarray(v, dtype=np.float64).reshape(xyz.shape), dtype=xyz.dtype, device=xyz.device)
    hv = curv.hvp(xyz, vt, str(params.get("hessian", "gn")))
    return {"hv": as_numpy(hv)}


geometry_hvp.compute_dtype = "float64"


def geometry_gn_solve(
    sites: PackedCartesian,
    restraints: PackedRestraints,
    rhs: Any,
    params: Optional[Any] = None,
    extra_diag: Optional[Any] = None,
) -> dict[str, Any]:
    """Solve (w 2 J^T W J + diag(extra_diag) + mu I) p = rhs with a sparse direct factorisation.

    ``rhs`` and ``extra_diag`` are (N, 3) or (3N,) cartesian arrays for sites, or (M,)
    for ADP parameters, or (3N + M,) for joint ("sites", "adp") blocks. ``params`` JSON:
    ``weight`` (geometry weight w, default 1), ``damping`` (mu relative to the median
    diagonal, default 1e-3), ``method`` = "sparse" (exact sparse factorisation, default) or
    "tridiagonal" (residue block-tridiagonal, needs ``groups``: atom-index lists in chain order).
    ``blocks`` = ["sites"] (default) | ["adp"] | ["sites", "adp"].
    """
    import torch
    from phridge.worker.geometry.adp import ADPPrior, ADPPriorTables
    from phridge.worker.geometry.curvature import BlockTridiagonalPreconditioner, GaussNewtonPreconditioner

    params = dict(params or {})
    blocks = params.get("blocks", None)
    block = params.get("block", None)
    weight = float(params.get("weight", 1.0))
    damping = float(params.get("damping", 1e-3))
    method = str(params.get("method", "sparse"))
    groups = params.get("groups", None)

    xyz = _sites_tensor(sites, restraints)
    n3 = 3 * int(xyz.shape[0])

    # Case 1: ADP-only solve
    if block == "adp" or blocks == ["adp"]:
        tables = ADPPriorTables.from_packed(restraints, sites=xyz, device=_DEVICE["device"])
        prior = ADPPrior(tables)
        n_adp = prior.t.n_params
        adp_params = params.get("adp_params", None)
        if adp_params is not None:
            adp_vec = torch.as_tensor(np.asarray(adp_params, dtype=np.float64), dtype=xyz.dtype, device=xyz.device)
        else:
            adp_vec = torch.full((n_adp,), math.log(max(float(prior.t.wilson_b or 20.0), 1.0)), dtype=xyz.dtype, device=xyz.device)

        coo_adp = prior.gn_sparse_coo(adp_vec, sites=xyz)
        ex_adp = None if extra_diag is None else np.asarray(extra_diag, dtype=np.float64).reshape(-1)
        kw_adp = dict(extra_diag=ex_adp, damping=damping, weight=weight, coo=coo_adp)

        if method == "sparse":
            M_adp: Any = GaussNewtonPreconditioner(prior, adp_vec, **kw_adp)
            stats = {"method": method, "nnz": M_adp.nnz, "mu": M_adp.mu, "ref_diagonal": M_adp.ref, "n_params": n_adp}
        elif method == "tridiagonal":
            if not groups:
                raise ValueError("method='tridiagonal' needs params.groups")
            param_dims = prior.t.param_dims.cpu().numpy()
            M_adp = BlockTridiagonalPreconditioner(
                prior, adp_vec, [np.asarray(g, dtype=np.int64) for g in groups], param_dims=param_dims, **kw_adp
            )
            stats = {"method": method, "nnz": M_adp.nnz_kept, "nnz_total": M_adp.nnz_total, "mu": M_adp.mu,
                     "ref_diagonal": M_adp.ref, "n_params": n_adp, "n_groups": len(groups)}
        else:
            raise ValueError("params.method must be 'sparse' or 'tridiagonal'")

        b = np.asarray(rhs, dtype=np.float64)
        shape = b.shape
        b2 = b.reshape(-1, n_adp) if b.size % n_adp == 0 else None
        if b2 is None:
            raise ValueError(f"rhs must have a multiple of {n_adp} entries, got {b.size}")
        sol = np.stack([M_adp.solve(row) for row in b2]).reshape(shape)
        return {"solution": sol, "stats": stats}

    # Case 2: Joint solve (sites + adp)
    if blocks == ["sites", "adp"] or blocks == ["adp", "sites"]:
        curv = _curvature(restraints)
        tables = ADPPriorTables.from_packed(restraints, sites=xyz, device=_DEVICE["device"])
        prior = ADPPrior(tables)
        n_adp = prior.t.n_params
        total_dim = n3 + n_adp

        adp_params = params.get("adp_params", None)
        if adp_params is not None:
            adp_vec = torch.as_tensor(np.asarray(adp_params, dtype=np.float64), dtype=xyz.dtype, device=xyz.device)
        else:
            adp_vec = torch.full((n_adp,), math.log(max(float(prior.t.wilson_b or 20.0), 1.0)), dtype=xyz.dtype, device=xyz.device)

        coo_adp = prior.gn_sparse_coo(adp_vec, sites=xyz)

        ex_sites = None
        ex_adp = None
        if extra_diag is not None:
            ex_all = np.asarray(extra_diag, dtype=np.float64).reshape(-1)
            if ex_all.shape[0] != total_dim:
                raise ValueError(f"extra_diag for joint solve must have {total_dim} entries, got {ex_all.shape[0]}")
            ex_sites = ex_all[:n3]
            ex_adp = ex_all[n3:]

        kw_sites = dict(extra_diag=ex_sites, damping=damping, weight=weight)
        kw_adp = dict(extra_diag=ex_adp, damping=damping, weight=weight, coo=coo_adp)

        if method == "sparse":
            M_sites: Any = GaussNewtonPreconditioner(curv, xyz, **kw_sites)
            M_adp = GaussNewtonPreconditioner(prior, adp_vec, **kw_adp)
            stats = {
                "method": method,
                "nnz_sites": M_sites.nnz,
                "nnz_adp": M_adp.nnz,
                "mu_sites": M_sites.mu,
                "mu_adp": M_adp.mu,
                "n_sites_params": n3,
                "n_adp_params": n_adp,
            }
        elif method == "tridiagonal":
            if not groups:
                raise ValueError("method='tridiagonal' needs params.groups")
            grp_arr = [np.asarray(g, dtype=np.int64) for g in groups]
            M_sites = BlockTridiagonalPreconditioner(curv, xyz, grp_arr, **kw_sites)
            param_dims = prior.t.param_dims.cpu().numpy()
            M_adp = BlockTridiagonalPreconditioner(prior, adp_vec, grp_arr, param_dims=param_dims, **kw_adp)
            stats = {
                "method": method,
                "nnz_sites": M_sites.nnz_kept,
                "nnz_adp": M_adp.nnz_kept,
                "mu_sites": M_sites.mu,
                "mu_adp": M_adp.mu,
                "n_sites_params": n3,
                "n_adp_params": n_adp,
                "n_groups": len(groups),
            }
        else:
            raise ValueError("params.method must be 'sparse' or 'tridiagonal'")

        b = np.asarray(rhs, dtype=np.float64)
        shape = b.shape
        b2 = b.reshape(-1, total_dim) if b.size % total_dim == 0 else None
        if b2 is None:
            raise ValueError(f"rhs must have a multiple of {total_dim} entries, got {b.size}")

        def solve_row(r_row: np.ndarray) -> np.ndarray:
            r_s = r_row[:n3]
            r_a = r_row[n3:]
            p_s = M_sites.solve(r_s)
            p_a = M_adp.solve(r_a)
            return np.concatenate([p_s, p_a])

        sol = np.stack([solve_row(row) for row in b2]).reshape(shape)
        return {"solution": sol, "stats": stats}

    # Case 3: Default sites-only solve
    curv = _curvature(restraints)
    kw = dict(
        extra_diag=None if extra_diag is None else np.asarray(extra_diag, dtype=np.float64).reshape(-1),
        damping=damping,
        weight=weight,
    )
    if method == "sparse":
        M = GaussNewtonPreconditioner(curv, xyz, **kw)
        stats = {"method": method, "nnz": M.nnz, "mu": M.mu, "ref_diagonal": M.ref, "n3": n3}
    elif method == "tridiagonal":
        if not groups:
            raise ValueError("method='tridiagonal' needs params.groups (list of atom-index lists in chain order)")
        M = BlockTridiagonalPreconditioner(curv, xyz, [np.asarray(g, dtype=np.int64) for g in groups], **kw)
        stats = {"method": method, "nnz": M.nnz_kept, "nnz_total": M.nnz_total, "mu": M.mu, "ref_diagonal": M.ref,
                 "n3": n3, "n_groups": len(groups)}
    else:
        raise ValueError("params.method must be 'sparse' or 'tridiagonal'")
    b = np.asarray(rhs, dtype=np.float64)
    shape = b.shape
    b2 = b.reshape(-1, n3) if b.size % n3 == 0 else None
    if b2 is None:
        raise ValueError(f"rhs must have a multiple of {n3} entries, got {b.size}")
    sol = np.stack([M.solve(row) for row in b2]).reshape(shape)
    return {"solution": sol, "stats": stats}


geometry_gn_solve.compute_dtype = "float64"


def adp_prior_eval(
    sites: PackedCartesian,
    adp_params: Any,
    restraints: PackedRestraints,
    params: Optional[Any] = None,
) -> dict[str, Any]:
    """Evaluate ADP prior energy, gradient, GN diagonal, and sparse COO triplets."""
    import torch
    from phridge.worker.geometry.adp import ADPPrior, ADPPriorTables
    from phridge.worker.geometry.curvature import as_numpy

    params = dict(params or {})
    xyz = _sites_tensor(sites, restraints)
    tables = ADPPriorTables.from_packed(restraints, sites=xyz, device=_DEVICE["device"])
    prior = ADPPrior(tables)

    adp_vec = torch.as_tensor(np.asarray(adp_params, dtype=np.float64), dtype=xyz.dtype, device=xyz.device)
    x = adp_vec.detach().clone().requires_grad_(True)
    energy = prior.energy_vec(x, sites=xyz)
    (grad,) = torch.autograd.grad(energy, x)
    diag = prior.gn_diagonal(x, sites=xyz)
    rows, cols, vals, m = prior.gn_sparse_coo(x, sites=xyz)

    return {
        "energy": float(energy.item()),
        "gradient": as_numpy(grad),
        "diagonal": as_numpy(diag),
        "gn_diagonal": as_numpy(diag),
        "rows": rows,
        "cols": cols,
        "vals": vals,
        "stats": {"n_params": m, "energy": float(energy.item())},
    }


adp_prior_eval.compute_dtype = "float64"


def adp_prior_hvp(
    sites: PackedCartesian,
    adp_params: Any,
    restraints: PackedRestraints,
    v: Any,
    params: Optional[Any] = None,
) -> dict[str, Any]:
    """Hessian-vector product of the ADP prior energy: params.hessian = 'gn' (default) | 'full'."""
    import torch
    from phridge.worker.geometry.adp import ADPPrior, ADPPriorTables
    from phridge.worker.geometry.curvature import as_numpy

    params = dict(params or {})
    xyz = _sites_tensor(sites, restraints)
    tables = ADPPriorTables.from_packed(restraints, sites=xyz, device=_DEVICE["device"])
    prior = ADPPrior(tables)

    adp_vec = torch.as_tensor(np.asarray(adp_params, dtype=np.float64), dtype=xyz.dtype, device=xyz.device)
    vt = torch.as_tensor(np.asarray(v, dtype=np.float64).reshape(adp_vec.shape), dtype=xyz.dtype, device=xyz.device)
    hv = prior.hvp(adp_vec, vt, sites=xyz, hessian=str(params.get("hessian", "gn")))
    return {"hv": as_numpy(hv)}


adp_prior_hvp.compute_dtype = "float64"
