"""``phridge-ensemble`` command line. Torch is imported only inside the worker op."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from phridge.tools.ensemble.options import EnsembleOptions
from phridge.tools.ensemble.ops import OP_NAME_FIT, OP_NAME_MAPS, OP_NAME_SCORES, register_ops
from phridge.tools.ensemble.report import format_report, write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="phridge-ensemble", description="Multi-crystal perturbation modes.")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Read a PDB campaign and write a job npz.")
    _add_io(prepare)
    prepare.add_argument("--out", required=True)

    fit = sub.add_parser("fit", help="Fit shared modes.")
    _add_io(fit)
    fit.add_argument("--out", required=True)
    fit.add_argument("--memory", action="store_true")
    fit.add_argument("--k", type=int, default=1)
    fit.add_argument("--sweeps", type=int, default=4)
    fit.add_argument("--likelihood", choices=("known_f", "intensity"), default="known_f")

    scores = sub.add_parser("scores", help="Score datasets against fixed modes.")
    scores.add_argument("--job", required=True, help="npz written by fit")
    scores.add_argument("--out", required=True)
    scores.add_argument("--memory", action="store_true")

    maps = sub.add_parser("maps", help="Pack Δ0 as a Miller array.")
    maps.add_argument("--job", required=True)
    maps.add_argument("--out", required=True)
    maps.add_argument("--memory", action="store_true")

    hold = sub.add_parser("holdout", help="K scan on a prepared or fitted npz.")
    hold.add_argument("--job", required=True)
    hold.add_argument("--out", required=True)
    hold.add_argument("--k", nargs="+", type=int, default=[0, 1])
    return parser


def _add_io(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--d-min", type=float, default=3.0)
    parser.add_argument("--labels", default="I(+),SIGI(+),I(-),SIGI(-)")


def _options_from_fit(args: argparse.Namespace) -> EnsembleOptions:
    return EnsembleOptions(
        d_min=float(args.d_min),
        labels=args.labels,
        k=int(args.k),
        n_sweeps=int(args.sweeps),
        likelihood=args.likelihood,
        fit_nuisance=False,
    )


def run(args: argparse.Namespace) -> dict:
    import numpy as np

    register_ops()
    if args.command == "prepare":
        from phridge.tools.ensemble.client import prepare_ensemble_job

        job = prepare_ensemble_job(args.model, args.data, EnsembleOptions(d_min=args.d_min, labels=args.labels))
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            hkl=job.hkl,
            centric=job.centric,
            epsilon=job.epsilon,
            cells=job.cells,
            i_obs=job.i_obs,
            sig_i=job.sig_i,
            mask=job.mask,
        )
        return {"out": str(path), "n_data": int(job.i_obs.shape[0])}
    if args.command == "fit":
        from phridge.client import Bridge
        from phridge.tools.ensemble.client import job_to_op_kwargs, prepare_ensemble_job

        options = _options_from_fit(args)
        job = prepare_ensemble_job(args.model, args.data, options)
        bridge = Bridge(memory=True) if args.memory else Bridge()
        raw = bridge.call(OP_NAME_FIT, **job_to_op_kwargs(job))
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        np.savez(
            out / "fit.npz",
            hkl=job.hkl,
            cells=job.cells,
            delta0=raw["delta0"],
            modes=raw["modes"],
            scores=raw["scores"],
            score_cov=raw["score_cov"],
            sigma2=raw["sigma2"],
            d=raw["d"],
            shell_index=raw["shell_index"],
            fom=raw["fom"],
            i_obs=job.i_obs,
            sig_i=job.sig_i,
            mask=job.mask,
            epsilon=job.epsilon,
            centric=job.centric,
            loglik_history=raw["loglik_history"],
        )
        text = format_report(raw, k_scan=(raw.get("heldout") or {}).get("k_scan"), half_split=(raw.get("heldout") or {}).get("half_split"))
        write_report(out / "report.txt", text)
        (out / "stats.json").write_text(json.dumps(raw["stats"], indent=2))
        return {"out": str(out), "stats": raw["stats"]}
    if args.command == "scores":
        from phridge.client import Bridge

        blob = np.load(args.job)
        bridge = Bridge(memory=True) if args.memory else Bridge()
        raw = bridge.call(
            OP_NAME_SCORES,
            i_obs=blob["i_obs"],
            sig_i=blob["sig_i"],
            mask=blob["mask"],
            f_mu=blob["delta0"] * 0.0,
            delta0=blob["delta0"],
            modes=blob["modes"],
            d=blob["d"],
            sigma2=blob["sigma2"],
            shell_index=blob["shell_index"],
            epsilon=blob["epsilon"],
            centric=blob["centric"],
            options={},
        )
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, scores=raw["scores"], score_cov=raw["score_cov"], fom=raw["fom"])
        return {"out": str(path), "loglik": raw["loglik"]}
    if args.command == "maps":
        from phridge.client import Bridge

        blob = np.load(args.job)
        bridge = Bridge(memory=True) if args.memory else Bridge()
        raw = bridge.call(OP_NAME_MAPS, hkl=blob["hkl"], delta0=blob["delta0"], modes=blob["modes"], cells=blob["cells"])
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, hkl=raw["delta0"].hkl, delta0=raw["delta0"].data, modes=raw["modes"])
        return {"out": str(path)}
    if args.command == "holdout":
        from phridge.legacy.ensemble_fit import known_f_target, phased_observed
        from phridge.tools.ensemble.holdout import k_scan

        blob = np.load(args.job)
        phase = np.zeros(blob["i_obs"].shape[1])
        if "delta0" in blob.files:
            phase = np.angle(blob["delta0"])
        f_obs = phased_observed(blob["i_obs"], np.broadcast_to(phase, blob["i_obs"].shape), blob["mask"])
        target = known_f_target(f_obs, np.zeros(blob["i_obs"].shape[1], dtype=np.complex128), None, np.ones(blob["i_obs"].shape))
        scanned = k_scan(target, blob["mask"].astype(np.float64), args.k, n_folds=2, n_sweeps=2)
        text = format_report({"stats": {"likelihood": "known_f", "k": args.k[-1]}, "loglik_history": [0.0]}, k_scan=scanned)
        path = write_report(args.out, text)
        return {"out": str(path)}
    raise SystemExit(f"unknown command {args.command}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
