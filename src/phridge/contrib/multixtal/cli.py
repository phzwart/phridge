"""``phridge-multixtal`` command-line entry point. Torch-free on this side."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional, Sequence

from phridge.contrib.multixtal.client import job_to_op_kwargs, load_manifest, prepare_job
from phridge.contrib.multixtal.op import OP_NAME, register_ops
from phridge.contrib.multixtal.options import MultixtalOptions
from phridge.contrib.multixtal.report import write_reports


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="phridge-multixtal", description="Multi-dataset covariance model (phase 1).")
    p.add_argument("--model", required=True, help="PDB with CRYST1")
    p.add_argument("--data", nargs="*", default=[], help="MTZ files (Bijvoet mates kept separate)")
    p.add_argument("--manifest", help="YAML/JSON list of files, labels, wavelength, notes")
    p.add_argument("--labels", default="I(+),SIGI(+),I(-),SIGI(-)")
    p.add_argument("--d-min", type=float, default=2.0)
    p.add_argument("--strong-isig", type=float, default=5.0)
    p.add_argument("--strong-min-datasets", type=float, default=0.5)
    p.add_argument("--rank", default="auto")
    p.add_argument("--rank-perms", type=int, default=50)
    p.add_argument("--anomalous", default="on")
    p.add_argument("--placement", choices=("fractional", "cartesian"), default="fractional")
    p.add_argument("--anom-sites", help="comma-separated 0-based scatterer indices")
    p.add_argument("--space-group", dest="space_group")
    p.add_argument("--cell", nargs=6, type=float)
    p.add_argument("--out", required=True)
    p.add_argument("--memory", action="store_true", help="In-process Bridge (no Redis)")
    p.add_argument("--unmerged", nargs="*", default=[], help="Unmerged AIMLESS/DIALS files (aligned with --data)")
    p.add_argument("--dose-table", dest="dose_table", help="CSV dataset,batch,dose_MGy")
    p.add_argument("--dose-rate", dest="dose_rate", type=float, default=0.05)
    p.add_argument("--dose-model", dest="dose_model", default="linear", choices=("linear", "exp", "learned"))
    p.add_argument("--dose-bins", dest="dose_bins", type=int, default=10)
    p.add_argument("--absorption-order", dest="absorption_order", type=int, default=4)
    p.add_argument("--damage-rank", dest="damage_rank", default="auto")
    return p


def options_from_args(args: argparse.Namespace) -> MultixtalOptions:
    rank: Any = args.rank
    if str(rank).strip().lower() != "auto":
        rank = int(rank)
    anom_sites = None
    if args.anom_sites:
        anom_sites = [int(x) for x in str(args.anom_sites).split(",") if x.strip()]
    damage_rank: Any = getattr(args, "damage_rank", "auto")
    if str(damage_rank).strip().lower() != "auto":
        damage_rank = int(damage_rank)
    return MultixtalOptions(
        d_min=float(args.d_min),
        strong_isig=float(args.strong_isig),
        strong_min_datasets=float(args.strong_min_datasets),
        rank=rank,
        rank_perms=int(args.rank_perms),
        anomalous=args.anomalous,
        placement=args.placement,
        anom_sites=anom_sites,
        labels=str(args.labels),
        space_group=args.space_group,
        cell=list(args.cell) if args.cell is not None else None,
        dose_model=getattr(args, "dose_model", "linear"),
        dose_bins=int(getattr(args, "dose_bins", 10)),
        dose_rate=float(getattr(args, "dose_rate", 0.05)),
        absorption_order=int(getattr(args, "absorption_order", 4)),
        damage_rank=damage_rank,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    register_ops()
    opts = options_from_args(args)
    manifest = load_manifest(args.manifest) if args.manifest else None
    if getattr(args, "unmerged", None) and manifest is None:
        from phridge.contrib.multixtal.options import DatasetManifest, DatasetSpec

        data = list(args.data) or [""] * len(args.unmerged)
        specs = []
        for i, path in enumerate(data):
            unmerged = args.unmerged[i] if i < len(args.unmerged) else None
            specs.append(
                DatasetSpec(
                    file=str(path or unmerged or f"dataset{i}"),
                    labels=opts.labels,
                    name=Path(path or unmerged).stem,
                    unmerged=unmerged,
                    dose_table=getattr(args, "dose_table", None),
                    dose_rate=getattr(args, "dose_rate", None),
                )
            )
        manifest = DatasetManifest(datasets=specs)
    elif getattr(args, "unmerged", None) and manifest is not None:
        for i, spec in enumerate(manifest.datasets):
            if i < len(args.unmerged) and not spec.unmerged:
                spec.unmerged = args.unmerged[i]
            if getattr(args, "dose_table", None) and not spec.dose_table:
                spec.dose_table = args.dose_table
    job = prepare_job(args.model, args.data, opts, manifest=manifest)
    from phridge.client import Bridge

    kwargs = job_to_op_kwargs(job)
    bridge = Bridge(memory=True) if args.memory else Bridge()
    raw = bridge.call(OP_NAME, **kwargs)
    paths = write_reports(Path(args.out), job, raw)
    return {"job": job, "raw": raw, "paths": paths}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
