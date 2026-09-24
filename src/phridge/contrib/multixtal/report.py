"""Write report.json/.md, CSVs, maps.mtz, run_config.yaml. Torch-free."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from phridge.contrib.multixtal.client import MultixtalJob
from phridge.contrib.multixtal.options import MultixtalOptions


def write_run_config(out_dir: Path, options: MultixtalOptions, extra: Optional[dict[str, Any]] = None) -> Path:
    path = out_dir / "run_config.yaml"
    payload = options.model_dump()
    if extra:
        payload.update(extra)
    path.write_text(_dump_yaml(payload))
    return path


def _dump_yaml(data: Any, indent: int = 0) -> str:
    pad = "  " * indent
    if isinstance(data, dict):
        lines = []
        for key, val in data.items():
            if isinstance(val, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.append(_dump_yaml(val, indent + 1))
            else:
                lines.append(f"{pad}{key}: {_yaml_scalar(val)}")
        return "\n".join(lines) + ("\n" if indent == 0 else "")
    if isinstance(data, list):
        lines = []
        for item in data:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.append(_dump_yaml(item, indent + 1))
            else:
                lines.append(f"{pad}- {_yaml_scalar(item)}")
        return "\n".join(lines)
    return f"{pad}{_yaml_scalar(data)}"


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value) if isinstance(value, float) else str(value)
    return json.dumps(str(value))


def write_reports(
    out_dir: Path,
    job: MultixtalJob,
    raw: dict[str, Any],
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = dict(raw.get("stats") or {})
    names = [d.name for d in job.datasets]
    report = {
        "inputs": {
            "n_datasets": len(job.datasets),
            "placement": job.placement,
            "d_min": job.options.d_min,
            "datasets": [
                {
                    "name": d.name,
                    "file": d.path,
                    "cell": list(d.cell),
                    "indexing": {
                        "operator": d.indexing.operator,
                        "correlation": d.indexing.correlation,
                        "margin": d.indexing.margin,
                    },
                    "notes": d.notes,
                }
                for d in job.datasets
            ],
            "logs": job.logs,
        },
        "strong_set": {
            "n_strong": stats.get("n_strong"),
            "n_reflections": stats.get("n_reflections"),
            "per_shell": stats.get("strong_per_shell"),
        },
        "rank": {
            "rank": stats.get("rank"),
            "singular_values": stats.get("singular_values"),
            "null_top": stats.get("null_top"),
            "scale_leakage_slopes": stats.get("scale_leakage_slopes"),
            "scale_leakage_significant": stats.get("scale_leakage_significant"),
            "delta_beta": stats.get("delta_beta"),
        },
        "explanation": {
            "chi2_model": _list(raw.get("chi2_model")),
            "chi2_mean": _list(raw.get("chi2_mean")),
            "chi2_in_sample": _list(raw.get("chi2_in_sample")),
            "chi2_loo": _list(raw.get("chi2_loo")),
            "flagged": _list(raw.get("flagged")),
            "f_explained": _list(raw.get("f_explained")),
        },
        "nuisance": {
            "k_sol": _list(raw.get("k_sol")),
            "b_sol": _list(raw.get("b_sol")),
        },
        "anomalous": {
            "q": _list(raw.get("q")),
            "q_se": _list(raw.get("q_se")),
        },
        "influence": raw.get("influence"),
        "damage": stats.get("damage"),
        "anom_damage": stats.get("anom_damage"),
        "dose_influence": raw.get("dose_influence"),
        "damage_skipped": stats.get("damage_skipped"),
    }
    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(report, indent=2, default=_json_default))
    md_path = out_dir / "report.md"
    md_path.write_text(_markdown(report, names))

    scores_path = out_dir / "scores.csv"
    scores = np.asarray(raw.get("scores"), dtype=np.float64)
    _write_csv(scores_path, ["dataset"] + [f"z{m+1}" for m in range(scores.shape[1] if scores.ndim == 2 else 0)],
               [[names[i] if i < len(names) else str(i), *scores[i].tolist()] for i in range(scores.shape[0])] if scores.size else [])

    per_path = out_dir / "per_dataset.csv"
    _write_csv(
        per_path,
        ["dataset", "k_sol", "b_sol", "chi2_model", "chi2_mean", "chi2_in_sample", "chi2_loo", "flagged", "f_explained", "q", "q_se"],
        [
            [
                names[i] if i < len(names) else str(i),
                _at(raw.get("k_sol"), i),
                _at(raw.get("b_sol"), i),
                _at(raw.get("chi2_model"), i),
                _at(raw.get("chi2_mean"), i),
                _at(raw.get("chi2_in_sample"), i),
                _at(raw.get("chi2_loo"), i),
                _at(raw.get("flagged"), i),
                _at(raw.get("f_explained"), i),
                _at(raw.get("q"), i),
                _at(raw.get("q_se"), i),
            ]
            for i in range(len(names) or 1)
        ],
    )

    inf_path = out_dir / "influence.csv"
    inf_rows: list[list[Any]] = []
    for block in raw.get("influence") or []:
        shares = np.asarray(block.get("shares"), dtype=np.float64)
        cooks = np.asarray(block.get("cooks"), dtype=np.float64)
        cons = np.asarray(block.get("consistency"), dtype=np.float64)
        n = shares.shape[0] if shares.ndim >= 1 else 0
        for d in range(n):
            share = shares[d]
            if np.ndim(share) == 0:
                share_v = float(share)
            else:
                share_v = float(np.mean(share))
            inf_rows.append(
                [
                    names[d] if d < len(names) else str(d),
                    block.get("quantity"),
                    "all",
                    share_v,
                    float(cooks[d]) if d < cooks.size else "",
                    float(cons[d]) if d < cons.size else "",
                    block.get("effective_n"),
                ]
            )
    _write_csv(inf_path, ["dataset", "quantity", "shell", "information_share", "cooks_distance", "consistency", "effective_n"], inf_rows)

    dmg_path = out_dir / "damage.csv"
    ell = np.asarray(raw.get("damage_loadings"), dtype=np.float64)
    dmg_rows: list[list[Any]] = []
    if ell.ndim == 2:
        for h in range(ell.shape[1]):
            dmg_rows.append([h, *[float(ell[k, h]) for k in range(ell.shape[0])]])
    _write_csv(dmg_path, ["h", *[f"ell{k+1}" for k in range(ell.shape[0] if ell.ndim == 2 else 0)]], dmg_rows)

    maps_path = out_dir / "maps.mtz"
    try:
        _write_maps_mtz(maps_path, job, raw)
    except Exception:
        maps_path.write_text("# maps.mtz not written (cctbx unavailable or empty coefficients)\n")

    cfg = write_run_config(
        out_dir,
        job.options,
        extra={"data": [d.path for d in job.datasets], "placement": job.placement, "logs": job.logs},
    )
    return {
        "report.json": json_path,
        "report.md": md_path,
        "scores.csv": scores_path,
        "per_dataset.csv": per_path,
        "influence.csv": inf_path,
        "maps.mtz": maps_path,
        "run_config.yaml": cfg,
        "damage.csv": dmg_path,
    }


def _markdown(report: dict[str, Any], names: list[str]) -> str:
    lines = ["# multixtal report", ""]
    lines.append(f"Datasets: {len(names)}")
    lines.append(f"Placement: {report['inputs']['placement']}")
    lines.append(f"Strong set: {report['strong_set']['n_strong']} / {report['strong_set']['n_reflections']}")
    lines.append(f"Rank: {report['rank']['rank']}")
    if report.get("damage_skipped"):
        lines.append("Damage: skipped (no unmerged observations or D = 0)")
    elif report.get("damage"):
        dmg = report["damage"]
        lines.append(f"Damage rank: {dmg.get('rank')}  attribution: {dmg.get('attribution')}")
        lines.append(f"Physical modes: {dmg.get('physical')}")
        lines.append(f"Smoothness: {dmg.get('smoothness')}")
    lines.append("")
    lines.append("## Indexing")
    for d in report["inputs"]["datasets"]:
        idx = d["indexing"]
        lines.append(f"- {d['name']}: `{idx['operator']}` corr={idx['correlation']} margin={idx['margin']}")
    lines.append("")
    lines.append("## Explanation (chi²/dof)")
    lines.append("| dataset | model | mean | in-sample | leave-one-out | flagged | f |")
    lines.append("|---|---|---|---|---|---|---|")
    expl = report["explanation"]
    for i, name in enumerate(names):
        lines.append(
            "| {name} | {m:.3f} | {mu:.3f} | {ins:.3f} | {loo:.3f} | {fl} | {f:.3f} |".format(
                name=name,
                m=_safe(expl["chi2_model"], i),
                mu=_safe(expl["chi2_mean"], i),
                ins=_safe(expl["chi2_in_sample"], i),
                loo=_safe(expl["chi2_loo"], i),
                fl=_safe(expl["flagged"], i),
                f=_safe(expl["f_explained"], i),
            )
        )
    return "\n".join(lines) + "\n"


def _write_maps_mtz(path: Path, job: MultixtalJob, raw: dict[str, Any]) -> None:
    from cctbx import miller
    from cctbx.array_family import flex
    from iotbx import mtz
    from phridge.client.convert import crystal_to_cctbx

    crystal = crystal_to_cctbx(job.xray.meta.crystal)
    indices = flex.miller_index([tuple(int(x) for x in row) for row in job.hkl])
    miller_set = miller.set(crystal, indices, anomalous_flag=False)

    def _complex_array(data: Any, label: str) -> Any:
        arr = np.asarray(data, dtype=np.complex128).reshape(-1)
        return miller.array(miller_set=miller_set, data=flex.complex_double(arr)).set_info(
            miller.array_info(labels=[label])
        )

    mtz_obj = mtz.object()
    mtz_obj = mtz_obj.set_title("phridge multixtal map coefficients")
    mtz_obj.add_history("mean anomalous: Bijvoet intercept, phases of iS, FOM from sigma_A")
    mtz_obj.add_history("factor ordinary: in-phase component only (loadings along S)")
    mtz_obj.add_history("damage loadings: in-phase only (ell_k along S, FOM from sigma_A)")
    first = _complex_array(raw.get("mean_anomalous"), "ANOM_MEAN")
    mtz_ds = first.as_mtz_dataset(column_root_label="ANOM_MEAN")
    slopes = np.asarray(raw.get("factor_anomalous"))
    if slopes.ndim == 2:
        for m in range(slopes.shape[0]):
            mtz_ds.add_miller_array(_complex_array(slopes[m], f"ANOM_Z{m+1}"), column_root_label=f"ANOM_Z{m+1}")
    ordinary = np.asarray(raw.get("factor_ordinary"))
    if ordinary.ndim == 2:
        for m in range(ordinary.shape[0]):
            mtz_ds.add_miller_array(_complex_array(ordinary[m], f"MODE_Z{m+1}"), column_root_label=f"MODE_Z{m+1}")
    dmg = np.asarray(raw.get("damage_maps"))
    if dmg.ndim == 2 and dmg.size:
        for k in range(dmg.shape[0]):
            mtz_ds.add_miller_array(_complex_array(dmg[k], f"DAM_{k+1}"), column_root_label=f"DAM_{k+1}")
    corr = raw.get("anom_corrected")
    if corr is not None and np.asarray(corr).size:
        mtz_ds.add_miller_array(_complex_array(corr, "ANOM_CORR"), column_root_label="ANOM_CORR")
    mtz_ds.mtz_object().write(str(path))


def _write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    arr = np.asarray(value)
    return arr.tolist()


def _at(value: Any, i: int) -> Any:
    if value is None:
        return ""
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    if i < arr.shape[0]:
        return arr[i].item() if np.ndim(arr[i]) == 0 else arr[i]
    return ""


def _safe(seq: list[Any], i: int) -> float:
    if i >= len(seq):
        return float("nan")
    try:
        return float(seq[i])
    except (TypeError, ValueError):
        return float("nan")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)
