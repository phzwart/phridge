"""Accuracy and timing: NUFFT engine vs stamp engine vs cctbx.

Synthetic space groups (cctbx ``random_structure``, C/N/O/S)::

    p1, p212121, r3, f423

``r3`` is hexagonal R 3 :H (SG 146). ``f423`` is HM ``F 4 3 2`` / Hall
``F 4 2 3`` (SG 209) — cctbx does not accept the compact symbol ``F423``.
``synthetic`` is an alias of ``p212121``.

Split: Phenix / cctbx client, torch worker over phridge (Redis)::

    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python -m phridge.worker.runner \\
        --device cpu --preload phridge.contrib.nufft_sf --consumer nufft-bench
    PYTHONPATH=src phenix.python -m phridge.contrib.nufft_sf.benchmark \\
        --redis-url redis://127.0.0.1:6379/1 --cases p1,p212121,r3,f423 \\
        --n-synthetic 1000 --d-min 2.0 --dtypes float32 --no-nufft
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

ROOT = Path(__file__).resolve().parents[4]

# cctbx random_structure symbols. R3 uses the hexagonal setting (R 3 :H).
_SYNTHETIC_SG = {
    "synthetic": "P212121",
    "synthetic_p212121": "P212121",
    "p212121": "P212121",
    "synthetic_p1": "P1",
    "p1": "P1",
    "synthetic_r3h": "R 3 :H",
    "synthetic_r3": "R 3 :H",
    "r3h": "R 3 :H",
    "r3": "R 3 :H",
    # HM is F 4 3 2 (SG 209); Hall is F 4 2 3 — cctbx rejects the compact "F423".
    "synthetic_f423": "F 4 3 2",
    "f423": "F 4 3 2",
}

_SYNTHETIC_TAG = {
    "synthetic": "p212121",
    "synthetic_p212121": "p212121",
    "p212121": "p212121",
    "synthetic_p1": "p1",
    "p1": "p1",
    "synthetic_r3h": "r3h",
    "synthetic_r3": "r3h",
    "r3h": "r3h",
    "r3": "r3h",
    "synthetic_f423": "f423",
    "f423": "f423",
}


def _is_synthetic(case: str) -> bool:
    return case in _SYNTHETIC_SG


def _synthetic_label(case: str, n_atoms: int) -> str:
    return f"synthetic_{_SYNTHETIC_TAG[case]}_{int(n_atoms)}"


@dataclass
class Row:
    engine: str
    case: str
    d_min: float
    device: str
    n_groups: int
    n_transforms: int
    r_vs_fft: float
    t_f: float
    t_f_grad: float
    peak_mem_mb: float
    n_atoms: int
    n_refl: int


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    elif device.startswith("mps"):
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.synchronize()


def _median(fn: Callable[[], Any], device: str, repeats: int, warmup: int = 1) -> tuple[Any, float]:
    for _ in range(max(0, warmup)):
        _sync(device)
        fn()
        _sync(device)
    times = []
    out = None
    for _ in range(max(1, repeats)):
        _sync(device)
        t0 = time.perf_counter()
        out = fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    return out, float(statistics.median(times))


def _peak_mb(device: str) -> float:
    if device.startswith("cuda"):
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024**2))
    try:
        import resource
        import sys

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux: KB; macOS: bytes
        return rss / (1024.0**2) if sys.platform == "darwin" else rss / 1024.0
    except Exception:
        return float("nan")


def _r_factor(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).sum() / max(np.abs(b).sum(), 1e-300))


def _stamp_engine(model, hkl, d_min: float, quality_factor: float, device: str):
    from phridge.sfcalc.engine.engine import EngineParams, StructureFactorEngine

    return StructureFactorEngine(
        model, hkl, EngineParams(d_min=d_min, quality_factor=quality_factor), device=device
    )


def _nufft_engine(model, hkl, d_min: float, tau: float, device: str):
    from phridge.sfcalc.engine.nufft_engine import NufftEngineParams, NufftStructureFactorEngine

    return NufftStructureFactorEngine(
        model, hkl, NufftEngineParams(d_min=d_min, tau=tau, n_max=2, eps=1e-6), device=device
    )


def _time_engine(eng, device: str, repeats: int, dtdf: np.ndarray) -> tuple[np.ndarray, float, float]:
    def _f():
        return eng.f_calc_numpy()

    f, t_f = _median(_f, device, repeats)

    def _fg():
        eng.gradients(dtdf)
        return None

    _, t_g = _median(_fg, device, repeats)
    return f, t_f, t_f + t_g


def _cctbx_available() -> bool:
    try:
        import cctbx  # noqa: F401

        return True
    except Exception:
        return False


def _load_cctbx_structure(case: str, n_synthetic: int):
    from cctbx import sgtbx
    from cctbx.development import random_structure
    from cctbx.array_family import flex

    if _is_synthetic(case):
        import random

        random.seed(0)
        flex.set_random_seed(0)
        elements = ("C", "N", "O", "S")
        n_repeat = max(1, int(round(int(n_synthetic) / len(elements))))
        return random_structure.xray_structure(
            space_group_info=sgtbx.space_group_info(_SYNTHETIC_SG[case]),
            elements=list(elements) * n_repeat,
            volume_per_atom=50,
            random_u_iso=True,
            random_occupancy=True,
        )
    pdb = ROOT / "examples" / case / f"{case}.pdb"
    if not pdb.exists():
        _download_pdb(case, pdb)
    if not pdb.exists():
        raise FileNotFoundError(pdb)
    import iotbx.pdb

    return iotbx.pdb.input(file_name=str(pdb)).xray_structure_simple()


def _download_pdb(code: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://files.rcsb.org/download/{code.upper()}.pdb"
    try:
        import urllib.request

        urllib.request.urlretrieve(url, dest)
    except Exception:
        return


def _model_from_cctbx(xs):
    from phridge.client.convert_xtal import scattering_table_from_cctbx, xray_from_cctbx
    from phridge.sfcalc.ops import scattering_model

    return scattering_model(xray_from_cctbx(xs), scattering_table_from_cctbx(xs))


def _toy_case(n_atoms: int = 200):
    from phridge.sfcalc.engine.engine import ScatteringModel
    from phridge.sfcalc.engine.symmetry import identity_ops

    rng = np.random.default_rng(0)
    rot, trans = identity_ops()
    n = int(n_atoms)
    return ScatteringModel(
        unit_cell=(40.0, 42.0, 45.0, 90.0, 90.0, 90.0),
        sites_frac=rng.random((n, 3)),
        occupancy=np.ones(n),
        u_iso=rng.uniform(0.02, 0.08, size=n),
        u_star=np.zeros((n, 6)),
        anisotropic=np.zeros(n, dtype=bool),
        fp=np.zeros(n),
        fdp=np.zeros(n),
        type_index=np.zeros(n, dtype=np.int64),
        gauss_a=np.array([[2.31, 1.02, 1.5886, 0.865]], dtype=np.float64),
        gauss_b=np.array([[20.8439, 10.2075, 0.5687, 51.6512]], dtype=np.float64),
        gauss_c=np.array([0.2156], dtype=np.float64),
        rot=rot,
        trans=trans,
        multiplicity=np.ones(n, dtype=np.int64),
    )


def _miller_from_model(model, d_min: float, max_index: int = 20) -> np.ndarray:
    from phridge.sfcalc.engine.cell import reciprocal_cartesian

    a, b, c = (float(x) for x in model.unit_cell[:3])
    hm = min(max_index, int(np.ceil(a / d_min) + 1))
    km = min(max_index, int(np.ceil(b / d_min) + 1))
    lm = min(max_index, int(np.ceil(c / d_min) + 1))
    hs = [
        (h, k, l)
        for h in range(-hm, hm + 1)
        for k in range(-km, km + 1)
        for l in range(-lm, lm + 1)
        if not (h == 0 and k == 0 and l == 0)
    ]
    hkl = np.array(hs, dtype=np.int64)
    dstar2 = np.sum(reciprocal_cartesian(model.unit_cell, hkl) ** 2, axis=1)
    return hkl[dstar2 <= (1.0 / d_min) ** 2 + 1e-12]


def run_case(
    case: str,
    d_min: float,
    device: str,
    repeats: int,
    n_synthetic: int,
    rng_seed: int = 0,
) -> list[Row]:
    rows: list[Row] = []
    cctbx_ok = _cctbx_available()
    xs = None
    if cctbx_ok and (case in ("1ee2", "6czg") or _is_synthetic(case)):
        try:
            xs = _load_cctbx_structure(case, n_synthetic)
            model = _model_from_cctbx(xs)
            fc = xs.structure_factors(d_min=d_min, algorithm="fft").f_calc()
            hkl = np.array(list(fc.indices()))
        except Exception as exc:
            print(f"skip cctbx case {case}: {exc}")
            xs = None
    if xs is None:
        if not _is_synthetic(case) and case != "synthetic":
            print(f"skip {case}: cctbx/PDB unavailable")
            return rows
        model = _toy_case(min(n_synthetic, 400))
        hkl = _miller_from_model(model, d_min)
        case = f"synthetic_p1_{model.n_scatterers}"
    elif _is_synthetic(case):
        case = _synthetic_label(case, xs.scatterers().size())

    rng = np.random.default_rng(rng_seed)
    dtdf = rng.normal(size=len(hkl)) + 1j * rng.normal(size=len(hkl))
    ref_fft: Optional[np.ndarray] = None
    if xs is not None:
        try:
            ref_fft = np.array(xs.structure_factors(d_min=d_min, algorithm="fft").f_calc().data())
        except Exception:
            ref_fft = None

    specs: list[tuple[str, Callable[[], Any]]] = [
        ("stamp_qf100", lambda: _stamp_engine(model, hkl, d_min, 100.0, device)),
        ("stamp_qf1000", lambda: _stamp_engine(model, hkl, d_min, 1000.0, device)),
        ("nufft_tau1e-3", lambda: _nufft_engine(model, hkl, d_min, 1e-3, device)),
        ("nufft_tau1e-4", lambda: _nufft_engine(model, hkl, d_min, 1e-4, device)),
    ]

    if device.startswith("cuda"):
        import torch

        torch.cuda.reset_peak_memory_stats()

    for name, factory in specs:
        try:
            eng = factory()
        except Exception as exc:
            print(f"skip {name} {case} d_min={d_min}: {exc}")
            continue
        n_groups = getattr(getattr(eng, "plan", None), "groups", None)
        n_groups_i = len(n_groups) if n_groups is not None else 0
        n_t = int(getattr(getattr(eng, "plan", None), "n_transforms", 0) or 0)
        f, t_f, t_fg = _time_engine(eng, device, repeats, dtdf)
        r = float("nan")
        if ref_fft is not None:
            r = _r_factor(f, ref_fft)
        rows.append(
            Row(
                engine=name,
                case=case,
                d_min=d_min,
                device=device,
                n_groups=n_groups_i,
                n_transforms=n_t,
                r_vs_fft=r,
                t_f=t_f,
                t_f_grad=t_fg,
                peak_mem_mb=_peak_mb(device),
                n_atoms=model.n_scatterers,
                n_refl=len(hkl),
            )
        )

    rows.extend(_cctbx_local_rows(xs, case, d_min, repeats, model.n_scatterers, len(hkl)))
    return rows


def _cctbx_local_rows(xs, case: str, d_min: float, repeats: int, n_atoms: int, n_refl: int) -> list[Row]:
    if xs is None:
        return []

    def _fft():
        return np.array(xs.structure_factors(d_min=d_min, algorithm="fft").f_calc().data())

    f_fft, t_fft = _median(_fft, "cpu", repeats)
    return [
        Row("cctbx_fft", case, d_min, "cpu", 0, 0, 0.0, t_fft, float("nan"), float("nan"), n_atoms, n_refl),
    ]


def _bridge_f_data(packed) -> np.ndarray:
    data = packed.data if hasattr(packed, "data") else packed
    return np.asarray(data)


def run_case_via_bridge(
    bridge: Any,
    case: str,
    d_min: float,
    device: str,
    repeats: int,
    n_synthetic: int,
    rng_seed: int = 0,
    dtypes: Optional[list[str]] = None,
    include_nufft: bool = True,
) -> list[Row]:
    """Phenix/cctbx client: local FFT, stamp + NUFFT on the torch worker."""
    from phridge.contrib.nufft_sf import register
    from phridge.models import SfEngineParams
    from phridge.packing import PackedMiller
    from phridge.sfcalc.client import _miller_template, _packed_xray

    register()
    rows: list[Row] = []
    if not _cctbx_available():
        print(f"skip {case}: cctbx required for --redis-url client")
        return rows
    try:
        xs = _load_cctbx_structure(case, n_synthetic)
        fc = xs.structure_factors(d_min=d_min, algorithm="fft").f_calc()
    except Exception as exc:
        print(f"skip cctbx case {case}: {exc}")
        return rows

    xray, table = _packed_xray(xs, None)
    hkl_tmpl = _miller_template(fc)
    n_atoms = int(xs.scatterers().size())
    n_refl = int(fc.size())
    if _is_synthetic(case):
        case = _synthetic_label(case, n_atoms)
    rng = np.random.default_rng(rng_seed)
    dtdf = PackedMiller(
        crystal=hkl_tmpl.meta.crystal,
        hkl=hkl_tmpl.hkl,
        data=rng.normal(size=n_refl) + 1j * rng.normal(size=n_refl),
        anomalous=bool(hkl_tmpl.meta.anomalous),
    )
    ref_fft = np.array(fc.data())

    specs: list[tuple[str, str, str, Any]] = []
    tags = {"float64": "f64", "float32": "f32", "float16": "f16", "fp64": "f64", "fp32": "f32", "fp16": "f16"}
    dtypes = dtypes or ["float64"]
    if str(device).startswith("mps"):
        dtypes = [d for d in dtypes if str(d).lower() not in ("float64", "fp64")] or ["float32"]
    for dt in dtypes:
        tag = tags.get(str(dt).lower(), str(dt))
        specs.extend(
            [
                (
                    f"stamp_qf1000_{tag}",
                    "sf_calc",
                    "sf_gradients",
                    SfEngineParams(d_min=d_min, quality_factor=1000.0, dtype=dt),
                ),
                (
                    f"stamp_cpp_qf1000_{tag}",
                    "sf_calc",
                    "sf_gradients",
                    SfEngineParams(d_min=d_min, quality_factor=1000.0, stamp_backend="cpp", dtype=dt),
                ),
                (
                    f"stamp_torch_qf1000_{tag}",
                    "sf_calc",
                    "sf_gradients",
                    SfEngineParams(d_min=d_min, quality_factor=1000.0, stamp_backend="torch", dtype=dt),
                ),
            ]
        )
    if include_nufft and not str(device).startswith("mps"):
        specs.extend(
            [
                (
                    "nufft_tau1e-3",
                    "nufft_sf_calc",
                    "nufft_sf_gradients",
                    {"engine": "nufft", "d_min": d_min, "tau": 1e-3, "n_max": 2, "eps": 1e-6},
                ),
                (
                    "nufft_tau1e-4",
                    "nufft_sf_calc",
                    "nufft_sf_gradients",
                    {"engine": "nufft", "d_min": d_min, "tau": 1e-4, "n_max": 2, "eps": 1e-6},
                ),
            ]
        )
    for name, calc_op, grad_op, params in specs:
        try:
            bind_op = "sf_bind" if calc_op == "sf_calc" else "nufft_sf_bind"
            handle = str(bridge.call(bind_op, xray=xray, table=table, hkl=hkl_tmpl, params=params))

            def _f(op=calc_op, hid=handle):
                return _bridge_f_data(bridge.call(op, handle=hid, hkl=hkl_tmpl))

            f, t_f = _median(_f, "cpu", repeats)

            def _fg(op=grad_op, hid=handle):
                bridge.call(op, handle=hid, d_target_d_f_calc=dtdf)
                return None

            _, t_g = _median(_fg, "cpu", repeats)
        except Exception as exc:
            print(f"skip {name} {case} d_min={d_min}: {exc}")
            continue
        r = _r_factor(f, ref_fft)
        rows.append(
            Row(
                engine=name,
                case=case,
                d_min=d_min,
                device=f"{device}+bridge",
                n_groups=0,
                n_transforms=0,
                r_vs_fft=r,
                t_f=t_f,
                t_f_grad=t_f + t_g,
                peak_mem_mb=float("nan"),
                n_atoms=n_atoms,
                n_refl=n_refl,
            )
        )
    rows.extend(_cctbx_local_rows(xs, case, d_min, repeats, n_atoms, n_refl))
    return rows


def format_table(rows: list[Row]) -> str:
    header = (
        "| engine | case | d_min | device | n_atoms | n_refl | n_groups | T | "
        "R(F) vs cctbx FFT | t(F) | t(F+grad) | peak mem (MB) |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        rf = "n/a" if not np.isfinite(r.r_vs_fft) else f"{r.r_vs_fft:.3e}"
        tg = "n/a" if not np.isfinite(r.t_f_grad) else f"{r.t_f_grad:.4f}"
        mem = "n/a" if not np.isfinite(r.peak_mem_mb) else f"{r.peak_mem_mb:.1f}"
        lines.append(
            f"| {r.engine} | {r.case} | {r.d_min:.1f} | {r.device} | {r.n_atoms} | {r.n_refl} | "
            f"{r.n_groups} | {r.n_transforms} | {rf} | {r.t_f:.4f} | {tg} | {mem} |"
        )
    return "\n".join(lines)


def write_docs(path: Path, table: str, commands: list[str]) -> None:
    text = path.read_text() if path.exists() else ""
    marker = "Results (fill after running the commands above):"
    block = (
        f"{marker}\n\n"
        + "\n".join(f"    {c}" for c in commands)
        + "\n\n"
        + table
        + "\n\nThe default worker engine is **not** changed in this PR. A go/no-go for\n"
        "making `engine: \"nufft\"` the CUDA default is a ≥5× `F+grad` speedup over\n"
        "the stamp engine at equal `R(F)` on the largest case.\n"
    )
    if marker in text:
        start = text.index(marker)
        path.write_text(text[:start] + block)
    else:
        path.write_text(text + "\n" + block)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="NUFFT vs stamp vs cctbx structure-factor benchmark")
    p.add_argument(
        "--cases",
        default="synthetic",
        help="comma list: 1ee2,6czg,p1,p212121,r3,f423 (aliases: synthetic=p212121, r3h, synthetic_*)",
    )
    p.add_argument("--d-min", default="2.0")
    p.add_argument("--device", default="cpu")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument(
        "--n-synthetic",
        default="1000,10000",
        help="comma list of ASU atom counts for the synthetic (cctbx random_structure) case",
    )
    p.add_argument(
        "--redis-url",
        default="",
        help="If set, run as a cctbx client: stamp/NUFFT via phridge, cctbx locally",
    )
    p.add_argument("--out", default="", help="markdown file to update (default: docs/nufft_engine.md)")
    p.add_argument(
        "--dtypes",
        default="float64",
        help="comma list of stamp dtypes: float64,float32,float16",
    )
    p.add_argument("--no-nufft", action="store_true", help="skip NUFFT rows")
    args = p.parse_args(argv)
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    dmins = [float(x) for x in args.d_min.split(",") if x.strip()]
    n_syns = [int(x) for x in str(args.n_synthetic).split(",") if x.strip()]
    dtypes = [x.strip() for x in str(args.dtypes).split(",") if x.strip()]
    bridge = None
    if args.redis_url:
        from phridge.client import Bridge

        bridge = Bridge(args.redis_url, timeout=3600.0)
    rows: list[Row] = []
    for case in cases:
        sizes = n_syns if _is_synthetic(case) else [0]
        for n_syn in sizes:
            for d_min in dmins:
                mode = "bridge" if bridge is not None else "in-process"
                extra = f" n={n_syn}" if _is_synthetic(case) else ""
                print(f"=== {case}{extra} d_min={d_min} device={args.device} dtypes={dtypes} ({mode}) ===")
                if bridge is not None:
                    rows.extend(
                        run_case_via_bridge(
                            bridge,
                            case,
                            d_min,
                            args.device,
                            args.repeats,
                            n_syn,
                            dtypes=dtypes,
                            include_nufft=not args.no_nufft,
                        )
                    )
                else:
                    rows.extend(run_case(case, d_min, args.device, args.repeats, n_syn))
    table = format_table(rows)
    print(table)
    if args.redis_url:
        cmd = (
            f"PYTHONPATH=src phenix.python -m phridge.contrib.nufft_sf.benchmark "
            f"--redis-url {args.redis_url} --cases {args.cases} --d-min {args.d_min} "
            f"--device {args.device} --repeats {args.repeats}"
        )
        worker_cmd = (
            "KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=src python -m phridge.worker.runner "
            f"--device {args.device} --preload phridge.contrib.nufft_sf"
        )
        commands = [worker_cmd, cmd]
    else:
        commands = [
            f"PYTHONPATH=src python -m phridge.contrib.nufft_sf.benchmark "
            f"--cases {args.cases} --d-min {args.d_min} --device {args.device} --repeats {args.repeats}"
        ]
    out = Path(args.out) if args.out else ROOT / "docs" / "nufft_engine.md"
    write_docs(out, table, commands)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
