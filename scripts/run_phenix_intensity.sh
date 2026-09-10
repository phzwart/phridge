#!/usr/bin/env bash
# ==============================================================================
# run_phenix_intensity.sh
#
# Standalone execution orchestrator for Phridge direct intensity likelihood
# refinement (`mli_quad`) in phenix.refine.
#
# Automatically manages:
#   1. In-process PyTorch by default (PHRIDGE_MEMORY=1 — no Redis sockets)
#   2. Optional Redis + background worker (--redis) for remote torch
#   3. Phenix environment (`phenix.python` with mli_quad hooks)
# ==============================================================================

set -euo pipefail

# ------------------------------------------------------------------------------
# 1. Default Environment Configuration (Tailored to your installation)
# ------------------------------------------------------------------------------
REPO_DIR="${PHRIDGE_REPO:-/Users/phzwart/Projects/phridge}"
PHENIX_ROOT="${PHENIX_DIR:-/Users/phzwart/Applications/phenix-2.0-5936}"
PHENIX_PYTHON="${PHENIX_PYTHON:-$PHENIX_ROOT/bin/phenix.python}"

VENV_DIR="${PHRIDGE_VENV:-$REPO_DIR/.venv}"
VENV_PYTHON="${PHRIDGE_PYTHON:-$VENV_DIR/bin/python}"
PHRIDGE_WORKER="${PHRIDGE_WORKER:-$VENV_DIR/bin/phridge-worker}"

REDIS_SERVER="${REDIS_SERVER:-$(command -v redis-server || echo "/opt/homebrew/bin/redis-server")}"
REDIS_CLI="${REDIS_CLI:-$(command -v redis-cli || echo "/opt/homebrew/bin/redis-cli")}"
REDIS_PORT="${PHRIDGE_REDIS_PORT:-6379}"
REDIS_URL="redis://127.0.0.1:${REDIS_PORT}/0"

# Default: Redis + fresh worker on Darwin (phenix.python + in-process torch
# often segfaults). Elsewhere default stays in-process unless PHRIDGE_MEMORY=0.
USE_REDIS=0
if [[ "$(uname -s)" == "Darwin" ]]; then
    USE_REDIS=1
fi
if [[ "${PHRIDGE_MEMORY:-}" == "0" || "${PHRIDGE_MEMORY:-}" == "false" ]]; then
    USE_REDIS=1
elif [[ "${PHRIDGE_MEMORY:-}" == "1" || "${PHRIDGE_MEMORY:-}" == "true" ]]; then
    USE_REDIS=0
fi
DEVICE="${PHRIDGE_DEVICE:-auto}"
LOG_DIR="${REPO_DIR}/logs"
WORKER_LOG="${LOG_DIR}/phridge-worker.log"
REDIS_LOG="${LOG_DIR}/redis.log"

STARTED_REDIS=0
STARTED_WORKER=0
WORKER_PID=""
KEEP_SERVICES=0
WORKER_ONLY=0

# ------------------------------------------------------------------------------
# 2. Help & Usage
# ------------------------------------------------------------------------------
show_help() {
    cat <<EOF
Usage:
    ./run_phenix_intensity.sh [options] [model.pdb data.mtz ...] [phenix_options ...]

Description:
    Orchestrates phenix.refine with Phridge direct intensity likelihood (mli_quad).
    Default on macOS: Redis + fresh worker (phenix.python + in-process torch often
    segfaults). Elsewhere default is in-process; pass --redis / --memory to force.

Options:
    --check-env         Verify Phenix, PyTorch (.venv), and optional Redis.
    --memory            In-process Bridge (PHRIDGE_MEMORY=1). On macOS this may
                        segfault; forces CPU + single-threaded BLAS as a mitigation.
    --redis             Redis + worker (recommended on macOS). Kills old workers,
                        flushes Redis DB, starts a fresh worker so src/ always loads.
    --worker-only       Start Redis and PyTorch worker in foreground (implies --redis).
    --keep-services     Keep Redis + worker after refine (only with --redis).
    --device DEVICE     PyTorch compute device ('auto', 'mps', 'cpu', 'cuda'). Default: ${DEVICE}
    --port PORT         Redis port (only with --redis). Default: ${REDIS_PORT}
    --nu VALUE          Student-t noise degrees of freedom (e.g. 5.0). Default: None (Gaussian).
    --fit-nu            Enable refinement of Student-t nu during scale updates (default: bins + TV).
    --nu-mode MODE      ν fit: bins (default, same shells as σ_A) or global (scalar).
    --precondition      Enable Gauss-Newton diagonal preconditioning of XYZ, occupancy, and ADP gradients (Phenix LBFGS).
    --stats-report      Print I/σ + σ_A(resolution) table after each scale update (default on).
    --no-stats-report   Disable the resolution stats report.
    --stats-bin-size N  Reflections per stats bin (default: 500).
    --sigma-a-bins N    Number of σ_A / ν resolution shells (default: auto).
    --tv-norm LAMBDA    Total-variation regularization on σ_A and ν bins (e.g. 0.04; default: 0).
    --fit-sigma-wilson  Intensity-only ML Wilson Σ₀/B_W (default; no model). Then fit σ_A.
    --no-fit-sigma-wilson  Keep moment-plot Σ₀/B_W (skip intensity ML Wilson).
    --omit-windows      After scale updates, write windowed omit map coefficients (.npz).
    --omit-box-size Å   Omit box edge length in Å (default: 10).
    --omit-mode MODE    boxes (default) or residue_blocks.
    --omit-prefix PATH  Output prefix → {prefix}_omit_windows.mtz (stitched omit coeffs).
    --keep-services     Keep Redis and worker running after refine (only with --redis).
    -h, --help          Show this help message.

Refinement Example:
    ./run_phenix_intensity.sh model.pdb data.mtz refinement.main.number_of_macro_cycles=5
    ./run_phenix_intensity.sh model.pdb data.mtz --nu 5.0 --fit-nu --tv-norm 0.04
    ./run_phenix_intensity.sh model.pdb data.mtz --omit-windows --omit-box-size 10
    ./run_phenix_intensity.sh --redis model.pdb data.mtz   # old Redis path

Environment Overrides:
    PHENIX_DIR          Phenix root directory (default: ${PHENIX_ROOT})
    PHRIDGE_VENV        Virtualenv containing torch and phridge (default: ${VENV_DIR})
    PHRIDGE_DEVICE      PyTorch device (default: ${DEVICE})
    PHRIDGE_MEMORY      In-process bridge (default 1). Set 0 for Redis.
    PHRIDGE_REDIS_PORT  Redis server port (default: ${REDIS_PORT})
    PHRIDGE_HEARTBEAT_INTERVAL  Seconds between alive prints (default 30)
    PHRIDGE_NU          Default Student-t nu degrees of freedom
    PHRIDGE_FIT_NU      Enable nu refinement ('1' or 'true')
    PHRIDGE_NU_MODE     ν fit mode: bins (default) or global
    PHRIDGE_PRECONDITION Enable XYZ/occ/ADP Gauss-Newton preconditioning ('1' or 'true')
    PHRIDGE_STATS_REPORT Print I/σ + σ_A bins after scale updates (default on; '0' to disable)
    PHRIDGE_STATS_BIN_SIZE  Reflections per stats bin (default 500)
    PHRIDGE_VERBOSE_TARGET  Print per-eval mli_quad banners ('1' to enable; default off)
    PHRIDGE_WEIGHT_METRIC   Weight-trial ranking: nll (default) or rfree
    PHRIDGE_SIGMA_A_BINS    Number of σ_A / ν resolution shells (bins mode)
    PHRIDGE_SIGMA_A_TV_NORM Total-variation penalty λ_TV on adjacent σ_A and ν bins
    PHRIDGE_FIT_SIGMA_WILSON  Intensity-only ML Wilson Σ₀/B_W then σ_A (default on; 0 = moment plot only)
    PHRIDGE_OMIT_WINDOWS    Write windowed omit coefficients after scale updates ('1')
    PHRIDGE_OMIT_BOX_SIZE   Omit box edge (Å, default 10)
    PHRIDGE_OMIT_MODE       boxes|residue_blocks
    PHRIDGE_OMIT_PREFIX     Output prefix → {prefix}_omit_windows.mtz
    PHRIDGE_OMIT_CHUNK_SIZE Window chunk size for batched map coeffs
    PHRIDGE_OMIT_RESOLUTION_FACTOR  FFT grid for stitching (default 0.25)
    PHRIDGE_OMIT_SAVE_NPZ   Also write diagnostic per-window *_omit_windows.npz
EOF
}

# ------------------------------------------------------------------------------
# 3. Environment Check
# ------------------------------------------------------------------------------
check_env() {
    echo "============================================================================"
    echo " Phridge Refinement Environment Verification"
    echo "============================================================================"
    echo ""

    # Check Phenix Python
    if [[ -x "$PHENIX_PYTHON" ]]; then
        echo -n "✓ Phenix Python:   $PHENIX_PYTHON "
        "$PHENIX_PYTHON" -c "import cctbx; print('(cctbx available)')" 2>/dev/null || echo "(! cctbx import failed)"
    else
        echo "✗ Phenix Python NOT found at $PHENIX_PYTHON"
    fi

    # Check .venv PyTorch
    if [[ -x "$VENV_PYTHON" ]]; then
        echo -n "✓ Worker Python:   $VENV_PYTHON "
        "$VENV_PYTHON" -c "import torch; print(f'(PyTorch {torch.__version__}, MPS={torch.backends.mps.is_available()})')" 2>/dev/null || echo "(! torch import failed)"
    else
        echo "✗ Worker Python NOT found at $VENV_PYTHON"
    fi

    # Check Redis Server
    if [[ -x "$REDIS_SERVER" ]]; then
        echo "✓ Redis Server:    $REDIS_SERVER"
    else
        echo "✗ Redis Server NOT found at $REDIS_SERVER (install via 'brew install redis')"
    fi

    # Check Redis CLI
    if [[ -x "$REDIS_CLI" ]]; then
        echo "✓ Redis CLI:       $REDIS_CLI"
        if "$REDIS_CLI" -p "$REDIS_PORT" ping 2>/dev/null | grep -q "PONG"; then
            echo "✓ Redis Status:    Active on port $REDIS_PORT"
        else
            echo "• Redis Status:    Inactive (will be started automatically on port $REDIS_PORT)"
        fi
    fi

    echo ""
    echo "Environment check complete."
    exit 0
}

# ------------------------------------------------------------------------------
# 4. Cleanup Trap + worker process management
# ------------------------------------------------------------------------------
# Match real worker CLIs only (not shells grepping phridge-worker.log).
_WORKER_PGREP_PAT='phridge-worker --|phridge-cctbx-worker --|-m phridge\.worker|-m phridge\.cctbx_worker'

list_worker_pids() {
    pgrep -f "$_WORKER_PGREP_PAT" 2>/dev/null || true
}

kill_all_phridge_workers() {
    local reason="${1:-restart}"
    local pids
    pids="$(list_worker_pids)"
    if [[ -z "$pids" ]]; then
        echo "• No existing phridge-worker processes ($reason)"
        return 0
    fi
    echo "• Killing old phridge-worker process(es) ($reason): $(echo "$pids" | tr '\n' ' ')"
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    local i
    for i in 1 2 3 4 5 6 7 8 9 10; do
        pids="$(list_worker_pids)"
        [[ -z "$pids" ]] && break
        sleep 0.2
    done
    pids="$(list_worker_pids)"
    if [[ -n "$pids" ]]; then
        echo "• Force-killing stubborn worker(s): $(echo "$pids" | tr '\n' ' ')"
        # shellcheck disable=SC2086
        kill -9 $pids 2>/dev/null || true
        sleep 0.2
    fi
    pids="$(list_worker_pids)"
    if [[ -n "$pids" ]]; then
        echo "Error: could not kill phridge-worker PIDs: $(echo "$pids" | tr '\n' ' ')" >&2
        return 1
    fi
    echo "✓ All prior phridge-worker processes are gone"
    return 0
}

flush_redis_job_state() {
    # Drop queued/in-flight job keys so a new worker does not resume stale work.
    if [[ ! -x "$REDIS_CLI" ]]; then
        return 0
    fi
    if ! "$REDIS_CLI" -p "$REDIS_PORT" ping 2>/dev/null | grep -q "PONG"; then
        return 0
    fi
    echo "• Flushing Redis DB for phridge jobs (port $REDIS_PORT)..."
    "$REDIS_CLI" -p "$REDIS_PORT" FLUSHDB >/dev/null 2>&1 || true
}

cleanup() {
    local exit_code=$?
    if [[ "$KEEP_SERVICES" -eq 1 ]]; then
        if [[ -n "$WORKER_PID" ]]; then
            echo "• Services left running (--keep-services active). Worker PID: $WORKER_PID"
        fi
        exit "$exit_code"
    fi

    if [[ "$STARTED_WORKER" -eq 1 ]]; then
        echo "• Stopping phridge-worker(s) started for this run..."
        kill_all_phridge_workers "cleanup" || true
    elif [[ -n "$WORKER_PID" ]] && kill -0 "$WORKER_PID" 2>/dev/null; then
        kill "$WORKER_PID" 2>/dev/null || true
        wait "$WORKER_PID" 2>/dev/null || true
    fi

    if [[ "$STARTED_REDIS" -eq 1 ]]; then
        if [[ -x "$REDIS_CLI" ]]; then
            echo "• Shutting down local Redis server (port $REDIS_PORT)..."
            "$REDIS_CLI" -p "$REDIS_PORT" shutdown nosave 2>/dev/null || true
        fi
    fi

    exit "$exit_code"
}

trap cleanup EXIT INT TERM

# ------------------------------------------------------------------------------
# 5. Argument Parsing
# ------------------------------------------------------------------------------
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        --check-env)
            check_env
            exit 0
            ;;
        --memory)
            USE_REDIS=0
            export PHRIDGE_MEMORY="1"
            # Mitigate phenix.python + torch segfaults (OpenMP / Accelerate clash)
            export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
            export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
            export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
            export PHRIDGE_DEVICE="${PHRIDGE_DEVICE:-cpu}"
            echo "⚠ --memory loads torch into phenix.python (can segfault on macOS)."
            echo "  Forcing PHRIDGE_DEVICE=${PHRIDGE_DEVICE}, OMP/MKL threads=1. Prefer --redis."
            shift
            ;;
        --redis)
            USE_REDIS=1
            export PHRIDGE_MEMORY="0"
            shift
            ;;
        --worker-only)
            WORKER_ONLY=1
            USE_REDIS=1
            export PHRIDGE_MEMORY="0"
            shift
            ;;
        --keep-services)
            KEEP_SERVICES=1
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --device=*)
            DEVICE="${1#*=}"
            shift
            ;;
        --port)
            REDIS_PORT="$2"
            REDIS_URL="redis://127.0.0.1:${REDIS_PORT}/0"
            shift 2
            ;;
        --port=*)
            REDIS_PORT="${1#*=}"
            REDIS_URL="redis://127.0.0.1:${REDIS_PORT}/0"
            shift
            ;;
        --nu)
            export PHRIDGE_NU="$2"
            shift 2
            ;;
        --nu=*)
            export PHRIDGE_NU="${1#*=}"
            shift
            ;;
        --fit-nu)
            export PHRIDGE_FIT_NU="1"
            shift
            ;;
        --no-fit-nu)
            export PHRIDGE_FIT_NU="0"
            shift
            ;;
        --nu-mode)
            export PHRIDGE_NU_MODE="$2"
            shift 2
            ;;
        --nu-mode=*)
            export PHRIDGE_NU_MODE="${1#*=}"
            shift
            ;;
        --precondition)
            export PHRIDGE_PRECONDITION="1"
            shift
            ;;
        --no-precondition)
            export PHRIDGE_PRECONDITION="0"
            shift
            ;;
        --stats-report)
            export PHRIDGE_STATS_REPORT="1"
            shift
            ;;
        --no-stats-report)
            export PHRIDGE_STATS_REPORT="0"
            shift
            ;;
        --stats-bin-size)
            export PHRIDGE_STATS_BIN_SIZE="$2"
            shift 2
            ;;
        --stats-bin-size=*)
            export PHRIDGE_STATS_BIN_SIZE="${1#*=}"
            shift
            ;;
        --verbose-target)
            export PHRIDGE_VERBOSE_TARGET="1"
            shift
            ;;
        --quiet-target)
            export PHRIDGE_VERBOSE_TARGET="0"
            shift
            ;;
        --weight-metric)
            export PHRIDGE_WEIGHT_METRIC="$2"
            shift 2
            ;;
        --weight-metric=*)
            export PHRIDGE_WEIGHT_METRIC="${1#*=}"
            shift
            ;;
        --sigma-a-bins)
            export PHRIDGE_SIGMA_A_BINS="$2"
            shift 2
            ;;
        --sigma-a-bins=*)
            export PHRIDGE_SIGMA_A_BINS="${1#*=}"
            shift
            ;;
        --tv-norm)
            export PHRIDGE_SIGMA_A_TV_NORM="$2"
            shift 2
            ;;
        --tv-norm=*)
            export PHRIDGE_SIGMA_A_TV_NORM="${1#*=}"
            shift
            ;;
        --fit-sigma-wilson)
            export PHRIDGE_FIT_SIGMA_WILSON="1"
            shift
            ;;
        --no-fit-sigma-wilson)
            export PHRIDGE_FIT_SIGMA_WILSON="0"
            shift
            ;;
        --omit-windows)
            export PHRIDGE_OMIT_WINDOWS="1"
            shift
            ;;
        --no-omit-windows)
            export PHRIDGE_OMIT_WINDOWS="0"
            shift
            ;;
        --omit-box-size)
            export PHRIDGE_OMIT_BOX_SIZE="$2"
            shift 2
            ;;
        --omit-box-size=*)
            export PHRIDGE_OMIT_BOX_SIZE="${1#*=}"
            shift
            ;;
        --omit-mode)
            export PHRIDGE_OMIT_MODE="$2"
            shift 2
            ;;
        --omit-mode=*)
            export PHRIDGE_OMIT_MODE="${1#*=}"
            shift
            ;;
        --omit-prefix)
            export PHRIDGE_OMIT_PREFIX="$2"
            shift 2
            ;;
        --omit-prefix=*)
            export PHRIDGE_OMIT_PREFIX="${1#*=}"
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

# Ensure log directory exists
mkdir -p "$LOG_DIR"

# ------------------------------------------------------------------------------
# 6. Start Redis + worker only when --redis (default is in-process memory)
# ------------------------------------------------------------------------------
echo "============================================================================"
echo " Starting Phridge Refinement Orchestrator"
echo "============================================================================"

if [[ "$USE_REDIS" -eq 0 ]]; then
    export PHRIDGE_MEMORY="1"
    unset PHRIDGE_REDIS_URL || true
    echo "✓ Mode: in-process memory (PHRIDGE_MEMORY=1) — no Redis / no worker daemon"
    echo "  (pass --redis for a remote torch worker; this script kills/restarts workers for you)"
    if [[ -n "$(list_worker_pids)" ]]; then
        echo "⚠ Note: phridge-worker process(es) still running, but this job will NOT use them."
        echo "  Optional cleanup: pkill -f phridge-worker"
    fi
else
    export PHRIDGE_MEMORY="0"
    echo "✓ Mode: Redis + phridge-worker (PHRIDGE_MEMORY=0)"
    echo "  Old workers are killed and a fresh worker is started so code changes load."

    if ! "$REDIS_CLI" -p "$REDIS_PORT" ping 2>/dev/null | grep -q "PONG"; then
        echo "• Starting Redis on port $REDIS_PORT..."
        if [[ ! -x "$REDIS_SERVER" ]]; then
            echo "Error: redis-server binary not found at $REDIS_SERVER" >&2
            exit 1
        fi
        "$REDIS_SERVER" --port "$REDIS_PORT" --daemonize yes --logfile "$REDIS_LOG"
        STARTED_REDIS=1

        READY=0
        for i in {1..30}; do
            if "$REDIS_CLI" -p "$REDIS_PORT" ping 2>/dev/null | grep -q "PONG"; then
                READY=1
                break
            fi
            sleep 0.2
        done
        if [[ "$READY" -ne 1 ]]; then
            echo "Error: Redis failed to start within 6 seconds. Check $REDIS_LOG" >&2
            exit 1
        fi
        echo "✓ Redis active on port $REDIS_PORT"
    else
        echo "✓ Redis already running on port $REDIS_PORT"
    fi

    # Always wipe stale workers + queued jobs before (re)starting
    kill_all_phridge_workers "before --redis start" || exit 1
    flush_redis_job_state

    if [[ "$WORKER_ONLY" -eq 1 ]]; then
        echo "• Starting PyTorch worker in foreground (device=${DEVICE}). Press Ctrl+C to exit."
        exec "$PHRIDGE_WORKER" --redis-url "$REDIS_URL" --device "$DEVICE"
    fi

    echo "• Launching fresh PyTorch worker daemon (device=${DEVICE})..."
    : > "$WORKER_LOG"  # truncate so this run's log is unambiguous
    "$PHRIDGE_WORKER" --redis-url "$REDIS_URL" --device "$DEVICE" >> "$WORKER_LOG" 2>&1 &
    WORKER_PID=$!
    STARTED_WORKER=1
    sleep 0.4
    if ! kill -0 "$WORKER_PID" 2>/dev/null; then
        echo "Error: phridge-worker exited immediately. Last log lines:" >&2
        tail -n 40 "$WORKER_LOG" >&2 || true
        exit 1
    fi
    echo "✓ PyTorch worker active (PID $WORKER_PID, log: $WORKER_LOG)"
    export PHRIDGE_REDIS_URL="$REDIS_URL"
fi

# ------------------------------------------------------------------------------
# 7. Execute phenix.refine with mli_quad
# ------------------------------------------------------------------------------
SCRIPT_PATH="${REPO_DIR}/scripts/phenix_refine_mli.py"

if [[ ${#ARGS[@]} -eq 0 ]]; then
    echo ""
    echo "No model or reflections specified. Showing phenix.refine usage options:"
    echo ""
    "$PHENIX_PYTHON" "$SCRIPT_PATH" --help || true
else
    echo ""
    echo "• Executing phenix.refine with target=mli_quad..."
    echo "----------------------------------------------------------------------------"
    "$PHENIX_PYTHON" "$SCRIPT_PATH" "${ARGS[@]}"
fi
