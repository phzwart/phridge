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

# Default: in-process Bridge(memory=True). Past jobs hung on Redis socket reads.
USE_REDIS=0
if [[ "${PHRIDGE_MEMORY:-1}" == "0" || "${PHRIDGE_MEMORY:-1}" == "false" ]]; then
    USE_REDIS=1
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
    Default: in-process torch (no Redis) — avoids past socket-read hangs.
    Pass --redis to start Redis + phridge-worker instead.

Options:
    --check-env         Verify Phenix, PyTorch (.venv), and optional Redis.
    --memory            In-process Bridge (default). Sets PHRIDGE_MEMORY=1.
    --redis             Use Redis + background PyTorch worker (PHRIDGE_MEMORY=0).
    --worker-only       Start Redis and PyTorch worker in foreground (implies --redis).
    --device DEVICE     PyTorch compute device ('auto', 'mps', 'cpu', 'cuda'). Default: ${DEVICE}
    --port PORT         Redis port (only with --redis). Default: ${REDIS_PORT}
    --nu VALUE          Student-t noise degrees of freedom (e.g. 5.0). Default: None (Gaussian).
    --fit-nu            Enable refinement of Student-t nu parameter during scale updates.
    --precondition      Enable Gauss-Newton diagonal preconditioning of XYZ, occupancy, and ADP gradients (Phenix LBFGS).
    --stats-report      Print I/σ + σ_A(resolution) table after each scale update (default on).
    --no-stats-report   Disable the resolution stats report.
    --stats-bin-size N  Reflections per stats bin (default: 500).
    --keep-services     Keep Redis and worker running after refine (only with --redis).
    -h, --help          Show this help message.

Refinement Example:
    ./run_phenix_intensity.sh model.pdb data.mtz refinement.main.number_of_macro_cycles=5
    ./run_phenix_intensity.sh model.pdb data.mtz --nu 5.0 --fit-nu
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
    PHRIDGE_PRECONDITION Enable XYZ/occ/ADP Gauss-Newton preconditioning ('1' or 'true')
    PHRIDGE_STATS_REPORT Print I/σ + σ_A bins after scale updates (default on; '0' to disable)
    PHRIDGE_STATS_BIN_SIZE  Reflections per stats bin (default 500)
    PHRIDGE_VERBOSE_TARGET  Print per-eval mli_quad banners ('1' to enable; default off)
    PHRIDGE_WEIGHT_METRIC   Weight-trial ranking: nll (default) or rfree
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
# 4. Cleanup Trap
# ------------------------------------------------------------------------------
cleanup() {
    local exit_code=$?
    if [[ "$KEEP_SERVICES" -eq 1 ]]; then
        if [[ -n "$WORKER_PID" ]]; then
            echo "• Services left running (--keep-services active). Worker PID: $WORKER_PID"
        fi
        exit "$exit_code"
    fi

    if [[ "$STARTED_WORKER" -eq 1 && -n "$WORKER_PID" ]]; then
        if kill -0 "$WORKER_PID" 2>/dev/null; then
            echo "• Stopping background PyTorch worker (PID $WORKER_PID)..."
            kill "$WORKER_PID" 2>/dev/null || true
            wait "$WORKER_PID" 2>/dev/null || true
        fi
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
    export PHRIDGE_MEMORY="${PHRIDGE_MEMORY:-1}"
    unset PHRIDGE_REDIS_URL || true
    echo "✓ Mode: in-process memory (PHRIDGE_MEMORY=1) — no Redis / no worker daemon"
    echo "  (use --redis if you need a remote torch worker; past hangs were Redis socket reads)"
else
    export PHRIDGE_MEMORY="0"
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

    if [[ "$WORKER_ONLY" -eq 1 ]]; then
        echo "• Starting PyTorch worker in foreground (device=${DEVICE}). Press Ctrl+C to exit."
        exec "$PHRIDGE_WORKER" --redis-url "$REDIS_URL" --device "$DEVICE"
    fi

    echo "• Launching PyTorch worker daemon (device=${DEVICE})..."
    "$PHRIDGE_WORKER" --redis-url "$REDIS_URL" --device "$DEVICE" >> "$WORKER_LOG" 2>&1 &
    WORKER_PID=$!
    STARTED_WORKER=1
    echo "✓ PyTorch worker active (PID $WORKER_PID, log: $WORKER_LOG)"
    sleep 0.5
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
