#!/bin/bash

#SBATCH --nodes=1
#SBATCH -n 48
#SBATCH --cpus-per-task=1
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=2500

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/euler}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STAGES="${STAGES:-download,paper,mango_mdd,mango_rmse,mango_rmse_random,compare}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_MODULES="${SKIP_MODULES:-0}"
STATUS_MANIFEST="$OUTPUT_ROOT/stage_status.tsv"

if [[ "$SKIP_MODULES" != "1" && "$DRY_RUN" != "1" ]]; then
    module load stack/2025-06
    module load gcc/12.2.0
    module load python/3.13.0
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$OUTPUT_ROOT"
cd "$REPO_ROOT"
: > "$STATUS_MANIFEST"
printf 'stage\tstatus\ttimestamp_utc\n' >> "$STATUS_MANIFEST"

has_stage() {
    [[ ",$STAGES," == *",$1,"* ]]
}

record_stage() {
    printf '%s\t%s\t%s\n' "$1" "$2" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS_MANIFEST"
}

run_stage() {
    local stage_name="$1"
    shift
    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY RUN [%s]:' "$stage_name"
        printf ' %q' "$@"
        printf '\n'
        record_stage "$stage_name" "planned"
        return 0
    fi
    record_stage "$stage_name" "running"
    if "$@"; then
        record_stage "$stage_name" "completed"
    else
        local exit_status=$?
        record_stage "$stage_name" "failed"
        return "$exit_status"
    fi
}

has_complete_forecast_run() {
    local run_dir="$1"
    if [[ -s "$run_dir/forecast_panel.csv" ]]; then
        return 0
    fi
    local panel
    for panel in "$run_dir"/*/forecast_panel.csv; do
        if [[ -s "$panel" ]]; then
            return 0
        fi
    done
    return 1
}

if has_stage download; then
    run_stage download "$PYTHON_BIN" "$REPO_ROOT/scripts/download_data.py"
fi

if has_stage paper; then
    run_stage paper "$PYTHON_BIN" "$REPO_ROOT/scripts/run_paper_hyperparameters.py" \
        --output-dir "$OUTPUT_ROOT/paper_hyperparameters"
fi

if has_stage mango_mdd; then
    run_stage mango_mdd "$PYTHON_BIN" "$REPO_ROOT/scripts/run_mango_mdd.py" \
        --output-dir "$OUTPUT_ROOT/mango_mdd"
fi

if has_stage mango_rmse; then
    run_stage mango_rmse "$PYTHON_BIN" "$REPO_ROOT/scripts/run_mango_rmse.py" \
        --output-dir "$OUTPUT_ROOT/mango_rmse"
fi

if has_stage mango_rmse_random; then
    run_stage mango_rmse_random "$PYTHON_BIN" "$REPO_ROOT/scripts/run_mango_rmse_random.py" \
        --output-dir "$OUTPUT_ROOT/mango_rmse_random"
fi

if has_stage compare; then
    if [[ "$DRY_RUN" != "1" ]]; then
        for required_run in paper_hyperparameters mango_mdd mango_rmse mango_rmse_random; do
            if ! has_complete_forecast_run "$OUTPUT_ROOT/$required_run"; then
                printf 'Comparison refused: required run is missing or incomplete: %s\n' \
                    "$OUTPUT_ROOT/$required_run" >&2
                record_stage compare "blocked_missing_${required_run}"
                exit 1
            fi
        done
    fi
    run_stage compare "$PYTHON_BIN" "$REPO_ROOT/scripts/compare_forecasts.py" \
        --paper-dir "$OUTPUT_ROOT/paper_hyperparameters" \
        --mango-mdd-dir "$OUTPUT_ROOT/mango_mdd" \
        --mango-rmse-dir "$OUTPUT_ROOT/mango_rmse" \
        --mango-rmse-random-dir "$OUTPUT_ROOT/mango_rmse_random" \
        --output-dir "$OUTPUT_ROOT/comparison"
fi
