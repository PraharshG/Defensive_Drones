#!/usr/bin/env bash
set -euo pipefail

RUNS="${RUNS:-1000}"
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-outputs_server}"
GENERATE_FIGURES="${GENERATE_FIGURES:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if command -v nproc >/dev/null 2>&1; then
  DEFAULT_JOBS="$(nproc)"
else
  DEFAULT_JOBS="1"
fi
JOBS="${JOBS:-$DEFAULT_JOBS}"

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r requirements.txt

simulation_cmd=(
  "$VENV_DIR/bin/python"
  -m defensive_drones.simulate
  --runs "$RUNS"
  --seed "$SEED"
  --out "$OUT_DIR"
  --jobs "$JOBS"
)

if [[ -n "${STRATEGIES:-}" ]]; then
  read -r -a strategy_args <<< "$STRATEGIES"
  simulation_cmd+=(--strategies "${strategy_args[@]}")
fi

echo "Running simulation: runs=$RUNS seed=$SEED jobs=$JOBS out=$OUT_DIR"
"${simulation_cmd[@]}"

if [[ "$GENERATE_FIGURES" == "1" ]]; then
  "$VENV_DIR/bin/python" scripts/generate_publication_figures.py \
    --input "$OUT_DIR/per_run_results.csv" \
    --out "$OUT_DIR/publication_figures"
fi

echo "Done. Outputs are in $OUT_DIR"
