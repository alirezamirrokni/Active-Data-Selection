#!/usr/bin/env bash
set -euo pipefail

for CONFIG_DIR in configs/llama-3.3-70b-versatile_*; do
  [ -d "$CONFIG_DIR" ] || continue
  echo "[run_all] ${CONFIG_DIR}"
  python generate_cache.py --config "${CONFIG_DIR}/generate.yaml"
  python run_experiment.py --config "${CONFIG_DIR}/random.yaml"
  python run_experiment.py --config "${CONFIG_DIR}/llm_select.yaml"
  python run_experiment.py --config "${CONFIG_DIR}/ours.yaml"
  python run_experiment.py --config "${CONFIG_DIR}/ours_llm.yaml"
done
python plot_results.py --runs outputs --out_dir figures
