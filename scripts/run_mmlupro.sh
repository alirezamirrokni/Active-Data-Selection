#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="configs/llama-3.3-70b-versatile_mmlupro-test_n500_seed42"

python generate_cache.py --config "${CONFIG_DIR}/generate.yaml"
python run_experiment.py --config "${CONFIG_DIR}/random.yaml"
python run_experiment.py --config "${CONFIG_DIR}/llm_select.yaml"
python run_experiment.py --config "${CONFIG_DIR}/ours.yaml"
python run_experiment.py --config "${CONFIG_DIR}/ours_llm.yaml"
python plot_results.py --runs outputs --out_dir figures
