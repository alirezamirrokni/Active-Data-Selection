import argparse

from dotenv import load_dotenv

from data_wrappers import build_data_wrapper
from main_llms import build_main_llm
from run_experiment import ensure_generations, flatten_batches, sample_batches
from utils import load_yaml, project_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--reset_generations",
        action="store_true",
        help="Delete the shared main-LLM generation cache before generating.",
    )
    args = parser.parse_args()

    load_dotenv()
    cfg = load_yaml(args.config)
    paths = project_paths(cfg)

    if args.reset_generations and paths["generation_cache"].exists():
        paths["generation_cache"].unlink()

    print(f"[generate] cache={paths['generation_cache']}")
    print(f"[generate] main_llm={cfg['main_llm'].get('model_name')}")
    print(f"[generate] dataset={cfg['data'].get('name')} split={cfg['data'].get('split')}")
    print(
        f"[generate] sampled batches={cfg['data'].get('num_batches')} "
        f"batch_size={cfg['data'].get('batch_size')} replacement=True"
    )

    data_wrapper = build_data_wrapper(cfg["data"])
    records_pool = data_wrapper.load_records()
    batches = sample_batches(records_pool, cfg)
    sampled_records = flatten_batches(batches)
    main_llm = build_main_llm(cfg["main_llm"])
    ensure_generations(sampled_records, data_wrapper, main_llm, paths["generation_cache"], allow_generate=True)
    print(f"[done] generation cache ready: {paths['generation_cache']}")


if __name__ == "__main__":
    main()
