import argparse

from dotenv import load_dotenv

from data_wrappers import build_data_wrapper
from main_llms import build_main_llm
from run_experiment import ensure_generations
from utils import load_yaml, project_paths


def select_generation_records(records, cfg):
    """Select the deterministic prefix used for building the shared generation cache.

    This is intentionally different from online method runs. Method runs sample
    batches with replacement, while data generation uses the first max_samples
    examples so the cache is easy to inspect and extend.
    """
    data_cfg = cfg["data"]
    max_samples = data_cfg.get("max_samples", data_cfg.get("max_examples", None))

    if max_samples is None:
        return records

    max_samples = int(max_samples)
    if max_samples <= 0:
        raise ValueError("data.max_samples must be positive.")

    if max_samples > len(records):
        print(
            f"[generate] requested max_samples={max_samples}, "
            f"but dataset has only {len(records)} examples; using all examples."
        )

    return records[:max_samples]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/generate.yaml", help="Path to generation YAML config.")
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
    print(f"[generate] max_samples={cfg['data'].get('max_samples', cfg['data'].get('max_examples', 'all'))}")

    data_wrapper = build_data_wrapper(cfg["data"])
    records_pool = data_wrapper.load_records()
    records = select_generation_records(records_pool, cfg)

    main_llm = build_main_llm(cfg["main_llm"])
    ensure_generations(records, data_wrapper, main_llm, paths["generation_cache"], allow_generate=True)

    print(f"[done] generation cache ready: {paths['generation_cache']}")


if __name__ == "__main__":
    main()
