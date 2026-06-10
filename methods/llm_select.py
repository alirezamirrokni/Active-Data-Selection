from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from main_llms import build_main_llm


DEFAULT_SELECTOR_PROMPT = """You are selecting examples for human review.

Task:
You will receive a batch of model-generated math answers. The goal is to select examples that are most likely to contain an error and therefore most worth sending to a human corrector.

Rules:
- Select at most {max_items} examples.
- Respect the total budget {budget}. Each item has a listed cost.
- Use only the problem and the model answer. Do not assume access to the gold answer.
- Prefer examples with suspicious reasoning, arithmetic mistakes, missing final answers, format violations, or unsupported conclusions.

Return only valid JSON in exactly this format:
{{"selected_indices": [0, 3, 4]}}

Batch:
{items}
"""


class LLMSelect:
    """LLM-based batch selection baseline.

    This baseline gives the whole batch to a selector LLM and asks it to select
    at most budget-feasible examples for review. We call it LLM-Select rather
    than ActiveLLM because it is only a batch selection baseline, not a claim to
    reproduce a specific prior method.
    """

    needs_score_model = False

    def __init__(self, cfg: Dict[str, Any], score_model=None, state: Dict[str, Any] | None = None):
        self.cfg = cfg
        self.policy = cfg["policy"]
        self.seed = int(cfg.get("seed", 0))
        selector_cfg = cfg.get("selector_llm") or cfg.get("main_llm")
        self.selector_llm = build_main_llm(selector_cfg)
        self.prompt_template = self.policy.get("prompt", DEFAULT_SELECTOR_PROMPT)

    @staticmethod
    def _max_items_under_budget(costs: np.ndarray, budget: float) -> int:
        if len(costs) == 0 or budget <= 0:
            return 0
        sorted_costs = np.sort(np.asarray(costs, dtype=float))
        total = 0.0
        count = 0
        for c in sorted_costs:
            if c <= 0:
                continue
            if total + c <= budget + 1e-12:
                total += float(c)
                count += 1
        return count

    @staticmethod
    def _truncate(text: str, max_chars: int = 1200) -> str:
        text = str(text)
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 20] + " ... [truncated]"

    def _format_items(self, batch_df: pd.DataFrame) -> str:
        chunks = []
        for i, row in batch_df.reset_index(drop=True).iterrows():
            chunks.append(
                f"Index: {i}\n"
                f"Cost: {float(row['cost']):g}\n"
                f"Problem:\n{self._truncate(row['question'], 900)}\n"
                f"Model answer:\n{self._truncate(row['model_answer'], 900)}\n"
            )
        return "\n---\n".join(chunks)

    def _build_prompt(self, batch_df: pd.DataFrame, budget: float, costs: np.ndarray) -> str:
        max_items = self._max_items_under_budget(costs, budget)
        return self.prompt_template.format(
            budget=f"{budget:g}",
            max_items=max_items,
            items=self._format_items(batch_df),
        )

    @staticmethod
    def _parse_indices(text: str) -> List[int]:
        text = str(text).strip()

        # Preferred path: JSON object with selected_indices.
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and isinstance(obj.get("selected_indices"), list):
                return [int(x) for x in obj["selected_indices"]]
        except Exception:
            pass

        # Robust path: extract first JSON-looking object.
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict) and isinstance(obj.get("selected_indices"), list):
                    return [int(x) for x in obj["selected_indices"]]
            except Exception:
                pass

        # Last-resort path: parse integers from the response.
        nums = re.findall(r"-?\d+", text)
        return [int(x) for x in nums]

    @staticmethod
    def _validate_indices(indices: List[int], costs: np.ndarray, budget: float) -> np.ndarray:
        n = len(costs)
        selected = np.zeros(n, dtype=int)
        spent = 0.0
        seen = set()
        for idx in indices:
            if idx in seen or idx < 0 or idx >= n:
                continue
            c = float(costs[idx])
            if c <= 0:
                continue
            if spent + c <= budget + 1e-12:
                selected[idx] = 1
                spent += c
                seen.add(idx)
        return selected

    def process_batch(self, batch_df: pd.DataFrame, t: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        budget = float(self.policy["budget_per_batch"])
        costs = batch_df["cost"].to_numpy(dtype=float)
        prompt = self._build_prompt(batch_df, budget, costs)

        try:
            response = self.selector_llm.generate(prompt)
            indices = self._parse_indices(response)
            selected = self._validate_indices(indices, costs, budget)
        except Exception as exc:
            print(f"[llm_select] selector call/parse failed at batch {t}: {exc}")
            selected = np.zeros(len(batch_df), dtype=int)

        out = batch_df.copy()
        out["eta"] = np.nan
        out["alpha"] = np.nan
        out["beta"] = np.nan
        out["selected"] = selected
        return out, {}
