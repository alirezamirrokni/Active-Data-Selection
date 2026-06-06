import re
from typing import Any, Dict, List, Optional

from datasets import load_dataset


def final_number_from_text(text: str | None) -> Optional[str]:
    if text is None:
        return None
    text = str(text)
    marker = re.search(r"####\s*([^\n]+)", text)
    if marker:
        nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", marker.group(1))
        if nums:
            return nums[-1].replace(",", "")
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if nums:
        return nums[-1].replace(",", "")
    return None


def numeric_equal(a: str | None, b: str | None, tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return str(a).strip() == str(b).strip()


class GSM8KWrapper:
    """GSM8K wrapper used by the active-selection experiments."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def load_records(self) -> List[Dict[str, Any]]:
        ds = load_dataset(
            self.cfg.get("hf_name", "openai/gsm8k"),
            self.cfg.get("hf_config", "main"),
            split=self.cfg.get("split", "test"),
        )
        max_examples = self.cfg.get("max_examples")
        if max_examples is not None:
            ds = ds.select(range(min(int(max_examples), len(ds))))

        records = []
        for idx, row in enumerate(ds):
            gold_final = final_number_from_text(row["answer"])
            records.append(
                {
                    "example_id": idx,
                    "question": row["question"],
                    "gold_answer": row["answer"],
                    "gold_final": gold_final,
                }
            )
        return records

    @staticmethod
    def parse_prediction(model_answer: str) -> str | None:
        return final_number_from_text(model_answer)

    @staticmethod
    def failure_label(pred_answer: str | None, gold_final: str | None) -> int:
        return int(not numeric_equal(pred_answer, gold_final))

    @staticmethod
    def build_prompt(question: str) -> str:
        return (
            "Solve this GSM8K problem. Keep the answer compact enough to fit the token budget.\n"
            "Use at most 4 short reasoning steps. Do not include extra commentary.\n"
            "End with exactly one final line in this format: #### <number>\n\n"
            f"Problem:\n{question}"
        )
