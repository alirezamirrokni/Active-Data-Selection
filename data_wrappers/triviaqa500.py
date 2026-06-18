import json
import re
import string
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from datasets import load_dataset


_TRIVIAQA_PUNCT = set(string.punctuation)


TRIVIAQA_MAIN_SYSTEM_PROMPT = """You are a careful trivia question-answering system.

Hard requirements:
- Answer the question accurately.
- Provide only the final short answer, not a sentence or explanation.
- End with exactly one final line in this format:
#### <answer>

The final line must contain only the marker #### followed by the answer."""


def normalize_triviaqa_answer(text: str | None) -> str:
    """Normalize answers with the standard TriviaQA exact-match protocol.

    This follows the TriviaQA/SQuAD-style normalization used by the official
    evaluator: lowercase, remove punctuation, remove English articles, and fix
    whitespace. Exact match is then computed against all normalized aliases.
    """
    if text is None:
        return ""

    def lower(s: str) -> str:
        return s.lower()

    def remove_punc(s: str) -> str:
        return "".join(ch for ch in s if ch not in _TRIVIAQA_PUNCT)

    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    return white_space_fix(remove_articles(remove_punc(lower(str(text)))))


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(x) for x in value if x is not None]
    return [str(value)]


def _answer_value(answer: Any) -> str:
    if isinstance(answer, dict):
        for key in ("value", "normalized_value"):
            val = answer.get(key)
            if val not in {None, ""}:
                return str(val)
        aliases = answer.get("aliases") or answer.get("normalized_aliases") or []
        if aliases:
            return str(aliases[0])
        return ""
    return str(answer if answer is not None else "")


def _normalized_aliases(answer: Any) -> List[str]:
    aliases: List[str] = []

    if isinstance(answer, dict):
        aliases.extend(_as_list(answer.get("normalized_aliases")))
        aliases.extend(normalize_triviaqa_answer(x) for x in _as_list(answer.get("aliases")))
        if answer.get("normalized_value"):
            aliases.append(str(answer["normalized_value"]))
        if answer.get("value"):
            aliases.append(normalize_triviaqa_answer(answer["value"]))
    else:
        aliases.append(normalize_triviaqa_answer(answer))

    # Preserve order while removing empty strings and duplicates.
    out: List[str] = []
    seen = set()
    for alias in aliases:
        norm = normalize_triviaqa_answer(alias)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def encode_gold_aliases(aliases: Sequence[str]) -> str:
    return json.dumps(list(aliases), ensure_ascii=False, sort_keys=True)


def decode_gold_aliases(gold_final: Any) -> List[str]:
    if gold_final is None:
        return []
    if isinstance(gold_final, list):
        return [normalize_triviaqa_answer(x) for x in gold_final if normalize_triviaqa_answer(x)]

    text = str(gold_final).strip()
    if not text:
        return []

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [normalize_triviaqa_answer(x) for x in obj if normalize_triviaqa_answer(x)]
    except Exception:
        pass

    # Backward-compatible fallback for hand-written configs/caches.
    if "|||" in text:
        return [normalize_triviaqa_answer(x) for x in text.split("|||") if normalize_triviaqa_answer(x)]
    return [normalize_triviaqa_answer(text)]


def triviaqa_exact_match(pred_answer: str | None, gold_final: Any) -> bool:
    pred_norm = normalize_triviaqa_answer(pred_answer)
    if not pred_norm:
        return False
    return pred_norm in set(decode_gold_aliases(gold_final))


def final_short_answer_from_text(text: str | None) -> Optional[str]:
    """Extract the short answer from a generated TriviaQA response.

    The generation prompt requests a final line of the form `#### <answer>`, but
    this parser is intentionally tolerant of common LLM variants such as
    `Final answer: ...` or a plain one-line answer.
    """
    if text is None:
        return None

    s = str(text).strip()
    if not s:
        return None

    marker_matches = re.findall(r"####\s*([^\n]+)", s)
    if marker_matches:
        return _clean_prediction(marker_matches[-1])

    patterns = [
        r"final\s+answer\s*[:\-]\s*([^\n]+)",
        r"answer\s*[:\-]\s*([^\n]+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, s, flags=re.IGNORECASE)
        if matches:
            return _clean_prediction(matches[-1])

    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if not lines:
        return None

    return _clean_prediction(lines[-1])


def _clean_prediction(text: str | None) -> Optional[str]:
    if text is None:
        return None
    ans = str(text).strip()
    ans = re.sub(r"^[-*\s]+", "", ans).strip()
    ans = re.sub(r"^(the\s+answer\s+is|answer\s+is|it\s+is|it's)\s+", "", ans, flags=re.IGNORECASE).strip()
    ans = ans.strip(" \t\r\n\"'`“”‘’")
    ans = ans.rstrip(".").strip()
    return ans or None


class TriviaQA500Wrapper:
    """Fixed 500-example TriviaQA validation subset for exact-match QA experiments."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.subset_size = int(cfg.get("subset_size", cfg.get("max_samples", 500)))
        self.subset_seed = int(cfg.get("subset_seed", 42))

    def load_records(self) -> List[Dict[str, Any]]:
        hf_name = self.cfg.get("hf_name", "mandarjoshi/trivia_qa")
        hf_config = self.cfg.get("hf_config", "rc.nocontext")
        split = self.cfg.get("split", "validation")

        ds = load_dataset(hf_name, hf_config, split=split)
        n_total = len(ds)
        if self.subset_size <= 0:
            raise ValueError("data.subset_size must be positive for triviaqa500.")
        if self.subset_size > n_total:
            raise ValueError(
                f"Requested triviaqa500 subset_size={self.subset_size}, but split has only {n_total} rows."
            )

        rng = np.random.default_rng(self.subset_seed)
        selected_source_indices = rng.permutation(n_total)[: self.subset_size]

        records = []
        for source_idx in selected_source_indices.tolist():
            row = ds[int(source_idx)]
            question = row.get("question") or row.get("Question") or row.get("input")
            answer = row.get("answer") or row.get("Answer")
            aliases = _normalized_aliases(answer)
            if not aliases:
                # Keep the failure mode explicit; exact match needs at least one
                # normalized reference alias.
                raise ValueError(f"TriviaQA example at source index {source_idx} has no answer aliases.")

            records.append(
                {
                    # Use the original validation-row index as the cache key so
                    # the selected examples remain traceable to the full split.
                    "example_id": int(source_idx),
                    "question": str(question),
                    "gold_answer": _answer_value(answer),
                    "gold_final": encode_gold_aliases(aliases),
                }
            )

        return records

    @staticmethod
    def parse_prediction(model_answer: str) -> str | None:
        return final_short_answer_from_text(model_answer)

    @staticmethod
    def failure_label(pred_answer: str | None, gold_final: str | None) -> int:
        return int(not triviaqa_exact_match(pred_answer, gold_final))

    @staticmethod
    def build_prompt(question: str) -> str:
        return (
            "Answer the following trivia question.\n"
            "Give only the final short answer; do not include explanation, reasoning, or extra text.\n"
            "End with exactly one final line in this format: #### <answer>\n\n"
            f"Question:\n{question}"
        )

    @staticmethod
    def main_system_prompt() -> str:
        return TRIVIAQA_MAIN_SYSTEM_PROMPT
