import ast
import json
import re
import string
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from datasets import load_dataset


_POPQA_PUNCT = set(string.punctuation)


POPQA_MAIN_SYSTEM_PROMPT = """You are a careful open-domain factual question-answering system.

Hard requirements:
- Answer the question accurately.
- Provide only the final short answer, not a sentence or explanation.
- The answer should be an entity, title, place, date, number, or short phrase.
- End with exactly one final line in this format:
#### <answer>

The final line must contain only the marker #### followed by the answer."""


def normalize_popqa_answer(text: str | None) -> str:
    """Normalize answers for PopQA-style exact match.

    PopQA provides a list of acceptable gold answers in `possible_answers`.
    We use the standard open-QA exact-match normalization: lowercase, remove
    punctuation, remove English articles, and normalize whitespace. A model
    prediction is correct if its normalized final answer exactly matches any
    normalized gold alias.
    """
    if text is None:
        return ""

    def lower(s: str) -> str:
        return s.lower()

    def remove_punc(s: str) -> str:
        return "".join(ch for ch in s if ch not in _POPQA_PUNCT)

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


def _maybe_parse_string_list(value: str) -> List[str]:
    text = str(value).strip()
    if not text:
        return []

    # Hugging Face discussions for PopQA note that `possible_answers` may be
    # represented as a string in some versions. Accept JSON strings,
    # Python-literal list strings, and a few simple delimited fallbacks.
    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(text)
            if isinstance(obj, list):
                return [str(x) for x in obj if x is not None]
            if isinstance(obj, tuple):
                return [str(x) for x in obj if x is not None]
            if isinstance(obj, str):
                return [obj]
        except Exception:
            pass

    for sep in ("|||", "||", "\t"):
        if sep in text:
            return [part.strip() for part in text.split(sep) if part.strip()]

    return [text]


def _raw_aliases(row: Dict[str, Any]) -> List[str]:
    aliases: List[str] = []

    if "possible_answers" in row:
        value = row.get("possible_answers")
        if isinstance(value, str):
            aliases.extend(_maybe_parse_string_list(value))
        else:
            aliases.extend(_as_list(value))

    # Robust fallbacks for alternate processed versions of the dataset.
    for key in (
        "answers",
        "answer",
        "aliases",
        "obj",
        "object",
        "object_entity",
        "object_entitiey",
    ):
        if key in row:
            value = row.get(key)
            if isinstance(value, str):
                aliases.extend(_maybe_parse_string_list(value))
            else:
                aliases.extend(_as_list(value))

    # Preserve order while removing empty raw strings and duplicates.
    out: List[str] = []
    seen = set()
    for alias in aliases:
        text = str(alias).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _normalized_aliases(row: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for alias in _raw_aliases(row):
        norm = normalize_popqa_answer(alias)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def _answer_value(row: Dict[str, Any]) -> str:
    aliases = _raw_aliases(row)
    return aliases[0] if aliases else ""


def encode_gold_aliases(aliases: Sequence[str]) -> str:
    return json.dumps(list(aliases), ensure_ascii=False, sort_keys=True)


def decode_gold_aliases(gold_final: Any) -> List[str]:
    if gold_final is None:
        return []
    if isinstance(gold_final, list):
        return [normalize_popqa_answer(x) for x in gold_final if normalize_popqa_answer(x)]

    text = str(gold_final).strip()
    if not text:
        return []

    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [normalize_popqa_answer(x) for x in obj if normalize_popqa_answer(x)]
    except Exception:
        pass

    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, (list, tuple)):
            return [normalize_popqa_answer(x) for x in obj if normalize_popqa_answer(x)]
    except Exception:
        pass

    if "|||" in text:
        return [normalize_popqa_answer(x) for x in text.split("|||") if normalize_popqa_answer(x)]
    return [normalize_popqa_answer(text)]


def popqa_exact_match(pred_answer: str | None, gold_final: Any) -> bool:
    pred_norm = normalize_popqa_answer(pred_answer)
    if not pred_norm:
        return False
    return pred_norm in set(decode_gold_aliases(gold_final))


def final_short_answer_from_text(text: str | None) -> Optional[str]:
    """Extract the short answer from a generated PopQA response.

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
    ans = re.sub(
        r"^(the\s+answer\s+is|answer\s+is|it\s+is|it's)\s+",
        "",
        ans,
        flags=re.IGNORECASE,
    ).strip()
    ans = ans.strip(" \t\r\n\"'`“”‘’")
    ans = ans.rstrip(".").strip()
    return ans or None


def _load_hf_dataset(hf_name: str, hf_config: Any, split: str):
    if hf_config in {None, "", "null", "none", "default"}:
        return load_dataset(hf_name, split=split)
    return load_dataset(hf_name, hf_config, split=split)


class PopQAWrapper:
    """Fixed-size PopQA subset for exact-match open-domain QA experiments."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.subset_size = int(cfg.get("subset_size", cfg.get("max_samples", 500)))
        self.subset_seed = int(cfg.get("subset_seed", 42))
        self.subset_strategy = str(cfg.get("subset_strategy", "random")).lower()

    def _select_source_indices(self, ds) -> List[int]:
        n_total = len(ds)
        if self.subset_size <= 0:
            raise ValueError("data.subset_size must be positive for popqa.")
        if self.subset_size > n_total:
            raise ValueError(
                f"Requested popqa subset_size={self.subset_size}, but split has only {n_total} rows."
            )

        if self.subset_strategy in {"random", "shuffle", "sample"}:
            rng = np.random.default_rng(self.subset_seed)
            return rng.permutation(n_total)[: self.subset_size].tolist()

        if self.subset_strategy in {"longtail", "lowest_popularity", "hard"}:
            scored = []
            for idx in range(n_total):
                row = ds[int(idx)]
                popularity = None
                for key in ("s_pop", "subj_pop", "subject_popularity", "pageviews"):
                    if key in row and row.get(key) is not None:
                        try:
                            popularity = float(row.get(key))
                            break
                        except Exception:
                            pass
                if popularity is None:
                    popularity = float("inf")
                scored.append((popularity, idx))
            scored.sort(key=lambda x: (x[0], x[1]))
            return [idx for _, idx in scored[: self.subset_size]]

        raise ValueError(
            f"Unknown popqa subset_strategy={self.subset_strategy!r}. "
            "Use 'random' or 'longtail'."
        )

    def load_records(self) -> List[Dict[str, Any]]:
        hf_name = self.cfg.get("hf_name", "akariasai/PopQA")
        hf_config = self.cfg.get("hf_config", None)
        split = self.cfg.get("split", "test")

        ds = _load_hf_dataset(hf_name, hf_config, split)
        selected_source_indices = self._select_source_indices(ds)

        records = []
        for source_idx in selected_source_indices:
            row = dict(ds[int(source_idx)])
            question = row.get("question") or row.get("Question") or row.get("input")
            aliases = _normalized_aliases(row)
            if not question:
                raise ValueError(f"PopQA example at source index {source_idx} has no question.")
            if not aliases:
                raise ValueError(f"PopQA example at source index {source_idx} has no possible_answers aliases.")

            records.append(
                {
                    # Use the original split-row index as the cache key so the
                    # selected examples remain traceable to the full PopQA split.
                    "example_id": int(source_idx),
                    "question": str(question),
                    "gold_answer": _answer_value(row),
                    "gold_final": encode_gold_aliases(aliases),
                }
            )

        return records

    @staticmethod
    def parse_prediction(model_answer: str) -> str | None:
        return final_short_answer_from_text(model_answer)

    @staticmethod
    def failure_label(pred_answer: str | None, gold_final: str | None) -> int:
        return int(not popqa_exact_match(pred_answer, gold_final))

    @staticmethod
    def build_prompt(question: str) -> str:
        return (
            "Answer the following open-domain factual question from PopQA.\n"
            "Give only the final short answer; do not include explanation, reasoning, or extra text.\n"
            "The answer should be an entity, title, place, date, number, or short phrase.\n"
            "End with exactly one final line in this format: #### <answer>\n\n"
            f"Question:\n{question}"
        )

    @staticmethod
    def main_system_prompt() -> str:
        return POPQA_MAIN_SYSTEM_PROMPT
