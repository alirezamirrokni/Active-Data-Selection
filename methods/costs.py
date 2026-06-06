def compute_cost(row: dict, variant: str) -> float:
    if variant == "constant":
        return 1.0
    if variant == "answer_length":
        return float(max(1, len(str(row.get("model_answer", "")).split())))
    if variant == "question_answer_length":
        q = len(str(row.get("question", "")).split())
        a = len(str(row.get("model_answer", "")).split())
        return float(max(1, q + a))
    raise ValueError(f"Unknown cost variant: {variant}")
