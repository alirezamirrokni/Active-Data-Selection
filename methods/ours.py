from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from utils import sigmoid


class OursSelection:
    """Our online risk- and budget-aware selection method."""

    needs_score_llm = True

    def __init__(self, cfg: Dict[str, Any], score_llm, state: Dict[str, Any] | None = None):
        if score_llm is None:
            raise ValueError("OursSelection requires a score_llm.")
        self.cfg = cfg
        self.score_llm = score_llm
        self.policy = cfg["policy"]
        state = state or {}
        self.alpha = float(state.get("alpha", self.policy.get("initial_alpha", 0.0)))
        theta = state.get("theta")
        self.theta = None if theta is None else np.array(theta, dtype=np.float32)

    def _ensure_theta(self, dim: int) -> None:
        if self.theta is None:
            self.theta = np.zeros(dim, dtype=np.float32)

    @staticmethod
    def _choose_beta(eta: np.ndarray, costs: np.ndarray, alpha: float, budget: float) -> float:
        def selected_cost(beta: float) -> float:
            return float(costs[eta > alpha + beta * costs].sum())

        if selected_cost(0.0) <= budget + 1e-12:
            return 0.0

        lo, hi = 0.0, 1.0
        while selected_cost(hi) > budget + 1e-12:
            hi *= 2.0
            if hi > 1e6:
                break
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if selected_cost(mid) <= budget + 1e-12:
                hi = mid
            else:
                lo = mid
        return hi

    @staticmethod
    def _safe_ratio(num: float, den: float) -> float:
        return 0.0 if den <= 0 else float(num / den)

    def process_batch(self, batch_df: pd.DataFrame, t: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        features = self.score_llm.encode_rows(batch_df.to_dict("records"))
        self._ensure_theta(features.shape[1])

        costs = batch_df["cost"].to_numpy(dtype=float)
        A = batch_df["A"].to_numpy(dtype=float)
        budget = float(self.policy["budget_per_batch"])
        epsilon = float(self.policy["epsilon"])
        gamma = float(self.policy["alpha_step_size"])
        theta_lr = float(self.policy["theta_step_size"])
        l2_reg = float(self.policy.get("l2_reg", 0.0))

        eta = sigmoid(features @ self.theta).astype(np.float32)
        beta = self._choose_beta(eta, costs, self.alpha, budget)
        selected = (eta > self.alpha + beta * costs).astype(int)

        n_sel = float(selected.sum())
        confirmation_rate = self._safe_ratio(float((selected * (1.0 - A)).sum()), n_sel)
        old_alpha = self.alpha
        self.alpha = max(0.0, self.alpha + gamma * (confirmation_rate - epsilon))

        if n_sel > 0:
            grad = ((selected * (eta - A))[:, None] * features).sum(axis=0) / n_sel
            if l2_reg > 0:
                grad = grad + l2_reg * self.theta
            self.theta = (self.theta - theta_lr * grad).astype(np.float32)

        out = batch_df.copy()
        out["eta"] = eta
        out["alpha"] = old_alpha
        out["beta"] = beta
        out["selected"] = selected
        return out, self.state_dict()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "alpha": float(self.alpha),
            "theta": None if self.theta is None else self.theta.astype(float).tolist(),
        }
