"""
Reward functions for task generation RL.

Binary reward: 1.0 if solver rollouts have mixed results (at least one pass
and one fail), 0.0 otherwise. Mixed results = the task is at the right
difficulty frontier.
"""

from typing import Dict, List


def compute_variance(scores: List[float]) -> float:
    if len(scores) < 2:
        return 0.0
    mean = sum(scores) / len(scores)
    return sum((s - mean) ** 2 for s in scores) / len(scores)


def compute_task_reward(
    raw_scores: List[float],
    hinted_scores: List[float],
    validity: float = 1.0,
    alpha: float = 1.0,
) -> Dict[str, float]:
    """Binary reward: 1.0 if mixed solver results, 0.0 otherwise."""
    if not raw_scores:
        return {"validity": validity, "p_raw": 0.0, "var_raw": 0.0, "total": 0.0}

    p_raw = sum(raw_scores) / len(raw_scores)
    var_raw = compute_variance(raw_scores)
    has_pass = any(s > 0 for s in raw_scores)
    has_fail = any(s == 0 for s in raw_scores)
    total = 1.0 if (has_pass and has_fail and validity > 0) else 0.0

    return {
        "validity": validity,
        "p_raw": p_raw,
        "var_raw": var_raw,
        "hint_gap": 0.0,
        "total": total,
    }
