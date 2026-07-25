from __future__ import annotations

from typing import Any

from BackTrading.bayesian.kernel import GPState, restore_gp_state


def warm_start_gp(
    previous_gp_state: GPState | None,
    n_dims: int,
) -> GPState | None:
    """跨窗口 GP warm-start。

    校验前一窗口 GP 状态是否可用于当前窗口。
    维度匹配时返回有效状态（优化器将用其超参初始化新 GP），
    不匹配时返回 None（从零开始）。
    """
    return restore_gp_state(previous_gp_state, n_dims)
