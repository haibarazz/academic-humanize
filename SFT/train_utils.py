"""
训练辅助工具。

当前仅包含：早停组件构建。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from transformers import EarlyStoppingCallback


def build_early_stopping_components(
    has_val: bool,
    enabled: bool,
    evaluation_strategy: str,
    save_steps: int,
    eval_steps: int,
    patience: int,
    threshold: float,
) -> Tuple[List[Any], Dict[str, Any], int, Optional[str]]:
    """
    构建早停所需的 callbacks 与 TrainingArguments 附加参数。

    返回：
    1) callbacks：传给 Trainer(callbacks=...)
    2) training_args_extras：传给 TrainingArguments 的附加参数
    3) resolved_save_steps：必要时调整后的 save_steps
    4) note：可选提示信息
    """
    callbacks: List[Any] = []
    extras: Dict[str, Any] = {}
    resolved_save_steps = int(save_steps)
    note: Optional[str] = None

    if not has_val or not enabled:
        return callbacks, extras, resolved_save_steps, note

    safe_patience = max(1, int(patience))
    safe_threshold = max(0.0, float(threshold))
    callbacks.append(
        EarlyStoppingCallback(
            early_stopping_patience=safe_patience,
            early_stopping_threshold=safe_threshold,
        )
    )

    extras.update(
        {
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "save_strategy": evaluation_strategy,
        }
    )

    # 使用 steps 策略时，save/eval 步长需要对齐，避免 Trainer 报错。
    if evaluation_strategy == "steps":
        resolved_save_steps = int(eval_steps)
        note = f"早停已启用，save_steps 自动对齐为 eval_steps={resolved_save_steps}"

    return callbacks, extras, resolved_save_steps, note
