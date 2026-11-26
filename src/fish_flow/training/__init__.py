"""`fish_flow.training` 训练模块统一导出入口。"""

from fish_flow.training.deepsets import (
    DeepSetsTrainHistory,
    DeepSetsTrainer,
    DeepSetsTrainingConfig,
    build_dynamic_validation_panel,
    evaluate_dynamic_validation_panel,
    train_deepsets,
)
from fish_flow.training.mlp import (
    MLPTrainHistory,
    MLPTrainer,
    MLPTrainingConfig,
    train_mlp,
)
from fish_flow.training.ppinn import (
    PPINNTrainHistory,
    PPINNTrainer,
    PPINNTrainingConfig,
    evaluate_ppinn_validation,
    train_ppinn,
)

__all__ = [
    "DeepSetsTrainHistory",
    "DeepSetsTrainer",
    "DeepSetsTrainingConfig",
    "MLPTrainHistory",
    "MLPTrainer",
    "MLPTrainingConfig",
    "PPINNTrainHistory",
    "PPINNTrainer",
    "PPINNTrainingConfig",
    "build_dynamic_validation_panel",
    "evaluate_dynamic_validation_panel",
    "evaluate_ppinn_validation",
    "train_deepsets",
    "train_mlp",
    "train_ppinn",
]
