"""统一训练命令行入口：直接读取 runs/ 下的单个 config.json 启动训练。"""
from __future__ import annotations

import argparse

from fish_flow.config import load_run_config, run_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Fish Flow 单文件配置训练启动脚本")
    parser.add_argument(
        "--config",
        required=True,
        help="runs/ 下的 config.json 路径或实验目录（如 runs/mlp 或 runs/mlp/config.json）",
    )
    parser.add_argument("--output-dir", default=None, help="覆盖模型保存目录（默认与 config.json 同目录）")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖配置中的 num_epochs")
    parser.add_argument("--seed", type=int, default=None, help="覆盖配置中的随机种子")
    parser.add_argument("--device", default=None, help="覆盖配置中的计算设备 (auto/cpu/cuda)")
    args = parser.parse_args()

    overrides = {
        "output_dir": args.output_dir,
        "num_epochs": args.epochs,
        "seed": args.seed,
        "device": args.device,
    }
    cfg, out_dir = load_run_config(args.config, overrides)
    task = str(cfg["task"]).strip().lower()
    _, history = run_from_config(args.config, overrides=overrides)

    if task == "ppinn":
        print(
            f"pPINN 训练完成 | 目录={out_dir} | best_epoch={history.best_epoch} | "
            f"{history.best_metric_name}={history.best_metric_value:.6f}"
        )
    else:
        print(
            f"{task.upper()} 训练完成 | 目录={out_dir} | best_epoch={history.best_epoch} | "
            f"best_score={history.best_selection_loss:.6f}"
        )


if __name__ == "__main__":
    main()
