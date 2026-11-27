"""训练损失可视化命令行入口：读取训练实验目录或 history.json，绘制收敛曲线并默认保存至 ckpt 同目录。"""
from __future__ import annotations

import argparse

from fish_flow.plot_loss import plot_training_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Fish Flow 训练过程损失可视化脚本")
    parser.add_argument(
        "--history",
        required=True,
        help="训练输出目录或 history.json 路径（如 runs/example/deepsets 或 .../history.json）",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出图像路径或目录（默认保存为 history.json / ckpt 同目录下的 loss_curve.png）",
    )
    parser.add_argument(
        "--task",
        default="auto",
        choices=["auto", "mlp", "deepsets", "ppinn"],
        help="训练任务类型（默认 auto 根据 history.json 自动识别）",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="纵轴使用线性坐标（默认使用对数坐标以清晰展示后期收敛）",
    )
    parser.add_argument("--dpi", type=int, default=180, help="保存图像分辨率 DPI")
    args = parser.parse_args()

    out_file = plot_training_history(
        args.history,
        output_path=args.output,
        task=args.task,
        log_scale=not args.linear,
        dpi=args.dpi,
    )
    print(f"训练损失曲线已保存 | 输出={out_file}")


if __name__ == "__main__":
    main()
