"""统一推理命令行入口：读取推理 config.json，执行编码器 / 解码器 / 串联推理并导出 .npz。"""
from __future__ import annotations

import argparse

from fish_flow.inference import load_infer_config, run_inference_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Fish Flow 单文件配置推理启动脚本")
    parser.add_argument(
        "--config",
        required=True,
        help="推理 config.json 路径或目录（如 runs/example/infer_cascade 或 .../config.json）",
    )
    parser.add_argument("--mode", choices=["encoder", "decoder", "cascade"], default=None, help="覆盖推理模式")
    parser.add_argument("--encoder-ckpt", default=None, help="覆盖编码器最优权重路径")
    parser.add_argument("--ppinn-ckpt", default=None, help="覆盖 pPINN 最优权重路径")
    parser.add_argument("--processed-dir", default=None, help="覆盖预处理数据目录")
    parser.add_argument("--re-list", type=int, nargs="+", default=None, help="覆盖推理雷诺数工况列表")
    parser.add_argument("--output-dir", default=None, help="覆盖输出目录（默认与 config.json 同目录）")
    parser.add_argument("--output-name", default=None, help="覆盖输出 .npz 文件名")
    parser.add_argument("--output-path", default=None, help="直接指定输出 .npz 完整路径")
    parser.add_argument("--device", default=None, help="覆盖计算设备 (auto/cpu/cuda)")
    args = parser.parse_args()

    overrides = {
        "mode": args.mode,
        "encoder_ckpt": args.encoder_ckpt,
        "ppinn_ckpt": args.ppinn_ckpt,
        "processed_dir": args.processed_dir,
        "re_list": args.re_list,
        "output_dir": args.output_dir,
        "output_name": args.output_name,
        "output_path": args.output_path,
        "device": args.device,
    }
    cfg, _, _ = load_infer_config(args.config, overrides)
    mode = str(cfg["mode"]).strip().lower()
    pred, out_npz = run_inference_from_config(args.config, overrides=overrides)

    if mode == "encoder":
        print(f"编码器推理完成 | 样本数={len(pred.re)} | 输出={out_npz}")
    elif mode == "decoder":
        print(f"解码器推理完成 | 流场形状={tuple(pred.uvp.shape)} | 输出={out_npz}")
    else:
        print(
            f"串联推理完成 | 样本数={len(pred.encoder.re)} | "
            f"流场形状={tuple(pred.decoder.uvp.shape)} | 输出={out_npz}"
        )


if __name__ == "__main__":
    main()
