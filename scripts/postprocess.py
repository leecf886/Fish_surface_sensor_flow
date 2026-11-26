"""统一后处理命令行入口：按推理模式对比预测结果(.npz)与原始工况(.h5)，输出指标与对比云图。"""
from __future__ import annotations

import argparse

from fish_flow.postprocess import run_postprocess


def main() -> None:
    parser = argparse.ArgumentParser(description="Fish Flow 推理结果后处理与对比云图脚本")
    parser.add_argument("--mode", required=True, choices=["encoder", "decoder", "cascade"], help="推理模式")
    parser.add_argument("--pred", required=True, help="模型推理输出 .npz 文件或输出目录路径")
    parser.add_argument("--raw", default="data/processed", help="原始预处理工况目录 (含 mesh.h5 与 ReXXX.h5)")
    parser.add_argument("--output-dir", default=None, help="后处理指标与云图输出目录（默认与推理结果同目录）")
    parser.add_argument("--re-list", type=int, nargs="+", default=None, help="仅评估指定雷诺数工况")
    parser.add_argument("--frame-idx", type=int, default=0, help="每个工况用于绘制对比云图的帧下标")
    parser.add_argument("--no-plot", action="store_true", help="仅计算并导出 metrics.json，不绘制对比云图")
    args = parser.parse_args()

    report = run_postprocess(
        mode=args.mode,
        pred_path=args.pred,
        raw_dir=args.raw,
        output_dir=args.output_dir,
        re_list=args.re_list,
        frame_idx=args.frame_idx,
        plot=not args.no_plot,
    )

    if "encoder_metrics" in report:
        em = report["encoder_metrics"]["overall"]
        print(
            f"[编码器指标] Re_RMSE={em['re_rmse']:.2f} | St_RMSE={em['st_rmse']:.4f} | "
            f"Phase_cRMSE={em['phase_crmse']:.4f} rad"
        )
    if "field_metrics" in report:
        fm = report["field_metrics"]["overall"]
        print(
            f"[流场指标] u_relL2={fm['u_rel_l2']:.4f} | v_relL2={fm['v_rel_l2']:.4f} | "
            f"p_gauge_relL2={fm['p_rel_l2_gauge_sameden']:.4f} | "
            f"wake_overall={fm.get('wake_overall_rel_l2_gauge_sameden', float('nan')):.4f}"
        )
    print(f"后处理完成 | 图像数量={len(report['figures'])}")


if __name__ == "__main__":
    main()
