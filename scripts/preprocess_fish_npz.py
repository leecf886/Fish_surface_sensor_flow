"""文件二：单线程循环读取各雷诺数工况 npz，按文件一确定的参数裁剪 ROI 并拼接元数据输出为精悍的 ReXXX.h5。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import h5py
import numpy as np


def parse_re_value(case_dir: Path) -> int:
    """从工况目录名（如 Re13500）提取整数雷诺数。"""
    return int(re.search(r"(\d+)", case_dir.name).group(1))


def convert_one_case(
    case_dir: Path,
    out_dir: Path,
    roi_cell_ids: np.ndarray,
    sensor_cell_ids: np.ndarray,
    L_m_default: float,
) -> tuple[int, float, tuple[int, ...], tuple[int, ...], float]:
    """读取单个工况的 two_cycles.npz 与 cycle_quality.json，裁剪并写入 ReXXX.h5。"""
    # 1. 读取该工况的 npz 流场数据与质量参数 json
    npz_path = case_dir / "two_cycles.npz"
    quality_path = case_dir / "cycle_quality.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))

    with np.load(npz_path) as data:
        fields = data["fields"]  # [2, 40, 60048, 3], float32: [Ux, Uy, kinematic_p]
        phase_cycle = data["phase"].astype(np.float64)  # [40], 归一化周期 [0, 0.975]
        t_star_2d = data["t_star"].astype(np.float64)  # [2, 40]

        n_cycles, n_phases, _, n_channels = fields.shape
        nt = n_cycles * n_phases

        # 2. 先基于全网格索引提取近壁候选传感器信号，再按 roi_cell_ids 截取密集区流场
        sensor_uvp = fields[:, :, sensor_cell_ids, :].reshape(nt, len(sensor_cell_ids), n_channels)
        sensor_input_flat = sensor_uvp.reshape(nt, -1)
        field_uvp = fields[:, :, roi_cell_ids, :].reshape(nt, len(roi_cell_ids), n_channels)

    # 3. 拼接周期相位与物理元数据（两个周期顺次展平为 nt=80 帧）
    phase_rad = np.mod(np.tile(phase_cycle * (2.0 * np.pi), n_cycles), 2.0 * np.pi)
    phase_sin = np.sin(phase_rad)
    phase_cos = np.cos(phase_rad)
    periodic_mask = np.ones(nt, dtype=np.int32)

    re_val = float(quality["Re"])
    u_inf = float(quality["U_inf_m_s"])
    L_m = float(quality.get("L_m", L_m_default))
    periods_star = np.asarray(quality["periods_star"], dtype=np.float64)
    st_val = float(1.0 / np.mean(periods_star))

    t_star = t_star_2d.reshape(nt)
    time_s = t_star * (L_m / u_inf)

    # 4. 写出精悍的 HDF5 工况文件（流场与传感器保持原生 float32 精度以控制体积）
    re_int = int(round(re_val))
    out_h5 = out_dir / f"Re{re_int}.h5"
    with h5py.File(out_h5, "w") as h5:
        # 元数据组：显式保存 St 与 uinf2 压强缩放约定
        h5.create_dataset("/meta/Re", data=np.array([re_val], dtype=np.float64))
        h5.create_dataset("/meta/U_inf", data=np.array([u_inf], dtype=np.float64))
        h5.create_dataset("/meta/L", data=np.array([L_m], dtype=np.float64))
        h5.create_dataset("/meta/St", data=np.array([st_val], dtype=np.float64))
        h5.create_dataset("/meta/periods_star", data=periods_star, dtype=np.float64)
        h5.create_dataset("/meta/Nt", data=np.array([nt], dtype=np.int32))
        h5.create_dataset("/meta/Nroi", data=np.array([len(roi_cell_ids)], dtype=np.int32))
        h5.create_dataset("/meta/Ncand", data=np.array([len(sensor_cell_ids)], dtype=np.int32))
        h5.attrs["pressure_scale_mode"] = "uinf2"
        h5.attrs["variables"] = ["Ux_m_s", "Uy_m_s", "kinematic_p_m2_s2"]

        # 快照时间与相位组
        h5.create_dataset("/snapshots/phase", data=phase_rad, dtype=np.float64)
        h5.create_dataset("/snapshots/phase_sin", data=phase_sin, dtype=np.float64)
        h5.create_dataset("/snapshots/phase_cos", data=phase_cos, dtype=np.float64)
        h5.create_dataset("/snapshots/periodic_mask", data=periodic_mask, dtype=np.int32)
        h5.create_dataset("/snapshots/t_star", data=t_star, dtype=np.float64)
        h5.create_dataset("/snapshots/time", data=time_s, dtype=np.float64)

        # 近壁传感器读数与核心 ROI 流场组
        h5.create_dataset("/sensors/sensor_uvp", data=sensor_uvp, dtype=np.float32)
        h5.create_dataset("/sensors/sensor_input_flat", data=sensor_input_flat, dtype=np.float32)
        h5.create_dataset("/field/uvp", data=field_uvp, dtype=np.float32)

    size_mb = out_h5.stat().st_size / (1024.0 * 1024.0)
    return re_int, st_val, sensor_uvp.shape, field_uvp.shape, size_mb


def main() -> None:
    parser = argparse.ArgumentParser(description="循环读取各 Re 工况 npz 并裁剪输出为 ReXXX.h5")
    parser.add_argument("--data-dir", type=Path, default=Path("data_files"), help="原始 Re* 工况目录")
    parser.add_argument("--params", type=Path, default=Path("configs/mesh_roi_params.json"), help="文件一生成的截取参数 JSON")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"), help="输出 ReXXX.h5 目录")
    args = parser.parse_args()

    # 读取文件一输出的 ROI 单元索引、传感器单元索引与几何尺度
    params = json.loads(args.params.read_text(encoding="utf-8"))
    roi_cell_ids = np.asarray(params["roi_cell_ids"], dtype=np.int64)
    sensor_cell_ids = np.asarray(params["sensor_cell_ids"], dtype=np.int64)
    L_m_default = float(params["L_m"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    case_dirs = sorted(
        (p for p in args.data_dir.iterdir() if p.is_dir() and re.match(r"^Re\d+$", p.name)),
        key=parse_re_value,
    )

    # 按雷诺数顺序单线程循环处理每个工况
    total_mb = 0.0
    for idx, case_dir in enumerate(case_dirs, start=1):
        re_int, st_val, sensor_shape, field_shape, size_mb = convert_one_case(
            case_dir=case_dir,
            out_dir=args.out_dir,
            roi_cell_ids=roi_cell_ids,
            sensor_cell_ids=sensor_cell_ids,
            L_m_default=L_m_default,
        )
        total_mb += size_mb
        print(
            f"[{idx:02d}/{len(case_dirs):02d}] Re{re_int}.h5 | "
            f"St={st_val:.4f} | sensor={sensor_shape} | field={field_shape} | {size_mb:.2f} MB"
        )

    print(f"全部处理完成: 共 {len(case_dirs)} 个工况, 输出目录 {args.out_dir}, 总体积 {total_mb:.2f} MB")


if __name__ == "__main__":
    main()
