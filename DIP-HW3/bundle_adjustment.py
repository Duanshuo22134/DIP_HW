"""DIP-HW3 的 Bundle Adjustment 实现。

本脚本从 2D 观测点优化恢复共享焦距、每个视角的相机外参和 3D 点云，
并导出带颜色的 OBJ 点云、loss 曲线和优化参数。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


IMAGE_SIZE = 1024
CX = IMAGE_SIZE / 2.0
CY = IMAGE_SIZE / 2.0


def load_observations(path: Path, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """读取 2D 观测数据。

    Args:
        path: `points2d.npz` 的路径。文件中每个 view 的形状为 `(N, 3)`，
            三列分别表示 `x, y, visibility`。
        device: 输出张量所在设备，例如 CPU 或 CUDA。

    Returns:
        observations: 形状为 `(V, N, 2)` 的 2D 像素坐标张量。
        visibility: 形状为 `(V, N)` 的可见性布尔张量。
    """
    data = np.load(path)
    keys = sorted(data.files)
    observations = np.stack([data[key][:, :2] for key in keys], axis=0).astype(np.float32)
    visibility = np.stack([data[key][:, 2] for key in keys], axis=0).astype(bool)
    return (
        torch.from_numpy(observations).to(device),
        torch.from_numpy(visibility).to(device),
    )


def euler_xyz_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    """将 XYZ 顺序的 Euler 角转换为旋转矩阵。

    Args:
        euler: 形状为 `(..., 3)` 的张量，最后一维依次为绕 x、y、z 轴的旋转角，
            单位为弧度。

    Returns:
        形状为 `(..., 3, 3)` 的旋转矩阵张量。
    """
    x, y, z = euler.unbind(dim=-1)
    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    cz, sz = torch.cos(z), torch.sin(z)

    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    rx = torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cx, -sx], dim=-1),
            torch.stack([zeros, sx, cx], dim=-1),
        ],
        dim=-2,
    )
    ry = torch.stack(
        [
            torch.stack([cy, zeros, sy], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-sy, zeros, cy], dim=-1),
        ],
        dim=-2,
    )
    rz = torch.stack(
        [
            torch.stack([cz, -sz, zeros], dim=-1),
            torch.stack([sz, cz, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )
    return rx @ ry @ rz


def project_points(
    points3d: torch.Tensor,
    euler: torch.Tensor,
    translation: torch.Tensor,
    focal: torch.Tensor,
    view_ids: torch.Tensor,
) -> torch.Tensor:
    """将 3D 点投影到指定视角的 2D 图像平面。

    Args:
        points3d: 形状为 `(N, 3)` 的世界坐标系 3D 点。
        euler: 形状为 `(V, 3)` 的每视角 Euler 角。
        translation: 形状为 `(V, 3)` 的每视角平移向量。
        focal: 共享焦距标量。
        view_ids: 本次参与投影的视角编号。

    Returns:
        形状为 `(B, N, 2)` 的投影坐标，其中 `B=len(view_ids)`。
    """
    rotation = euler_xyz_to_matrix(euler[view_ids])
    t = translation[view_ids]
    camera_points = torch.einsum("vij,nj->vni", rotation, points3d) + t[:, None, :]
    z = camera_points[..., 2].clamp(max=-1e-4)
    u = -focal * camera_points[..., 0] / z + CX
    v = focal * camera_points[..., 1] / z + CY
    return torch.stack([u, v], dim=-1)


def initialize_points(observations: torch.Tensor, visibility: torch.Tensor, focal: float, depth: float) -> torch.Tensor:
    """根据平均 2D 观测初始化一个粗略的正面 3D 点云。

    初始化只需要给优化一个合理起点：先对每个点在所有可见视角中的 2D 坐标
    求平均，再假设其深度约为 `depth`，按针孔模型反投影回 3D 空间。

    Args:
        observations: 形状为 `(V, N, 2)` 的 2D 观测。
        visibility: 形状为 `(V, N)` 的可见性 mask。
        focal: 初始焦距。
        depth: 初始相机到物体的大致距离。

    Returns:
        形状为 `(N, 3)` 的初始 3D 点坐标。
    """
    mask = visibility.float()
    denom = mask.sum(dim=0).clamp_min(1.0)
    mean_xy = (observations * mask[..., None]).sum(dim=0) / denom[:, None]

    x = (mean_xy[:, 0] - CX) * depth / focal
    y = -(mean_xy[:, 1] - CY) * depth / focal
    z = torch.zeros_like(x)
    points = torch.stack([x, y, z], dim=-1)
    points = points + 0.01 * torch.randn_like(points)
    return points


def save_obj(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """保存带顶点颜色的 OBJ 点云。

    Args:
        path: 输出 OBJ 文件路径。
        points: 形状为 `(N, 3)` 的 3D 点坐标。
        colors: 形状为 `(N, 3)` 的 RGB 颜色，数值范围应为 `[0, 1]`。
    """
    colors = np.clip(colors, 0.0, 1.0)
    with path.open("w", encoding="utf-8") as f:
        for p, c in zip(points, colors):
            f.write(f"v {p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")


def save_loss_curve(path: Path, losses: list[float]) -> None:
    """保存优化过程中的 loss 曲线。

    如果环境中安装了 Matplotlib，则保存为图片；否则退化为保存同名 `.txt`
    文本文件，避免因为可视化依赖缺失而中断主流程。

    Args:
        path: 输出曲线图片路径。
        losses: 每一次迭代记录的 loss 数值。
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        np.savetxt(path.with_suffix(".txt"), np.asarray(losses, dtype=np.float32))
        print("matplotlib is not installed; saved loss values as txt instead.")
        return

    plt.figure(figsize=(7, 4))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Mean robust reprojection loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def optimize(args: argparse.Namespace) -> Dict[str, object]:
    """执行 Bundle Adjustment 优化主循环。

    优化变量包括 3D 点坐标、每个视角的 Euler 角、每个视角的平移向量，以及
    所有视角共享的焦距。目标函数是可见点上的鲁棒 2D 重投影误差，附加一个
    深度约束用于减少点跑到相机后方的情况。

    Args:
        args: 命令行参数，包含数据路径、迭代次数、学习率和初始化设置。

    Returns:
        一个字典，包含优化后的 `points`、`euler`、`translation`、`focal`
        以及完整的 `losses` 列表。
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    observations, visibility = load_observations(args.points2d, device)
    num_views, _, _ = observations.shape

    init_focal = IMAGE_SIZE / (2.0 * math.tan(math.radians(args.init_fov_deg) / 2.0))
    init_depth = args.init_depth

    points = initialize_points(observations, visibility, init_focal, init_depth).detach().requires_grad_(True)
    euler_init = torch.zeros(num_views, 3, device=device)
    if args.init_yaw_range_deg > 0:
        yaw = torch.linspace(
            -math.radians(args.init_yaw_range_deg),
            math.radians(args.init_yaw_range_deg),
            num_views,
            device=device,
        )
        euler_init[:, 1] = yaw
    euler = euler_init.detach().requires_grad_(True)

    translation_init = torch.zeros(num_views, 3, device=device)
    translation_init[:, 2] = -init_depth
    translation = translation_init.detach().requires_grad_(True)
    log_focal = torch.tensor(math.log(init_focal), device=device, requires_grad=True)

    optimizer = torch.optim.Adam(
        [
            {"params": [points], "lr": args.lr_points},
            {"params": [euler, translation], "lr": args.lr_cameras},
            {"params": [log_focal], "lr": args.lr_focal},
        ]
    )

    all_views = torch.arange(num_views, device=device)
    losses: list[float] = []

    for step in range(1, args.iters + 1):
        if args.view_batch_size and args.view_batch_size < num_views:
            perm = torch.randperm(num_views, device=device)[: args.view_batch_size]
            view_ids = perm.sort().values
        else:
            view_ids = all_views

        optimizer.zero_grad(set_to_none=True)
        focal = torch.exp(log_focal)
        projected = project_points(points, euler, translation, focal, view_ids)
        target = observations[view_ids]
        mask = visibility[view_ids]

        residual = projected - target
        robust = F.smooth_l1_loss(projected[mask], target[mask], beta=args.huber_beta, reduction="mean")

        visible_depth = (torch.einsum("vij,nj->vni", euler_xyz_to_matrix(euler[view_ids]), points) + translation[view_ids, None, :])[
            ..., 2
        ][mask]
        depth_penalty = F.relu(visible_depth + 1e-3).mean() if visible_depth.numel() > 0 else torch.zeros((), device=device)
        center_penalty = 1e-4 * (points.mean(dim=0) ** 2).sum()
        loss = robust + args.depth_weight * depth_penalty + center_penalty
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        losses.append(loss_value)
        if step == 1 or step % args.log_every == 0 or step == args.iters:
            with torch.no_grad():
                rmse = torch.sqrt((residual[mask].pow(2).sum(dim=-1)).mean()).item()
                print(
                    f"iter {step:5d}/{args.iters}  loss={loss_value:.6f}  "
                    f"rmse={rmse:.3f}px  f={float(torch.exp(log_focal)):.2f}"
                )

    return {
        "points": points.detach().cpu().numpy(),
        "euler": euler.detach().cpu().numpy(),
        "translation": translation.detach().cpu().numpy(),
        "focal": float(torch.exp(log_focal).detach().cpu()),
        "losses": losses,
    }


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        包含路径、训练超参数和初始化参数的 argparse 命名空间。
    """
    parser = argparse.ArgumentParser(description="PyTorch Bundle Adjustment for DIP-HW3")
    parser.add_argument("--points2d", type=Path, default=Path("data/points2d.npz"))
    parser.add_argument("--colors", type=Path, default=Path("data/points3d_colors.npy"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/task1"))
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda, or leave empty for auto")
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--view-batch-size", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init-fov-deg", type=float, default=55.0)
    parser.add_argument("--init-depth", type=float, default=3.0)
    parser.add_argument("--init-yaw-range-deg", type=float, default=70.0)
    parser.add_argument("--lr-points", type=float, default=0.01)
    parser.add_argument("--lr-cameras", type=float, default=0.003)
    parser.add_argument("--lr-focal", type=float, default=0.001)
    parser.add_argument("--huber-beta", type=float, default=4.0)
    parser.add_argument("--depth-weight", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    """脚本入口：解析参数、运行优化并保存全部 Task 1 输出结果。"""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    result = optimize(args)
    colors = np.load(args.colors).astype(np.float32)

    save_obj(args.out_dir / "reconstruction.obj", result["points"], colors)
    save_loss_curve(args.out_dir / "loss_curve.png", result["losses"])
    np.savez(
        args.out_dir / "optimized_parameters.npz",
        points=result["points"],
        euler=result["euler"],
        translation=result["translation"],
        focal=np.asarray(result["focal"], dtype=np.float32),
        losses=np.asarray(result["losses"], dtype=np.float32),
    )

    print("Saved:")
    print(f"  {args.out_dir / 'reconstruction.obj'}")
    print(f"  {args.out_dir / 'loss_curve.png'}")
    print(f"  {args.out_dir / 'optimized_parameters.npz'}")


if __name__ == "__main__":
    main()
