import argparse
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_samples(path, pose_id):
    rows = []
    with open(path, "r", newline="") as file:
        for row in csv.DictReader(file):
            if int(row["pose_id"]) == pose_id:
                rows.append(row)
    rows.sort(key=lambda r: int(r["dir_idx"]))
    return rows


def load_guides(path):
    guides = {}
    with open(path, "r", newline="") as file:
        for row in csv.DictReader(file):
            sample_id = int(row["sample_id"])
            guides.setdefault(sample_id, []).append(
                [float(row["x"]), float(row["y"]), float(row["z"])]
            )
    return {sample_id: np.asarray(points, dtype=np.float32) for sample_id, points in guides.items()}


def read_ply_xyz(path):
    ply_to_numpy = {
        "char": "i1",
        "int8": "i1",
        "uchar": "u1",
        "uint8": "u1",
        "short": "i2",
        "int16": "i2",
        "ushort": "u2",
        "uint16": "u2",
        "int": "i4",
        "int32": "i4",
        "uint": "u4",
        "uint32": "u4",
        "float": "f4",
        "float32": "f4",
        "double": "f8",
        "float64": "f8",
    }

    with open(path, "rb") as file:
        header = []
        while True:
            line = file.readline()
            if not line:
                raise ValueError(f"Invalid PLY file without end_header: {path}")
            decoded = line.decode("ascii").strip()
            header.append(decoded)
            if decoded == "end_header":
                break

        fmt = None
        vertex_count = None
        properties = []
        in_vertex = False
        for line in header:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                if parts[1] == "list":
                    raise ValueError(f"PLY list properties are not supported for vertices: {path}")
                properties.append((parts[2], parts[1]))

        property_names = [name for name, _ in properties]
        for name in ("x", "y", "z"):
            if name not in property_names:
                raise ValueError(f"PLY file missing vertex property '{name}': {path}")

        if fmt == "ascii":
            data = np.loadtxt(file, max_rows=vertex_count)
            return data[:, [property_names.index("x"), property_names.index("y"), property_names.index("z")]].astype(np.float32)

        endian = "<" if fmt == "binary_little_endian" else ">" if fmt == "binary_big_endian" else None
        if endian is None:
            raise ValueError(f"Unsupported PLY format '{fmt}': {path}")
        dtype = np.dtype([(name, endian + ply_to_numpy[prop_type]) for name, prop_type in properties])
        data = np.frombuffer(file.read(dtype.itemsize * vertex_count), dtype=dtype, count=vertex_count)
        return np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)


def crop_obstacles(points, origin, radius, z_min, z_max):
    xy_dist = np.linalg.norm(points[:, :2] - origin[None, :2], axis=1)
    mask = (xy_dist <= radius) & (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    return points[mask]


def draw_sector(ax, origin, radius, sector_idx, sector_num, color):
    width = 2.0 * math.pi / sector_num
    center = sector_idx * width
    a0 = center - width / 2.0
    a1 = center + width / 2.0
    angles = np.linspace(a0, a1, 24)
    z = origin[2] - 0.08
    verts = [[origin[0], origin[1], z]]
    verts += [[origin[0] + radius * math.cos(a), origin[1] + radius * math.sin(a), z] for a in angles]
    poly = Poly3DCollection([verts], facecolor=color, edgecolor=color, alpha=0.08, linewidth=0.8)
    ax.add_collection3d(poly)
    ax.plot(
        [origin[0], origin[0] + radius * math.cos(a0)],
        [origin[1], origin[1] + radius * math.sin(a0)],
        [z, z],
        color=color,
        alpha=0.25,
        linewidth=0.8,
    )


def equalize_axes(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    span = (maxs - mins).max()
    span = max(span, 1.0)
    ax.set_xlim(center[0] - span / 2.0, center[0] + span / 2.0)
    ax.set_ylim(center[1] - span / 2.0, center[1] + span / 2.0)
    ax.set_zlim(max(0.0, center[2] - span / 3.0), center[2] + span / 3.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--map-id", type=int, default=0)
    parser.add_argument("--pose-id", type=int, default=0)
    parser.add_argument("--output", default="docs/yopo_guides_pose0.png")
    parser.add_argument("--radius", type=float, default=10.0)
    parser.add_argument("--obstacle-radius", type=float, default=13.0)
    parser.add_argument("--obstacle-mode", choices=["low", "mid", "full"], default="mid")
    parser.add_argument("--obstacle-z-min", type=float, default=None)
    parser.add_argument("--obstacle-z-max", type=float, default=None)
    parser.add_argument("--max-obstacle-points", type=int, default=0)
    parser.add_argument("--obstacle-alpha", type=float, default=0.14)
    parser.add_argument("--obstacle-size", type=float, default=2.2)
    args = parser.parse_args()

    samples = load_samples(os.path.join(args.dataset, f"samples-{args.map_id}.csv"), args.pose_id)
    guides = load_guides(os.path.join(args.dataset, f"guides-{args.map_id}.csv"))
    if not samples:
        raise ValueError(f"No samples found for pose_id={args.pose_id}")

    origin = np.array(
        [float(samples[0]["px"]), float(samples[0]["py"]), float(samples[0]["pz"])],
        dtype=np.float32,
    )
    sector_num = len(samples)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(sector_num, 10)))

    fig = plt.figure(figsize=(13, 10), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"YOPO-Omni 2.5D Guide Visualization: map {args.map_id}, pose {args.pose_id}", pad=18)

    all_points = [origin[None, :]]
    pointcloud_path = os.path.join(args.dataset, f"pointcloud-{args.map_id}.ply")
    if os.path.exists(pointcloud_path):
        if args.obstacle_z_min is not None and args.obstacle_z_max is not None:
            obstacle_z_min = args.obstacle_z_min
            obstacle_z_max = args.obstacle_z_max
        elif args.obstacle_mode == "low":
            obstacle_z_min = 0.0
            obstacle_z_max = origin[2] + 0.7
        elif args.obstacle_mode == "mid":
            obstacle_z_min = max(0.0, origin[2] - 1.0)
            obstacle_z_max = origin[2] + 1.4
        else:
            obstacle_z_min = 0.0
            obstacle_z_max = origin[2] + 8.0

        obstacles = crop_obstacles(
            read_ply_xyz(pointcloud_path),
            origin,
            args.obstacle_radius,
            obstacle_z_min,
            obstacle_z_max,
        )
        if args.max_obstacle_points > 0 and obstacles.shape[0] > args.max_obstacle_points:
            rng = np.random.default_rng(0)
            obstacles = obstacles[rng.choice(obstacles.shape[0], args.max_obstacle_points, replace=False)]
        if obstacles.shape[0] > 0:
            ax.scatter(
                obstacles[:, 0],
                obstacles[:, 1],
                obstacles[:, 2],
                c="0.2",
                s=args.obstacle_size,
                alpha=args.obstacle_alpha,
                depthshade=False,
                label="obstacles",
            )
            all_points.append(obstacles)

    for idx, row in enumerate(samples):
        color = colors[idx]
        draw_sector(ax, origin, args.radius, idx, sector_num, color)

        sample_id = int(row["sample_id"])
        goal = np.array([float(row["goal_wx"]), float(row["goal_wy"]), float(row["goal_wz"])], dtype=np.float32)
        vdes = np.array([float(row["vdes_bx"]), float(row["vdes_by"]), float(row["vdes_bz"])], dtype=np.float32)
        guide_mask = int(float(row["guide_mask"]))
        selected_topology = int(row["selected_topology"])

        arrow_len = args.radius * 0.38
        ax.quiver(
            origin[0],
            origin[1],
            origin[2] + 0.18,
            vdes[0] * arrow_len,
            vdes[1] * arrow_len,
            vdes[2] * arrow_len,
            color=color,
            linewidth=1.4,
            arrow_length_ratio=0.16,
        )
        ax.text(
            origin[0] + vdes[0] * (arrow_len + 0.35),
            origin[1] + vdes[1] * (arrow_len + 0.35),
            origin[2] + 0.2,
            f"d{idx}->t{selected_topology}",
            color=color,
            fontsize=8,
        )

        ax.scatter(goal[0], goal[1], goal[2], marker="x", color=color, s=35, alpha=0.9)
        all_points.append(goal[None, :])

        guide = guides.get(sample_id)
        if guide_mask and guide is not None and guide.shape[0] > 0:
            ax.plot(guide[:, 0], guide[:, 1], guide[:, 2], color=color, linewidth=2.2, alpha=0.9)
            ax.scatter(guide[-1, 0], guide[-1, 1], guide[-1, 2], color=color, s=18, alpha=0.9)
            all_points.append(guide)
        else:
            ax.plot(
                [origin[0], goal[0]],
                [origin[1], goal[1]],
                [origin[2], goal[2]],
                color=color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.35,
            )

    ax.scatter(origin[0], origin[1], origin[2], color="black", s=80, marker="o", label="sample point")
    ax.text(origin[0], origin[1], origin[2] + 0.45, "sample", color="black", fontsize=9)

    points = np.vstack(all_points)
    equalize_axes(ax, points)
    ax.set_xlabel("X world (m)")
    ax.set_ylabel("Y world (m)")
    ax.set_zlabel("Z / height (m)")
    ax.view_init(elev=36, azim=-58)
    ax.grid(True, alpha=0.25)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
