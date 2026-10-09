#!/usr/bin/env python3
"""
Create an interaction replay file from a recorded GROVE demo.

The deployment alignment policy stops at a bottleneck pose. This script takes
the demonstration trajectory after that bottleneck and stores it relative to the
bottleneck so deploy_bbox_seg4ch.py can replay it from the live aligned pose.
"""

import argparse
import os

import numpy as np


def quat_inv(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    out = q.copy()
    out[:3] *= -1.0
    n = float(np.dot(q, q))
    if n > 1e-8:
        out /= n
    return out


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float32)


def make_relative_replay(poses: np.ndarray, anchor_pose: np.ndarray):
    anchor_inv = quat_inv(anchor_pose[3:7])
    rel_pos = poses[:, :3].astype(np.float32) - anchor_pose[:3].astype(np.float32)
    rel_quat = []
    for q in poses[:, 3:7]:
        rel_quat.append(quat_mul(q, anchor_inv))
    return rel_pos.astype(np.float32), np.asarray(rel_quat, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("demo_dir",
                        help="Demo folder containing robot_states.npy.")
    parser.add_argument("--out", default=None,
                        help="Output .npz path. Defaults to demo_dir/interaction_replay.npz.")
    parser.add_argument("--start_idx", type=int, default=None,
                        help="Override bottleneck index. Default: min z from EEF pose.")
    parser.add_argument("--start_offset", type=int, default=0,
                        help="Additional frames after bottleneck before replay starts.")
    parser.add_argument("--end_idx", type=int, default=None,
                        help="Optional exclusive end index.")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Keep every Nth pose.")
    parser.add_argument("--dt", type=float, default=1.0 / 15.0,
                        help="Source waypoint dt in seconds. Recorder is usually 15 Hz.")
    args = parser.parse_args()

    demo_dir = os.path.abspath(args.demo_dir)
    states_path = os.path.join(demo_dir, "robot_states.npy")
    if not os.path.exists(states_path):
        raise FileNotFoundError(states_path)

    states = np.load(states_path)
    if states.ndim != 2 or states.shape[1] < 14:
        raise ValueError(f"{states_path}: expected robot_states shape (T, >=14), got {states.shape}")

    poses = states[:, 7:14].astype(np.float32)
    if args.start_idx is None:
        bottleneck_idx = int(np.argmin(poses[:, 2]))
    else:
        bottleneck_idx = int(args.start_idx)
    start_idx = max(0, min(len(poses) - 1, bottleneck_idx + int(args.start_offset)))
    end_idx = len(poses) if args.end_idx is None else max(start_idx + 1, min(len(poses), int(args.end_idx)))
    downsample = max(1, int(args.downsample))

    replay_poses = poses[start_idx:end_idx:downsample]
    anchor_pose = poses[bottleneck_idx]
    rel_pos, rel_quat = make_relative_replay(replay_poses, anchor_pose)

    out_path = args.out or os.path.join(demo_dir, "interaction_replay.npz")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        rel_pos=rel_pos,
        rel_quat=rel_quat,
        poses=replay_poses,
        anchor_pose=anchor_pose,
        bottleneck_idx=np.array(bottleneck_idx, dtype=np.int32),
        start_idx=np.array(start_idx, dtype=np.int32),
        end_idx=np.array(end_idx, dtype=np.int32),
        downsample=np.array(downsample, dtype=np.int32),
        dt=np.array(float(args.dt) * downsample, dtype=np.float32),
        source_demo=np.array(demo_dir),
    )

    total_disp = float(np.linalg.norm(replay_poses[-1, :3] - replay_poses[0, :3]))
    print(f"saved {out_path}")
    print(f"  source demo: {demo_dir}")
    print(f"  bottleneck_idx={bottleneck_idx} start_idx={start_idx} end_idx={end_idx} downsample={downsample}")
    print(f"  waypoints={len(replay_poses)} dt={float(args.dt) * downsample:.3f}s total_disp={total_disp*1000:.1f}mm")
    print(f"  anchor_pose={np.array2string(anchor_pose, precision=5)}")


if __name__ == "__main__":
    main()
