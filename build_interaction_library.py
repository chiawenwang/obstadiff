#!/usr/bin/env python3
"""
Build the interaction replay library from decomposed demonstrations.

Each input episode is expected to contain:
  robot_states.npy       columns 7:14 are EEF [x,y,z,qx,qy,qz,qw]
  alignment_end_idx.npy  first bottleneck/interaction frame
  head_camera_seg.npy    (optional) 3-class mask, class 2 = target
  head_camera_depth.npy  (optional) depth in mm

When seg and depth are present, each entry's metadata.json gets the target
image position and depth at the bottleneck frame (target_state_uvd), which
deploy_bbox_seg4ch.py uses to pick a replay and trigger the bottleneck switch.

The output library mirrors the Learning a Thousand Tasks decomposition:
the learned policy handles alignment to a bottleneck, then the interaction
trajectory is replayed relative to the bottleneck reached online.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import scipy.spatial.transform as st
except Exception:
    st = None


def quat_inv(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float32)
    out = q.copy()
    out[:3] *= -1.0
    denom = float(np.dot(q, q))
    if denom > 0:
        out /= denom
    return out


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = [float(x) for x in q1]
    x2, y2, z2, w2 = [float(x) for x in q2]
    out = np.asarray([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=np.float32)
    norm = float(np.linalg.norm(out))
    return out / norm if norm > 0 else out


def make_relative_replay(poses: np.ndarray, anchor_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rel_pos = poses[:, :3].astype(np.float32) - anchor_pose[:3].astype(np.float32)
    rel_quat = []
    if st is not None:
        anchor_rot = st.Rotation.from_quat(anchor_pose[3:7])
        for quat in poses[:, 3:7]:
            rel_quat.append((st.Rotation.from_quat(quat) * anchor_rot.inv()).as_quat())
    else:
        anchor_inv = quat_inv(anchor_pose[3:7])
        for quat in poses[:, 3:7]:
            rel_quat.append(quat_mul(quat, anchor_inv))
    return rel_pos.astype(np.float32), np.asarray(rel_quat, dtype=np.float32)


def load_alignment_end(ep_dir: Path) -> int:
    end_path = ep_dir / "alignment_end_idx.npy"
    if not end_path.exists():
        raise FileNotFoundError(str(end_path))
    return int(np.asarray(np.load(end_path)).item())


def target_box_features(box, depth_mm: np.ndarray,
                        depth_center_frac: float = 0.4,
                        depth_percentile: float = 20.0) -> Dict[str, object]:
    if box is None:
        return {}
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    H, W = depth_mm.shape[:2]
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    feat: Dict[str, object] = {
        "target_u": float((x0 + x1) * 0.5),
        "target_v": float((y0 + y1) * 0.5),
        "bbox": [int(x0), int(y0), int(x1), int(y1)],
        "bbox_area_frac": float((w * h) / max(1, W * H)),
        "bbox_aspect": float(w / max(1, h)),
    }

    x0 = max(0, min(W - 1, x0)); x1 = max(0, min(W, x1))
    y0 = max(0, min(H - 1, y0)); y1 = max(0, min(H, y1))
    crop_frac = float(np.clip(depth_center_frac, 0.05, 1.0))
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    cw = max(1.0, (x1 - x0) * crop_frac)
    ch = max(1.0, (y1 - y0) * crop_frac)
    dx0 = max(0, min(W - 1, int(round(cx - 0.5 * cw))))
    dx1 = max(0, min(W, int(round(cx + 0.5 * cw))))
    dy0 = max(0, min(H - 1, int(round(cy - 0.5 * ch))))
    dy1 = max(0, min(H, int(round(cy + 0.5 * ch))))
    roi = depth_mm[dy0:dy1, dx0:dx1]
    valid = roi[(roi > 150) & (roi < 65000)]
    if valid.size:
        pct = float(np.clip(depth_percentile, 0.0, 100.0))
        feat["target_depth_mm"] = float(np.percentile(valid, pct))
        feat["target_depth_valid_px"] = int(valid.size)
        feat["target_depth_crop_frac"] = crop_frac
        feat["target_depth_percentile"] = pct
        feat["target_state_uvd"] = [
            feat["target_u"],
            feat["target_v"],
            feat["target_depth_mm"],
        ]
    return feat


def target_bbox_from_seg(seg: np.ndarray, target_class: int = 2):
    mask = np.asarray(seg) == int(target_class)
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def export_episode(ep_dir: Path, out_dir: Path, dt: float,
                   depth_center_frac: float,
                   depth_percentile: float) -> Dict[str, object]:
    states_path = ep_dir / "robot_states.npy"
    if not states_path.exists():
        raise FileNotFoundError(str(states_path))

    states = np.load(states_path)
    if states.ndim != 2 or states.shape[1] < 14:
        raise ValueError(f"{states_path} must be (T, >=14), got {states.shape}")

    alignment_end = load_alignment_end(ep_dir)
    if alignment_end < 0 or alignment_end >= states.shape[0]:
        raise ValueError(f"{ep_dir.name}: alignment_end_idx={alignment_end} outside T={states.shape[0]}")

    poses = states[alignment_end:, 7:14].astype(np.float32)
    if poses.shape[0] < 2:
        raise ValueError(f"{ep_dir.name}: interaction phase has only {poses.shape[0]} pose(s)")

    anchor_pose = poses[0].copy()
    rel_pos, rel_quat = make_relative_replay(poses, anchor_pose)

    ep_out = out_dir / ep_dir.name
    ep_out.mkdir(parents=True, exist_ok=True)
    replay_path = ep_out / "interaction_replay.npz"
    np.savez_compressed(
        replay_path,
        rel_pos=rel_pos,
        rel_quat=rel_quat,
        poses=poses,
        anchor_pose=anchor_pose,
        dt=np.asarray(dt, dtype=np.float32),
        source_episode=np.asarray(ep_dir.name),
        alignment_end_idx=np.asarray(alignment_end, dtype=np.int32),
        n_alignment=np.asarray(alignment_end + 1, dtype=np.int32),
        n_interaction=np.asarray(poses.shape[0], dtype=np.int32),
    )

    meta = {
        "name": ep_dir.name,
        "source_episode": ep_dir.name,
        "replay": replay_path.name,
        "dt": float(dt),
        "alignment_end_idx": int(alignment_end),
        "n_alignment": int(alignment_end + 1),
        "n_interaction": int(poses.shape[0]),
        "anchor_pose": [float(x) for x in anchor_pose],
        "final_pose": [float(x) for x in poses[-1]],
    }
    seg_path = ep_dir / "head_camera_seg.npy"
    depth_path = ep_dir / "head_camera_depth.npy"
    if seg_path.exists() and depth_path.exists():
        seg = np.load(seg_path, mmap_mode="r")
        depth = np.load(depth_path, mmap_mode="r")
        if alignment_end < seg.shape[0] and alignment_end < depth.shape[0]:
            bbox = target_bbox_from_seg(seg[alignment_end], target_class=2)
            meta.update(target_box_features(
                bbox,
                np.asarray(depth[alignment_end]),
                depth_center_frac=depth_center_frac,
                depth_percentile=depth_percentile,
            ))
    with open(ep_out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/final_data_decomp",
                        help="Directory containing demo_decomp_* episodes.")
    parser.add_argument("--out", default="data/interaction_library",
                        help="Output replay library directory.")
    parser.add_argument("--dt", type=float, default=1.0 / 15.0,
                        help="Replay waypoint spacing in seconds (recorder runs at 15 Hz).")
    parser.add_argument("--target_depth_center_frac", type=float, default=0.4,
                        help="Center crop fraction for target depth metadata.")
    parser.add_argument("--target_depth_percentile", type=float, default=20.0,
                        help="Depth percentile for target-state metadata.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    episodes = sorted(p for p in root.glob("demo_decomp_*") if p.is_dir())
    if not episodes:
        raise SystemExit(f"No demo_decomp_* directories found in {root}")

    out_dir.mkdir(parents=True, exist_ok=True)
    metas: List[Dict[str, object]] = []
    failures: List[str] = []
    for ep_dir in episodes:
        try:
            meta = export_episode(
                ep_dir,
                out_dir,
                args.dt,
                depth_center_frac=args.target_depth_center_frac,
                depth_percentile=args.target_depth_percentile,
            )
            metas.append(meta)
            target_msg = ""
            if "target_state_uvd" in meta:
                target_msg = f" target_uvd={meta['target_state_uvd']}"
            print(f"OK {ep_dir.name}: {meta['n_interaction']} interaction poses{target_msg}")
        except Exception as exc:
            failures.append(f"{ep_dir.name}: {exc}")
            print(f"FAIL {ep_dir.name}: {exc}")

    library = {
        "source_root": root.name,
        "dt": float(args.dt),
        "num_replays": len(metas),
        "replays": metas,
        "failures": failures,
    }
    with open(out_dir / "library_metadata.json", "w", encoding="utf-8") as f:
        json.dump(library, f, indent=2)
        f.write("\n")

    print(f"\nWrote {len(metas)} replay(s) to {out_dir}")
    if failures:
        raise SystemExit(f"{len(failures)} episode(s) failed; see output above")


if __name__ == "__main__":
    main()
