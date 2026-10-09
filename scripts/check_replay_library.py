#!/usr/bin/env python3
"""Inspect an ObstaDiff interaction replay library."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def replay_motion_mm(npz_path: Path) -> float | None:
    data = np.load(npz_path)
    if "rel_pos" in data:
        rel_pos = np.asarray(data["rel_pos"], dtype=np.float64)
        if rel_pos.ndim == 2 and rel_pos.shape[1] == 3 and len(rel_pos):
            return float(np.linalg.norm(rel_pos[-1] - rel_pos[0]) * 1000.0)
    if "poses" in data:
        poses = np.asarray(data["poses"], dtype=np.float64)
        if poses.ndim == 2 and poses.shape[1] >= 3 and len(poses):
            return float(np.linalg.norm(poses[-1, :3] - poses[0, :3]) * 1000.0)
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True, help="Interaction library directory.")
    args = parser.parse_args()

    root = Path(args.library)
    entries = sorted(p for p in root.iterdir() if p.is_dir())
    if not entries:
        raise SystemExit(f"No replay entries found under {root}")

    print(f"Found {len(entries)} replay entries under {root}\n")
    errors: list[str] = []
    no_target: list[str] = []

    for idx, entry in enumerate(entries):
        meta_path = entry / "metadata.json"
        replay_path = entry / "interaction_replay.npz"
        if not replay_path.exists():
            if not any(entry.iterdir()):
                print(f"{idx:03d} {entry.name}: empty directory (ignored by deployment)")
                continue
            errors.append(f"{entry.name}: missing interaction_replay.npz")
            continue

        # Like load_interaction_library() in deploy_bbox_seg4ch.py: metadata.json
        # is optional and target-state keys stored in the npz take precedence.
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
            except json.JSONDecodeError as exc:
                errors.append(f"{entry.name}: bad metadata JSON: {exc}")
                continue
        data = np.load(replay_path, allow_pickle=True)
        for key in ("target_u", "target_v", "target_depth_mm", "target_state_uvd"):
            if key in data:
                value = np.asarray(data[key])
                meta[key] = value.item() if value.shape == () else value.tolist()

        motion = replay_motion_mm(replay_path)
        # Same lookup as replay_target_state() in deploy_bbox_seg4ch.py.
        uvd = meta.get("target_state_uvd")
        if uvd is None and all(k in meta for k in ("target_u", "target_v", "target_depth_mm")):
            uvd = [meta["target_u"], meta["target_v"], meta["target_depth_mm"]]
        u, v, d = uvd[:3] if uvd is not None else (None, None, None)
        if uvd is None:
            no_target.append(entry.name)
        motion_text = "unknown" if motion is None else f"{motion:.1f} mm"
        print(f"{idx:03d} {entry.name}: motion={motion_text}, u={u}, v={v}, depth_mm={d}")

    if errors:
        print("\nProblems:")
        for err in errors:
            print(f"  - {err}")
        return 1

    if no_target:
        print(f"\nWarning: {len(no_target)} entr{'y' if len(no_target) == 1 else 'ies'} "
              "without target_state_uvd; the automatic bottleneck switch skips "
              "them (only --manual_bottleneck_switch can use them):")
        for name in no_target:
            print(f"  - {name}")

    print("\nReplay library check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
