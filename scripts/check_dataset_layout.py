#!/usr/bin/env python3
"""Check the expected demonstration layout for ObstaDiff retraining."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def describe_array(path: Path) -> tuple[tuple[int, ...], str]:
    arr = np.load(path, mmap_mode="r")
    return tuple(arr.shape), str(arr.dtype)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to final_data_decomp.")
    parser.add_argument("--limit", type=int, default=5, help="Number of demos to inspect.")
    args = parser.parse_args()

    root = Path(args.dataset)
    demos = sorted(p for p in root.glob("demo_decomp_*") if p.is_dir())
    if not demos:
        raise SystemExit(f"No demo_decomp_* folders found under {root}")

    required = [
        "head_camera_rgb.npy",
        "head_camera_cam4.npy",
        "robot_states.npy",
        "alignment_end_idx.npy",
    ]

    errors: list[str] = []
    print(f"Found {len(demos)} demos under {root}")

    for demo in demos[: args.limit]:
        print(f"\n{demo.name}")
        missing = [name for name in required if not (demo / name).exists()]
        if missing:
            errors.append(f"{demo.name}: missing {missing}")
            continue

        rgb_shape, rgb_dtype = describe_array(demo / "head_camera_rgb.npy")
        cam4_shape, cam4_dtype = describe_array(demo / "head_camera_cam4.npy")
        state_shape, state_dtype = describe_array(demo / "robot_states.npy")
        align = np.load(demo / "alignment_end_idx.npy")
        align_idx = int(np.asarray(align).reshape(-1)[0])

        print(f"  rgb:    shape={rgb_shape}, dtype={rgb_dtype}")
        print(f"  cam4:   shape={cam4_shape}, dtype={cam4_dtype}")
        print(f"  states: shape={state_shape}, dtype={state_dtype}")
        print(f"  alignment_end_idx: {align_idx}")

        if len(rgb_shape) != 4:
            errors.append(f"{demo.name}: rgb should be rank 4")
        if len(cam4_shape) != 4:
            errors.append(f"{demo.name}: cam4 should be rank 4")
        if state_shape[0] < 2 or len(state_shape) != 2 or state_shape[1] < 14:
            errors.append(f"{demo.name}: robot_states should be (T, >=14)")
        if align_idx < 1 or align_idx >= state_shape[0]:
            errors.append(f"{demo.name}: alignment_end_idx outside trajectory length")

    if errors:
        print("\nProblems:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("\nDataset layout check passed for inspected demos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
