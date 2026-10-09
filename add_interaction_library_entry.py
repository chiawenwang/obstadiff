#!/usr/bin/env python3
"""
Add a recorded interaction replay to a simple retrieval library.

The library format is intentionally lightweight:

interaction_library/
  black_pepper_close/
    interaction_replay.npz
    metadata.json

deploy_bbox_seg4ch.py can load this directory with --interaction_library and
select an entry by key plus optional bbox/depth metadata.
"""

import argparse
import json
import os
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True,
                        help="Interaction library directory.")
    parser.add_argument("--name", required=True,
                        help="Entry name, e.g. black_pepper_close.")
    parser.add_argument("--replay", required=True,
                        help="Path to interaction_replay.npz.")
    parser.add_argument("--label", default=None,
                        help="Object/task label, e.g. black_pepper.")
    parser.add_argument("--condition", default=None,
                        help="Optional condition, e.g. clear, occluded, different_plant.")
    parser.add_argument("--bbox_area_frac", type=float, default=None,
                        help="Optional target bbox area divided by image area.")
    parser.add_argument("--bbox_aspect", type=float, default=None,
                        help="Optional target bbox width/height.")
    parser.add_argument("--target_depth_mm", type=float, default=None,
                        help="Optional median target depth in mm.")
    args = parser.parse_args()

    if not os.path.exists(args.replay):
        raise FileNotFoundError(args.replay)

    entry_dir = os.path.join(os.path.abspath(args.library), args.name)
    os.makedirs(entry_dir, exist_ok=True)
    dst_replay = os.path.join(entry_dir, "interaction_replay.npz")
    shutil.copy2(args.replay, dst_replay)

    metadata = {
        "name": args.name,
        "label": args.label or args.name,
        "condition": args.condition or "",
        "replay": "interaction_replay.npz",
    }
    for key in ("bbox_area_frac", "bbox_aspect", "target_depth_mm"):
        value = getattr(args, key)
        if value is not None:
            metadata[key] = float(value)

    meta_path = os.path.join(entry_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    print(f"added interaction library entry: {entry_dir}")
    print(f"  replay:   {dst_replay}")
    print(f"  metadata: {meta_path}")


if __name__ == "__main__":
    main()
