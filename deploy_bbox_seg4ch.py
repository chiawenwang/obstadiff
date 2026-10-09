"""
Deploy the bbox-target seg4ch GROVE checkpoint on Sawyer.

This matches the bbox post-processing / recorder order:
  - target/object class 2 is a filled bounding-box region, not SAM-refined.
  - obstacle/vegetation class 1 comes from YOLO-World boxes refined by MobileSAM.
  - background class 0 is everything else.
  - cam_high is [bg_mask, obstacle_mask, target_box_mask, depth_norm].
"""

import time
import argparse
import os
import sys
import csv
import json
import signal
import multiprocessing as mp
import subprocess
from collections import deque
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.spatial.transform as st
import torch
import dill
import hydra
import cv2


# Match data_collection/record_full_data.py and home.py.
HOME_JOINTS_LIST = [
    0.0273525390625,
    -0.77991015625,
    -1.30016015625,
    1.673435546875,
    -0.903478515625,
    -1.5015849609375,
    -0.1209189453125,
]

# Camera intrinsics that produced the training depth (RealSense L515-like)
FX, FY = 386.020, 385.415
CX, CY = 321.556, 240.143

# Depth normalization used at training (encode_4ch with depth_max_mm=2000)
DEPTH_MAX_MM = 2000.0

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DEPLOY_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_COLLECTION_DIR = os.path.join(PROJECT_ROOT, "data_collection")
BBOX_CHECKPOINT_DEFAULT = os.path.join(
    DEPLOY_DIR,
    "checkpoints",
    "4ch_bbox_postprocessed_best_epoch0050_val0.03307.ckpt",
)
WORLD_WEIGHTS_DEFAULT = os.path.join(DATA_COLLECTION_DIR, "yolov8s-worldv2.pt")
SAM_WEIGHTS_DEFAULT = os.path.join(DATA_COLLECTION_DIR, "mobile_sam.pt")
WORLD_OBJECT_CLASSES_DEFAULT = ["bell pepper", "purple pepper", "dark pepper", "small dark fruit"]
WORLD_OBSTACLE_CLASSES_DEFAULT = ["leaves", "leaf foliage", "plant stem", "green plant"]
WORLD_CONF_DEFAULT = 0.05
_ACTIVE_CONTROLLER = None
_ACTIVE_SHM = None
_ACTIVE_COMMAND_LOGGER = None
_CLEANING_UP = False
_PATH_TRACE = None
_SCHEDULED_CHUNKS = None
_SCHEDULED_WAYPOINT_TRACE = None
_TARGET_STATE_TRACE = None
_PATH_SUMMARY_PATH = "/tmp/bbox_deploy_path_summary.png"
_DISTANCE_ERROR_PATH = "/tmp/bbox_deploy_distance_error.png"
_TARGET_STATE_ERROR_PATH = "/tmp/bbox_deploy_target_state_error.png"
_TARGET_STATE_STRICT_THRESH = None
_TARGET_STATE_PLATEAU_THRESH = None

# Per-step displacement safety clip (mirrors the original deploy script).
# At freq=10Hz and dt=0.1s this caps EEF speed to ~0.5 m/s and rotation to ~1.5 rad/s.
MAX_STEP_DISP = 0.05    # metres per inter-waypoint step
MAX_STEP_ROT  = 0.15    # radians per inter-waypoint step

# Hard chunk-wide displacement cap: if the policy commands more than this total
# translation across all 8 waypoints, scale the whole chunk down or refuse.
MAX_CHUNK_DISP = 0.40   # metres total across n_action_steps


def patch_huggingface_hub_for_old_diffusers():
    """Old diffusers versions import cached_download, which newer HF Hub removed."""
    try:
        import huggingface_hub
        if not hasattr(huggingface_hub, "cached_download"):
            huggingface_hub.cached_download = huggingface_hub.hf_hub_download
    except Exception:
        pass


def ensure_ros_python_paths():
    """Let conda Python import ROS/Intera packages without relying on shell PYTHONPATH."""
    ros_paths = [
        "/opt/ros/noetic/lib/python3/dist-packages",
        "/path/to/sawyer_robot_ws/devel/lib/python3/dist-packages",
    ]
    existing = os.environ.get("PYTHONPATH", "")
    parts = [p for p in existing.split(os.pathsep) if p]
    changed = False
    for path in ros_paths:
        if os.path.isdir(path):
            if path not in sys.path:
                sys.path.insert(0, path)
            if path not in parts:
                parts.insert(0, path)
                changed = True
    if changed:
        os.environ["PYTHONPATH"] = os.pathsep.join(parts)
    # This deployment is for the right Sawyer. Do not inherit a stale local
    # ROS master from the shell (sourcing ROS sets localhost), because then the
    # camera/EEF wait silently hangs. Override with OBSTADIFF_ROS_MASTER_URI /
    # OBSTADIFF_ROS_HOSTNAME instead.
    os.environ["ROS_MASTER_URI"] = os.environ.get(
        "OBSTADIFF_ROS_MASTER_URI", "http://robot.local:11311")
    os.environ["ROS_HOSTNAME"] = os.environ.get(
        "OBSTADIFF_ROS_HOSTNAME", "workstation.local")


def cleanup_processes(controller=None, shm=None, command_logger=None):
    global _CLEANING_UP
    if _CLEANING_UP:
        return
    _CLEANING_UP = True
    if controller is not None:
        try:
            controller.stop(wait=True)
        except Exception:
            try:
                if controller.is_alive():
                    controller.terminate()
                    controller.join(timeout=2.0)
            except Exception:
                pass
    for child in mp.active_children():
        try:
            child.terminate()
        except Exception:
            pass
    for child in mp.active_children():
        try:
            child.join(timeout=1.0)
        except Exception:
            pass
    for child in mp.active_children():
        try:
            if child.is_alive():
                child.kill()
                child.join(timeout=1.0)
        except Exception:
            pass
    if command_logger is not None:
        try:
            command_logger.close()
        except Exception:
            pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    if shm is not None:
        try:
            shm.shutdown()
        except Exception:
            pass


def kill_existing_deploy_processes():
    """Kill stale deploy_bbox_seg4ch.py instances before starting a new one."""
    script_path = os.path.abspath(__file__)
    current = os.getpid()
    ancestors = {current}
    pid = os.getppid()
    while pid and pid not in ancestors:
        ancestors.add(pid)
        try:
            ppid = int(subprocess.check_output(
                ["ps", "-o", "ppid=", "-p", str(pid)],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip())
        except Exception:
            break
        pid = ppid
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid=,comm=,args="],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return
    stale_pids = []
    for line in out.splitlines():
        if script_path not in line:
            continue
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        comm = parts[1]
        args = parts[2]
        # Only kill direct Python deploy scripts. Do not kill the active
        # conda-run parent that launched this process.
        if (
            pid not in ancestors
            and "python" in comm
            and "conda run" not in args
        ):
            stale_pids.append(pid)
    if not stale_pids:
        return
    print(f"[startup-cleanup] killing stale deploy processes: {stale_pids}")
    for pid in stale_pids:
        try:
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        except Exception as e:
            print(f"[startup-cleanup] SIGINT failed for {pid}: {e}")
    time.sleep(1.0)
    for pid in stale_pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            print(f"[startup-cleanup] SIGKILL stale deploy pid {pid}")
        except ProcessLookupError:
            pass
        except Exception as e:
            print(f"[startup-cleanup] SIGKILL failed for {pid}: {e}")


def handle_sigint(signum, frame):
    print("\n[shutdown] Ctrl-C received; cleaning up deploy processes...")
    try:
        render_path_summary(_PATH_TRACE, _SCHEDULED_CHUNKS, _PATH_SUMMARY_PATH)
        print(f"[path] saved summary to {_PATH_SUMMARY_PATH}")
    except Exception:
        pass
    try:
        render_distance_error(_PATH_TRACE, _SCHEDULED_WAYPOINT_TRACE, _DISTANCE_ERROR_PATH)
        print(f"[error] saved distance error to {_DISTANCE_ERROR_PATH}")
    except Exception:
        pass
    try:
        render_target_state_error(
            _TARGET_STATE_TRACE,
            _TARGET_STATE_ERROR_PATH,
            strict_thresh=_TARGET_STATE_STRICT_THRESH,
            plateau_thresh=_TARGET_STATE_PLATEAU_THRESH,
        )
        print(f"[target-state] saved error to {_TARGET_STATE_ERROR_PATH}")
    except Exception:
        pass
    cleanup_processes(
        controller=_ACTIVE_CONTROLLER,
        shm=_ACTIVE_SHM,
        command_logger=_ACTIVE_COMMAND_LOGGER,
    )
    os._exit(130)


def clip_waypoint_chain(action: np.ndarray, current_pose: np.ndarray,
                        max_step_disp: float = MAX_STEP_DISP,
                        max_step_rot: float = MAX_STEP_ROT):
    """
    Walk the predicted action chunk in order; clip each waypoint so it sits
    within (max_step_disp, max_step_rot) of the previous waypoint. The first
    waypoint is clipped against current_pose. Subsequent waypoints chain off
    the (already-clipped) previous one, so the trajectory still moves in the
    policy's intended direction but never exceeds the per-step velocity cap.

    Args:
        action       : (n, 7) float32 — predicted waypoints [x,y,z, qx,qy,qz,qw]
        current_pose : (7,) float32   — current EEF [x,y,z, qx,qy,qz,qw]

    Returns:
        clipped_action : (n, 7) float32
        n_pos_clipped  : how many waypoints had translation clipped
        n_rot_clipped  : how many waypoints had rotation clipped
    """
    n = action.shape[0]
    out = np.empty_like(action)
    prev_pos = current_pose[:3].astype(np.float32)
    prev_rot = st.Rotation.from_quat(current_pose[3:7])
    n_pos_clipped = 0
    n_rot_clipped = 0
    for i in range(n):
        tgt_pos = action[i, :3].astype(np.float32)
        tgt_rot = st.Rotation.from_quat(action[i, 3:7])
        # translation clip
        disp = tgt_pos - prev_pos
        d = float(np.linalg.norm(disp))
        if d > max_step_disp:
            tgt_pos = prev_pos + disp * (max_step_disp / d)
            n_pos_clipped += 1
        # rotation clip (rotvec form, in current frame)
        delta = (tgt_rot * prev_rot.inv()).as_rotvec()
        dn = float(np.linalg.norm(delta))
        if dn > max_step_rot:
            delta = delta * (max_step_rot / dn)
            tgt_rot = st.Rotation.from_rotvec(delta) * prev_rot
            n_rot_clipped += 1
        out[i, :3]  = tgt_pos
        out[i, 3:7] = tgt_rot.as_quat()
        prev_pos = tgt_pos
        prev_rot = tgt_rot
    return out, n_pos_clipped, n_rot_clipped


def smooth_action_chunk(action: np.ndarray, prev_action: Optional[np.ndarray],
                        alpha: float) -> np.ndarray:
    """Blend absolute EEF waypoint chunks. alpha=1 uses new action unchanged."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if prev_action is None or alpha >= 0.999 or prev_action.shape != action.shape:
        return action
    if alpha <= 0.001:
        return prev_action.copy()

    out = action.copy()
    out[:, :3] = alpha * action[:, :3] + (1.0 - alpha) * prev_action[:, :3]

    for i in range(action.shape[0]):
        prev_rot = st.Rotation.from_quat(prev_action[i, 3:7])
        new_rot = st.Rotation.from_quat(action[i, 3:7])
        delta = (new_rot * prev_rot.inv()).as_rotvec()
        out[i, 3:7] = (st.Rotation.from_rotvec(alpha * delta) * prev_rot).as_quat()
    return out


def drop_backward_prefix(action: np.ndarray, current_pose: np.ndarray,
                         min_net_disp: float = 0.005) -> tuple[np.ndarray, int]:
    """
    Remove leading waypoints that initially move opposite the chunk's net
    direction from the current EEF pose. This avoids the common
    "back up, then go forward" artifact from stale early waypoints.
    """
    if action.shape[0] <= 1:
        return action, 0
    current_pos = current_pose[:3].astype(np.float32)
    net = action[-1, :3].astype(np.float32) - current_pos
    net_norm = float(np.linalg.norm(net))
    if net_norm < min_net_disp:
        return action, 0
    direction = net / net_norm

    keep_idx = 0
    for i in range(action.shape[0] - 1):
        rel = action[i, :3].astype(np.float32) - current_pos
        if float(np.dot(rel, direction)) >= -0.002:
            break
        keep_idx = i + 1
    if keep_idx > 0:
        return action[keep_idx:], keep_idx
    return action, 0


def drop_near_pose_prefix(action: np.ndarray, pose: np.ndarray,
                          radius_m: float) -> tuple[np.ndarray, int]:
    """Drop leading waypoints that sit near the pose used for inference."""
    if action.shape[0] <= 1 or radius_m <= 0:
        return action, 0
    pose_pos = pose[:3].astype(np.float32)
    keep_idx = 0
    # Always leave at least one waypoint.
    for i in range(action.shape[0] - 1):
        d = float(np.linalg.norm(action[i, :3].astype(np.float32) - pose_pos))
        if d > radius_m:
            break
        keep_idx = i + 1
    if keep_idx > 0:
        return action[keep_idx:], keep_idx
    return action, 0


def format_action_chunk(name: str, action: np.ndarray, max_rows: int = 8) -> str:
    lines = [f"  {name}:"]
    for i, row in enumerate(action[:max_rows]):
        lines.append(
            f"    {i:02d}: pos=({row[0]:+.4f},{row[1]:+.4f},{row[2]:+.4f}) "
            f"quat=({row[3]:+.4f},{row[4]:+.4f},{row[5]:+.4f},{row[6]:+.4f})"
        )
    if action.shape[0] > max_rows:
        lines.append(f"    ... {action.shape[0] - max_rows} more")
    return "\n".join(lines)


def load_interaction_replay(path: Optional[str]) -> Optional[Dict[str, np.ndarray]]:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"interaction replay not found: {path}")
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        if "rel_pos" in data and "rel_quat" in data:
            rel_pos = np.asarray(data["rel_pos"], dtype=np.float32)
            rel_quat = np.asarray(data["rel_quat"], dtype=np.float32)
        elif "poses" in data:
            poses = np.asarray(data["poses"], dtype=np.float32)
            if poses.ndim != 2 or poses.shape[1] != 7:
                raise ValueError(f"{path}: poses must have shape (N, 7)")
            anchor = poses[0]
            rel_pos, rel_quat = make_relative_replay(poses, anchor)
        else:
            raise ValueError(f"{path}: expected rel_pos/rel_quat or poses")
        replay_dt = float(np.asarray(data["dt"]).reshape(-1)[0]) if "dt" in data else 0.1
    else:
        poses = np.asarray(data, dtype=np.float32)
        if poses.ndim != 2 or poses.shape[1] != 7:
            raise ValueError(f"{path}: .npy replay must have shape (N, 7)")
        rel_pos, rel_quat = make_relative_replay(poses, poses[0])
        replay_dt = 0.1
    if rel_pos.shape[0] != rel_quat.shape[0] or rel_pos.shape[1] != 3 or rel_quat.shape[1] != 4:
        raise ValueError(f"{path}: rel_pos must be (N,3), rel_quat must be (N,4)")
    return {"rel_pos": rel_pos, "rel_quat": rel_quat, "dt": np.array(replay_dt, dtype=np.float32)}


def replay_translation_path_length(replay: Dict[str, np.ndarray]) -> float:
    rel_pos = np.asarray(replay["rel_pos"], dtype=np.float32)
    if rel_pos.ndim != 2 or rel_pos.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(rel_pos, axis=0), axis=1).sum())


def load_interaction_library(library_dir: Optional[str], min_translation_m: float = 0.0):
    if not library_dir:
        return []
    library_dir = os.path.abspath(library_dir)
    if not os.path.isdir(library_dir):
        raise FileNotFoundError(f"interaction library dir not found: {library_dir}")
    entries = []
    for root, _dirs, files in os.walk(library_dir):
        for name in sorted(files):
            if not name.endswith(".npz"):
                continue
            path = os.path.join(root, name)
            try:
                replay = load_interaction_replay(path)
            except Exception as e:
                print(f"[library] skipping {path}: {e}")
                continue
            total_translation = replay_translation_path_length(replay)
            if total_translation < float(min_translation_m):
                print(
                    f"[library] skipping {path}: replay translation "
                    f"{total_translation*1000:.1f}mm < min "
                    f"{float(min_translation_m)*1000:.1f}mm"
                )
                continue
            metadata = {"name": os.path.splitext(name)[0], "path": path}
            meta_path = os.path.join(root, "metadata.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        metadata.update(json.load(f))
                except Exception as e:
                    print(f"[library] WARN: failed to read {meta_path}: {e}")
            try:
                data = np.load(path, allow_pickle=True)
                for key in ("label", "object_label", "condition", "bbox_area_frac",
                            "bbox_aspect", "target_depth_mm", "target_u",
                            "target_v", "target_state_uvd", "anchor_pose"):
                    if key in data:
                        value = np.asarray(data[key])
                        if value.shape == ():
                            metadata[key] = value.item()
                        else:
                            metadata[key] = value.tolist()
            except Exception:
                pass
            metadata["replay_total_translation_m"] = total_translation
            entries.append({"path": path, "replay": replay, "metadata": metadata})
    print(f"[library] loaded {len(entries)} interaction replay(s) from {library_dir}")
    return entries


def target_box_features(box, depth_mm: Optional[np.ndarray],
                        depth_center_frac: float = 0.4,
                        depth_percentile: float = 20.0):
    if box is None:
        return {}
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    H = int(depth_mm.shape[0]) if depth_mm is not None else 1
    W = int(depth_mm.shape[1]) if depth_mm is not None else 1
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    feat = {
        "target_u": float((x0 + x1) * 0.5),
        "target_v": float((y0 + y1) * 0.5),
        "bbox_area_frac": float((w * h) / max(1, W * H)),
        "bbox_aspect": float(w / max(1, h)),
    }
    if depth_mm is not None:
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
    if all(k in feat for k in ("target_u", "target_v", "target_depth_mm")):
        feat["target_state_uvd"] = [
            feat["target_u"],
            feat["target_v"],
            feat["target_depth_mm"],
        ]
    return feat


def _matches_key(metadata: dict, key: Optional[str]) -> bool:
    if not key:
        return True
    key_l = str(key).lower()
    fields = [
        metadata.get("name", ""),
        metadata.get("label", ""),
        metadata.get("object_label", ""),
        metadata.get("condition", ""),
        os.path.basename(metadata.get("path", "")),
    ]
    return any(key_l in str(v).lower() for v in fields)


def infer_interaction_key(object_classes) -> Optional[str]:
    if not object_classes:
        return None
    joined = " ".join(str(c).lower() for c in object_classes)
    if "black" in joined or "dark" in joined:
        return "black_pepper"
    if "green" in joined:
        return "green_pepper"
    if "purple" in joined:
        return "purple_pepper"
    if "pepper" in joined:
        return "pepper"
    return None


def select_interaction_from_library(library_entries, interaction_key: Optional[str],
                                    box, depth_mm: Optional[np.ndarray],
                                    anchor_pose: Optional[np.ndarray] = None,
                                    depth_center_frac: float = 0.4,
                                    depth_percentile: float = 20.0):
    if not library_entries:
        return None, None
    candidates = [e for e in library_entries if _matches_key(e["metadata"], interaction_key)]
    if not candidates:
        print(f"[library] no entries matched key '{interaction_key}', using full library")
        candidates = library_entries
    feat = target_box_features(
        box, depth_mm,
        depth_center_frac=depth_center_frac,
        depth_percentile=depth_percentile,
    )

    def score(entry):
        meta = entry["metadata"]
        s = 0.0
        if "bbox_area_frac" in feat and "bbox_area_frac" in meta:
            s += abs(float(feat["bbox_area_frac"]) - float(meta["bbox_area_frac"])) * 8.0
        if "bbox_aspect" in feat and "bbox_aspect" in meta:
            s += abs(np.log(max(1e-3, float(feat["bbox_aspect"])) /
                            max(1e-3, float(meta["bbox_aspect"])))) * 1.0
        if "target_depth_mm" in feat and "target_depth_mm" in meta:
            s += abs(float(feat["target_depth_mm"]) - float(meta["target_depth_mm"])) / 800.0
        if anchor_pose is not None and "anchor_pose" in meta:
            try:
                ref = np.asarray(meta["anchor_pose"], dtype=np.float32).reshape(-1)
                if ref.shape[0] >= 7:
                    # Strongest automatic signal: choose the replay recorded
                    # from the most similar bottleneck EEF pose.
                    pos_dist = float(np.linalg.norm(anchor_pose[:3] - ref[:3]))
                    qdot = abs(float(np.dot(anchor_pose[3:7], ref[3:7])))
                    qdot = float(np.clip(qdot, 0.0, 1.0))
                    rot_dist = 2.0 * np.arccos(qdot)
                    s += pos_dist * 20.0 + rot_dist * 0.25
            except Exception:
                pass
        return s

    best = min(candidates, key=score)
    return best["replay"], best


def replay_target_state(metadata: dict) -> Optional[np.ndarray]:
    if "target_state_uvd" in metadata:
        arr = np.asarray(metadata["target_state_uvd"], dtype=np.float32).reshape(-1)
        if arr.shape[0] >= 3:
            return arr[:3]
    keys = ("target_u", "target_v", "target_depth_mm")
    if all(k in metadata for k in keys):
        return np.asarray([metadata["target_u"], metadata["target_v"],
                           metadata["target_depth_mm"]], dtype=np.float32)
    return None


def normalize_target_state_uvd(uvd: np.ndarray,
                               image_width: float,
                               image_height: float,
                               depth_scale_mm: float) -> np.ndarray:
    scale = np.asarray([
        max(1.0, float(image_width)),
        max(1.0, float(image_height)),
        max(1.0, float(depth_scale_mm)),
    ], dtype=np.float32)
    return np.asarray(uvd, dtype=np.float32) / scale


def nearest_target_state_error(library_entries, interaction_key: Optional[str],
                               live_feat: dict,
                               image_width: float = 640.0,
                               image_height: float = 480.0,
                               depth_scale_mm: float = 1000.0,
                               weights=(0.2, 0.2, 0.6),
                               depth_gate_thresh: Optional[float] = None,
                               depth_ready_entries: Optional[set] = None):
    if not library_entries or not all(k in live_feat for k in ("target_u", "target_v", "target_depth_mm")):
        return None
    live = np.asarray([
        live_feat["target_u"],
        live_feat["target_v"],
        live_feat["target_depth_mm"],
    ], dtype=np.float32)
    valid_for_switch = not bool(live_feat.get("ignore_target_depth_error", False))
    candidates = [e for e in library_entries if _matches_key(e["metadata"], interaction_key)]
    if not candidates:
        candidates = library_entries
    best_depth = None
    best_depth_score = None
    best_uv = None
    best_uv_score = None
    for entry in candidates:
        ref = replay_target_state(entry["metadata"])
        if ref is None:
            continue
        delta = live - ref
        live_norm = normalize_target_state_uvd(
            live, image_width=image_width, image_height=image_height,
            depth_scale_mm=depth_scale_mm)
        ref_norm = normalize_target_state_uvd(
            ref, image_width=image_width, image_height=image_height,
            depth_scale_mm=depth_scale_mm)
        norm_delta = live_norm - ref_norm
        if live_feat.get("ignore_target_depth_error", False):
            delta[2] = 0.0
            norm_delta[2] = 0.0
        weight_arr = np.asarray(weights, dtype=np.float32)
        weight_arr = np.maximum(weight_arr, 0.0)
        if float(weight_arr.sum()) <= 0:
            weight_arr = np.asarray([0.2, 0.2, 0.6], dtype=np.float32)
        weight_arr = weight_arr / float(weight_arr.sum())
        uv_weights = weight_arr[:2].copy()
        if float(uv_weights.sum()) <= 0:
            uv_weights = np.asarray([0.5, 0.5], dtype=np.float32)
        uv_weights = uv_weights / float(uv_weights.sum())
        uv_dist = float(np.sqrt(np.sum(uv_weights * np.square(norm_delta[:2]))))
        uv_sum = float(abs(norm_delta[0]) + abs(norm_delta[1]))
        # One-sided depth gate: if the live target is already closer than the
        # replay reference, treat depth error as zero. This avoids blocking the
        # switch on negative depth deltas such as -0.4.
        depth_dist = float(max(0.0, norm_delta[2]))
        combined_dist = float(np.sqrt(np.sum(weight_arr * np.square(norm_delta))))
        depth_gate_pass = (
            depth_gate_thresh is None
            or depth_dist <= float(depth_gate_thresh)
        )
        entry_key = entry.get("path", entry["metadata"].get("name", str(id(entry))))
        # Depth readiness is latched per replay entry. Once this specific
        # replay has been close enough in depth, later frames only need the UV
        # gate for this same replay; other replays must satisfy depth
        # independently before they can switch.
        depth_entered_now = bool(depth_gate_pass and valid_for_switch)
        if depth_entered_now and depth_ready_entries is not None:
            depth_ready_entries.add(entry_key)
        depth_ready = (
            depth_gate_thresh is None
            or depth_entered_now
            or (depth_ready_entries is not None and entry_key in depth_ready_entries)
        )
        candidate = (
            entry, ref, delta, live_norm, ref_norm, norm_delta,
            uv_dist, uv_sum, depth_dist, combined_dist,
            depth_gate_pass, depth_ready, entry_key,
        )
        if best_depth_score is None or depth_dist < best_depth_score:
            best_depth_score = depth_dist
            best_depth = candidate
        if depth_ready and (best_uv_score is None or uv_sum < best_uv_score):
            best_uv_score = uv_sum
            best_uv = candidate

    best = best_uv if best_uv is not None else best_depth
    if best is None:
        return None
    (entry, ref, delta, live_norm, ref_norm, norm_delta,
     uv_dist, uv_sum, depth_dist, combined_dist,
     depth_gate_pass, depth_ready, entry_key) = best
    print(f"[target-state] nearest replay '{entry['metadata'].get('name', entry['metadata'].get('path', ''))}' found")
    return {
        "entry": entry,
        "ref": ref,
        "delta": delta,
        "live_norm": live_norm,
        "ref_norm": ref_norm,
        "norm_delta": norm_delta,
        "dist": uv_sum,
        "uv_dist": uv_dist,
        "uv_sum": uv_sum,
        "depth_dist": depth_dist,
        "combined_dist": combined_dist,
        "depth_gate_pass": depth_gate_pass,
        "depth_ready": depth_ready,
        "entry_key": entry_key,
        "live": live,
        "valid_for_switch": valid_for_switch,
    }


def make_relative_replay(poses: np.ndarray, anchor_pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Represent absolute poses as base-frame position deltas plus relative quats."""
    anchor_rot = st.Rotation.from_quat(anchor_pose[3:7])
    rel_pos = poses[:, :3].astype(np.float32) - anchor_pose[:3].astype(np.float32)
    rel_quat = []
    for q in poses[:, 3:7]:
        rel_quat.append((st.Rotation.from_quat(q) * anchor_rot.inv()).as_quat())
    return rel_pos.astype(np.float32), np.asarray(rel_quat, dtype=np.float32)


def compose_interaction_replay(replay: Dict[str, np.ndarray], anchor_pose: np.ndarray) -> np.ndarray:
    """Place a relative replay trajectory at the current bottleneck pose."""
    anchor_pose = anchor_pose.astype(np.float32)
    anchor_rot = st.Rotation.from_quat(anchor_pose[3:7])
    rel_pos = np.asarray(replay["rel_pos"], dtype=np.float32)
    rel_quat = np.asarray(replay["rel_quat"], dtype=np.float32)
    poses = np.empty((rel_pos.shape[0], 7), dtype=np.float32)
    poses[:, :3] = anchor_pose[:3] + rel_pos
    for i, q_rel in enumerate(rel_quat):
        poses[i, 3:7] = (st.Rotation.from_quat(q_rel) * anchor_rot).as_quat()
    return poses


def schedule_pose_chunk(controller, poses: np.ndarray, start_time: float, step_dt: float):
    if controller is None:
        return
    for i in range(poses.shape[0]):
        pos = poses[i, :3]
        quat = poses[i, 3:7]
        rotvec = st.Rotation.from_quat(quat).as_rotvec()
        pose6 = np.concatenate([pos, rotvec])
        controller.schedule_waypoint(pose6, start_time + i * step_dt)


def action_position_activity(action: np.ndarray, current_pose: np.ndarray) -> float:
    """Mean absolute translational movement in a scheduled chunk."""
    if action.size == 0:
        return 0.0
    prev = current_pose[:3].reshape(1, 3)
    pts = np.concatenate([prev, action[:, :3]], axis=0)
    return float(np.mean(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def image_sharpness(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def render_inference_view(rgb: np.ndarray, seg_idx: np.ndarray, box, frame_idx: int,
                          target_px: int, obstacle_px: int,
                          target_depth_mm: Optional[float] = None,
                          raw_target_depth_mm: Optional[float] = None) -> np.ndarray:
    H, W = 240, 320
    rgb_r = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    rgb_bgr = cv2.cvtColor(rgb_r, cv2.COLOR_RGB2BGR)
    scale_x = W / float(rgb.shape[1])
    scale_y = H / float(rgb.shape[0])
    if box is not None:
        x0, y0, x1, y1 = box
        p0 = (int(round(x0 * scale_x)), int(round(y0 * scale_y)))
        p1 = (int(round(x1 * scale_x)), int(round(y1 * scale_y)))
        cv2.rectangle(rgb_bgr, p0, p1, (0, 255, 255), 2)

    seg_r = cv2.resize(seg_idx, (W, H), interpolation=cv2.INTER_NEAREST)
    seg_bgr = cv2.cvtColor(_seg_to_rgb(seg_r), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(rgb_bgr, 0.65, seg_bgr, 0.35, 0)
    panel = np.concatenate([rgb_bgr, overlay], axis=1)
    cv2.putText(panel, f"frame {frame_idx} target={target_px} obstacle={obstacle_px}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    if target_depth_mm is not None and np.isfinite(target_depth_mm):
        depth_msg = f"target depth={target_depth_mm:.0f}mm"
        if raw_target_depth_mm is not None and np.isfinite(raw_target_depth_mm):
            if abs(float(raw_target_depth_mm) - float(target_depth_mm)) > 1.0:
                depth_msg += f" raw={raw_target_depth_mm:.0f}mm"
        cv2.putText(panel, depth_msg, (8, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(panel, "target depth=NA", (8, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, "RGB + bbox", (8, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, "seg overlay", (W + 8, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return panel


def render_clean_preview_frame(rgb: np.ndarray, seg_idx: np.ndarray, box=None,
                               panel: str = "side_by_side",
                               draw_bbox: bool = True) -> np.ndarray:
    """Render clean deployment visuals without text for recording."""
    H, W = 240, 320
    rgb_r = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    rgb_bgr = cv2.cvtColor(rgb_r, cv2.COLOR_RGB2BGR)
    if draw_bbox and box is not None:
        scale_x = W / float(rgb.shape[1])
        scale_y = H / float(rgb.shape[0])
        x0, y0, x1, y1 = box
        p0 = (int(round(x0 * scale_x)), int(round(y0 * scale_y)))
        p1 = (int(round(x1 * scale_x)), int(round(y1 * scale_y)))
        cv2.rectangle(rgb_bgr, p0, p1, (0, 0, 255), 2)

    seg_r = cv2.resize(seg_idx, (W, H), interpolation=cv2.INTER_NEAREST)
    seg_bgr = cv2.cvtColor(_seg_to_rgb(seg_r), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(rgb_bgr, 0.65, seg_bgr, 0.35, 0)
    panel = str(panel or "side_by_side").lower()
    if panel == "rgb":
        return rgb_bgr
    if panel == "seg":
        return seg_bgr
    if panel == "overlay":
        return overlay
    return np.concatenate([rgb_bgr, overlay], axis=1)


def render_path_summary(path_trace, scheduled_chunks, out_path: str,
                        size=(1200, 700)) -> Optional[np.ndarray]:
    if not path_trace and not scheduled_chunks:
        return None
    canvas = np.full((size[1], size[0], 3), 245, dtype=np.uint8)
    margin = 60
    gap = 40
    plot_w = (size[0] - 2 * margin - gap) // 2
    plot_h = size[1] - 2 * margin
    xy_rect = (margin, margin, plot_w, plot_h)
    xz_rect = (margin + plot_w + gap, margin, plot_w, plot_h)

    actual = None
    if path_trace:
        actual = np.asarray(path_trace, dtype=np.float32)
    chunk_arrays = [np.asarray(c, dtype=np.float32) for c in (scheduled_chunks or []) if len(c) > 0]

    points_xyz = []
    if actual is not None and actual.size:
        points_xyz.append(actual[:, 1:4])
    points_xyz.extend(chunk_arrays)
    if not points_xyz:
        return None
    all_xyz = np.concatenate(points_xyz, axis=0)

    def bounds(cols):
        vals = all_xyz[:, cols]
        lo = vals.min(axis=0)
        hi = vals.max(axis=0)
        span = np.maximum(hi - lo, 0.05)
        lo -= span * 0.12
        hi += span * 0.12
        return lo, hi

    xy_lo, xy_hi = bounds([0, 1])
    xz_lo, xz_hi = bounds([0, 2])

    def project(points, rect, lo, hi, cols):
        x, y, w, h = rect
        vals = points[:, cols]
        denom = np.maximum(hi - lo, 1e-6)
        u = (vals[:, 0] - lo[0]) / denom[0]
        v = (vals[:, 1] - lo[1]) / denom[1]
        px = x + np.clip(u, 0, 1) * w
        py = y + (1.0 - np.clip(v, 0, 1)) * h
        return np.stack([px, py], axis=1).astype(np.int32)

    def draw_panel(rect, title, lo, hi, cols, ylabel):
        x, y, w, h = rect
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (80, 80, 80), 1)
        for i in range(1, 5):
            gx = x + i * w // 5
            gy = y + i * h // 5
            cv2.line(canvas, (gx, y), (gx, y + h), (220, 220, 220), 1)
            cv2.line(canvas, (x, gy), (x + w, gy), (220, 220, 220), 1)
        cv2.putText(canvas, title, (x, y - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"x [{lo[0]:.3f}, {hi[0]:.3f}]",
                    (x, y + h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{ylabel} [{lo[1]:.3f}, {hi[1]:.3f}]",
                    (x + 180, y + h + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
        for c in chunk_arrays:
            pts = project(c, rect, lo, hi, cols)
            if len(pts) >= 2:
                cv2.polylines(canvas, [pts], False, (0, 150, 255), 1, cv2.LINE_AA)
        if actual is not None and actual.shape[0] >= 2:
            pts = project(actual[:, 1:4], rect, lo, hi, cols)
            cv2.polylines(canvas, [pts], False, (255, 80, 20), 3, cv2.LINE_AA)
            cv2.circle(canvas, tuple(pts[0]), 6, (40, 180, 40), -1, cv2.LINE_AA)
            cv2.circle(canvas, tuple(pts[-1]), 6, (40, 40, 220), -1, cv2.LINE_AA)

    draw_panel(xy_rect, "EEF path top view (x-y)", xy_lo, xy_hi, [0, 1], "y")
    draw_panel(xz_rect, "EEF path side view (x-z)", xz_lo, xz_hi, [0, 2], "z")
    cv2.putText(canvas, "blue = actual EEF path, orange = scheduled policy chunks, green=start, red=end",
                (margin, size[1] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        cv2.imwrite(out_path, canvas)
    return canvas


def record_scheduled_waypoints(trace, poses: np.ndarray, start_time: float,
                               step_dt: float, phase_id: int):
    if trace is None or poses is None:
        return
    for i in range(poses.shape[0]):
        trace.append([
            float(start_time + i * step_dt),
            float(poses[i, 0]),
            float(poses[i, 1]),
            float(poses[i, 2]),
            int(phase_id),
        ])


def render_distance_error(path_trace, scheduled_waypoints, out_path: str,
                          csv_path: Optional[str] = None,
                          size=(1200, 650)) -> Optional[np.ndarray]:
    if not path_trace or not scheduled_waypoints:
        return None
    actual = np.asarray(path_trace, dtype=np.float64)
    desired = np.asarray(scheduled_waypoints, dtype=np.float64)
    if actual.ndim != 2 or actual.shape[1] < 4 or desired.ndim != 2 or desired.shape[1] < 5:
        return None
    desired = desired[np.argsort(desired[:, 0])]
    actual = actual[np.argsort(actual[:, 0])]
    t0 = max(float(actual[0, 0]), float(desired[0, 0]))
    t1 = min(float(actual[-1, 0]), float(desired[-1, 0]))
    mask = (actual[:, 0] >= t0) & (actual[:, 0] <= t1)
    if int(mask.sum()) < 2:
        return None
    t = actual[mask, 0]
    actual_xyz = actual[mask, 1:4]
    desired_xyz = np.stack([
        np.interp(t, desired[:, 0], desired[:, 1]),
        np.interp(t, desired[:, 0], desired[:, 2]),
        np.interp(t, desired[:, 0], desired[:, 3]),
    ], axis=1)
    err_mm = np.linalg.norm(actual_xyz - desired_xyz, axis=1) * 1000.0
    rel_t = t - t[0]

    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t_rel_s", "actual_x", "actual_y", "actual_z",
                             "desired_x", "desired_y", "desired_z", "error_mm"])
            for ti, a, d, e in zip(rel_t, actual_xyz, desired_xyz, err_mm):
                writer.writerow([ti, a[0], a[1], a[2], d[0], d[1], d[2], e])

    canvas = np.full((size[1], size[0], 3), 245, dtype=np.uint8)
    margin_l, margin_r, margin_t, margin_b = 80, 40, 70, 90
    x0, y0 = margin_l, margin_t
    w = size[0] - margin_l - margin_r
    h = size[1] - margin_t - margin_b
    max_err = max(10.0, float(np.percentile(err_mm, 98)) * 1.15, float(err_mm.max()) * 1.05)
    max_t = max(1e-3, float(rel_t[-1]))

    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (80, 80, 80), 1)
    for i in range(1, 6):
        gx = x0 + i * w // 6
        gy = y0 + i * h // 5
        cv2.line(canvas, (gx, y0), (gx, y0 + h), (220, 220, 220), 1)
        cv2.line(canvas, (x0, gy), (x0 + w, gy), (220, 220, 220), 1)
    pts = []
    for ti, e in zip(rel_t, err_mm):
        px = int(x0 + np.clip(ti / max_t, 0, 1) * w)
        py = int(y0 + h - np.clip(e / max_err, 0, 1) * h)
        pts.append([px, py])
    if len(pts) >= 2:
        cv2.polylines(canvas, [np.asarray(pts, dtype=np.int32)], False,
                      (30, 90, 220), 2, cv2.LINE_AA)

    mean_e = float(np.mean(err_mm))
    p95_e = float(np.percentile(err_mm, 95))
    max_e = float(np.max(err_mm))
    cv2.putText(canvas, "EEF tracking distance error over deployment",
                (x0, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"mean={mean_e:.1f} mm   p95={p95_e:.1f} mm   max={max_e:.1f} mm",
                (x0, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, "time since first scheduled waypoint (s)",
                (x0 + w // 2 - 170, size[1] - 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(canvas, "error (mm)", (12, y0 + h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"0", (x0 - 25, y0 + h + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{max_err:.0f}", (x0 - 55, y0 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{max_t:.1f}s", (x0 + w - 45, y0 + h + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1, cv2.LINE_AA)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        cv2.imwrite(out_path, canvas)
    return canvas


def render_target_state_error(target_state_trace, out_path: str,
                              csv_path: Optional[str] = None,
                              size=(1200, 650),
                              strict_thresh: Optional[float] = None,
                              plateau_thresh: Optional[float] = None) -> Optional[np.ndarray]:
    if not target_state_trace:
        return None
    arr = np.asarray(target_state_trace, dtype=object)
    if arr.shape[0] < 2:
        return None
    numeric = np.asarray([row[:12] for row in target_state_trace], dtype=np.float64)
    # columns:
    # t, live_u, live_v, live_d, ref_u, ref_v, ref_d, du, dv, dd, norm_delta, switched
    t = numeric[:, 0]
    rel_t = t - t[0]
    delta = numeric[:, 10]
    switched = numeric[:, 11] > 0.5
    if csv_path:
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "t_rel_s", "live_u_px", "live_v_px", "live_d_mm",
                "ref_u_px", "ref_v_px", "ref_d_mm",
                "du_px", "dv_px", "dd_mm", "delta_uvd_normalized", "switched",
                "du_norm", "dv_norm", "dd_norm",
                "target_depth_valid_for_switch",
                "entry_path",
            ])
            for row in target_state_trace:
                depth_valid = row[16] if len(row) > 16 else 1.0
                writer.writerow([row[0] - t[0]] + list(row[1:12]) + list(row[13:16]) + [depth_valid, row[12]])

    canvas = np.full((size[1], size[0], 3), 245, dtype=np.uint8)
    margin_l, margin_r, margin_t, margin_b = 80, 40, 70, 90
    x0, y0 = margin_l, margin_t
    w = size[0] - margin_l - margin_r
    h = size[1] - margin_t - margin_b
    max_t = max(1e-3, float(rel_t[-1]))
    max_delta = max(1.0, float(np.percentile(delta, 98)) * 1.15, float(delta.max()) * 1.05)
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (80, 80, 80), 1)
    for i in range(1, 6):
        gx = x0 + i * w // 6
        gy = y0 + i * h // 5
        cv2.line(canvas, (gx, y0), (gx, y0 + h), (220, 220, 220), 1)
        cv2.line(canvas, (x0, gy), (x0 + w, gy), (220, 220, 220), 1)
    _draw_threshold_line(canvas, x0, y0, w, h, max_delta, strict_thresh,
                         "strict", (40, 130, 220))
    _draw_threshold_line(canvas, x0, y0, w, h, max_delta, plateau_thresh,
                         "plateau", (20, 150, 20))
    _draw_y_value_label(canvas, x0, y0, h, max_delta, float(np.min(delta)),
                        "min", (120, 90, 160))
    _draw_y_value_label(canvas, x0, y0, h, max_delta, float(delta[-1]),
                        "last", (170, 70, 210))
    pts = []
    for ti, e in zip(rel_t, delta):
        px = int(x0 + np.clip(ti / max_t, 0, 1) * w)
        py = int(y0 + h - np.clip(e / max_delta, 0, 1) * h)
        pts.append([px, py])
    if len(pts) >= 2:
        cv2.polylines(canvas, [np.asarray(pts, dtype=np.int32)], False,
                      (170, 70, 210), 2, cv2.LINE_AA)
    if switched.any():
        idx = int(np.argmax(switched))
        sx = int(x0 + np.clip(rel_t[idx] / max_t, 0, 1) * w)
        cv2.line(canvas, (sx, y0), (sx, y0 + h), (20, 150, 20), 2, cv2.LINE_AA)
        cv2.putText(canvas, "switch", (sx + 6, y0 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 120, 20), 1, cv2.LINE_AA)

    cv2.putText(canvas, "Nearest replay target-state error",
                (x0, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "weighted normalized delta = sqrt(0.2(du/W)^2 + 0.2(dv/H)^2 + 0.6(dd/D)^2)",
                (x0, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, "time (s)", (x0 + w // 2 - 40, size[1] - 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(canvas, "normalized error", (8, y0 + h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{max_delta:.2f}", (x0 - 58, y0 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{max_t:.1f}s", (x0 + w - 45, y0 + h + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1, cv2.LINE_AA)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        cv2.imwrite(out_path, canvas)
    return canvas


def _draw_threshold_line(canvas, x0, y0, w, h, max_delta, value, label, color):
    if value is None:
        return
    try:
        value = float(value)
    except Exception:
        return
    if value < 0:
        return
    py = int(y0 + h - np.clip(value / max(1e-6, max_delta), 0, 1) * h)
    cv2.line(canvas, (x0, py), (x0 + w, py), color, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{label}={value:.3f}", (x0 + w - 150, max(y0 + 14, py - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def _draw_y_value_label(canvas, x0, y0, h, max_delta, value, label, color):
    try:
        value = float(value)
    except Exception:
        return
    py = int(y0 + h - np.clip(value / max(1e-6, max_delta), 0, 1) * h)
    cv2.line(canvas, (x0 - 7, py), (x0, py), color, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{label} {value:.3f}", (4, max(y0 + 14, min(y0 + h - 4, py + 4))),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)


def render_live_target_state_error(target_state_trace, size=(720, 320),
                                   strict_thresh: Optional[float] = None,
                                   plateau_thresh: Optional[float] = None) -> np.ndarray:
    canvas = np.full((size[1], size[0], 3), 245, dtype=np.uint8)
    x0, y0 = 55, 45
    w, h = size[0] - 80, size[1] - 90
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (80, 80, 80), 1)
    cv2.putText(canvas, "live nearest-replay target-state error",
                (x0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "weighted normalized delta = sqrt(0.2(du/W)^2 + 0.2(dv/H)^2 + 0.6(dd/D)^2)",
                (x0, size[1] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
    if not target_state_trace:
        cv2.putText(canvas, "waiting for library target-state matches...",
                    (x0 + 20, y0 + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (80, 80, 80), 1, cv2.LINE_AA)
        return canvas
    numeric = np.asarray([row[:12] for row in target_state_trace], dtype=np.float64)
    t = numeric[:, 0]
    rel_t = t - t[0]
    delta = numeric[:, 10]
    switched = numeric[:, 11] > 0.5
    max_t = max(1e-3, float(rel_t[-1]), 10.0)
    # Show last 30 seconds; if shorter, show from zero.
    t_min = max(0.0, float(rel_t[-1]) - 30.0)
    plot_t = rel_t - t_min
    mask = plot_t >= 0
    plot_t = plot_t[mask]
    plot_delta = delta[mask]
    plot_switched = switched[mask]
    max_t_window = max(1e-3, min(30.0, max_t - t_min))
    max_delta = max(1.0, float(np.percentile(plot_delta, 95)) * 1.25, float(plot_delta.max()) * 1.05)
    for i in range(1, 5):
        gx = x0 + i * w // 5
        gy = y0 + i * h // 4
        cv2.line(canvas, (gx, y0), (gx, y0 + h), (220, 220, 220), 1)
        cv2.line(canvas, (x0, gy), (x0 + w, gy), (220, 220, 220), 1)
    _draw_threshold_line(canvas, x0, y0, w, h, max_delta, strict_thresh,
                         "strict", (40, 130, 220))
    _draw_threshold_line(canvas, x0, y0, w, h, max_delta, plateau_thresh,
                         "plateau", (20, 150, 20))
    _draw_y_value_label(canvas, x0, y0, h, max_delta, float(np.min(plot_delta)),
                        "min", (120, 90, 160))
    _draw_y_value_label(canvas, x0, y0, h, max_delta, float(delta[-1]),
                        "last", (170, 70, 210))
    pts = []
    for ti, e in zip(plot_t, plot_delta):
        px = int(x0 + np.clip(ti / max_t_window, 0, 1) * w)
        py = int(y0 + h - np.clip(e / max_delta, 0, 1) * h)
        pts.append([px, py])
    if len(pts) >= 2:
        cv2.polylines(canvas, [np.asarray(pts, dtype=np.int32)], False,
                      (170, 70, 210), 2, cv2.LINE_AA)
    if len(pts) >= 1:
        cv2.circle(canvas, tuple(pts[-1]), 4, (40, 40, 220), -1, cv2.LINE_AA)
    if plot_switched.any():
        idx = int(np.argmax(plot_switched))
        sx = int(x0 + np.clip(plot_t[idx] / max_t_window, 0, 1) * w)
        cv2.line(canvas, (sx, y0), (sx, y0 + h), (20, 150, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"current norm_delta={float(delta[-1]):.3f}  u={numeric[-1,1]:.1f} v={numeric[-1,2]:.1f} d={numeric[-1,3]:.0f}",
                (x0, y0 + h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"max y={max_delta:.2f}",
                (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50, 50, 50), 1, cv2.LINE_AA)
    return canvas


def render_live_target_depth(depth_trace, size=(720, 320)) -> np.ndarray:
    canvas = np.full((size[1], size[0], 3), 245, dtype=np.uint8)
    x0, y0 = 55, 45
    w, h = size[0] - 80, size[1] - 90
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (80, 80, 80), 1)
    cv2.putText(canvas, "live target bbox depth",
                (x0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "yellow=filtered depth used by switching, gray=raw bbox median",
                (x0, size[1] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
    if not depth_trace:
        cv2.putText(canvas, "waiting for target depth...",
                    (x0 + 20, y0 + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (80, 80, 80), 1, cv2.LINE_AA)
        return canvas

    arr = np.asarray(depth_trace, dtype=np.float64)
    # columns: t, filtered_depth_mm, raw_depth_mm, depth_ignored
    rel_t = arr[:, 0] - arr[0, 0]
    filt = arr[:, 1]
    raw = arr[:, 2]
    ignored = arr[:, 3] > 0.5
    t_min = max(0.0, float(rel_t[-1]) - 30.0)
    mask = rel_t >= t_min
    plot_t = rel_t[mask] - t_min
    plot_filt = filt[mask]
    plot_raw = raw[mask]
    plot_ignored = ignored[mask]
    max_t_window = max(1e-3, min(30.0, max(float(rel_t[-1]), 10.0) - t_min))
    valid_vals = np.concatenate([
        plot_filt[np.isfinite(plot_filt)],
        plot_raw[np.isfinite(plot_raw)],
    ])
    if valid_vals.size == 0:
        return canvas
    y_min = max(0.0, float(np.percentile(valid_vals, 5)) - 100.0)
    y_max = max(y_min + 100.0, float(np.percentile(valid_vals, 95)) + 150.0, float(np.max(valid_vals)) * 1.02)

    for i in range(1, 5):
        gx = x0 + i * w // 5
        gy = y0 + i * h // 4
        cv2.line(canvas, (gx, y0), (gx, y0 + h), (220, 220, 220), 1)
        cv2.line(canvas, (x0, gy), (x0 + w, gy), (220, 220, 220), 1)

    def to_point(ti, depth):
        px = int(x0 + np.clip(ti / max_t_window, 0, 1) * w)
        py = int(y0 + h - np.clip((depth - y_min) / max(1e-6, y_max - y_min), 0, 1) * h)
        return [px, py]

    raw_pts = [to_point(ti, d) for ti, d in zip(plot_t, plot_raw) if np.isfinite(d)]
    filt_pts = [to_point(ti, d) for ti, d in zip(plot_t, plot_filt) if np.isfinite(d)]
    if len(raw_pts) >= 2:
        cv2.polylines(canvas, [np.asarray(raw_pts, dtype=np.int32)], False,
                      (150, 150, 150), 1, cv2.LINE_AA)
    if len(filt_pts) >= 2:
        cv2.polylines(canvas, [np.asarray(filt_pts, dtype=np.int32)], False,
                      (0, 210, 255), 2, cv2.LINE_AA)
    for ti, d, bad in zip(plot_t, plot_raw, plot_ignored):
        if bad and np.isfinite(d):
            cv2.circle(canvas, tuple(to_point(ti, d)), 4, (0, 0, 255), -1, cv2.LINE_AA)

    last_f = float(filt[-1])
    last_r = float(raw[-1])
    cv2.putText(canvas, f"filtered={last_f:.0f}mm raw={last_r:.0f}mm",
                (x0, y0 + h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{y_max:.0f}mm", (x0 - 52, y0 + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{y_min:.0f}mm", (x0 - 52, y0 + h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50, 50, 50), 1, cv2.LINE_AA)
    return canvas


def render_live_target_uv(target_state_trace, size=(720, 320)) -> np.ndarray:
    canvas = np.full((size[1], size[0], 3), 245, dtype=np.uint8)
    x0, y0 = 55, 45
    w, h = size[0] - 80, size[1] - 90
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (80, 80, 80), 1)
    cv2.putText(canvas, "live target center u/v",
                (x0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "blue=u, green=v, thin lines=current nearest replay ref",
                (x0, size[1] - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
    if not target_state_trace:
        cv2.putText(canvas, "waiting for target-state matches...",
                    (x0 + 20, y0 + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (80, 80, 80), 1, cv2.LINE_AA)
        return canvas

    numeric = np.asarray([row[:12] for row in target_state_trace], dtype=np.float64)
    rel_t = numeric[:, 0] - numeric[0, 0]
    live_u = numeric[:, 1]
    live_v = numeric[:, 2]
    ref_u = numeric[:, 4]
    ref_v = numeric[:, 5]
    switched = numeric[:, 11] > 0.5
    t_min = max(0.0, float(rel_t[-1]) - 30.0)
    mask = rel_t >= t_min
    plot_t = rel_t[mask] - t_min
    plot_live_u = live_u[mask]
    plot_live_v = live_v[mask]
    plot_ref_u = ref_u[mask]
    plot_ref_v = ref_v[mask]
    plot_switched = switched[mask]
    max_t_window = max(1e-3, min(30.0, max(float(rel_t[-1]), 10.0) - t_min))

    vals = np.concatenate([
        plot_live_u[np.isfinite(plot_live_u)],
        plot_live_v[np.isfinite(plot_live_v)],
        plot_ref_u[np.isfinite(plot_ref_u)],
        plot_ref_v[np.isfinite(plot_ref_v)],
    ])
    if vals.size == 0:
        return canvas
    y_min = max(0.0, float(np.percentile(vals, 2)) - 25.0)
    y_max = min(640.0, max(y_min + 50.0, float(np.percentile(vals, 98)) + 25.0))

    for i in range(1, 5):
        gx = x0 + i * w // 5
        gy = y0 + i * h // 4
        cv2.line(canvas, (gx, y0), (gx, y0 + h), (220, 220, 220), 1)
        cv2.line(canvas, (x0, gy), (x0 + w, gy), (220, 220, 220), 1)

    def to_point(ti, value):
        px = int(x0 + np.clip(ti / max_t_window, 0, 1) * w)
        py = int(y0 + h - np.clip((value - y_min) / max(1e-6, y_max - y_min), 0, 1) * h)
        return [px, py]

    def draw_series(values, color, thickness):
        pts = [to_point(ti, v) for ti, v in zip(plot_t, values) if np.isfinite(v)]
        if len(pts) >= 2:
            cv2.polylines(canvas, [np.asarray(pts, dtype=np.int32)], False,
                          color, thickness, cv2.LINE_AA)
        if pts:
            cv2.circle(canvas, tuple(pts[-1]), 3, color, -1, cv2.LINE_AA)

    draw_series(plot_ref_u, (230, 160, 120), 1)
    draw_series(plot_ref_v, (120, 190, 120), 1)
    draw_series(plot_live_u, (220, 80, 40), 2)
    draw_series(plot_live_v, (40, 170, 70), 2)
    if plot_switched.any():
        idx = int(np.argmax(plot_switched))
        sx = int(x0 + np.clip(plot_t[idx] / max_t_window, 0, 1) * w)
        cv2.line(canvas, (sx, y0), (sx, y0 + h), (20, 150, 20), 2, cv2.LINE_AA)

    cv2.putText(canvas, f"u={float(live_u[-1]):.1f} ref_u={float(ref_u[-1]):.1f}  "
                        f"v={float(live_v[-1]):.1f} ref_v={float(ref_v[-1]):.1f}",
                (x0, y0 + h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{y_max:.0f}px", (x0 - 48, y0 + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50, 50, 50), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{y_min:.0f}px", (x0 - 48, y0 + h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50, 50, 50), 1, cv2.LINE_AA)
    return canvas


class PolicyCommandLogger:
    def __init__(self, log_dir: Optional[str]):
        self.log_dir = log_dir
        self.summary_writer = None
        self.summary_file = None
        if not log_dir:
            return
        os.makedirs(log_dir, exist_ok=True)
        self.summary_file = open(os.path.join(log_dir, "policy_command_summary.csv"),
                                 "w", newline="")
        self.summary_writer = csv.DictWriter(
            self.summary_file,
            fieldnames=[
                "frame", "stage", "idx", "t_rel", "x", "y", "z",
                "qx", "qy", "qz", "qw", "source_idx",
                "target_px", "obstacle_px", "chunk_disp_mm",
            ],
        )
        self.summary_writer.writeheader()
        self.summary_file.flush()

    def close(self):
        if self.summary_file is not None:
            self.summary_file.close()
            self.summary_file = None

    def log(self, frame_idx: int, raw: np.ndarray, smoothed: np.ndarray,
            plan: np.ndarray, scheduled: np.ndarray, source_indices: np.ndarray,
            action_dt: float, target_px: int, obstacle_px: int,
            chunk_disp_mm: float):
        if not self.log_dir:
            return
        np.savez_compressed(
            os.path.join(self.log_dir, f"policy_commands_f{frame_idx:05d}.npz"),
            raw=raw,
            smoothed=smoothed,
            plan=plan,
            scheduled=scheduled,
            scheduled_source_indices=source_indices,
            action_dt=np.array(action_dt, dtype=np.float32),
            target_px=np.array(target_px, dtype=np.int32),
            obstacle_px=np.array(obstacle_px, dtype=np.int32),
            chunk_disp_mm=np.array(chunk_disp_mm, dtype=np.float32),
        )
        if self.summary_writer is None:
            return
        for stage, arr, sources in [
            ("raw", raw, np.arange(raw.shape[0])),
            ("smoothed", smoothed, np.arange(smoothed.shape[0])),
            ("plan", plan, source_indices),
            ("scheduled", scheduled, source_indices[:scheduled.shape[0]]),
        ]:
            for i, row in enumerate(arr):
                src = int(sources[i]) if i < len(sources) else -1
                self.summary_writer.writerow({
                    "frame": frame_idx,
                    "stage": stage,
                    "idx": i,
                    "t_rel": i * action_dt,
                    "x": row[0],
                    "y": row[1],
                    "z": row[2],
                    "qx": row[3],
                    "qy": row[4],
                    "qz": row[5],
                    "qw": row[6],
                    "source_idx": src,
                    "target_px": target_px,
                    "obstacle_px": obstacle_px,
                    "chunk_disp_mm": chunk_disp_mm,
                })
        self.summary_file.flush()


DEPLOYMENT_RUN_FIELDS = [
    "run_id",
    "run_name",
    "run_folder",
    "date_time",
    "operator",
    "model",
    "test_setting",
    "target_variety",
    "target_position_id",
    "target_orientation_id",
    "obstacle_config_id",
    "obstacle_position_id",
    "checkpoint_id",
    "inference_seed",
    "start_time",
    "end_time",
    "duration_s",
    "t_alignment",
    "t_interaction",
    "alignment_timeout_s",
    "timed_out",
    "switch_fired",
    "switch_reason",
    "switch_distance",
    "selected_replay_index",
    "selected_replay_path",
    "selected_replay_name",
    "trajectory_file",
    "action_file",
    "path_summary_path",
    "distance_error_path",
    "distance_error_csv",
    "target_state_error_path",
    "target_state_error_csv",
    "success",
    "collision",
    "collision_count",
    "redo_attempts",
    "failure_mode",
    "notes",
]


class DeploymentRunLogger:
    """Automatic timing/file logger for one deployment trial."""

    @staticmethod
    def safe_name(name: str) -> str:
        safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(name).strip())
        safe = "_".join(part for part in safe.split("_") if part)
        return safe or "unnamed_run"

    def __init__(self, log_dir: Optional[str], run_id: Optional[str], args):
        self.enabled = bool(log_dir)
        self.args = args
        self.root = os.path.abspath(log_dir) if log_dir else None
        self.run_id = run_id or time.strftime("OBSTADIFF_%Y%m%d_%H%M%S")
        self.run_name = args.experiment_run_name or self.run_id
        self.run_folder_name = self.safe_name(self.run_name) if args.experiment_run_name else self.run_id
        self.run_dir = None
        self.summary_csv = None
        self.record = {}
        if not self.enabled:
            return
        self.run_dir = os.path.join(self.root, "runs", self.run_folder_name)
        if os.path.exists(self.run_dir) and args.experiment_run_name:
            self.run_folder_name = f"{self.run_folder_name}_{self.run_id}"
            self.run_dir = os.path.join(self.root, "runs", self.run_folder_name)
        os.makedirs(self.run_dir, exist_ok=True)
        self.summary_csv = os.path.join(self.root, "deployment_runs.csv")
        self.record = {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "run_folder": self.run_folder_name,
            "date_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "operator": args.experiment_operator,
            "model": args.experiment_model,
            "test_setting": args.experiment_setting,
            "target_variety": args.experiment_target_variety,
            "target_position_id": args.experiment_target_position_id,
            "target_orientation_id": args.experiment_target_orientation_id,
            "obstacle_config_id": args.experiment_obstacle_config_id,
            "obstacle_position_id": args.experiment_obstacle_position_id,
            "checkpoint_id": os.path.basename(args.checkpoint),
            "inference_seed": args.experiment_seed,
            "switch_fired": "false",
            "timed_out": "false",
            "alignment_timeout_s": args.alignment_timeout,
            "success": "",
            "collision": "",
            "collision_count": "",
            "redo_attempts": "",
            "failure_mode": "",
            "notes": "",
        }
        self.save_json()
        print(f"[experiment] logging deployment run -> {self.run_dir}")

    def _abs(self, path: Optional[str]) -> str:
        return os.path.abspath(path) if path else ""

    def mark_start(self):
        if not self.enabled:
            return
        self.record["start_time"] = f"{time.time():.6f}"
        self.save_json()

    def mark_switch(self, reason: str, distance=None, selected_entry=None,
                    selected_index=None, align_start_time=None):
        if not self.enabled:
            return
        now = time.time()
        self.record["switch_fired"] = "true"
        self.record["switch_reason"] = reason
        if distance is not None and np.isfinite(distance):
            self.record["switch_distance"] = f"{float(distance):.6f}"
        if selected_index is not None:
            self.record["selected_replay_index"] = str(selected_index)
        if selected_entry is not None:
            meta = selected_entry.get("metadata", {})
            self.record["selected_replay_path"] = selected_entry.get("path", "")
            self.record["selected_replay_name"] = meta.get("name", "")
        if align_start_time is not None:
            self.record["t_alignment"] = f"{max(0.0, now - align_start_time):.3f}"
        self.save_json()

    def mark_end(self, interaction_start_time=None):
        if not self.enabled:
            return
        now = time.time()
        self.record["end_time"] = f"{now:.6f}"
        try:
            start = float(self.record.get("start_time", "nan"))
            self.record["duration_s"] = f"{max(0.0, now - start):.3f}"
        except Exception:
            pass
        if interaction_start_time is not None:
            self.record["t_interaction"] = f"{max(0.0, now - interaction_start_time):.3f}"
        self.save_json()

    def mark_alignment_timeout(self, align_start_time, timeout_s: float):
        if not self.enabled:
            return
        now = time.time()
        self.record["timed_out"] = "true"
        self.record["switch_fired"] = "false"
        self.record["alignment_timeout_s"] = f"{float(timeout_s):.3f}"
        self.record["t_alignment"] = f"{max(0.0, now - align_start_time):.3f}"
        self.record["success"] = "false"
        self.record["failure_mode"] = "missed_bottleneck"
        self.record["notes"] = (
            self.record.get("notes", "") + " alignment timeout before bottleneck switch"
        ).strip()
        self.save_json()

    def attach_paths(self, trajectory_file="", action_file="", path_summary_path="",
                     distance_error_path="", distance_error_csv="",
                     target_state_error_path="", target_state_error_csv=""):
        if not self.enabled:
            return
        self.record.update({
            "trajectory_file": self._abs(trajectory_file),
            "action_file": self._abs(action_file),
            "path_summary_path": self._abs(path_summary_path),
            "distance_error_path": self._abs(distance_error_path),
            "distance_error_csv": self._abs(distance_error_csv),
            "target_state_error_path": self._abs(target_state_error_path),
            "target_state_error_csv": self._abs(target_state_error_csv),
        })
        self.save_json()

    def trajectory_path(self) -> str:
        if not self.enabled:
            return ""
        return os.path.join(self.run_dir, "eef_trajectory.csv")

    def save_json(self):
        if not self.enabled:
            return
        with open(os.path.join(self.run_dir, "deployment_run.json"), "w") as f:
            json.dump(self.record, f, indent=2, sort_keys=True)

    def append_summary(self):
        if not self.enabled:
            return
        os.makedirs(self.root, exist_ok=True)
        rows = []
        if os.path.exists(self.summary_csv):
            with open(self.summary_csv, newline="") as f:
                rows = list(csv.DictReader(f))
        rows = [r for r in rows if r.get("run_id") != self.run_id]
        rows.append(dict(self.record))
        tmp = self.summary_csv + ".tmp"
        with open(tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=DEPLOYMENT_RUN_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in DEPLOYMENT_RUN_FIELDS})
        os.replace(tmp, self.summary_csv)
        print(f"[experiment] saved run record -> {self.summary_csv}")


def write_eef_trajectory_csv(path: str, eef_trace) -> Optional[str]:
    if not path or not eef_trace:
        return None
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    t0 = float(eef_trace[0][0])
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_rel_s", "time", "x", "y", "z", "qx", "qy", "qz", "qw"])
        for row in eef_trace:
            writer.writerow([float(row[0]) - t0] + [float(v) for v in row])
    print(f"[experiment] saved EEF trajectory -> {path}")
    return path


# ── Live segmentation — matches training pipeline from segment_manual.py ──

def _make_csrt():
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    raise RuntimeError("CSRT tracker missing. Install opencv-contrib-python.")


class LiveSegmenter:
    """
    Live segmenter mirroring segment_manual.py training pipeline.

    Outputs (H, W) uint8 with values {0=bg, 1=veg, 2=tomato}.

    Pipeline per frame:
      1. (first call) Auto-detect target tomato bbox from HSV-red ∩ near-depth,
         filtering to round, foreground blobs.
      2. CSRT tracker updates the bbox each frame.
      3. Tomato mask = depth-gated red AND inside the tracked bbox (+padding).
      4. SAM optionally refines the tomato mask if backend='sam_vith'.
      5. Vegetation = green ∩ near-depth, excluding tomato pixels.
      6. Morphological cleanup + fill enclosed holes.

    Backends:
      backend="color"    : tracked color+depth segmentation, ~30Hz.
      backend="sam_vith" : same + SAM refinement of the tomato silhouette, ~1Hz.
    """

    def __init__(self, backend: str = "color",
                 sam_ckpt: str = os.path.join(
                     DATA_COLLECTION_DIR, "assets", "sam_vit_h_4b8939.pth"),
                 depth_min_mm: float = 150.0,
                 depth_max_mm: float = 1500.0,
                 depth_align_margin_px: int = 8,
                 require_depth_for_tomato: bool = True,
                 bbox_pad_px: int = 25):
        self.backend = backend
        self.depth_min_mm = float(depth_min_mm)
        self.depth_max_mm = float(depth_max_mm)
        self.depth_align_margin_px = int(depth_align_margin_px)
        self.require_depth_for_tomato = bool(require_depth_for_tomato)
        self.bbox_pad_px = int(bbox_pad_px)

        self._tracker = None
        self._prev_box = None        # (x, y, w, h)
        self._sam_predictor = None
        self._warned_shape_mismatch = False

        if backend == "sam_vith":
            import sys
            sys.path.insert(0, DATA_COLLECTION_DIR)
            from segment_anything import sam_model_registry, SamPredictor
            print("Loading SAM ViT-H ...")
            sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).to("cuda")
            self._sam_predictor = SamPredictor(sam)
            print("SAM loaded.")
        print(f"Segmenter: backend={backend}, depth=[{depth_min_mm},{depth_max_mm}]mm")

    def _ensure_aligned(self, rgb: np.ndarray, depth_mm: np.ndarray) -> np.ndarray:
        """If RGB and depth shapes don't match, resize depth to RGB shape (NEAREST).
        This isn't a true alignment fix but masks the most common publisher mismatch
        (e.g., depth published at 640x480 while RGB is 848x480 or 1280x720)."""
        H_rgb, W_rgb = rgb.shape[:2]
        H_dep, W_dep = depth_mm.shape[:2]
        if (H_rgb, W_rgb) != (H_dep, W_dep):
            if not self._warned_shape_mismatch:
                print(f"[segmenter] WARN: rgb {rgb.shape[:2]} != depth {depth_mm.shape[:2]} — "
                      f"resizing depth to match. For proper alignment relaunch the camera "
                      f"with `align_depth:=true` and matching color/depth resolutions.")
                self._warned_shape_mismatch = True
            depth_mm = cv2.resize(depth_mm, (W_rgb, H_rgb), interpolation=cv2.INTER_NEAREST)
        return depth_mm

    def _near_mask(self, depth_mm: np.ndarray) -> np.ndarray:
        """Foreground depth mask, dilated by `depth_align_margin_px` to tolerate small
        RGB-depth registration errors."""
        near = (depth_mm > self.depth_min_mm) & (depth_mm < self.depth_max_mm)
        if self.depth_align_margin_px > 0:
            k = 2 * self.depth_align_margin_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            near = cv2.dilate(near.astype(np.uint8), kernel) > 0
        return near

    # color helpers (mirror segment_manual.py thresholds)
    @staticmethod
    def _green(rgb):
        rf = rgb.astype(np.float32) / 255.0
        r, g, b = rf[:, :, 0], rf[:, :, 1], rf[:, :, 2]
        cmax = np.maximum(np.maximum(r, g), b)
        sat = np.where(cmax > 0, (cmax - np.minimum(np.minimum(r, g), b)) / cmax, 0)
        return (g > r) & (g > b) & (sat > 0.15) & (cmax > 0.1)

    @staticmethod
    def _red(rgb):
        """Strict red: high saturation + high value, rejects pink walls / warm light /
        brown surfaces. Tomatoes are deeply saturated red and pass easily."""
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        return ((h < 10) | (h > 160)) & (s > 100) & (v > 80)

    def segment(self, rgb: np.ndarray, depth_mm: np.ndarray) -> np.ndarray:
        depth_mm = self._ensure_aligned(rgb, depth_mm)
        return self._segment_color_depth(rgb, depth_mm)

    def _segment_color_depth(self, rgb, depth_mm):
        """3-class per-pixel segmentation:
          0=bg, 1=vegetation (green ∩ near-depth), 2=tomato (red ∩ near-depth).
        No bbox, no tracker — pure color+depth thresholding."""
        H, W = rgb.shape[:2]
        out = np.zeros((H, W), dtype=np.uint8)
        near = self._near_mask(depth_mm)

        # Vegetation: green ∩ near
        veg = self._green(rgb) & near
        out[veg] = 1

        # Tomato: red ∩ near (overrides veg in overlap)
        red = self._red(rgb)
        tom = red & near if self.require_depth_for_tomato else red
        out[tom] = 2
        return out

    def _auto_bbox(self, rgb, depth_mm):
        """Find the most plausible foreground tomato in the current frame."""
        H, W = rgb.shape[:2]
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h_, s_, v_ = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        red_strict = ((h_ < 10) | (h_ > 160)) & (s_ > 100) & (v_ > 80)
        near = (depth_mm > self.depth_min_mm) & (depth_mm < self.depth_max_mm)
        cand = red_strict & near
        n_lab, lab = cv2.connectedComponents(cand.astype(np.uint8))
        if n_lab <= 1:
            return None
        best_score = -1.0
        best_box = None
        for i in range(1, n_lab):
            comp = (lab == i)
            n = int(comp.sum())
            if n < 200:
                continue
            ys, xs = np.where(comp)
            x0, y0 = int(xs.min()), int(ys.min())
            x1, y1 = int(xs.max()), int(ys.max())
            w, h = x1 - x0, y1 - y0
            if w < 10 or h < 10:
                continue
            aspect = max(w, h) / max(1, min(w, h))
            if aspect > 2.0:                      # reject elongated red panels
                continue
            d_med = float(np.median(depth_mm[ys, xs][depth_mm[ys, xs] > 0]) or 1000)
            score = n / max(1.0, d_med)            # large + close wins
            if score > best_score:
                best_score = score
                best_box = (x0, y0, w, h)
        return best_box

    def _segment_tracked(self, rgb, depth_mm):
        H, W = rgb.shape[:2]
        out = np.zeros((H, W), dtype=np.uint8)

        # 1) Update tracker
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, box = self._tracker.update(bgr)
        if not ok:
            box = self._prev_box
        bx, by, bw, bh = [int(v) for v in box]

        # 2) Expanded bbox region with padding (in-bbox mask)
        pad = self.bbox_pad_px
        x0 = max(0, bx - pad); y0 = max(0, by - pad)
        x1 = min(W, bx + bw + pad); y1 = min(H, by + bh + pad)
        in_box = np.zeros((H, W), dtype=bool); in_box[y0:y1, x0:x1] = True

        near = self._near_mask(depth_mm)
        red  = self._red(rgb)
        green = self._green(rgb)

        # 3) Vegetation: green + near, EXCLUDING the tomato bbox region
        veg = green & near & ~in_box
        out[veg] = 1

        # 4) Tomato candidate: red ∩ in_bbox (+ optional near). Take largest blob.
        tom_cand = red & in_box
        if self.require_depth_for_tomato:
            tom_cand = tom_cand & near
        nl, lab = cv2.connectedComponents(tom_cand.astype(np.uint8))
        if nl > 1:
            sizes = [(i, int((lab == i).sum())) for i in range(1, nl)]
            best_id = max(sizes, key=lambda x: x[1])[0]
            tom_mask = (lab == best_id)
        else:
            tom_mask = np.zeros((H, W), dtype=bool)

        # 5) Optional SAM refinement of the tomato silhouette
        if self.backend == "sam_vith" and tom_mask.sum() > 80:
            tom_mask = self._refine_with_sam(rgb, tom_mask, (x0, y0, x1, y1))

        # 6) Morphological cleanup + hole fill
        if tom_mask.sum() > 0:
            u8 = tom_mask.astype(np.uint8) * 255
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, k)
            contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                filled = np.zeros_like(u8)
                cv2.drawContours(filled, [largest], -1, 255, thickness=cv2.FILLED)
                tom_mask = filled > 0
            out[tom_mask] = 2  # tomato wins over vegetation

        # 7) Re-seat CSRT to the actual tomato extent so it doesn't drift onto leaves
        if tom_mask.sum() > 100:
            ys, xs = np.where(tom_mask)
            nx0, ny0 = int(xs.min()), int(ys.min())
            nx1, ny1 = int(xs.max()), int(ys.max())
            new_box = (nx0, ny0, nx1 - nx0, ny1 - ny0)
            self._prev_box = new_box
            try:
                self._tracker = _make_csrt()
                self._tracker.init(bgr, new_box)
            except Exception:
                pass
        return out

    def _refine_with_sam(self, rgb, color_mask, bbox_xyxy):
        x0, y0, x1, y1 = bbox_xyxy
        ys, xs = np.where(color_mask)
        cx, cy = int(np.median(xs)), int(np.median(ys))
        self._sam_predictor.set_image(rgb)
        masks, scores, _ = self._sam_predictor.predict(
            box=np.array([[x0, y0, x1, y1]], dtype=np.float32),
            point_coords=np.array([[cx, cy]], dtype=np.float32),
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=True,
        )
        # pick whichever mask is most "red-covering" inside the bbox
        red = self._red(rgb)
        best_idx, best_score = 0, -1
        for i, m in enumerate(masks):
            m = m.astype(bool); n = int(m.sum())
            if n < 80 or n > 60000:
                continue
            rf = float((m & red).sum()) / max(1, n)
            iou = (m & color_mask).sum() / max(1, (m | color_mask).sum())
            s = 0.4 * float(scores[i]) + 0.3 * rf + 0.3 * iou
            if s > best_score:
                best_score, best_idx = s, i
        out = masks[best_idx].astype(bool)
        # constrain to bbox to prevent SAM from leaking onto leaves
        in_box = np.zeros_like(out); in_box[y0:y1, x0:x1] = True
        out = out & in_box
        return out if 80 < out.sum() < 60000 else color_mask

    # (legacy methods retained below for compatibility — not used in the new pipeline)
    def _segment_hsv_depth(self, rgb, depth_mm):
        H, W = rgb.shape[:2]
        depth_mm = self._ensure_aligned(rgb, depth_mm)
        near = self._near_mask(depth_mm)
        out = np.zeros((H, W), dtype=np.uint8)

        # Vegetation: green pixels in (dilated) near-depth
        veg = self._green(rgb) & near
        out[veg] = 1

        # Tomato: red pixels. Depth is OPTIONAL — red is specific enough that
        # depth misalignment was cutting the boundary off when intersecting.
        red = self._red(rgb)
        tom = red & near if self.require_depth_for_tomato else red
        out[tom] = 2
        return out

    def _segment_sam(self, rgb, depth_mm):
        H, W = rgb.shape[:2]
        depth_mm = self._ensure_aligned(rgb, depth_mm)
        near = self._near_mask(depth_mm)
        out = np.zeros((H, W), dtype=np.uint8)

        self._sam_predictor.set_image(rgb)

        # Vegetation: sample positive points on green & near pixels
        veg_seed = self._green(rgb) & near
        if veg_seed.sum() > 100:
            ys, xs = np.where(veg_seed)
            idx = np.random.default_rng(0).choice(len(xs), size=min(6, len(xs)), replace=False)
            pts = np.array([[int(xs[i]), int(ys[i])] for i in idx], dtype=np.float32)
            masks, scores, _ = self._sam_predictor.predict(
                point_coords=pts, point_labels=np.ones(len(pts), dtype=np.int32),
                multimask_output=True)
            plant_mask = masks[int(np.argmax(scores))].astype(bool) & near
            out[plant_mask] = 1

        # Tomato: prompt SAM with the red centroid (depth optional). Use the largest
        # red blob in the image regardless of depth — most robust to misalignment.
        red = self._red(rgb)
        red_seed = red & near if self.require_depth_for_tomato else red
        if red_seed.sum() > 50:
            # use the largest connected red blob as the centroid
            nl, lab = cv2.connectedComponents(red_seed.astype(np.uint8))
            if nl > 1:
                sizes = [(i, int((lab == i).sum())) for i in range(1, nl)]
                best_id = max(sizes, key=lambda x: x[1])[0]
                ys, xs = np.where(lab == best_id)
                cx, cy = int(np.median(xs)), int(np.median(ys))
                # optional bbox prompt from the red blob extent
                x0, y0 = int(xs.min()), int(ys.min())
                x1, y1 = int(xs.max()), int(ys.max())
                pad = 10
                box = np.array([[max(0, x0-pad), max(0, y0-pad),
                                 min(W-1, x1+pad), min(H-1, y1+pad)]], dtype=np.float32)
                masks, scores, _ = self._sam_predictor.predict(
                    point_coords=np.array([[cx, cy]], dtype=np.float32),
                    point_labels=np.array([1], dtype=np.int32),
                    box=box,
                    multimask_output=True)
                tom_mask = masks[int(np.argmax(scores))].astype(bool)
                # only intersect with depth if explicitly requested
                if self.require_depth_for_tomato:
                    tom_mask = tom_mask & near
                if 80 < int(tom_mask.sum()) < 60000:
                    out[tom_mask] = 2
                else:
                    # fallback: use just the red blob if SAM produced a weird mask
                    out[red_seed] = 2
        return out


# ── Bbox-target + YOLO/SAM live segmentation ───────────────────────────────


class BBoxYoloSamSegmenter:
    """
    Live segmenter matching postprocess_bbox_seg:
      0 = background
      1 = obstacle/vegetation, from YOLO-World obstacle boxes + MobileSAM masks
      2 = target/object, filled bbox only
    """

    def __init__(self,
                 yolo_weights=WORLD_WEIGHTS_DEFAULT,
                 sam_weights=SAM_WEIGHTS_DEFAULT,
                 device="cuda",
                 object_classes=None,
                 obstacle_classes=None,
                 conf=WORLD_CONF_DEFAULT,
                 draw_box=True,
                 depth_min_mm=150.0,
                 depth_front_buffer_mm=250.0,
                 yolo_every=1,
                 sam_every=1,
                 min_object_box_area_frac=0.002,
                 min_object_box_side_px=20.0,
                 min_obstacle_box_area_frac=0.001,
                 min_obstacle_box_side_px=12.0):
        self.yolo_weights = yolo_weights
        self.sam_weights = sam_weights
        self.device = device
        self.object_classes = list(object_classes or WORLD_OBJECT_CLASSES_DEFAULT)
        self.obstacle_classes = list(obstacle_classes or WORLD_OBSTACLE_CLASSES_DEFAULT)
        self.conf = float(conf)
        self.draw_box = bool(draw_box)
        self.depth_min_mm = float(depth_min_mm)
        self.depth_front_buffer_mm = float(depth_front_buffer_mm)
        self.yolo_every = max(1, int(yolo_every))
        self.sam_every = max(1, int(sam_every))
        self.min_object_box_area_frac = float(min_object_box_area_frac)
        self.min_object_box_side_px = float(min_object_box_side_px)
        self.min_obstacle_box_area_frac = float(min_obstacle_box_area_frac)
        self.min_obstacle_box_side_px = float(min_obstacle_box_side_px)

        self._world = None
        self._sam = None
        self._tracker = None
        self._box = None
        self._prompted = False
        self._frame_idx = 0
        self._last_obs_boxes = []
        self._last_obs_mask = None
        self.last_timing = {
            "yolo_ms": 0.0,
            "tracker_ms": 0.0,
            "sam_ms": 0.0,
            "total_ms": 0.0,
            "ran_yolo": False,
            "ran_sam": False,
        }
        self._load_models()

    def _load_models(self):
        from ultralytics import YOLO, SAM
        print(f"[bbox-seg] Loading YOLO-World {self.yolo_weights} on {self.device}...")
        self._world = YOLO(self.yolo_weights)
        self._world.set_classes(self.object_classes + self.obstacle_classes)
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self._world.predict(dummy, device=self.device, verbose=False)

        print(f"[bbox-seg] Loading MobileSAM {self.sam_weights} on {self.device}...")
        self._sam = SAM(self.sam_weights)
        try:
            self._sam.predict(dummy, bboxes=[[10, 10, 100, 100]],
                              device=self.device, verbose=False)
        except Exception:
            pass
        print(f"[bbox-seg] object classes:   {self.object_classes}")
        print(f"[bbox-seg] obstacle classes: {self.obstacle_classes}")

    @staticmethod
    def _clip_box(box, width, height):
        x0, y0, x1, y1 = [int(round(float(v))) for v in box]
        x0 = max(0, min(width - 1, x0))
        x1 = max(0, min(width, x1))
        y0 = max(0, min(height - 1, y0))
        y1 = max(0, min(height, y1))
        if x1 <= x0 + 2 or y1 <= y0 + 2:
            return None
        return (x0, y0, x1, y1)

    @staticmethod
    def _box_area(box):
        x0, y0, x1, y1 = box
        return max(0, x1 - x0) * max(0, y1 - y0)

    def _box_passes_size_filter(self, box, width, height, *, is_object: bool) -> bool:
        cbox = self._clip_box(box, width, height)
        if cbox is None:
            return False
        x0, y0, x1, y1 = cbox
        bw = x1 - x0
        bh = y1 - y0
        area_frac = self._box_area(cbox) / float(max(1, width * height))
        if is_object:
            return (
                bw >= self.min_object_box_side_px
                and bh >= self.min_object_box_side_px
                and area_frac >= self.min_object_box_area_frac
            )
        return (
            bw >= self.min_obstacle_box_side_px
            and bh >= self.min_obstacle_box_side_px
            and area_frac >= self.min_obstacle_box_area_frac
        )

    def _detect_boxes(self, rgb):
        H, W = rgb.shape[:2]
        det = self._world.predict(rgb, conf=self.conf, device=self.device, verbose=False)
        if not det or det[0].boxes is None or len(det[0].boxes) == 0:
            return [], []
        boxes = det[0].boxes.xyxy
        cls = det[0].boxes.cls
        score = det[0].boxes.conf
        try:
            boxes = boxes.cpu().numpy()
            cls = cls.cpu().numpy().astype(int)
            score = score.cpu().numpy()
        except Exception:
            boxes = np.asarray(boxes)
            cls = np.asarray(cls).astype(int)
            score = np.asarray(score)

        n_obj = len(self.object_classes)
        obj_boxes, obs_boxes = [], []
        dropped_obj = 0
        dropped_obs = 0
        for b, c, s in zip(boxes, cls, score):
            item = (tuple(float(v) for v in b), float(s))
            if int(c) < n_obj:
                if self._box_passes_size_filter(item[0], W, H, is_object=True):
                    obj_boxes.append(item)
                else:
                    dropped_obj += 1
            else:
                if self._box_passes_size_filter(item[0], W, H, is_object=False):
                    obs_boxes.append(item)
                else:
                    dropped_obs += 1
        if dropped_obj or dropped_obs:
            print(f"[bbox-seg] filtered small YOLO boxes: object={dropped_obj} obstacle={dropped_obs}")
        return obj_boxes, obs_boxes

    def _choose_nearest_object_box(self, obj_boxes, shape):
        H, W = shape[:2]
        clipped = []
        for box, score in obj_boxes:
            cbox = self._clip_box(box, W, H)
            if cbox is not None:
                clipped.append((cbox, score))
        if not clipped:
            return None
        if self._box is None:
            return max(clipped, key=lambda x: x[1])[0]

        px0, py0, px1, py1 = self._box
        pcx, pcy = (px0 + px1) * 0.5, (py0 + py1) * 0.5
        parea = max(1, self._box_area(self._box))

        def rank(item):
            box, score = item
            x0, y0, x1, y1 = box
            cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
            dist = np.hypot(cx - pcx, cy - pcy)
            area_ratio = abs(np.log(max(1, self._box_area(box)) / parea))
            return score - 0.001 * dist - 0.15 * area_ratio

        return max(clipped, key=rank)[0]

    def _init_tracker(self, rgb, box):
        self._box = box
        try:
            self._tracker = _make_csrt()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            x0, y0, x1, y1 = box
            self._tracker.init(bgr, (x0, y0, x1 - x0, y1 - y0))
        except Exception as e:
            self._tracker = None
            print(f"[bbox-seg] WARN: CSRT unavailable/failed ({e}); holding last bbox on misses.")

    def _prompt_initial_box(self, rgb):
        self._prompted = True
        if not self.draw_box:
            return
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        print("\n[bbox-seg] Draw target/object bbox on first live frame.")
        print("           ENTER/SPACE = accept, c/ESC = skip and use YOLO.")
        win = "Select target bbox"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, bgr)
        try:
            cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)
            cv2.moveWindow(win, 80, 80)
        except Exception:
            pass
        cv2.waitKey(300)
        roi = cv2.selectROI(win, bgr, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(win)
        x, y, w, h = [int(v) for v in roi]
        box = self._clip_box((x, y, x + w, y + h), rgb.shape[1], rgb.shape[0])
        if box is not None:
            print(f"[bbox-seg] initial target bbox = {box}")
            self._init_tracker(rgb, box)
        else:
            print("[bbox-seg] no drawn bbox; using YOLO initial object box.")

    def _update_tracker_box(self, rgb):
        if self._tracker is None:
            return None
        try:
            ok, box = self._tracker.update(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        except Exception:
            return None
        if not ok:
            return None
        x, y, w, h = [int(round(v)) for v in box]
        return self._clip_box((x, y, x + w, y + h), rgb.shape[1], rgb.shape[0])

    def _object_depth(self, depth_mm, box):
        if box is None:
            return None
        x0, y0, x1, y1 = box
        roi = depth_mm[y0:y1, x0:x1]
        valid = roi[(roi > self.depth_min_mm) & (roi < 65000)]
        if valid.size < 10:
            return None
        return float(np.median(valid))

    def _sam_obstacle_mask(self, rgb, depth_mm, obs_boxes, object_box):
        H, W = rgb.shape[:2]
        boxes = []
        for box, _score in obs_boxes:
            cbox = self._clip_box(box, W, H)
            if cbox is not None:
                boxes.append(list(cbox))
        if not boxes:
            return np.zeros((H, W), dtype=bool)

        try:
            res = self._sam.predict(rgb, bboxes=boxes, device=self.device, verbose=False)
        except Exception as e:
            print(f"[bbox-seg] WARN: MobileSAM obstacle pass failed: {e}")
            return np.zeros((H, W), dtype=bool)
        if not res or res[0].masks is None:
            return np.zeros((H, W), dtype=bool)

        masks_t = res[0].masks.data
        try:
            masks = masks_t.cpu().numpy()
        except Exception:
            masks = np.asarray(masks_t)

        obs = np.zeros((H, W), dtype=bool)
        for m in masks:
            if m.shape != (H, W):
                m = cv2.resize((m > 0.5).astype(np.uint8), (W, H),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                m = m > 0.5
            if int(m.sum()) >= 30:
                obs |= m

        target_depth = self._object_depth(depth_mm, object_box)
        valid_depth = (depth_mm > self.depth_min_mm) & (depth_mm < 65000)
        if target_depth is not None:
            obs &= valid_depth & (depth_mm <= target_depth + self.depth_front_buffer_mm)
        else:
            obs &= valid_depth
        return obs

    def segment(self, rgb: np.ndarray, depth_mm: np.ndarray) -> np.ndarray:
        t_total = time.time()
        yolo_ms = 0.0
        tracker_ms = 0.0
        sam_ms = 0.0
        if not self._prompted:
            self._prompt_initial_box(rgb)

        H, W = rgb.shape[:2]
        if depth_mm.shape[:2] != (H, W):
            depth_mm = cv2.resize(depth_mm, (W, H), interpolation=cv2.INTER_NEAREST)

        run_yolo = (self._frame_idx % self.yolo_every == 0) or self._box is None
        run_sam = (self._frame_idx % self.sam_every == 0) or self._last_obs_mask is None
        obj_boxes, obs_boxes = ([], self._last_obs_boxes)
        if run_yolo:
            t_yolo = time.time()
            obj_boxes, obs_boxes = self._detect_boxes(rgb)
            yolo_ms = (time.time() - t_yolo) * 1000.0
            self._last_obs_boxes = obs_boxes

        t_tracker = time.time()
        tracker_box = self._update_tracker_box(rgb)
        tracker_ms = (time.time() - t_tracker) * 1000.0
        if tracker_box is not None:
            self._box = tracker_box

        if obj_boxes:
            yolo_box = self._choose_nearest_object_box(obj_boxes, rgb.shape)
            if yolo_box is not None:
                self._init_tracker(rgb, yolo_box)

        obj_mask = np.zeros((H, W), dtype=bool)
        if self._box is not None:
            x0, y0, x1, y1 = self._box
            obj_mask[y0:y1, x0:x1] = True

        if run_sam:
            t_sam = time.time()
            obs_mask = self._sam_obstacle_mask(rgb, depth_mm, obs_boxes, self._box)
            sam_ms = (time.time() - t_sam) * 1000.0
            self._last_obs_mask = obs_mask
        elif self._last_obs_mask is not None and self._last_obs_mask.shape == (H, W):
            obs_mask = self._last_obs_mask.copy()
        else:
            obs_mask = np.zeros((H, W), dtype=bool)
        obs_mask &= ~obj_mask

        out = np.zeros((H, W), dtype=np.uint8)
        out[obs_mask] = 1
        out[obj_mask] = 2
        self._frame_idx += 1
        self.last_timing = {
            "yolo_ms": yolo_ms,
            "tracker_ms": tracker_ms,
            "sam_ms": sam_ms,
            "total_ms": (time.time() - t_total) * 1000.0,
            "ran_yolo": run_yolo,
            "ran_sam": run_sam,
        }
        return out


# ── 4-channel cam_high assembly (mirrors training _encode_4ch exactly) ─────

SEG_PALETTE = np.array([
    [255, 255, 255],   # 0 bg     — white
    [ 30, 160,  60],   # 1 obstacle/vegetation — green
    [220,  40,  40],   # 2 target/object bbox  — red
], dtype=np.uint8)


def _seg_to_rgb(seg_idx: np.ndarray) -> np.ndarray:
    out = np.zeros((*seg_idx.shape, 3), dtype=np.uint8)
    for c in range(SEG_PALETTE.shape[0]):
        out[seg_idx == c] = SEG_PALETTE[c]
    return out


def render_debug_image(rgb: np.ndarray, depth_mm: np.ndarray, seg_idx: np.ndarray,
                       cam4: np.ndarray, info: dict) -> np.ndarray:
    """
    Compose a (~520 x 1320) BGR image: RGB | Seg | Depth | 4-ch encoder view.
    cam4 is (4, 240, 320) float32 in [0,1] — the exact policy input.
    Returns BGR uint8 image ready for cv2.imwrite.
    """
    H, W = 240, 320
    panel_pad = 6
    title_h = 26
    panels = []

    def labeled(panel_bgr, title):
        h, w = panel_bgr.shape[:2]
        canvas = np.full((h + title_h, w, 3), 30, dtype=np.uint8)
        cv2.putText(canvas, title, (6, title_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        canvas[title_h:, :] = panel_bgr
        return canvas

    # 1) RGB
    rgb_r = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    panels.append(labeled(cv2.cvtColor(rgb_r, cv2.COLOR_RGB2BGR), "RGB"))

    # 2) Segmentation (color-coded)
    seg_r = cv2.resize(seg_idx, (W, H), interpolation=cv2.INTER_NEAREST)
    seg_rgb = _seg_to_rgb(seg_r)
    panels.append(labeled(cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR),
                          f"Seg (target={int((seg_r==2).sum())} obst={int((seg_r==1).sum())})"))

    # 3) Raw depth (mm) — turbo colormap, range [0, DEPTH_MAX_MM]
    d = depth_mm.astype(np.float32)
    d = np.clip(d / DEPTH_MAX_MM, 0, 1) * 255
    d = cv2.resize(d.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    d_color = cv2.applyColorMap(d, cv2.COLORMAP_TURBO)
    panels.append(labeled(d_color, f"Depth (0-{int(DEPTH_MAX_MM)}mm)"))

    # 4) 4-channel encoder view (THE actual policy input) — 2x2 tile of channels
    # cam4 is (4, 240, 320) float32 in [0,1]. Show each channel as grayscale,
    # tiled bg|obstacle over target|depth_norm, then label each quadrant.
    half_h, half_w = H // 2, W // 2
    ch_imgs = []
    for c in range(4):
        ci = (cam4[c] * 255).clip(0, 255).astype(np.uint8)
        ci = cv2.resize(ci, (half_w, half_h), interpolation=cv2.INTER_NEAREST)
        ch_imgs.append(cv2.cvtColor(ci, cv2.COLOR_GRAY2BGR))
    top    = np.concatenate([ch_imgs[0], ch_imgs[1]], axis=1)   # bg | obstacle
    bottom = np.concatenate([ch_imgs[2], ch_imgs[3]], axis=1)   # target | depth_norm
    enc_view = np.concatenate([top, bottom], axis=0)            # (H, W, 3)
    # Quadrant labels
    for (row, col, txt) in [(0,0,"bg"), (0,1,"obst"), (1,0,"target"), (1,1,"depth")]:
        cv2.putText(enc_view, txt, (col*half_w + 4, row*half_h + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.line(enc_view, (half_w, 0), (half_w, H), (200, 200, 200), 1)
    cv2.line(enc_view, (0, half_h), (W, half_h), (200, 200, 200), 1)
    panels.append(labeled(enc_view, "4-ch Encoder Input (policy sees this)"))

    # Concatenate horizontally with padding
    canvas = np.full((H + title_h, panel_pad, 3), 50, dtype=np.uint8)
    composite = panels[0]
    for p in panels[1:]:
        composite = np.concatenate([composite, canvas, p], axis=1)

    # Top info bar
    bar_h = 24
    bar = np.full((bar_h, composite.shape[1], 3), 20, dtype=np.uint8)
    info_str = (f"frame {info.get('frame', '?'):>4}  "
                f"inf {info.get('inf_ms', 0.0):5.1f}ms  "
                f"chunk_disp {info.get('chunk_disp_mm', 0.0):5.1f}mm  "
                f"clip pos {info.get('n_pclip', 0)}/{info.get('n_act', 8)} "
                f"rot {info.get('n_rclip', 0)}/{info.get('n_act', 8)}")
    cv2.putText(bar, info_str, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    composite = np.concatenate([bar, composite], axis=0)
    return composite


def encode_4ch_cam_high(seg_idx: np.ndarray, depth_mm: np.ndarray,
                       out_hw=(240, 320), depth_max_mm=DEPTH_MAX_MM) -> np.ndarray:
    """
    Returns (4, 240, 320) float32 in [0, 1].

    Channels: [bg_mask, obstacle_mask, target_mask, depth_normalized]
    """
    oH, oW = out_hw
    H, W = seg_idx.shape
    # one-hot masks (still in source resolution)
    ch = np.empty((4, H, W), dtype=np.uint8)
    for c in range(3):
        ch[c] = np.where(seg_idx == c, 255, 0).astype(np.uint8)
    d = np.clip(depth_mm.astype(np.float32) / float(depth_max_mm), 0.0, 1.0) * 255.0
    ch[3] = d.astype(np.uint8)
    # resize (NEAREST to preserve binary masks); produce (240, 320, 4)
    img4 = np.moveaxis(ch, 0, -1)
    if (H, W) != (oH, oW):
        img4 = cv2.resize(img4, (oW, oH), interpolation=cv2.INTER_NEAREST)
    img4 = np.moveaxis(img4, -1, 0).astype(np.float32) / 255.0   # (4, oH, oW)
    return img4


def check_depth_alignment(image_topic: str, depth_topic: str, timeout_s: float = 10.0):
    """Verify RGB and depth topics are publishing and that depth is the aligned variant.

    Checks:
      1. Both topics exist and publish within `timeout_s`.
      2. The depth topic name contains 'aligned_depth_to_color'. If not, warns
         loudly because depth will not be in the RGB frame.
      3. Compares RGB and depth image dimensions; warns if they differ.
    """
    import rospy
    from sensor_msgs.msg import Image as RosImage, CameraInfo
    print("\n[align-check] verifying depth-to-RGB alignment...")

    # warn if depth topic isn't the aligned variant
    if "aligned_depth_to_color" not in depth_topic:
        print(f"[align-check] WARNING: depth topic '{depth_topic}' is NOT "
              "'aligned_depth_to_color/image_raw'. Depth will be in the depth "
              "sensor frame, not aligned to RGB. Relaunch the camera with "
              "`align_depth:=true` and use /camera/aligned_depth_to_color/image_raw.")

    rgb_msg = None
    depth_msg = None
    t0 = time.time()
    while time.time() - t0 < timeout_s and (rgb_msg is None or depth_msg is None):
        try:
            if rgb_msg is None:
                rgb_msg = rospy.wait_for_message(image_topic, RosImage, timeout=1.0)
        except rospy.ROSException:
            pass
        try:
            if depth_msg is None:
                depth_msg = rospy.wait_for_message(depth_topic, RosImage, timeout=1.0)
        except rospy.ROSException:
            pass

    if rgb_msg is None:
        print(f"[align-check] FAIL: no RGB on {image_topic} within {timeout_s}s.")
        return False
    if depth_msg is None:
        print(f"[align-check] FAIL: no depth on {depth_topic} within {timeout_s}s. "
              "Relaunch camera with `align_depth:=true`.")
        return False

    print(f"[align-check] RGB   : {rgb_msg.width}x{rgb_msg.height} enc={rgb_msg.encoding}")
    print(f"[align-check] Depth : {depth_msg.width}x{depth_msg.height} enc={depth_msg.encoding}")
    if (rgb_msg.width, rgb_msg.height) != (depth_msg.width, depth_msg.height):
        print(f"[align-check] WARNING: shape mismatch — depth will be auto-resized to "
              f"RGB shape, but the alignment between depth pixels and RGB pixels is "
              f"NOT guaranteed. Relaunch with matching resolutions, e.g.:\n"
              f"    roslaunch realsense2_camera rs_camera.launch align_depth:=true \\\n"
              f"      color_width:=640 color_height:=480 depth_width:=640 depth_height:=480")
        return True
    print("[align-check] OK: shapes match.\n")
    return True


@torch.no_grad()
def sanity_check(policy, dev):
    """Quick dummy-input check: predicts without raising; non-NaN output."""
    obs = {
        "cam_high": torch.zeros(1, 2, 4, 240, 320, dtype=torch.float32, device=dev),
        "eef_pose": torch.zeros(1, 2, 7, dtype=torch.float32, device=dev),
    }
    out = policy.predict_action(obs)
    a = out["action"].detach().cpu().numpy()
    assert a.shape == (1, 8, 7), a.shape
    assert np.isfinite(a).all()
    print(f"  sanity check: action shape {a.shape}, finite ✓")


def run_sanity_check_with_cudnn_fallback(policy, dev, allow_cpu_fallback=False):
    try:
        sanity_check(policy, dev)
        return dev
    except RuntimeError as e:
        msg = str(e)
        if "CUDNN_STATUS_NOT_INITIALIZED" not in msg and "cuDNN error" not in msg:
            raise
        print("  WARN: cuDNN failed during sanity check; disabling cuDNN and retrying.")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        torch.backends.cudnn.enabled = False
        try:
            sanity_check(policy, dev)
            return dev
        except RuntimeError as e2:
            msg2 = str(e2)
            if "CUBLAS_STATUS_NOT_INITIALIZED" not in msg2 and "CUDA error" not in msg2:
                raise
            if not allow_cpu_fallback:
                print("  ERROR: CUDA/CUBLAS failed after cuDNN fallback, and CPU fallback is disabled.")
                print("         Kill stale deploy processes or free GPU memory, then rerun.")
                raise
            print("  WARN: CUDA/CUBLAS failed after cuDNN fallback; moving policy to CPU.")
            try:
                policy.to("cpu")
                torch.cuda.empty_cache()
            except Exception:
                pass
            cpu_dev = torch.device("cpu")
            sanity_check(policy, cpu_dev)
            return cpu_dev


def _strip_repeat_args(argv):
    """Remove supervisor-only args before spawning a single child deployment."""
    flags_with_values = {
        "--repeat_run_count",
        "--repeat_start_index",
        "--repeat_increment",
        "--repeat_run_name_prefix",
        "--experiment_run_name",
    }
    flags_without_values = {
        "--repeat_runs",
        "--prompt_experiment_run_name",
    }
    out = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in flags_without_values:
            i += 1
            continue
        if arg in flags_with_values:
            i += 2
            continue
        out.append(arg)
        i += 1
    return out


def run_repeat_supervisor(args):
    base_args = _strip_repeat_args(sys.argv[1:])
    script = os.path.abspath(__file__)
    prefix = (
        args.repeat_run_name_prefix
        or args.experiment_run_name
        or "-".join(x for x in [args.experiment_setting, "ObstaDiff"] if x)
        or "ObstaDiff"
    )
    count = int(args.repeat_run_count)
    idx = int(args.repeat_start_index)
    increment = int(args.repeat_increment)
    if increment == 0:
        increment = 1
    completed = 0
    print("[repeat] ObstaDiff repeat mode active.")
    print("[repeat] Press ENTER to start the suggested run name, type a custom name, or type q to quit.")
    while count <= 0 or completed < count:
        suggested = f"{prefix}-{idx:03d}"
        try:
            entered = input(f"[repeat] next run name [{suggested}]: ").strip()
        except EOFError:
            entered = ""
        if entered.lower() in ("q", "quit", "exit", "stop"):
            print("[repeat] stopping before next run.")
            break
        run_name = entered or suggested
        child_cmd = [sys.executable, script] + base_args + ["--experiment_run_name", run_name]
        print(f"[repeat] starting {run_name}")
        rc = subprocess.call(child_cmd)
        print(f"[repeat] run {run_name} exited with code {rc}")
        completed += 1
        idx += increment
        if count <= 0:
            try:
                cont = input("[repeat] continue? ENTER=yes, q=quit: ").strip()
            except EOFError:
                cont = ""
            if cont.lower() in ("q", "quit", "exit", "stop"):
                break
    print(f"[repeat] finished {completed} run(s).")


def main():
    global _ACTIVE_CONTROLLER, _ACTIVE_SHM, _ACTIVE_COMMAND_LOGGER
    global _PATH_TRACE, _SCHEDULED_CHUNKS, _SCHEDULED_WAYPOINT_TRACE
    global _TARGET_STATE_TRACE
    global _PATH_SUMMARY_PATH, _DISTANCE_ERROR_PATH, _TARGET_STATE_ERROR_PATH
    global _TARGET_STATE_STRICT_THRESH, _TARGET_STATE_PLATEAU_THRESH
    signal.signal(signal.SIGINT, handle_sigint)
    ensure_ros_python_paths()
    kill_existing_deploy_processes()
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", default=BBOX_CHECKPOINT_DEFAULT,
                        help="Path to the bbox-postprocessed 4ch checkpoint.")
    parser.add_argument("--image_topic", default="/camera/color/image_raw")
    parser.add_argument("--depth_topic", default="/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--frequency", type=float, default=10.0,
                        help="Policy inference Hz (15 matches recording; 10 is safer).")
    parser.add_argument("--max_pos_speed", type=float, default=0.3,
                        help="Controller hard cap on EEF linear speed (m/s).")
    parser.add_argument("--max_step_disp", type=float, default=MAX_STEP_DISP,
                        help="Per-waypoint translation cap (m). At freq=10Hz, 0.05 = 0.5 m/s.")
    parser.add_argument("--max_step_rot", type=float, default=MAX_STEP_ROT,
                        help="Per-waypoint rotation cap (rad).")
    parser.add_argument("--max_chunk_disp", type=float, default=MAX_CHUNK_DISP,
                        help="Reject chunks whose total commanded translation exceeds this (m).")
    parser.add_argument("--overlap_steps", type=int, default=2,
                        help="How many action steps to overlap between chunks. Higher replans more often.")
    parser.add_argument("--action_time_scale", type=float, default=1.0,
                        help="Stretch waypoint timing by this factor. 2.0 makes the same action chunk execute half as fast.")
    parser.add_argument("--action_smoothing", type=float, default=1.0,
                        help="Blend new action chunks with the previous chunk. 1.0=no smoothing, 0.4=smoother/slower.")
    parser.add_argument("--action_start_offset", type=int, default=1,
                        help="Skip this many leading predicted waypoints before scheduling. Helps avoid stale first-step backtracking.")
    parser.add_argument("--drop_inference_pose_radius", type=float, default=0.0,
                        help="Drop leading waypoints within this many meters of the EEF pose used for inference. Try 0.01-0.03.")
    parser.add_argument("--allow_backward_prefix", action="store_true",
                        help="Do not drop leading waypoints that move opposite the chunk's net direction.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--disable_cudnn", action="store_true",
                        help="Disable cuDNN convolutions. Useful if CUDA reports CUDNN_STATUS_NOT_INITIALIZED.")
    parser.add_argument("--allow_cpu_fallback", action="store_true",
                        help="If CUDA fails, fall back to CPU policy inference. Default is fail-fast so GPU runs stay on GPU.")
    parser.add_argument("--num_inference_steps", type=int, default=None,
                        help="Override diffusion inference steps. Use e.g. 16 for faster deployment.")
    parser.add_argument("--yolo_weights", default=WORLD_WEIGHTS_DEFAULT,
                        help="YOLO-World weights for target/obstacle box detection.")
    parser.add_argument("--sam_weights", default=SAM_WEIGHTS_DEFAULT,
                        help="MobileSAM weights for obstacle mask refinement.")
    parser.add_argument("--yolo_device", default=None,
                        help="Device for YOLO/SAM. Defaults to --device. Use cpu if CUDA memory is tight.")
    parser.add_argument("--cpu_segmentation", action="store_true",
                        help="Run YOLO/SAM on CPU to leave CUDA memory for the diffusion policy.")
    parser.add_argument("--object_classes", nargs="+", default=None,
                        help="YOLO-World target classes. Defaults to bell/dark pepper.")
    parser.add_argument("--obstacle_classes", nargs="+", default=None,
                        help="YOLO-World obstacle classes. Defaults to leaves/plant stem/green plant.")
    parser.add_argument("--yolo_conf", type=float, default=WORLD_CONF_DEFAULT,
                        help="YOLO confidence threshold for both target and obstacle boxes.")
    parser.add_argument("--min_object_box_area_frac", type=float, default=0.002,
                        help="Ignore YOLO target/object boxes smaller than this image area fraction.")
    parser.add_argument("--min_object_box_side_px", type=float, default=20.0,
                        help="Ignore YOLO target/object boxes with width or height below this many pixels.")
    parser.add_argument("--min_obstacle_box_area_frac", type=float, default=0.001,
                        help="Ignore YOLO obstacle boxes smaller than this image area fraction before SAM.")
    parser.add_argument("--min_obstacle_box_side_px", type=float, default=12.0,
                        help="Ignore YOLO obstacle boxes with width or height below this many pixels before SAM.")
    parser.add_argument("--yolo_every", type=int, default=1,
                        help="Run YOLO every N control loops. Higher is faster; target tracker still updates every loop.")
    parser.add_argument("--sam_every", type=int, default=1,
                        help="Run MobileSAM every N control loops and reuse the last obstacle mask between runs.")
    parser.add_argument("--no_draw_box", dest="draw_box", action="store_false",
                        help="Skip the first-frame manual target bbox and initialize from YOLO.")
    parser.set_defaults(draw_box=True)
    parser.add_argument("--depth_min_mm", type=float, default=150.0,
                        help="Depth lower bound for valid obstacle pixels (mm).")
    parser.add_argument("--depth_front_buffer_mm", type=float, default=250.0,
                        help="Keep obstacle pixels no farther than target_depth + this buffer.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Skip Sawyer controller; just inference + log.")
    parser.add_argument("--save_debug_dir", type=str, default=None,
                        help="If set, dump a side-by-side PNG showing what the policy "
                             "sees (RGB | seg | depth | 4-ch encoder view) every N steps.")
    parser.add_argument("--debug_every", type=int, default=1,
                        help="Save a debug image every N control steps (default: every step).")
    parser.add_argument("--log_policy_commands_dir", type=str, default=None,
                        help="If set, save raw/smoothed/scheduled policy command chunks for each inference.")
    parser.add_argument("--print_policy_commands", action="store_true",
                        help="Print raw/smoothed/scheduled policy command chunks to the terminal.")
    parser.add_argument("--show_inference_view", action="store_true",
                        help="Show RGB+bbox and segmentation overlay every policy inference.")
    parser.add_argument("--record_clean_preview_video", type=str, default=None,
                        help="If set, record a clean no-text deployment preview video. "
                             "Use 'auto' with --experiment_log_dir to save in the run folder.")
    parser.add_argument("--clean_preview_panel", type=str, default="side_by_side",
                        choices=("side_by_side", "rgb", "seg", "overlay"),
                        help="Which clean preview view to record.")
    parser.add_argument("--clean_preview_fps", type=float, default=10.0,
                        help="FPS for --record_clean_preview_video.")
    parser.add_argument("--clean_preview_no_bbox", action="store_true",
                        help="Do not draw the red target bbox in the clean preview video.")
    parser.add_argument("--show_target_state_error", action="store_true",
                        help="Show live nearest-replay target-state (u,v,d) error plot.")
    parser.add_argument("--interaction_replay_path", type=str, default=None,
                        help="Optional .npz/.npy replay to execute after the alignment bottleneck switch.")
    parser.add_argument("--interaction_library", type=str, default=None,
                        help="Directory of interaction replay .npz files. The best replay is selected at bottleneck.")
    parser.add_argument("--min_interaction_replay_disp", type=float, default=0.010,
                        help="Skip interaction replays whose total translation is below this many meters. "
                             "This prevents accidentally recorded stationary replays from being selected.")
    parser.add_argument("--interaction_key", type=str, default=None,
                        help="Optional key/label filter for interaction library selection, e.g. black_pepper.")
    parser.add_argument("--manual_bottleneck_switch", action="store_true",
                        help="Allow SPACE in the inference preview to switch from alignment to interaction replay.")
    parser.add_argument("--auto_bottleneck_switch", action="store_true",
                        help="Automatically switch when alignment commands stay small for several inferences.")
    parser.add_argument("--bottleneck_action_norm_thresh", type=float, default=0.010,
                        help="Auto-switch threshold: mean scheduled translational step in meters. Default is 10 mm, matching the observed alignment plateau.")
    parser.add_argument("--bottleneck_target_state_thresh", type=float, default=0.25,
                        help="Legacy target-state threshold. Specific depth/UV thresholds below control switching.")
    parser.add_argument("--bottleneck_target_depth_thresh", type=float, default=0.005,
                        help="Normalized depth gate for switching: require |dd|/D <= this before considering u/v.")
    parser.add_argument("--bottleneck_target_uv_thresh", type=float, default=0.2,
                        help="Normalized image-plane switch threshold after depth gate passes: require |du|/W + |dv|/H <= this.")
    parser.add_argument("--bottleneck_target_state_plateau_thresh", type=float, default=0.28,
                        help="Good-enough normalized target-state error for plateau switching when strict threshold is not reached.")
    parser.add_argument("--bottleneck_target_state_plateau_eps", type=float, default=0.05,
                        help="Max range of recent normalized target-state errors considered a stable plateau.")
    parser.add_argument("--bottleneck_target_state_plateau_steps", type=int, default=8,
                        help="Number of recent target-state errors used for good-enough plateau switching.")
    parser.add_argument("--bottleneck_plateau_action_norm_thresh", type=float, default=0.018,
                        help="Plateau switch also requires mean scheduled translational step below this many meters.")
    parser.add_argument("--target_state_norm_width", type=float, default=640.0,
                        help="Width scale W for normalized target-state error.")
    parser.add_argument("--target_state_norm_height", type=float, default=480.0,
                        help="Height scale H for normalized target-state error.")
    parser.add_argument("--target_state_norm_depth_mm", type=float, default=1000.0,
                        help="Depth scale D in millimeters for normalized target-state error.")
    parser.add_argument("--target_depth_center_frac", type=float, default=0.4,
                        help="Fraction of target bbox side length used for target-depth estimation. 0.4 uses the center 40%% crop.")
    parser.add_argument("--target_depth_percentile", type=float, default=20.0,
                        help="Percentile of valid depth pixels in the target center crop. Lower values prefer the closer target surface over background.")
    parser.add_argument("--target_state_weight_u", type=float, default=0.2,
                        help="Weight for normalized u error in nearest-replay target-state distance.")
    parser.add_argument("--target_state_weight_v", type=float, default=0.2,
                        help="Weight for normalized v error in nearest-replay target-state distance.")
    parser.add_argument("--target_state_weight_d", type=float, default=0.6,
                        help="Weight for normalized depth error in nearest-replay target-state distance.")
    parser.add_argument("--target_state_max_depth_jump_mm", type=float, default=400.0,
                        help="Reject one-frame target-depth jumps larger than this many mm when computing nearest-replay target-state error. 0 disables.")
    parser.add_argument("--target_state_depth_ema", type=float, default=0.35,
                        help="EMA alpha for accepted target depth in nearest-replay target-state error. 1.0 disables smoothing.")
    parser.add_argument("--bottleneck_stable_steps", type=int, default=5,
                        help="Auto-switch after this many consecutive low-motion alignment chunks.")
    parser.add_argument("--bottleneck_min_alignment_time", type=float, default=8.0,
                        help="Do not auto-switch until alignment has run for at least this many seconds.")
    parser.add_argument("--bottleneck_timeout", type=float, default=30.0,
                        help="Optional fallback: switch after this many seconds even without low-motion detection. 0 disables.")
    parser.add_argument("--alignment_timeout", type=float, default=16.0,
                        help="Stop and mark the run as missed_bottleneck if no interaction switch occurs within this many alignment seconds. 0 disables.")
    parser.add_argument("--interaction_time_scale", type=float, default=1.0,
                        help="Stretch interaction replay timing by this factor.")
    parser.add_argument("--replay_dt", type=float, default=None,
                        help="Override interaction replay waypoint dt in seconds.")
    parser.add_argument("--stop_and_go", action="store_true",
                        help="Execute a chunk, wait for motion/image to settle, then run the next inference.")
    parser.add_argument("--stop_and_go_steps", type=int, default=1,
                        help="In stop-and-go mode, execute at most this many scheduled waypoints per inference.")
    parser.add_argument("--settle_time", type=float, default=0.35,
                        help="Extra seconds to wait after the last stop-and-go waypoint before checking image sharpness.")
    parser.add_argument("--stable_blur_threshold", type=float, default=80.0,
                        help="Minimum Laplacian variance considered sharp/stable for stop-and-go inference.")
    parser.add_argument("--stable_frames", type=int, default=2,
                        help="Number of consecutive sharp frames required before the next stop-and-go inference.")
    parser.add_argument("--stable_timeout", type=float, default=2.0,
                        help="Max seconds to wait for a sharp image before continuing anyway.")
    parser.add_argument("--path_summary_path", type=str,
                        default="/tmp/bbox_deploy_path_summary.png",
                        help="Where to save the final actual/scheduled path summary image.")
    parser.add_argument("--distance_error_path", type=str,
                        default="/tmp/bbox_deploy_distance_error.png",
                        help="Where to save actual-vs-commanded EEF distance error plot.")
    parser.add_argument("--distance_error_csv", type=str, default=None,
                        help="Optional CSV dump of actual/desired EEF positions and distance error.")
    parser.add_argument("--target_state_error_path", type=str,
                        default="/tmp/bbox_deploy_target_state_error.png",
                        help="Where to save nearest-replay target-state error plot.")
    parser.add_argument("--target_state_error_csv", type=str, default=None,
                        help="Optional CSV dump of live/replay target-state errors.")
    parser.add_argument("--experiment_log_dir", type=str, default=None,
                        help="If set, automatically log deployment timing, selected replay, EEF trajectory, and output paths here.")
    parser.add_argument("--experiment_run_id", type=str, default=None,
                        help="Optional run id for --experiment_log_dir. Defaults to OBSTADIFF_YYYYMMDD_HHMMSS.")
    parser.add_argument("--experiment_run_name", type=str, default="",
                        help="Human-readable run name stored in the deployment record.")
    parser.add_argument("--prompt_experiment_run_name", action="store_true",
                        help="Prompt in the terminal for a run name before starting deployment logging.")
    parser.add_argument("--repeat_runs", action="store_true",
                        help="Stay in this parent process and launch one ObstaDiff deployment after another.")
    parser.add_argument("--repeat_run_count", type=int, default=0,
                        help="Number of repeated runs to launch. 0 means keep asking until q/quit.")
    parser.add_argument("--repeat_start_index", type=int, default=1,
                        help="Starting numeric suffix for suggested repeat run names.")
    parser.add_argument("--repeat_increment", type=int, default=1,
                        help="Amount to add to the suggested run-number suffix after each trial.")
    parser.add_argument("--repeat_run_name_prefix", type=str, default="",
                        help="Prefix for suggested repeat run names, e.g. TAG-ObstaDiff.")
    parser.add_argument("--experiment_operator", type=str, default="",
                        help="Operator name stored in the deployment run record.")
    parser.add_argument("--experiment_model", type=str, default="obstadiff",
                        help="Model name stored in the deployment run record.")
    parser.add_argument("--experiment_setting", type=str, default="",
                        help="Experiment setting stored in the deployment run record, e.g. TPG, OLG, TAG.")
    parser.add_argument("--experiment_target_variety", type=str, default="",
                        help="Target variety stored in the deployment run record.")
    parser.add_argument("--experiment_target_position_id", type=str, default="",
                        help="Target position id stored in the deployment run record.")
    parser.add_argument("--experiment_target_orientation_id", type=str, default="",
                        help="Target orientation id stored in the deployment run record.")
    parser.add_argument("--experiment_obstacle_config_id", type=str, default="",
                        help="Obstacle config id stored in the deployment run record.")
    parser.add_argument("--experiment_obstacle_position_id", type=str, default="",
                        help="Obstacle position id stored in the deployment run record.")
    parser.add_argument("--experiment_seed", type=str, default="",
                        help="Inference seed/condition seed stored in the deployment run record.")
    parser.add_argument("--no_show_path_summary", dest="show_path_summary",
                        action="store_false",
                        help="Save final path/error images without opening display windows.")
    parser.set_defaults(show_path_summary=True)
    args = parser.parse_args()
    if args.repeat_runs:
        run_repeat_supervisor(args)
        return
    if args.prompt_experiment_run_name and not args.experiment_run_name:
        try:
            args.experiment_run_name = input("Experiment run name: ").strip()
        except EOFError:
            args.experiment_run_name = ""
    run_logger = DeploymentRunLogger(
        args.experiment_log_dir,
        args.experiment_run_id,
        args,
    )
    if run_logger.enabled:
        if not args.log_policy_commands_dir:
            args.log_policy_commands_dir = os.path.join(run_logger.run_dir, "action_logs")
        if not args.distance_error_csv:
            args.distance_error_csv = os.path.join(run_logger.run_dir, "distance_error.csv")
        if not args.target_state_error_csv:
            args.target_state_error_csv = os.path.join(run_logger.run_dir, "target_state_error.csv")
        if args.path_summary_path == "/tmp/bbox_deploy_path_summary.png":
            args.path_summary_path = os.path.join(run_logger.run_dir, "path_summary.png")
        if args.distance_error_path == "/tmp/bbox_deploy_distance_error.png":
            args.distance_error_path = os.path.join(run_logger.run_dir, "distance_error.png")
        if args.target_state_error_path == "/tmp/bbox_deploy_target_state_error.png":
            args.target_state_error_path = os.path.join(run_logger.run_dir, "target_state_error.png")
        if args.record_clean_preview_video == "auto":
            args.record_clean_preview_video = os.path.join(run_logger.run_dir, "clean_preview.mp4")
    elif args.record_clean_preview_video == "auto":
        args.record_clean_preview_video = "/tmp/bbox_deploy_clean_preview.mp4"
    _PATH_SUMMARY_PATH = args.path_summary_path
    _DISTANCE_ERROR_PATH = args.distance_error_path
    _TARGET_STATE_ERROR_PATH = args.target_state_error_path
    _TARGET_STATE_STRICT_THRESH = args.bottleneck_target_uv_thresh
    _TARGET_STATE_PLATEAU_THRESH = None
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("[cuda] cuDNN disabled by --disable_cudnn")
    if (args.interaction_replay_path or args.interaction_library) and not args.manual_bottleneck_switch:
        args.auto_bottleneck_switch = True
        print("[phase] enabling auto bottleneck switch because an interaction replay/library was provided")
    if args.manual_bottleneck_switch and not args.show_inference_view:
        args.show_inference_view = True
        print("[phase] enabling --show_inference_view so SPACE can trigger the bottleneck switch")
    interaction_replay = load_interaction_replay(args.interaction_replay_path)
    if interaction_replay is not None:
        print(f"[phase] loaded interaction replay {args.interaction_replay_path}: "
              f"{len(interaction_replay['rel_pos'])} waypoints, "
              f"source_dt={float(interaction_replay['dt']):.3f}s")
    interaction_library = load_interaction_library(
        args.interaction_library,
        min_translation_m=args.min_interaction_replay_disp,
    )
    if interaction_library and not args.interaction_key:
        args.interaction_key = infer_interaction_key(args.object_classes)
        if args.interaction_key:
            print(f"[library] inferred interaction_key={args.interaction_key} from object classes")
        else:
            print("[library] no interaction_key provided/inferred; selecting from full library")

    from diffusion_policy.real_world.sawyer_interpolation_controller import SawyerInterpolationController
    from multiprocessing.managers import SharedMemoryManager

    # ── Load policy ────────────────────────────────────────────────────────
    print(f"Loading checkpoint {args.checkpoint}")
    patch_huggingface_hub_for_old_diffusers()
    payload = torch.load(open(args.checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.get("use_ema", False) else workspace.model
    requested_dev = torch.device(args.device)
    dev = requested_dev
    try:
        policy.eval().to(dev)
    except RuntimeError as e:
        msg = str(e)
        if "CUDA" not in msg and "CUBLAS" not in msg and "cuDNN" not in msg:
            raise
        if not args.allow_cpu_fallback:
            print(f"  ERROR: failed to move policy to {dev}: {e}")
            print("         CPU fallback is disabled; free GPU memory or kill stale deploy processes.")
            raise
        print(f"  WARN: failed to move policy to {dev}: {e}")
        print("  WARN: falling back to CPU policy inference.")
        dev = torch.device("cpu")
        policy.eval().to(dev)
    if args.num_inference_steps is not None and hasattr(policy, "num_inference_steps"):
        policy.num_inference_steps = int(args.num_inference_steps)
        print(f"  overriding diffusion num_inference_steps={policy.num_inference_steps}")
    print(f"  n_obs_steps={cfg.n_obs_steps}  n_action_steps={cfg.n_action_steps}  use_ema=True")

    n_obs_steps    = int(cfg.n_obs_steps)
    n_action_steps = int(cfg.n_action_steps)

    dev = run_sanity_check_with_cudnn_fallback(policy, dev, allow_cpu_fallback=args.allow_cpu_fallback)
    if dev.type == "cpu" and args.yolo_device and str(args.yolo_device).startswith("cuda"):
        print("  WARN: policy fell back to CPU; forcing YOLO/SAM to CPU as well.")
        args.cpu_segmentation = True
        args.yolo_device = None

    # ── Start segmenter (BEFORE rospy + controller) ────────────────────────
    seg_device = "cpu" if args.cpu_segmentation else (args.yolo_device or args.device)
    segmenter = BBoxYoloSamSegmenter(
        yolo_weights=args.yolo_weights,
        sam_weights=args.sam_weights,
        device=seg_device,
        object_classes=args.object_classes,
        obstacle_classes=args.obstacle_classes,
        conf=args.yolo_conf,
        draw_box=args.draw_box,
        depth_min_mm=args.depth_min_mm,
        depth_front_buffer_mm=args.depth_front_buffer_mm,
        yolo_every=args.yolo_every,
        sam_every=args.sam_every,
        min_object_box_area_frac=args.min_object_box_area_frac,
        min_object_box_side_px=args.min_object_box_side_px,
        min_obstacle_box_area_frac=args.min_obstacle_box_area_frac,
        min_obstacle_box_side_px=args.min_obstacle_box_side_px,
    )

    # ── Start Sawyer controller (handles its own ROS init + homing) ────────
    shm = SharedMemoryManager(); shm.start()
    _ACTIVE_SHM = shm
    if args.dry_run:
        controller = None
    else:
        print(f"Starting Sawyer controller — homing to HOME_JOINTS_LIST={HOME_JOINTS_LIST}")
        controller = SawyerInterpolationController(
            shm_manager=shm,
            frequency=200,
            max_pos_speed=args.max_pos_speed,
            max_rot_speed=args.max_pos_speed * 3.0,
            joints_init=HOME_JOINTS_LIST,
            joints_init_speed=0.3,
            verbose=False,
        )
        _ACTIVE_CONTROLLER = controller
        controller.start()
        if not controller.ready_event.wait(timeout=30.0):
            raise RuntimeError("Sawyer controller did not become ready within 30s — "
                               "check robot enable / clock sync / "
                               "/tmp/sawyer_controller_stderr.txt")
        if not controller.is_alive():
            raise RuntimeError("Sawyer controller process died during startup. "
                               "Check /tmp/sawyer_controller_stderr.txt")
        time.sleep(1.0)
        try:
            state = controller.get_state()
            actual_q = state.get("ActualQ", None)
            if actual_q is not None and len(actual_q) == 7:
                actual_list = [round(float(v), 3) for v in actual_q]
                home_err = max(abs(float(a) - float(h))
                               for a, h in zip(actual_q, HOME_JOINTS_LIST))
                print(f"  homed joints: {actual_list}  (max|q - q_home|={home_err:.4f} rad)")
        except Exception as e:
            print(f"  WARN: could not read current joints ({e}); proceeding")
        print("Sawyer controller ready (homed).")
        print("  pausing 2s before starting policy (Ctrl-C to abort)...")
        time.sleep(2.0)

    # ── ROS subscribers ────────────────────────────────────────────────────
    import rospy
    from sensor_msgs.msg import Image, JointState
    from intera_core_msgs.msg import EndpointState
    rospy.init_node("deploy_bbox_seg4ch", anonymous=True, disable_signals=True)
    pred_path_pub = None
    try:
        from nav_msgs.msg import Path
        from geometry_msgs.msg import PoseStamped
        pred_path_pub = rospy.Publisher("/diffusion_policy/predicted_trajectory",
                                       Path, queue_size=1)
    except Exception:
        pass

    rgb_buf:   Optional[np.ndarray] = None
    depth_buf: Optional[np.ndarray] = None
    latest_eef: Optional[np.ndarray] = None
    img_hist  = deque(maxlen=n_obs_steps)
    eef_hist  = deque(maxlen=n_obs_steps)
    path_trace = []
    eef_trace = []
    scheduled_chunks = []
    scheduled_waypoint_trace = []
    target_state_trace = []
    target_depth_trace = []
    depth_ready_entries = set()
    clean_preview_writer = None
    clean_preview_video_path = args.record_clean_preview_video
    if clean_preview_video_path:
        os.makedirs(os.path.dirname(os.path.abspath(clean_preview_video_path)) or ".", exist_ok=True)
        print(f"[video] clean preview will be saved to {clean_preview_video_path}")
    _PATH_TRACE = path_trace
    _SCHEDULED_CHUNKS = scheduled_chunks
    _SCHEDULED_WAYPOINT_TRACE = scheduled_waypoint_trace
    _TARGET_STATE_TRACE = target_state_trace
    target_state_depth_mm: Optional[float] = None
    target_state_recent = deque(maxlen=max(1, int(args.bottleneck_target_state_plateau_steps)))

    def on_rgb(msg: Image):
        nonlocal rgb_buf
        if msg.encoding in ("rgb8", "bgr8"):
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            rgb_buf = arr if msg.encoding == "rgb8" else arr[:, :, ::-1].copy()
        else:
            print(f"[rgb] unsupported encoding {msg.encoding}")

    def on_depth(msg: Image):
        nonlocal depth_buf
        depth_buf = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)

    def on_endpoint(msg: EndpointState):
        nonlocal latest_eef
        p, q = msg.pose.position, msg.pose.orientation
        latest_eef = np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float32)
        path_trace.append([time.time(), latest_eef[0], latest_eef[1], latest_eef[2]])
        eef_trace.append([time.time(), *[float(v) for v in latest_eef]])

    def wait_for_stable_image():
        if not args.stop_and_go:
            return
        sharp_count = 0
        deadline = time.time() + max(0.0, args.stable_timeout)
        last_score = 0.0
        while not rospy.is_shutdown() and time.time() < deadline:
            if rgb_buf is not None:
                last_score = image_sharpness(rgb_buf)
                if last_score >= args.stable_blur_threshold:
                    sharp_count += 1
                    if sharp_count >= max(1, args.stable_frames):
                        print(f"[stop-go] image stable: sharpness={last_score:.1f}")
                        return
                else:
                    sharp_count = 0
            rate.sleep()
        print(f"[stop-go] continuing after stable wait timeout; sharpness={last_score:.1f}")

    # ── Verify depth-to-RGB alignment BEFORE subscribing ─────────────────
    print(f"[ros] ROS_MASTER_URI={os.environ.get('ROS_MASTER_URI')}")
    print(f"[ros] ROS_HOSTNAME={os.environ.get('ROS_HOSTNAME')}")
    check_depth_alignment(args.image_topic, args.depth_topic)

    rospy.Subscriber(args.image_topic, Image, on_rgb,      queue_size=1)
    rospy.Subscriber(args.depth_topic, Image, on_depth,    queue_size=1)
    rospy.Subscriber("/robot/limb/right/endpoint_state", EndpointState, on_endpoint, queue_size=1)

    rate = rospy.Rate(args.frequency)
    dt = 1.0 / args.frequency
    action_time_scale = max(1.0, float(args.action_time_scale))
    action_dt = dt * action_time_scale

    # ── Warm up: wait for obs ──────────────────────────────────────────────
    print("Waiting for camera + endpoint state...")
    last_wait_print = 0.0
    while not rospy.is_shutdown():
        if rgb_buf is not None and depth_buf is not None and latest_eef is not None:
            break
        now = time.time()
        if now - last_wait_print > 2.0:
            missing = []
            if rgb_buf is None:
                missing.append(args.image_topic)
            if depth_buf is None:
                missing.append(args.depth_topic)
            if latest_eef is None:
                missing.append("/robot/limb/right/endpoint_state")
            print(f"[wait] still waiting for: {', '.join(missing)}")
            last_wait_print = now
        rate.sleep()

    # Pre-fill history with current frame (so first inference has 2 steps)
    while len(img_hist) < n_obs_steps:
        if len(img_hist) == 0:
            print("[bbox-seg] preparing first observation; bbox selector should appear now.")
        seg_idx = segmenter.segment(rgb_buf, depth_buf)
        img_hist.append(encode_4ch_cam_high(seg_idx, depth_buf))
        eef_hist.append(latest_eef.copy())
        rate.sleep()

    print("Ready — running policy.")
    run_logger.mark_start()

    # ── Debug image dir setup ──────────────────────────────────────────────
    if args.save_debug_dir:
        import os as _os
        _os.makedirs(args.save_debug_dir, exist_ok=True)
        print(f"Debug images -> {args.save_debug_dir}/  (every {args.debug_every} step(s))")
    command_logger = PolicyCommandLogger(args.log_policy_commands_dir)
    _ACTIVE_COMMAND_LOGGER = command_logger
    if args.log_policy_commands_dir:
        print(f"Policy command logs -> {args.log_policy_commands_dir}/")

    # ── Receding-horizon control loop ──────────────────────────────────────
    overlap_steps = max(0, min(int(args.overlap_steps), n_action_steps - 1))
    exec_steps = max(1, n_action_steps - overlap_steps)
    action_start_offset = max(0, min(int(args.action_start_offset), n_action_steps - 1))
    next_chunk_time = time.time() + action_dt
    frame_idx = 0
    prev_action_chunk = None
    action_smoothing = float(np.clip(args.action_smoothing, 0.0, 1.0))
    phase = "ALIGNMENT"
    align_start_time = time.time()
    interaction_start_time = None
    stable_bottleneck_count = 0
    interaction_executed = False
    print(f"Timing: policy_dt={dt:.3f}s action_dt={action_dt:.3f}s "
          f"action_time_scale={action_time_scale:.2f} exec_steps={exec_steps} "
          f"action_smoothing={action_smoothing:.2f} "
          f"action_start_offset={action_start_offset}")
    if args.manual_bottleneck_switch or args.auto_bottleneck_switch:
        print("[phase] ALIGNMENT active. "
              "Press SPACE in the preview to switch manually; press q to stop.")

    def record_clean_preview_frame(rgb_frame, seg_frame):
        nonlocal clean_preview_writer
        if not clean_preview_video_path or rgb_frame is None or seg_frame is None:
            return
        try:
            clean_frame = render_clean_preview_frame(
                rgb_frame, seg_frame, getattr(segmenter, "_box", None),
                panel=args.clean_preview_panel,
                draw_bbox=not args.clean_preview_no_bbox,
            )
            if clean_preview_writer is None:
                vh, vw = clean_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                clean_preview_writer = cv2.VideoWriter(
                    clean_preview_video_path,
                    fourcc,
                    max(1.0, float(args.clean_preview_fps)),
                    (vw, vh),
                )
                if not clean_preview_writer.isOpened():
                    print(f"[video] WARN: failed to open {clean_preview_video_path}")
                    clean_preview_writer = None
            if clean_preview_writer is not None:
                clean_preview_writer.write(clean_frame)
        except Exception as e:
            print(f"[video] failed to record clean preview frame: {e}")

    def record_clean_preview_for_duration(until_time: float):
        if not clean_preview_video_path:
            now = time.time()
            if until_time > now:
                time.sleep(until_time - now)
            return
        frame_dt = 1.0 / max(1.0, float(args.clean_preview_fps))
        next_frame_t = 0.0
        while time.time() < until_time and not rospy.is_shutdown():
            now = time.time()
            if now >= next_frame_t and rgb_buf is not None and depth_buf is not None:
                try:
                    seg_frame = segmenter.segment(rgb_buf, depth_buf)
                    record_clean_preview_frame(rgb_buf, seg_frame)
                except Exception as e:
                    print(f"[video] failed to segment interaction preview frame: {e}")
                next_frame_t = now + frame_dt
            time.sleep(min(0.01, max(0.0, until_time - time.time())))

    def execute_interaction_replay(anchor_pose: np.ndarray, reason: str,
                                   switch_distance=None,
                                   selected_entry_override=None) -> bool:
        nonlocal interaction_executed, next_chunk_time, phase, interaction_start_time
        replay_to_execute = interaction_replay
        selected_entry = None
        selected_index = None
        if selected_entry_override is not None:
            selected_entry = selected_entry_override
            replay_to_execute = selected_entry["replay"]
            try:
                selected_index = interaction_library.index(selected_entry)
            except Exception:
                selected_index = None
            meta = selected_entry["metadata"]
            print(f"[library] selected from switch match {selected_entry['path']}")
            print(f"[library] metadata={meta}")
        elif interaction_library:
            replay_to_execute, selected_entry = select_interaction_from_library(
                interaction_library,
                args.interaction_key,
                getattr(segmenter, "_box", None),
                depth_buf,
                anchor_pose,
                depth_center_frac=args.target_depth_center_frac,
                depth_percentile=args.target_depth_percentile,
            )
            if selected_entry is not None:
                meta = selected_entry["metadata"]
                try:
                    selected_index = interaction_library.index(selected_entry)
                except ValueError:
                    selected_index = None
                print(f"[library] selected {selected_entry['path']}")
                print(f"[library] metadata={meta}")
        if replay_to_execute is None:
            print(f"[phase] bottleneck reached by {reason}, but no interaction replay was provided. Holding.")
            phase = "HOLD"
            return False
        phase = "INTERACTION_REPLAY"
        interaction_executed = True
        interaction_start_time = time.time()
        run_logger.mark_switch(
            reason,
            distance=switch_distance,
            selected_entry=selected_entry,
            selected_index=selected_index,
            align_start_time=align_start_time,
        )
        replay_poses = compose_interaction_replay(replay_to_execute, anchor_pose)
        replay_dt = float(replay_to_execute["dt"]) * max(1.0, float(args.interaction_time_scale))
        if args.replay_dt is not None:
            replay_dt = float(args.replay_dt) * max(1.0, float(args.interaction_time_scale))
        chunk_start = max(time.time() + replay_dt, next_chunk_time)
        scheduled_chunks.append(replay_poses[:, :3].copy())
        record_scheduled_waypoints(scheduled_waypoint_trace, replay_poses,
                                   chunk_start, replay_dt, phase_id=1)
        print(f"[phase] SWITCH -> INTERACTION_REPLAY ({reason})")
        print(f"[phase] scheduling {len(replay_poses)} interaction waypoints at dt={replay_dt:.3f}s")
        if args.print_policy_commands:
            print(format_action_chunk("interaction_replay", replay_poses, max_rows=16))
        schedule_pose_chunk(controller, replay_poses, chunk_start, replay_dt)
        sleep_until = chunk_start + max(0, len(replay_poses) - 1) * replay_dt + args.settle_time
        record_clean_preview_for_duration(sleep_until)
        phase = "DONE"
        print("[phase] interaction replay complete.")
        run_logger.mark_end(interaction_start_time=interaction_start_time)
        return True

    try:
        while not rospy.is_shutdown():
            if (
                phase == "ALIGNMENT"
                and args.alignment_timeout > 0
                and (time.time() - align_start_time) >= args.alignment_timeout
            ):
                elapsed = time.time() - align_start_time
                print(
                    f"[phase] ALIGNMENT TIMEOUT: no bottleneck switch after "
                    f"{elapsed:.1f}s >= {args.alignment_timeout:.1f}s"
                )
                run_logger.mark_alignment_timeout(
                    align_start_time,
                    args.alignment_timeout,
                )
                break
            wait_for_stable_image()
            t_loop = time.time()
            scheduled_exec_steps = exec_steps

            # 1) segment, build cam_high
            seg_idx = segmenter.segment(rgb_buf, depth_buf)
            seg_timing = getattr(segmenter, "last_timing", {})
            t_encode = time.time()
            cam4 = encode_4ch_cam_high(seg_idx, depth_buf)        # (4,240,320) float
            encode_ms = (time.time() - t_encode) * 1000.0
            img_hist.append(cam4)
            inference_eef = latest_eef.copy()
            eef_hist.append(inference_eef)

            t_policy_prep = time.time()
            cam = np.stack(list(img_hist), axis=0)[None]          # (1, T_obs, 4, 240, 320)
            eef = np.stack(list(eef_hist), axis=0)[None]          # (1, T_obs, 7)
            obs_t = {
                "cam_high": torch.from_numpy(cam).to(dev),
                "eef_pose": torch.from_numpy(eef).to(dev),
            }
            prep_ms = (time.time() - t_policy_prep) * 1000.0
            t_policy = time.time()
            with torch.no_grad():
                out = policy.predict_action(obs_t)
                action_raw = out["action"][0].detach().cpu().numpy()  # (8, 7)
            action_smoothed = smooth_action_chunk(action_raw, prev_action_chunk, action_smoothing)
            prev_action_chunk = action_smoothed.copy()
            action_plan = action_smoothed[action_start_offset:]
            source_indices = np.arange(action_start_offset, n_action_steps, dtype=np.int32)
            dropped_inference_pose = 0
            if args.drop_inference_pose_radius > 0:
                action_plan, dropped_inference_pose = drop_near_pose_prefix(
                    action_plan, inference_eef, args.drop_inference_pose_radius)
                source_indices = source_indices[dropped_inference_pose:]
            dropped_backward = 0
            if not args.allow_backward_prefix:
                action_plan, dropped_backward = drop_backward_prefix(action_plan, latest_eef)
                source_indices = source_indices[dropped_backward:]
            policy_ms = (time.time() - t_policy) * 1000.0
            loop_compute_ms = (time.time() - t_loop) * 1000.0

            # ── Velocity limiting ─────────────────────────────────────────
            # (a) chunk-wide sanity: if total translation is implausibly large,
            #     reject this chunk and hold (interpolator will coast on the
            #     previous chunk's last waypoint until the next inference).
            steps_disp = np.linalg.norm(np.diff(action_plan[:, :3], axis=0), axis=1)
            first_step = float(np.linalg.norm(action_plan[0, :3] - latest_eef[:3]))
            chunk_total = float(steps_disp.sum()) + first_step
            if chunk_total > args.max_chunk_disp:
                print(f"[velocity] REJECTED chunk: total disp {chunk_total*1000:.0f}mm "
                      f"> max_chunk_disp {args.max_chunk_disp*1000:.0f}mm — holding")
                # advance the schedule clock so we replan again on the next tick
                next_chunk_time = time.time() + action_dt * exec_steps
                rate.sleep()
                continue

            # (b) per-waypoint chain clip: walk the chunk, clamping each step
            #     to (max_step_disp, max_step_rot) relative to the previous one.
            action, n_pclip, n_rclip = clip_waypoint_chain(
                action_plan, latest_eef,
                max_step_disp=args.max_step_disp,
                max_step_rot=args.max_step_rot,
            )
            if args.stop_and_go:
                action = action[:max(1, min(int(args.stop_and_go_steps), action.shape[0]))]
            scheduled_exec_steps = min(
                exec_steps,
                max(1, action.shape[0] - min(overlap_steps, action.shape[0] - 1)),
            )
            if args.stop_and_go:
                scheduled_exec_steps = action.shape[0]

            target_px = int((seg_idx == 2).sum())
            obstacle_px = int((seg_idx == 1).sum())
            chunk_disp = float(np.linalg.norm(action[-1, :3] - action[0, :3]))
            action_activity = action_position_activity(action, latest_eef)
            live_target_feat = target_box_features(
                getattr(segmenter, "_box", None),
                depth_buf,
                depth_center_frac=args.target_depth_center_frac,
                depth_percentile=args.target_depth_percentile,
            )
            depth_filter_msg = ""
            raw_target_depth = None
            depth_error_ignored = False
            if "target_depth_mm" in live_target_feat:
                raw_target_depth = float(live_target_feat["target_depth_mm"])
                if (
                    target_state_depth_mm is not None
                    and args.target_state_max_depth_jump_mm > 0
                    and abs(raw_target_depth - target_state_depth_mm) > args.target_state_max_depth_jump_mm
                ):
                    live_target_feat["ignore_target_depth_error"] = True
                    depth_error_ignored = True
                    depth_filter_msg = (
                        f"depth_zero raw_d={raw_target_depth:.0f} "
                    )
                else:
                    alpha = float(np.clip(args.target_state_depth_ema, 0.0, 1.0))
                    if target_state_depth_mm is None or alpha >= 1.0:
                        target_state_depth_mm = raw_target_depth
                    else:
                        target_state_depth_mm = (
                            alpha * raw_target_depth
                            + (1.0 - alpha) * target_state_depth_mm
                        )
                        live_target_feat["target_depth_mm"] = float(target_state_depth_mm)
                        live_target_feat["target_state_uvd"] = [
                            live_target_feat["target_u"],
                            live_target_feat["target_v"],
                            live_target_feat["target_depth_mm"],
                        ]
                        depth_filter_msg = (
                            f"depth_ema raw_d={raw_target_depth:.0f}->"
                            f"{target_state_depth_mm:.0f} "
                        )
            if "target_depth_mm" in live_target_feat:
                target_depth_trace.append([
                    time.time(),
                    float(live_target_feat["target_depth_mm"]),
                    float(raw_target_depth) if raw_target_depth is not None else np.nan,
                    1.0 if depth_error_ignored else 0.0,
                ])
            nn_target = nearest_target_state_error(
                interaction_library,
                args.interaction_key,
                live_target_feat,
                image_width=args.target_state_norm_width,
                image_height=args.target_state_norm_height,
                depth_scale_mm=args.target_state_norm_depth_mm,
                weights=(
                    args.target_state_weight_u,
                    args.target_state_weight_v,
                    args.target_state_weight_d,
                ),
                depth_gate_thresh=args.bottleneck_target_depth_thresh,
                depth_ready_entries=depth_ready_entries,
            )
            preview_key = -1
            if args.show_inference_view:
                try:
                    view = render_inference_view(
                        rgb_buf, seg_idx, getattr(segmenter, "_box", None),
                        frame_idx, target_px, obstacle_px,
                        target_depth_mm=live_target_feat.get("target_depth_mm"),
                        raw_target_depth_mm=raw_target_depth)
                    cv2.putText(view, f"phase={phase}  SPACE=switch  q=quit",
                                (8, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                                (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow("bbox inference preview", view)
                    preview_key = cv2.waitKey(1) & 0xFF
                    if preview_key == ord("q"):
                        print("[preview] q pressed; stopping deployment")
                        break
                except Exception as e:
                    print(f"[preview] failed to render inference view: {e}")
            if clean_preview_video_path:
                record_clean_preview_frame(rgb_buf, seg_idx)
            command_logger.log(
                frame_idx=frame_idx,
                raw=action_raw,
                smoothed=action_smoothed,
                plan=action_plan,
                scheduled=action,
                source_indices=source_indices,
                action_dt=action_dt,
                target_px=target_px,
                obstacle_px=obstacle_px,
                chunk_disp_mm=chunk_disp * 1000.0,
            )
            if args.print_policy_commands:
                print(f"[policy commands] frame={frame_idx} "
                      f"offset={action_start_offset}+inferdrop{dropped_inference_pose}+backdrop{dropped_backward} "
                      f"source={source_indices.tolist()}")
                print(format_action_chunk("raw", action_raw))
                print(format_action_chunk("smoothed", action_smoothed))
                print(format_action_chunk("scheduled", action))
            target_state_msg = (
                f"target_uvd={live_target_feat.get('target_u', np.nan):.1f},"
                f"{live_target_feat.get('target_v', np.nan):.1f},"
                f"{live_target_feat.get('target_depth_mm', np.nan):.0f} "
                f"depth_px={live_target_feat.get('target_depth_valid_px', 0)} "
            )
            if nn_target is not None:
                target_state_msg += (
                    f"uv_sum={nn_target['uv_sum']:.3f} "
                    f"depth_norm={nn_target['depth_dist']:.3f} "
                )
                if not nn_target.get("valid_for_switch", True):
                    target_state_msg += "depth_invalid_uv_allowed "
                elif not nn_target.get("depth_ready", False):
                    target_state_msg += "depth_not_ready "
                elif not nn_target.get("depth_gate_pass", False):
                    target_state_msg += "depth_ready_latched "
            target_state_msg += depth_filter_msg
            print(f"[policy] total={loop_compute_ms:5.1f}ms  "
                  f"seg={seg_timing.get('total_ms', 0.0):5.1f}ms "
                  f"(yolo={seg_timing.get('yolo_ms', 0.0):5.1f}"
                  f"{'*' if seg_timing.get('ran_yolo', False) else ' '} "
                  f"sam={seg_timing.get('sam_ms', 0.0):5.1f}"
                  f"{'*' if seg_timing.get('ran_sam', False) else ' '} "
                  f"trk={seg_timing.get('tracker_ms', 0.0):4.1f}) "
                  f"enc={encode_ms:4.1f} prep={prep_ms:4.1f} policy={policy_ms:5.1f}  "
                  f"target_px={target_px:6d} "
                  f"obstacle_px={obstacle_px:6d}  "
                  f"chunk_disp={chunk_disp*1000:5.1f}mm  "
                  f"activity={action_activity*1000:4.1f}mm  "
                  f"{target_state_msg}"
                  f"offset={action_start_offset}+inferdrop{dropped_inference_pose}+backdrop{dropped_backward} "
                  f"exec={scheduled_exec_steps}/{action.shape[0]} "
                  f"clipped: pos={n_pclip}/{len(action)} rot={n_rclip}/{len(action)}")

            # Debug dump: render the exact policy input next to RGB/seg/depth
            if args.save_debug_dir and (frame_idx % args.debug_every == 0):
                try:
                    info = {
                        "frame":         frame_idx,
                        "inf_ms":        loop_compute_ms,
                        "chunk_disp_mm": chunk_disp * 1000.0,
                        "n_pclip":       n_pclip,
                        "n_rclip":       n_rclip,
                        "n_act":         len(action),
                    }
                    debug_img = render_debug_image(rgb_buf, depth_buf, seg_idx, cam4, info)
                    out_path = f"{args.save_debug_dir}/f{frame_idx:05d}.png"
                    cv2.imwrite(out_path, debug_img)
                except Exception as e:
                    print(f"[debug] failed to render frame {frame_idx}: {e}")
            frame_idx += 1

            switch_reason = None
            if args.manual_bottleneck_switch and preview_key in (32, 13):
                switch_reason = "manual SPACE"
            if args.auto_bottleneck_switch:
                uv_sum = None if nn_target is None else float(nn_target["uv_sum"])
                depth_dist = None if nn_target is None else float(nn_target["depth_dist"])
                use_target_state_switch = (
                    uv_sum is not None
                    and depth_dist is not None
                    and args.bottleneck_target_uv_thresh > 0
                )
                if use_target_state_switch:
                    depth_ok = (
                        args.bottleneck_target_depth_thresh <= 0
                        or nn_target.get("depth_ready", False)
                        or not nn_target.get("valid_for_switch", True)
                    )
                    if not depth_ok:
                        print(
                            f"[phase] waiting: no replay has entered depth range yet "
                            f"(best depth_norm={depth_dist:.3f}, "
                            f"depth_thresh={args.bottleneck_target_depth_thresh:.3f})"
                        )
                    elif uv_sum <= args.bottleneck_target_uv_thresh:
                        depth_status = (
                            "invalid_depth_allowed"
                            if not nn_target.get("valid_for_switch", True)
                            else f"depth_ready={nn_target.get('depth_ready', False)}"
                        )
                        switch_reason = (
                            f"auto depth-or-invalid+uv: {depth_status} "
                            f"current_depth_norm={depth_dist:.3f}, "
                            f"uv_sum={uv_sum:.3f} <= {args.bottleneck_target_uv_thresh:.3f}"
                        )
                    else:
                        print(
                            f"[phase] waiting: depth already entered once for this replay, but "
                            f"uv_sum={uv_sum:.3f} > uv_thresh={args.bottleneck_target_uv_thresh:.3f}"
                        )
                else:
                    print("[phase] waiting: no nearest-replay target-state match")

            if nn_target is not None:
                live = nn_target["live"]
                ref = nn_target["ref"]
                delta_uvd = nn_target["delta"]
                norm_delta_uvd = nn_target["norm_delta"]
                target_state_trace.append([
                    time.time(),
                    float(live[0]), float(live[1]), float(live[2]),
                    float(ref[0]), float(ref[1]), float(ref[2]),
                    float(delta_uvd[0]), float(delta_uvd[1]), float(delta_uvd[2]),
                    float(nn_target["dist"]),
                    1.0 if switch_reason is not None else 0.0,
                    nn_target["entry"]["path"],
                    float(norm_delta_uvd[0]), float(norm_delta_uvd[1]), float(norm_delta_uvd[2]),
                    1.0 if nn_target.get("valid_for_switch", True) else 0.0,
                ])
                if args.show_target_state_error:
                    try:
                        live_err_view = render_live_target_state_error(
                            target_state_trace,
                            strict_thresh=args.bottleneck_target_uv_thresh,
                            plateau_thresh=None,
                        )
                        cv2.imshow("live target-state error", live_err_view)
                        depth_view = render_live_target_depth(target_depth_trace)
                        cv2.imshow("live target depth", depth_view)
                        uv_view = render_live_target_uv(target_state_trace)
                        cv2.imshow("live target u/v", uv_view)
                        key_err = cv2.waitKey(1) & 0xFF
                        if key_err == ord("q"):
                            print("[target-state] q pressed; stopping deployment")
                            break
                    except Exception as e:
                        print(f"[target-state] failed to render live error: {e}")

            if switch_reason is not None:
                switch_distance = None
                selected_entry_override = None
                if nn_target is not None:
                    switch_distance = float(nn_target["dist"])
                    selected_entry_override = nn_target["entry"]
                execute_interaction_replay(
                    latest_eef.copy(),
                    switch_reason,
                    switch_distance=switch_distance,
                    selected_entry_override=selected_entry_override,
                )
                break

            # 2) schedule the clipped future waypoints
            chunk_start = max(next_chunk_time, time.time() + action_dt)
            scheduled_chunks.append(action[:, :3].copy())
            record_scheduled_waypoints(scheduled_waypoint_trace, action,
                                       chunk_start, action_dt, phase_id=0)
            if controller is not None:
                schedule_pose_chunk(controller, action, chunk_start, action_dt)

            # publish RViz path (optional)
            if pred_path_pub is not None and pred_path_pub.get_num_connections() > 0:
                from nav_msgs.msg import Path
                from geometry_msgs.msg import PoseStamped
                path = Path()
                path.header.stamp = rospy.Time.now()
                path.header.frame_id = "base"
                for i in range(action.shape[0]):
                    ps = PoseStamped()
                    ps.header.stamp = rospy.Time.from_sec(chunk_start + i * action_dt)
                    ps.header.frame_id = "base"
                    ps.pose.position.x = float(action[i, 0])
                    ps.pose.position.y = float(action[i, 1])
                    ps.pose.position.z = float(action[i, 2])
                    ps.pose.orientation.x = float(action[i, 3])
                    ps.pose.orientation.y = float(action[i, 4])
                    ps.pose.orientation.z = float(action[i, 5])
                    ps.pose.orientation.w = float(action[i, 6])
                    path.poses.append(ps)
                pred_path_pub.publish(path)

            # 3) sleep until just before replan time
            next_chunk_time = chunk_start + scheduled_exec_steps * action_dt
            if args.stop_and_go:
                sleep_until = chunk_start + max(0, action.shape[0] - 1) * action_dt + args.settle_time
            else:
                sleep_until = next_chunk_time - min(dt, action_dt)
            now = time.time()
            if sleep_until > now:
                time.sleep(sleep_until - now)
            if args.stop_and_go:
                next_chunk_time = time.time() + action_dt

    except KeyboardInterrupt:
        print("\n[shutdown] Ctrl-C received; stopping controller and child processes...")
    finally:
        try:
            if clean_preview_writer is not None:
                clean_preview_writer.release()
                print(f"[video] saved clean preview to {clean_preview_video_path}")
        except Exception as e:
            print(f"[video] failed to close clean preview video: {e}")
        images_to_show = []
        try:
            path_img = render_path_summary(path_trace, scheduled_chunks, args.path_summary_path)
            if path_img is not None:
                print(f"[path] saved summary to {args.path_summary_path}")
                images_to_show.append(("deployment path summary", path_img))
        except Exception as e:
            print(f"[path] failed to render final path summary: {e}")
        try:
            err_img = render_distance_error(
                path_trace,
                scheduled_waypoint_trace,
                args.distance_error_path,
                csv_path=args.distance_error_csv,
            )
            if err_img is not None:
                print(f"[error] saved distance error to {args.distance_error_path}")
                if args.distance_error_csv:
                    print(f"[error] saved distance error csv to {args.distance_error_csv}")
                images_to_show.append(("deployment distance error", err_img))
        except Exception as e:
            print(f"[error] failed to render distance error: {e}")
        try:
            target_err_img = render_target_state_error(
                target_state_trace,
                args.target_state_error_path,
                csv_path=args.target_state_error_csv,
                strict_thresh=args.bottleneck_target_uv_thresh,
                plateau_thresh=None,
            )
            if target_err_img is not None:
                print(f"[target-state] saved error to {args.target_state_error_path}")
                if args.target_state_error_csv:
                    print(f"[target-state] saved error csv to {args.target_state_error_csv}")
                images_to_show.append(("target-state nearest replay error", target_err_img))
        except Exception as e:
            print(f"[target-state] failed to render nearest replay error: {e}")
        if args.show_path_summary and images_to_show:
            for name, img in images_to_show:
                cv2.imshow(name, img)
            print("[plots] showing final plots; press any key in an image window to close.")
            cv2.waitKey(0)
        try:
            if run_logger.enabled:
                if "end_time" not in run_logger.record:
                    run_logger.mark_end(interaction_start_time=interaction_start_time)
                trajectory_file = write_eef_trajectory_csv(
                    run_logger.trajectory_path(),
                    eef_trace,
                ) or ""
                action_file = ""
                if args.log_policy_commands_dir:
                    action_file = os.path.join(
                        os.path.abspath(args.log_policy_commands_dir),
                        "policy_command_summary.csv",
                    )
                run_logger.attach_paths(
                    trajectory_file=trajectory_file,
                    action_file=action_file,
                    path_summary_path=args.path_summary_path,
                    distance_error_path=args.distance_error_path,
                    distance_error_csv=args.distance_error_csv,
                    target_state_error_path=args.target_state_error_path,
                    target_state_error_csv=args.target_state_error_csv,
                )
                run_logger.append_summary()
        except Exception as e:
            print(f"[experiment] failed to save deployment run record: {e}")
        cleanup_processes(controller=controller, shm=shm, command_logger=command_logger)
        _ACTIVE_CONTROLLER = None
        _ACTIVE_SHM = None
        _ACTIVE_COMMAND_LOGGER = None


if __name__ == "__main__":
    main()
