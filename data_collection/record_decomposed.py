"""Two-phase recorder: alignment + interaction in one episode.

Inspired by:
  * arxiv 2511.10110 — decomposing manipulation into a coarse "alignment"
    phase (only the final pose matters) and a precise "interaction" phase
    (the trajectory matters).
  * the project data-collection format — the same `bottleneck_pose.npy`
    convention that marks the start of the interaction trajectory.

State machine
─────────────
    IDLE
    ──SPACE──>  ALIGNMENT       (recording, frames tagged phase=0)
    ──'B'────>  TRANSITION      (PAUSED — robot sits at alignment target;
                                  operator settles, gripper preps)
    ──'B'────>  INTERACTION     (recording, frames tagged phase=1)
    ──SPACE──>  SAVE → home → IDLE

Recordings are saved under  final_data_decomp/demo_decomp_NNN/  with the
*identical* layout as record_full_data.py, plus three extra phase files:

    head_camera_rgb.npy             (T, H, W, 3)   uint8        ← inherited
    head_camera_depth.npy           (T, H, W)      uint16       ← inherited
    head_camera_seg.npy             (T, H, W)      uint8        ← inherited
    head_camera_cam4.npy            (T, 4, 240, 320) float16    ← inherited
    head_camera_rgb_intrinsic_matrix.npy (3, 3)    float64      ← inherited
    head_camera_ws_rgb.png                                       ← inherited
    robot_states.npy                (T, 15)        float64      ← inherited

    phase_index.npy                 (T,)           uint8        NEW
                                       0 = alignment, 1 = interaction
    alignment_end_idx.npy           ()             int32        NEW
                                       index of LAST alignment frame
    bottleneck_pose.npy             (7,)           float64      OVERRIDDEN
                                       EEF pose at alignment-end (the
                                       canonical alignment target — see
                                       thousand_tasks framework)
    bottleneck_posevec.npy          (7,)           float64      OVERRIDDEN
                                       duplicate of bottleneck_pose

Because DecomposedRecorder subclasses FullStateRecorder, every feature of
record_full_data.py carries over unchanged:
  * YOLO-World target box + depth-gated obstacle pixels for 4ch masks
  * CSRT bbox tracker that follows the target across frames
  * Per-frame depth gating with adaptive `target_depth + buffer`
  * Persisted home bbox (`.home_bbox.json`) reloaded on launch
  * Post-hoc segmentation pass at save time → seg + cam4 ready for training
  * IR-emitter alternation for clean RGB
  * Live 4-channel preview window with HSV trackbars
  * Auto-home after each take

────────────────────────────────────────────────────────────────────────
DIFFUSION-POLICY TRAINING WITH THE DECOMPOSED DATA
────────────────────────────────────────────────────────────────────────
The per-frame data fields (RGB, depth, seg, cam4, eef_pose, action) are
exactly the same as in record_full_data.py, so the existing
`GroveFinalDataImageDataset` loader works on these episodes with NO code
changes — just point `dataset_path` at `final_data_decomp/` instead of
`final_data/`.

Phase-aware training options (in order of complexity):

  (A) Single policy, both phases combined.
      Treat each episode as one continuous trajectory; let the diffusion
      policy learn the alignment→interaction transition implicitly from
      the data distribution. Easiest. Works because the diffusion model
      naturally captures multi-modal action distributions.

  (B) Two specialised policies (split at the bottleneck).
      Use `alignment_end_idx.npy` to slice each episode into two sub-
      episodes. Train a coarse "alignment" policy on phase=0 frames
      (input: scene observation; output: trajectory toward bottleneck)
      and a precise "interaction" policy on phase=1 frames (input:
      scene observation conditioned on bottleneck_pose; output: fine
      manipulation trajectory). Matches the arxiv 2511.10110
      decomposition exactly.

  (C) Conditional single policy.
      Feed `phase_index` (or `dist_to_bottleneck`) as an extra low-dim
      observation channel. Single network, but it knows which mode it
      is currently in.

Recommended starting point: (A) on `final_data_decomp/`, no loader
changes, no extra code. If accuracy on the interaction phase is
insufficient, move to (B) and train two checkpoints. The extra phase
files are metadata — they don't break the existing single-policy loader.

Usage:
    cd ~/Documents/sawyer_robot_ws && source ./devel/setup.bash && source ./sci01_intera.sh
    conda activate robopol
    cd /path/to/package/data_collection
    python record_decomposed.py
"""

import os
import sys
import time
import argparse

import numpy as np
import cv2
import rospy
from pynput import keyboard

# Re-use everything from the existing single-phase recorder.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import record_full_data as rfd


DECOMPOSED_SAVE_ROOT = os.path.expanduser(
    "./data/final_data_decomp/")


class DecomposedRecorder(rfd.FullStateRecorder):
    """Two-phase variant of FullStateRecorder. Overrides the keyboard state
    machine and tags each saved frame with its phase index."""

    PHASE_IDLE        = "idle"
    PHASE_ALIGNMENT   = "alignment"
    PHASE_TRANSITION  = "transition"
    PHASE_INTERACTION = "interaction"

    def __init__(self, task_name="demo_decomp", save_root=DECOMPOSED_SAVE_ROOT,
                 **kwargs):
        super().__init__(task_name=task_name, **kwargs)
        # Save under a dedicated subfolder so decomposed demos don't get
        # mixed with the existing single-phase alignment-only ones.
        self.save_root = os.path.expanduser(save_root)
        os.makedirs(self.save_root, exist_ok=True)

        self.phase = self.PHASE_IDLE
        self.alignment_end_idx = -1
        # Remember the most recent save folder so save_data overrides can
        # append phase metadata after the parent finishes.
        self._last_save_folder = None
        self._print_phase_banner()

    # ── Helpers ─────────────────────────────────────────────────────────

    def _print_phase_banner(self):
        print("\n" + "=" * 60)
        print(" DECOMPOSED RECORDER — alignment + interaction")
        print("=" * 60)
        print("  SPACE: start ALIGNMENT  /  stop + save after INTERACTION")
        print("  'B'  : mark phase boundary (alignment→pause→interaction)")
        print("  ESC  : quit")
        print("=" * 60)

    def reset_buffers(self):
        super().reset_buffers()
        self.alignment_end_idx = -1
        self.phase = self.PHASE_IDLE

    def get_next_folder_path(self):
        """Override to remember the path we just allocated."""
        folder = super().get_next_folder_path()
        self._last_save_folder = folder
        return folder

    # ── State machine ───────────────────────────────────────────────────

    def on_press(self, key):
        """Custom state machine — does NOT call super().on_press()."""
        if key == keyboard.Key.esc:
            self._stop_display = True
            return False
        if key == keyboard.Key.space:
            self._on_space()
            return
        if hasattr(key, "char") and key.char and key.char.lower() == "b":
            self._on_phase_boundary()
            return

    def _on_space(self):
        if self.phase == self.PHASE_IDLE:
            self._start_alignment()
        elif self.phase == self.PHASE_ALIGNMENT:
            print("[!] SPACE pressed during ALIGNMENT. Press 'B' first to "
                  "mark the bottleneck, then 'B' again to start interaction, "
                  "then SPACE to stop the whole take.")
        elif self.phase == self.PHASE_TRANSITION:
            print("[!] SPACE pressed during TRANSITION pause. Press 'B' to "
                  "begin interaction recording, then SPACE to stop.")
        elif self.phase == self.PHASE_INTERACTION:
            self._stop_and_save()

    def _on_phase_boundary(self):
        if self.phase == self.PHASE_ALIGNMENT:
            # End alignment → enter transition (PAUSE capture, robot stays put)
            self.alignment_end_idx = len(self.rs_rgb) - 1
            self.is_recording = False
            self.phase = self.PHASE_TRANSITION
            print(f"\n■ ALIGNMENT done — captured {self.alignment_end_idx + 1} frames")
            print(f"    Bottleneck = frame {self.alignment_end_idx} "
                  f"(this becomes bottleneck_pose.npy)")
            print(f"    Settle / prep gripper, then press 'B' to start INTERACTION.")
        elif self.phase == self.PHASE_TRANSITION:
            # Resume capture as INTERACTION
            self.is_recording = True
            self.phase = self.PHASE_INTERACTION
            n_align = self.alignment_end_idx + 1
            print(f"\n● INTERACTION phase — recording from frame {n_align}...")
            print(f"    Press SPACE to stop the whole take and save.")
        else:
            print(f"[!] 'B' has no effect in phase '{self.phase}' "
                  f"(only valid during ALIGNMENT or TRANSITION).")

    # ── Start / stop helpers ────────────────────────────────────────────

    def _start_alignment(self):
        # Make sure we have a target bbox before any recording starts.
        if rfd._HOME_BBOX is None:
            if self.latest_rgb_bgr is None:
                print("[bbox] no camera frame yet — wait a moment")
                return
            print("[bbox] no home bbox set — draw one to start recording.")
            self._select_target_bbox()
            if rfd._HOME_BBOX is None:
                print("[bbox] cancelled — recording aborted")
                return

        self.reset_buffers()
        if self.latest_rgb_bgr is not None:
            rfd._TARGET_BBOX = rfd._HOME_BBOX
            rfd.init_tracker_from_bbox(self.latest_rgb_bgr, rfd._HOME_BBOX)
            rfd._TRACKER_FREEZE = 15
            print(f"[bbox] applied home bbox {rfd._HOME_BBOX}")
        self.recording_start_bbox = rfd._TARGET_BBOX
        self.is_recording = True
        self.phase = self.PHASE_ALIGNMENT
        print(f"\n● ALIGNMENT phase — recording...")
        print(f"    Move the arm toward the alignment target pose.")
        print(f"    Press 'B' when at the alignment target to mark the bottleneck.")

    def _stop_and_save(self):
        self.is_recording = False
        n_align = self.alignment_end_idx + 1 if self.alignment_end_idx >= 0 else 0
        n_inter = len(self.rs_rgb) - n_align
        self.phase = self.PHASE_IDLE
        print(f"\n■ STOPPED — alignment={n_align}  interaction={n_inter}  "
              f"total={len(self.rs_rgb)}")
        if n_align == 0:
            print("[!] No alignment frames captured — bottleneck/phase files "
                  "will be skipped.")
        if n_inter <= 0:
            print("[!] No interaction frames captured — did you forget the "
                  "second 'B' press?")
        self.set_emitter(True)
        self.save_data()
        # Move home after save
        self.limb.move_to_joint_positions(
            dict(zip(self.joint_names, self.home_joints)))
        # Restore tracker on home pose so the next take starts clean
        if rfd._HOME_BBOX is not None and self.latest_rgb_bgr is not None:
            time.sleep(0.3)
            rfd._TARGET_BBOX = rfd._HOME_BBOX
            rfd.init_tracker_from_bbox(self.latest_rgb_bgr, rfd._HOME_BBOX)
            print(f"[track] home pose reached — bbox reset; ready for next take")

    # ── Save: parent does the heavy lifting, we add phase metadata ──────

    def save_data(self):
        super().save_data()
        if self._last_save_folder is None or not os.path.isdir(self._last_save_folder):
            print("[phase] no save folder available — skipping phase metadata")
            return
        folder = self._last_save_folder

        T = min(len(self.rs_rgb), len(self.rs_depth), len(self.full_state))
        if T <= 0:
            return

        # phase_index: 0 for [0, alignment_end_idx], 1 for [alignment_end_idx+1, T)
        phase = np.zeros(T, dtype=np.uint8)
        end = self.alignment_end_idx
        if end < 0:
            # Should not happen — operator didn't press 'B'. Mark all as
            # alignment for now and warn.
            print("[phase] WARNING: no bottleneck pressed; all frames marked "
                  "as alignment (phase=0).")
        else:
            end = min(end, T - 1)
            phase[end + 1:] = 1

        np.save(os.path.join(folder, "phase_index.npy"), phase)
        np.save(os.path.join(folder, "alignment_end_idx.npy"),
                np.array(end, dtype=np.int32))

        # Bottleneck pose = EEF pose at the LAST alignment frame.
        # full_state row = [j0..j6, x, y, z, qx, qy, qz, qw, gripper]
        # → EEF pose is slice [7:14].
        if end >= 0:
            state_at_end = np.array(self.full_state[end], dtype=np.float64)
            bottleneck = state_at_end[7:14]
            # Overwrite the bottleneck files the parent wrote (which used
            # the min-Z heuristic). For decomposed demos, the alignment-end
            # is the canonical bottleneck.
            np.save(os.path.join(folder, "bottleneck_pose.npy"), bottleneck)
            np.save(os.path.join(folder, "bottleneck_posevec.npy"), bottleneck)
            print(f"[phase] alignment_end_idx={end}  "
                  f"interaction_frames={T - end - 1}  "
                  f"bottleneck_xyz=({bottleneck[0]:+.3f},{bottleneck[1]:+.3f},"
                  f"{bottleneck[2]:+.3f})")
            print(f"[phase] phase_index.npy {phase.shape} {phase.dtype}  "
                  f"({int((phase==0).sum())} align, {int((phase==1).sum())} inter)")


def main():
    parser = argparse.ArgumentParser(
        description="Two-phase recorder (alignment + interaction).")
    parser.add_argument("--task_name", default="demo_decomp",
                        help="Folder name prefix under the save root.")
    parser.add_argument("--save_root", default=DECOMPOSED_SAVE_ROOT,
                        help="Directory for combined alignment+interaction demos.")
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument("--no_yolo", dest="use_yolo", action="store_false")
    parser.add_argument("--yolo_weights", default=rfd.WORLD_WEIGHTS_DEFAULT)
    parser.add_argument("--sam_weights",  default=None,
                        help="Ignored. Kept for compatibility; SAM is disabled.")
    parser.add_argument("--yolo_device",  default="cuda")
    parser.set_defaults(use_yolo=True)
    args = parser.parse_args()

    rospy.init_node("decomposed_recorder")

    rec = DecomposedRecorder(
        task_name=args.task_name,
        save_root=args.save_root,
        do_annotate=args.annotate,
        use_yolo=args.use_yolo,
        yolo_weights=args.yolo_weights,
        sam_weights=args.sam_weights,
        yolo_device=args.yolo_device,
    )

    listener = keyboard.Listener(on_press=rec.on_press)
    listener.start()
    try:
        rec._display_loop()
    finally:
        rec._stop_display = True
        listener.stop()


if __name__ == "__main__":
    main()
