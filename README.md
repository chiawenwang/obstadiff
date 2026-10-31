# ObstaDiff: Generalizable Diffusion Policy Learning via Obstacle-aware Representations

Jiawen Wang, Kevin Yao, Khalid Jawed

[**Paper**](https://arxiv.org/abs/2609.10918) · [**Project Website**](https://obstadiff.structures.computer/) · [**Video**](PLACEHOLDER_VIDEO_URL)

<!-- ![teaser](media/teaser.gif) -->

Most imitation-learning approaches for manipulation assume clean backgrounds and
lack explicit mechanisms for obstacle-aware motion generation. **ObstaDiff**
extracts a structured target–obstacle–background (TOB) representation, enabling a
downstream diffusion alignment policy to generate end-effector trajectories
toward a target-centered bottleneck pose while reasoning about surrounding
obstacles. At the bottleneck, the policy hands off to a replayed interaction
trajectory. On 366 real-robot greenhouse executions ObstaDiff achieves 75.41%
average task success and 8.20% average obstacle collision rate.

This repository contains the official ObstaDiff implementation. The diffusion
policy code is built on top of
[Diffusion Policy](https://github.com/real-stanford/diffusion_policy).

## 🛠️ Installation

Real-robot scripts target Python 3.9, ROS Noetic, the Intera SDK for Sawyer, and
an aligned RealSense RGB-D stream.

```bash
conda create -n obstadiff python=3.9 -y
conda activate obstadiff

# Install the tested PyTorch build first (pick the CUDA tag matching your driver):
pip install torch==1.12.1 torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

pip install -r requirements.txt

# ultralytics pulls in plain opencv-python, which shadows the contrib build and
# lacks the CSRT tracker used by the recorders. Keep only the contrib build:
pip uninstall -y opencv-python
pip install --force-reinstall --no-deps opencv-contrib-python==4.6.0.66
python -c "import cv2; assert hasattr(cv2, 'TrackerCSRT_create'); print('cv2 ok')"
```

Run the scripts from the repository root so the `diffusion_policy` package is
importable.

For real-robot recording/deployment, source the robot workspace in the same shell:

```bash
source /path/to/ros/noetic/setup.bash
source /path/to/intera/workspace/devel/setup.bash
```

> **Note:** the deployment scripts prepend a Sawyer/Intera workspace path to
> `sys.path` (`"/path/to/sawyer_robot_ws/devel/lib/python3/dist-packages"` in
> `deploy_bbox_seg4ch.py` and `diffusion_policy/real_world/sawyer_interpolation_controller.py`).
> Edit that to point at your own Intera workspace. The scripts ignore the
> shell's `ROS_MASTER_URI`/`ROS_HOSTNAME` (sourcing ROS sets them to localhost)
> and default to `http://robot.local:11311` and `workstation.local`; if your
> hostnames differ, set `OBSTADIFF_ROS_MASTER_URI` and `OBSTADIFF_ROS_HOSTNAME`.

## 📁 Example Data

We provide a small example bundle (one decomposed demonstration dataset, a
prebuilt interaction replay library, and the detector weights) so the training
and replay-library steps can be run end to end without collecting new data.

👉 **[Download example data (Google Drive)](PLACEHOLDER_GDRIVE_URL)**

```bash
mkdir -p data
# Download obstadiff_example_data.zip from the link above into ./data
# (or: pip install gdown && gdown --id PLACEHOLDER_GDRIVE_FILE_ID -O data/obstadiff_example_data.zip)
cd data && unzip obstadiff_example_data.zip && rm obstadiff_example_data.zip && cd ..
```

Expected layout after extraction:

```text
data/
  final_data_decomp/          # decomposed demonstrations (alignment + interaction)
  interaction_library/        # prebuilt bottleneck replay library
  yolo_weights.pt             # target/obstacle detector
```

Full datasets and trained checkpoints are not included; they are
produced locally and passed to the scripts by path.

## 🤖 Usage

### 1. Record demonstrations (optional)

Skip this if you use the example data. The decomposed recorder captures one full
episode and stores the alignment/interaction split.

```bash
python data_collection/record_decomposed.py \
  --task_name demo_decomp \
  --save_root data/final_data_decomp \
  --yolo_weights data/yolo_weights.pt \
  --yolo_device cuda
```

Keyboard controls: `SPACE` start alignment · `B` mark bottleneck / start
interaction · `SPACE` stop & save · `ESC` quit.

Each episode is saved with per-step arrays plus `phase_index.npy`
(`0` = alignment, `1` = interaction), `alignment_end_idx.npy`, and the recorded
`bottleneck_pose`. Check a dataset with:

```bash
python scripts/check_dataset_layout.py --dataset data/final_data_decomp
```

### 2. Train ObstaDiff

```bash
python train.py \
  --config-name=train_diffusion_unet_grove_alignment_tob4ch_structured_workspace \
  task.dataset_path=data/final_data_decomp
```

This uses `StructuredObsEncoder`: a semantic-mask branch, a depth branch, spatial
softmax in both, and the 7-D end-effector pose. The included config uses horizon
16, two observation steps, eight action steps, a DDIM scheduler with 100 training
diffusion steps, AdamW at learning rate 1e-4, batch size 32, and runs 150 epochs.

### 3. Build the interaction replay library

The policy learns alignment; the contact-rich interaction is replayed from
demonstrations after the bottleneck switch. (The example data ships a prebuilt
library — this step is only needed to build your own.)

```bash
# One replay per demo: the interaction phase after alignment_end_idx, stored
# relative to the bottleneck pose, plus the target's image position and depth
# at the bottleneck frame (target_state_uvd)
python build_interaction_library.py \
  --root data/final_data_decomp \
  --out data/interaction_library

# Check the library
python scripts/check_replay_library.py --library data/interaction_library
```

Deployment picks the replay whose `target_state_uvd` is closest to the live
target and uses it to trigger the bottleneck switch, so every entry needs it.
The builder fills it from each demo's `head_camera_seg.npy` (class 2 = target)
and `head_camera_depth.npy`; the checker warns about entries without it.

To add a single hand-picked segment instead, use `make_interaction_replay.py`
followed by `add_interaction_library_entry.py`. Those entries carry no target
image position, so they are only usable with `--manual_bottleneck_switch`.

### 4. Deploy

Real deployment needs a trained checkpoint, detector weights, the replay library,
live ROS topics, and the robot safety setup.

```bash
python deploy_bbox_seg4ch.py \
  --checkpoint /path/to/obstadiff_checkpoint.ckpt \
  --yolo_weights data/yolo_weights.pt \
  --interaction_library data/interaction_library \
  --image_topic /camera/color/image_raw \
  --depth_topic /camera/aligned_depth_to_color/image_raw \
  --device cuda:0 --yolo_device cuda:0 \
  --bottleneck_target_depth_thresh 0.005 \
  --bottleneck_target_uv_thresh 0.2 \
  --target_state_norm_width 640 --target_state_norm_height 480 \
  --target_state_norm_depth_mm 1000 \
  --show_inference_view --show_target_state_error
```

At the bottleneck the script picks, among depth-ready replay candidates, the one
with the smallest normalized image-plane error `|du|/W + |dv|/H` and switches when
it drops below the UV threshold (paper settings: 5 mm depth gate after
normalizing by 1000 mm, 0.2 image-plane threshold). Useful setup flags:

- `--dry_run` — run perception and inference without commanding the arm
- `--manual_bottleneck_switch` — switch to the interaction replay manually
- `--interaction_replay_path replay.npz` — use one replay instead of a library
- `--record_clean_preview_video path.mp4` — save the preview video

## 🙏 Acknowledgement

The diffusion policy code in this repository is built on
[Diffusion Policy](https://github.com/real-stanford/diffusion_policy)
(Chi et al., RSS 2023).

## Citation

```bibtex
@inproceedings{wang2026obstadiff,
  title     = {ObstaDiff: Generalizable Diffusion Policy Learning via Obstacle-aware Representations},
  author    = {Wang, Jiawen and Yao, Kevin and Jawed, Khalid},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026}
}
```

## License

Released under the [MIT License](LICENSE).
