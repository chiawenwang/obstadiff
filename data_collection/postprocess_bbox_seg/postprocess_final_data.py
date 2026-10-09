"""Post-process final_data demos with object-as-bounding-box segmentation.

Object class:
    Filled YOLO-World bounding box, no SAM.

Vegetation class:
    YOLO-World vegetation boxes refined with MobileSAM, depth-filtered to the
    foreground around the object of interest.
"""

import argparse
import glob
import os
import shutil
import sys
import types

cv2 = None
np = None
rfd = None


def _stub_ros_imports():
    stubs = {
        "rospy": {
            "logwarn_throttle": lambda *a, **k: None,
            "logwarn_once": lambda *a, **k: None,
            "init_node": lambda *a, **k: None,
            "wait_for_message": lambda *a, **k: None,
            "Subscriber": lambda *a, **k: None,
            "sleep": lambda *a, **k: None,
            "is_shutdown": lambda: False,
            "get_time": lambda: 0.0,
        },
        "intera_interface": {
            "Limb": object,
            "Gripper": object,
            "RobotEnable": lambda *a, **k: None,
            "CHECK_VERSION": None,
        },
        "pynput": {},
        "pynput.keyboard": {
            "Listener": object,
            "Key": type("K", (), {"space": None, "esc": None}),
        },
        "sensor_msgs": {},
        "sensor_msgs.msg": {"Image": object, "CameraInfo": object},
        "dynamic_reconfigure": {},
        "dynamic_reconfigure.client": {"Client": lambda *a, **k: None},
    }
    for name, attrs in stubs.items():
        if name not in sys.modules:
            m = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(m, k, v)
            sys.modules[name] = m


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_COLLECTION_DIR = os.path.dirname(THIS_DIR)


def _load_runtime_modules():
    global cv2, np, rfd
    if cv2 is not None and np is not None and rfd is not None:
        return
    try:
        import cv2 as cv2_mod
        import numpy as np_mod
    except ImportError as e:
        raise SystemExit(
            "Missing vision dependency. Run this from the same environment "
            "used for record_full_data.py / visualize_demo.py. "
            f"Original error: {e}")

    sys.path.insert(0, DATA_COLLECTION_DIR)
    _stub_ros_imports()
    import record_full_data as rfd_mod
    cv2 = cv2_mod
    np = np_mod
    rfd = rfd_mod


SOURCE_LINK_FILES = (
    "head_camera_rgb.npy",
    "head_camera_depth.npy",
    "head_camera_rgb_intrinsic_matrix.npy",
    "head_camera_ws_rgb.png",
    "robot_states.npy",
    "bottleneck_pose.npy",
    "bottleneck_posevec.npy",
)

OBJECT_SOURCE_NONE = 0
OBJECT_SOURCE_YOLO = 1
OBJECT_SOURCE_PREVIOUS = 2
OBJECT_SOURCE_MANUAL = 3
OBJECT_SOURCE_TRACKER = 4


def resolve_demo_paths(root, demo, process_all):
    if process_all:
        paths = sorted(glob.glob(os.path.join(root, "demo_task_full_*")))
        if not paths:
            raise SystemExit(f"No demos found under {root}")
        return paths
    if demo is None:
        paths = sorted(glob.glob(os.path.join(root, "demo_task_full_*")))
        if not paths:
            raise SystemExit(f"No demos found under {root}")
        return [paths[-1]]
    if os.path.isabs(demo) or os.path.exists(demo):
        return [os.path.abspath(demo)]
    return [os.path.join(root, demo)]


def clip_box(box, width, height):
    x0, y0, x1, y1 = [int(round(float(v))) for v in box]
    x0 = max(0, min(width - 1, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height - 1, y0))
    y1 = max(0, min(height, y1))
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return None
    return (x0, y0, x1, y1)


def box_area(box):
    x0, y0, x1, y1 = box
    return max(0, x1 - x0) * max(0, y1 - y0)


def detect_world_boxes(rgb_bgr, conf):
    det = rfd._WORLD_MODEL.predict(
        rgb_bgr, conf=conf, device=rfd._MODEL_DEVICE, verbose=False)
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

    n_object = rfd._n_plant_classes()
    object_boxes = []
    veg_boxes = []
    for b, c, s in zip(boxes, cls, score):
        item = (tuple(float(v) for v in b), float(s))
        if int(c) < n_object:
            object_boxes.append(item)
        else:
            veg_boxes.append(item)
    return object_boxes, veg_boxes


def choose_object_box(object_boxes, shape, previous_box=None):
    height, width = shape[:2]
    clipped = []
    for box, score in object_boxes:
        cbox = clip_box(box, width, height)
        if cbox is not None:
            clipped.append((cbox, score))
    if not clipped:
        if previous_box is not None:
            return previous_box, OBJECT_SOURCE_PREVIOUS
        return None, OBJECT_SOURCE_NONE

    if previous_box is None:
        return max(clipped, key=lambda x: x[1])[0], OBJECT_SOURCE_YOLO

    px0, py0, px1, py1 = previous_box
    pcx, pcy = (px0 + px1) * 0.5, (py0 + py1) * 0.5

    def rank(item):
        box, score = item
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        dist = np.hypot(cx - pcx, cy - pcy)
        return score - 0.001 * dist

    return max(clipped, key=rank)[0], OBJECT_SOURCE_YOLO


def manual_select_box(rgb_bgr, demo_name, frame_idx):
    win = f"Relabel object box: {demo_name} frame {frame_idx}"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(win, rgb_bgr)
    cv2.waitKey(1)
    roi = cv2.selectROI(win, rgb_bgr, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(win)
    x, y, w, h = [int(v) for v in roi]
    if w <= 2 or h <= 2:
        return None
    return (x, y, x + w, y + h)


def object_box_mask(shape, box):
    mask = np.zeros(shape[:2], dtype=bool)
    if box is None:
        return mask
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = True
    return mask


def object_depth(depth_mm, box):
    if box is None:
        return None
    x0, y0, x1, y1 = box
    roi = depth_mm[y0:y1, x0:x1]
    valid = roi[(roi > 0) & (roi < 65000)]
    if valid.size < 10:
        return None
    return float(np.median(valid))


def sam_vegetation_mask(rgb_bgr, veg_boxes, depth_mm, object_box,
                        front_buffer_mm, min_depth_mm):
    height, width = rgb_bgr.shape[:2]
    boxes = []
    for box, _score in veg_boxes:
        cbox = clip_box(box, width, height)
        if cbox is not None:
            boxes.append(list(cbox))
    if not boxes:
        return np.zeros((height, width), dtype=bool)

    try:
        sam_res = rfd._SAM_MODEL.predict(
            rgb_bgr, bboxes=boxes, device=rfd._MODEL_DEVICE, verbose=False)
    except Exception as e:
        print(f"  WARN: MobileSAM vegetation pass failed: {e}")
        return np.zeros((height, width), dtype=bool)
    if not sam_res or sam_res[0].masks is None:
        return np.zeros((height, width), dtype=bool)

    masks_t = sam_res[0].masks.data
    try:
        masks = masks_t.cpu().numpy()
    except Exception:
        masks = np.asarray(masks_t)

    veg = np.zeros((height, width), dtype=bool)
    for m in masks:
        if m.shape != (height, width):
            m = cv2.resize((m > 0.5).astype(np.uint8), (width, height),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            m = m > 0.5
        if int(m.sum()) >= 30:
            veg |= m

    target_depth = object_depth(depth_mm, object_box)
    valid_depth = (depth_mm > min_depth_mm) & (depth_mm < 65000)
    if target_depth is not None:
        foreground = valid_depth & (depth_mm <= target_depth + front_buffer_mm)
        veg &= foreground
    else:
        veg &= valid_depth
    return veg


def choose_nearest_box(object_boxes, shape, previous_box):
    if previous_box is None:
        return choose_object_box(object_boxes, shape, None)

    height, width = shape[:2]
    clipped = []
    for box, score in object_boxes:
        cbox = clip_box(box, width, height)
        if cbox is not None:
            clipped.append((cbox, score))
    if not clipped:
        return None, OBJECT_SOURCE_NONE

    px0, py0, px1, py1 = previous_box
    pcx, pcy = (px0 + px1) * 0.5, (py0 + py1) * 0.5
    parea = max(1, box_area(previous_box))

    def rank(item):
        box, score = item
        x0, y0, x1, y1 = box
        cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        dist = np.hypot(cx - pcx, cy - pcy)
        area_ratio = abs(np.log(max(1, box_area(box)) / parea))
        return score - 0.001 * dist - 0.15 * area_ratio

    return max(clipped, key=rank)[0], OBJECT_SOURCE_YOLO


def update_tracker_box(rgb_bgr, prompt_tracking):
    tracker = prompt_tracking.get("tracker")
    if tracker is None:
        return None
    try:
        ok, box = tracker.update(rgb_bgr)
    except Exception:
        return None
    if not ok:
        return None
    x, y, w, h = [int(round(v)) for v in box]
    return clip_box((x, y, x + w, y + h), rgb_bgr.shape[1], rgb_bgr.shape[0])


def update_prompt_box(rgb_bgr, object_boxes, prompt_tracking, previous_box, frame_idx):
    if prompt_tracking is None:
        return None
    tracker_box = update_tracker_box(rgb_bgr, prompt_tracking)
    last_box = tracker_box or previous_box or prompt_tracking.get("box")
    if frame_idx == 0:
        return prompt_tracking.get("box"), OBJECT_SOURCE_MANUAL

    yolo_box, yolo_source = choose_nearest_box(
        object_boxes, rgb_bgr.shape, last_box)
    if yolo_box is not None:
        prompt_tracking["box"] = yolo_box
        prompt_tracking["tracker"] = make_tracker(rgb_bgr, yolo_box)
        return yolo_box, yolo_source

    if tracker_box is not None:
        prompt_tracking["box"] = tracker_box
        return tracker_box, OBJECT_SOURCE_TRACKER

    return last_box, OBJECT_SOURCE_PREVIOUS


def make_segmentation(rgb_bgr, depth_mm, conf, previous_box,
                      demo_name, frame_idx, front_buffer_mm, min_depth_mm,
                      prompt_tracking=None):
    object_boxes, veg_boxes = detect_world_boxes(rgb_bgr, conf)
    if prompt_tracking is not None:
        obj_box, source = update_prompt_box(
            rgb_bgr, object_boxes, prompt_tracking, previous_box, frame_idx)
    else:
        obj_box, source = choose_object_box(object_boxes, rgb_bgr.shape, previous_box)

    obj_mask = object_box_mask(rgb_bgr.shape, obj_box)
    veg_mask = sam_vegetation_mask(
        rgb_bgr, veg_boxes, depth_mm, obj_box, front_buffer_mm, min_depth_mm)
    veg_mask &= ~obj_mask

    seg = np.zeros(rgb_bgr.shape[:2], dtype=np.uint8)
    seg[veg_mask] = 1
    seg[obj_mask] = 2
    return seg, obj_box, source


def make_tracker(rgb_bgr, box):
    if box is None:
        return None
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        tracker = cv2.legacy.TrackerCSRT_create()
    elif hasattr(cv2, "TrackerCSRT_create"):
        tracker = cv2.TrackerCSRT_create()
    else:
        print("  WARN: CSRT tracker unavailable; manual box will stay fixed.")
        return None
    x0, y0, x1, y1 = box
    try:
        tracker.init(rgb_bgr, (x0, y0, x1 - x0, y1 - y0))
        return tracker
    except Exception as e:
        print(f"  WARN: could not initialize CSRT tracker: {e}")
        return None


def prompt_initial_box(rgb, demo_name):
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    print(f"\n[{demo_name}] Draw the object bounding box before processing.")
    print("ENTER/SPACE = accept, c/ESC = skip and use YOLO object boxes.")
    manual = manual_select_box(rgb_bgr, demo_name, 0)
    box = clip_box(manual, rgb_bgr.shape[1], rgb_bgr.shape[0]) if manual else None
    if box is None:
        print("  no manual box selected; using YOLO object boxes.")
        return None
    print(f"  manual object bbox = {box}")
    return {
        "box": box,
        "tracker": make_tracker(rgb_bgr, box),
    }


def encode_cam4(seg_idx, depth_mm):
    return rfd.encode_4ch(seg_idx, depth_mm).astype(np.float16)


def draw_box_on_preview(preview, box, orig_shape):
    """Match visualize_demo.py: draw the object bbox on the RGB panel only."""
    if box is None:
        return preview
    bar_h, title_h = 22, 20
    panel_h, panel_w = 240, 320
    orig_h, orig_w = orig_shape
    x0, y0, x1, y1 = box
    bx0 = int(x0 * panel_w / orig_w)
    by0 = int(y0 * panel_h / orig_h) + bar_h + title_h
    bx1 = int(x1 * panel_w / orig_w)
    by1 = int(y1 * panel_h / orig_h) + bar_h + title_h
    cv2.rectangle(preview, (bx0, by0), (bx1, by1), (0, 255, 0), 2)
    return preview


def record_preview_frame(rgb, depth_mm, seg, cam4, box, frame_idx):
    frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    preview = rfd.render_preview(
        frame_bgr, depth_mm, seg, cam4.astype(np.float32), False, frame_idx + 1,
        show_seg_legend=False)
    return draw_box_on_preview(preview, box, frame_bgr.shape[:2])


def review_video(video_path, window_name=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARN: could not open preview video: {video_path}")
        return
    win = window_name or os.path.basename(video_path)
    print(f"\nReviewing {video_path}")
    print("SPACE = pause/resume, r = replay, q/ESC/ENTER = next demo")
    paused = False
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok:
                    break
            cv2.imshow(win, frame)
        key = cv2.waitKey(0 if paused else 33) & 0xFF
        if key in (27, ord("q"), 13):
            break
        if key == ord(" "):
            paused = not paused
        elif key == ord("r"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            paused = False
    cap.release()
    cv2.destroyWindow(win)


def link_or_copy_source_files(src_dir, dst_dir, copy_arrays):
    for name in SOURCE_LINK_FILES:
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if not os.path.exists(src) or os.path.exists(dst):
            continue
        if copy_arrays:
            shutil.copy2(src, dst)
        else:
            os.symlink(os.path.relpath(src, dst_dir), dst)


def process_demo(demo_path, out_root, args):
    demo_name = os.path.basename(os.path.normpath(demo_path))
    out_dir = os.path.join(out_root, demo_name)
    os.makedirs(out_dir, exist_ok=True)

    rgb = np.load(os.path.join(demo_path, "head_camera_rgb.npy"))
    depth = np.load(os.path.join(demo_path, "head_camera_depth.npy"))
    T = min(len(rgb), len(depth))
    rgb = rgb[:T]
    depth = depth[:T]
    prompt_tracking = None
    if args.draw_box:
        prompt_tracking = prompt_initial_box(rgb[0], demo_name)

    segs = np.zeros((T, rgb.shape[1], rgb.shape[2]), dtype=np.uint8)
    cam4 = np.zeros((T, 4, rfd.ENCODE_HW[0], rfd.ENCODE_HW[1]), dtype=np.float16)
    video = None
    video_path = os.path.join(out_dir, "record_preview.mp4")

    previous_box = None
    first_box = None
    print(f"\nProcessing {demo_name}: {T} frames")
    for i in range(T):
        rgb_bgr = cv2.cvtColor(rgb[i], cv2.COLOR_RGB2BGR)
        seg, box, _source = make_segmentation(
            rgb_bgr=rgb_bgr,
            depth_mm=depth[i],
            conf=args.conf,
            previous_box=previous_box,
            demo_name=demo_name,
            frame_idx=i,
            front_buffer_mm=args.front_buffer_mm,
            min_depth_mm=args.min_depth_mm,
            prompt_tracking=prompt_tracking,
        )
        if i == 0:
            first_box = box
        previous_box = box if box is not None else previous_box
        segs[i] = seg
        cam4[i] = encode_cam4(seg, depth[i])
        if args.save_video:
            preview = record_preview_frame(rgb[i], depth[i], seg, cam4[i], box, i)
            if video is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                h, w = preview.shape[:2]
                video = cv2.VideoWriter(video_path, fourcc, args.fps, (w, h))
                if not video.isOpened():
                    raise RuntimeError(f"Could not open video writer: {video_path}")
            video.write(preview)
        if (i + 1) % 10 == 0 or i == T - 1:
            print(f"  {i + 1}/{T} frames")

    if video is not None:
        video.release()

    np.save(os.path.join(out_dir, "head_camera_seg.npy"), segs)
    np.save(os.path.join(out_dir, "head_camera_cam4.npy"), cam4)
    if T > 0:
        cv2.imwrite(os.path.join(out_dir, "record_preview_first.png"),
                    record_preview_frame(rgb[0], depth[0], segs[0], cam4[0],
                                         first_box, 0))
    link_or_copy_source_files(demo_path, out_dir, args.copy_source_arrays)
    print(f"Saved: {out_dir}")
    if args.review_video and args.save_video:
        review_video(video_path, window_name=f"review {demo_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.path.join(DATA_COLLECTION_DIR, "final_data"))
    parser.add_argument("--out-root",
                        default=os.path.join(DATA_COLLECTION_DIR, "final_data_bbox_postprocessed"))
    parser.add_argument("--demo", default=None,
                        help="Demo folder name or absolute path. Defaults to latest.")
    parser.add_argument("--all", action="store_true",
                        help="Process every demo under --data-root.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--yolo-classes", nargs="+", default=None,
                        help="Override object class prompts for YOLO-World.")
    parser.add_argument("--veg-classes", nargs="+", default=None,
                        help="Override vegetation class prompts for YOLO-World.")
    parser.add_argument("--front-buffer-mm", type=float, default=250.0,
                        help="Keep vegetation pixels with depth <= object depth + buffer.")
    parser.add_argument("--min-depth-mm", type=float, default=150.0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--review-video", action="store_true",
                        help="Play record_preview.mp4 after each processed demo.")
    parser.add_argument("--draw-box", dest="draw_box", action="store_true",
                        help="Draw the object bbox on the first frame before processing.")
    parser.add_argument("--no-draw-box", dest="draw_box", action="store_false",
                        help="Skip first-frame drawing and use YOLO object boxes.")
    parser.add_argument("--relabel", dest="draw_box", action="store_true",
                        help="Deprecated alias for --draw-box.")
    parser.add_argument("--copy-source-arrays", action="store_true",
                        help="Copy source arrays instead of symlinking them.")
    parser.set_defaults(draw_box=True)
    args = parser.parse_args()

    _load_runtime_modules()
    if args.yolo_classes is not None:
        rfd.WORLD_PLANT_CLASSES = list(args.yolo_classes)
    if args.veg_classes is not None:
        rfd.WORLD_VEG_CLASSES = list(args.veg_classes)
    rfd.WORLD_CONF = float(args.conf)
    if not rfd.load_yolo(device=args.device):
        raise SystemExit("Could not load YOLO-World + MobileSAM.")

    demo_paths = resolve_demo_paths(args.data_root, args.demo, args.all)
    os.makedirs(args.out_root, exist_ok=True)
    for demo_path in demo_paths:
        if not os.path.isdir(demo_path):
            print(f"Skipping missing demo: {demo_path}")
            continue
        process_demo(demo_path, args.out_root, args)


if __name__ == "__main__":
    main()
