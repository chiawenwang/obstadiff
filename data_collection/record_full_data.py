import numpy as np
import cv2
import json
import os
import argparse
import threading
import time

try:
    import rospy
except Exception:
    class _RospyFallback:
        @staticmethod
        def logwarn_throttle(_period, msg):
            print(f"[WARN] {msg}")

        @staticmethod
        def logwarn_once(msg):
            print(f"[WARN] {msg}")

        @staticmethod
        def is_shutdown():
            return True

        @staticmethod
        def sleep(_seconds):
            return None

    rospy = _RospyFallback()

try:
    import intera_interface
    from intera_interface import Gripper
except Exception:
    intera_interface = None
    Gripper = None

try:
    from sensor_msgs.msg import Image, CameraInfo
except Exception:
    Image = None
    CameraInfo = None

try:
    from pynput import keyboard
except Exception:
    keyboard = None
from PIL import Image as PILImage
try:
    import dynamic_reconfigure.client
except Exception:
    dynamic_reconfigure = None


# ─── Bypass cv_bridge ────────────────────────────────────────────────────────
# cv_bridge pulls in system libgdal/libtiff that conflict with conda env libs
# (LIBTIFF_4.0 / TIFFReadRGBATileExt). Decode sensor_msgs/Image directly.

def imgmsg_to_bgr8(msg):
    enc = msg.encoding
    if enc in ("bgr8", "rgb8"):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        if enc == "rgb8":
            arr = arr[:, :, ::-1]
        return np.ascontiguousarray(arr)
    if enc == "mono8":
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        return np.stack([arr, arr, arr], axis=-1)
    if enc in ("bgra8", "rgba8"):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
        arr = arr[:, :, :3]
        if enc == "rgba8":
            arr = arr[:, :, ::-1]
        return np.ascontiguousarray(arr)
    raise ValueError(f"Unsupported image encoding: {enc}")


def imgmsg_to_depth16(msg):
    return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).copy()


# ─── 4-channel segmentation preview (matches deploy_seg4ch.py pipeline) ─────
# Class scheme: 0=bg, 1=obstacle, 2=plant. Wong / Nature Methods palette,
# first combination from the reference image (BGR for OpenCV).
SEG_PALETTE_BGR = np.array([
    [167, 121, 204],   # 0 bg       — reddish purple #CC79A7
    [178, 114,   0],   # 1 obstacle — blue           #0072B2
    [  0,  94, 213],   # 2 plant    — vermillion     #D55E00
], dtype=np.uint8)
SEG_FILL_ALPHA = {
    0: 0.80,  # background
    1: 0.80,  # obstacle
    2: 0.80,  # target / plant
}
SEG_CONTOUR_W = {
    0: 1,
    1: 2,
    2: 2,
}

# Plant (dark "purple" / "black" bell pepper) HSV thresholds.
# OpenCV HSV: H in [0,180], S/V in [0,255].
# Reference sample: RGB(20,24,18) -> HSV(H=50, S=64, V=24) in OpenCV space.
# The pepper reads as very dark green-ish in raw RGB (despite looking purple
# to the eye), so the hue band sits around 50 with low V to exclude
# normal-brightness foliage. S_min skips pure black (shadows on leaves).
# Mutable so the trackbars can update them live.
THRESHOLDS = {
    "H_low":  30,
    "H_high": 75,
    "S_min":  20,
    "S_max":  255,
    "V_min":  5,     # > 0 to skip pure black
    "V_max":  60,    # cap on darkness — bright leaves are V>>80, peppers V<~50
    "depth_min_mm":  150,
    "depth_max_mm":  900,    # fallback upper bound when no target bbox
    "depth_buffer_mm": 250,  # how far behind the target counts as "its plant"
}

# Dilation margin (pixels) to tolerate RGB-depth registration error.
DEPTH_ALIGN_MARGIN_PX  = 8

# 4-channel encoding params (must match deploy_seg4ch.py exactly)
DEPTH_MAX_MM = 2000.0
ENCODE_HW    = (240, 320)

# Real-time detection stack (one pass per frame):
#   YOLO-World detects the pepper/object box.
#   Obstacles are pixel-level color evidence gated by target-plane depth.
#   No SAM is used in the live recorder/deployment path.
WORLD_WEIGHTS_DEFAULT = "yolov8s-worldv2.pt"   # ~30MB, real-time on GPU
WORLD_PLANT_CLASSES   = ["bell pepper", "purple pepper", "dark pepper", "small dark fruit"]
WORLD_VEG_CLASSES     = ["leaves", "leaf foliage", "plant stem", "green plant"]
WORLD_CONF            = 0.05    # YOLO-World scores are typically much lower than COCO
_WORLD_MODEL = None
_MODEL_DEVICE = "cuda"

def _all_world_classes():
    return list(WORLD_PLANT_CLASSES) + list(WORLD_VEG_CLASSES)

def _n_plant_classes():
    return len(WORLD_PLANT_CLASSES)


def _class_name(model, class_id):
    names = getattr(model, "names", None)
    try:
        if isinstance(names, dict):
            return str(names.get(int(class_id), "")).lower()
        if names is not None:
            return str(names[int(class_id)]).lower()
    except Exception:
        pass
    return ""


def _name_matches(name, prompts, extra_keywords=()):
    if not name:
        return False
    text = name.replace("_", " ").replace("-", " ")
    for prompt in prompts:
        prompt = prompt.lower()
        if prompt in text or text in prompt:
            return True
    return any(k in text for k in extra_keywords)

# Target-ROI rectangle drawn at startup (or re-drawn via 'r'). When set, it
# acts as the target bbox prompt. At the beginning of every recording it is
# reset to _HOME_BBOX so each take starts from the same segmentation prompt.
_TARGET_BBOX = None   # (x0, y0, x1, y1) — live tracker bbox, updated per frame
_HOME_BBOX   = None   # bbox at home pose; restored after each take so the
                      #   operator only draws the rectangle once per session
_TRACKER = None       # cv2 CSRT tracker instance — None if disabled / lost
_TRACKER_LOST = 0     # consecutive frames where CSRT.update returned False
_TRACKER_FREEZE = 0   # >0 → skip CSRT update briefly after
                      #   SPACE so the new recording starts from _HOME_BBOX

# Persist the home bbox to disk so subsequent script launches reuse it
# without making the operator redraw the rectangle.
_HOME_BBOX_FILE = os.path.expanduser(
    "./data/final_data/.home_bbox.json")


def _save_home_bbox_to_disk():
    if _HOME_BBOX is None:
        return
    try:
        os.makedirs(os.path.dirname(_HOME_BBOX_FILE), exist_ok=True)
        with open(_HOME_BBOX_FILE, "w") as f:
            json.dump({"bbox": list(_HOME_BBOX)}, f)
    except Exception as e:
        print(f"[bbox] could not persist home bbox: {e}")


def _load_home_bbox_from_disk():
    if not os.path.exists(_HOME_BBOX_FILE):
        return None
    try:
        with open(_HOME_BBOX_FILE) as f:
            data = json.load(f)
        bbox = tuple(int(v) for v in data.get("bbox", []))
        if len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            return bbox
    except Exception as e:
        print(f"[bbox] could not load home bbox: {e}")
    return None


def _delete_home_bbox_on_disk():
    if os.path.exists(_HOME_BBOX_FILE):
        try:
            os.remove(_HOME_BBOX_FILE)
        except Exception:
            pass


def _make_csrt():
    """Return a fresh CSRT tracker, or None if opencv-contrib isn't available."""
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
        return cv2.legacy.TrackerCSRT_create()
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    return None


def init_tracker_from_bbox(rgb_bgr, bbox_xyxy):
    """Seed a fresh CSRT tracker with `bbox_xyxy`. Returns True on success."""
    global _TRACKER, _TRACKER_LOST
    x0, y0, x1, y1 = bbox_xyxy
    w, h = int(x1 - x0), int(y1 - y0)
    if w < 5 or h < 5:
        _TRACKER = None
        return False
    t = _make_csrt()
    if t is None:
        print("[track] CSRT unavailable (need opencv-contrib-python); "
              "bbox will stay fixed in image coords.")
        _TRACKER = None
        return False
    try:
        t.init(rgb_bgr, (int(x0), int(y0), w, h))
        _TRACKER = t
        _TRACKER_LOST = 0
        print("[track] CSRT tracker initialized")
        return True
    except Exception as e:
        print(f"[track] CSRT init failed: {e}")
        _TRACKER = None
        return False


def update_tracker_bbox(rgb_bgr):
    """Step CSRT on the current frame. Returns updated (x0, y0, x1, y1) on
    success, or None if the tracker isn't initialized / lost the target."""
    global _TRACKER, _TRACKER_LOST
    if _TRACKER is None:
        return None
    try:
        ok, box = _TRACKER.update(rgb_bgr)
    except Exception as e:
        rospy.logwarn_throttle(5.0, f"CSRT update failed: {e}")
        return None
    if not ok:
        _TRACKER_LOST += 1
        if _TRACKER_LOST >= 30:
            print(f"[track] CSRT lost target for {_TRACKER_LOST} frames — "
                  f"holding last bbox; press 'r' to re-select.")
        return None
    _TRACKER_LOST = 0
    bx, by, bw, bh = (int(v) for v in box)
    return (bx, by, bx + bw, by + bh)


def load_yolo(weights=WORLD_WEIGHTS_DEFAULT,
              sam_weights=None, device="cuda"):
    """Load YOLO-World detection only. `sam_weights` is accepted for backward
    compatibility but intentionally ignored."""
    global _WORLD_MODEL, _MODEL_DEVICE
    try:
        from ultralytics import YOLO
    except ImportError as e:
        print(f"[YOLO] ultralytics not installed: {e}")
        return False
    try:
        print(f"[YOLO-World] Loading {weights} on {device} ...")
        _WORLD_MODEL = YOLO(weights)
        all_classes = _all_world_classes()
        if hasattr(_WORLD_MODEL, "set_classes"):
            _WORLD_MODEL.set_classes(all_classes)
        else:
            print("[YOLO] model does not support text classes; using checkpoint class names.")
        _MODEL_DEVICE = device
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        _WORLD_MODEL.predict(dummy, device=device, verbose=False)
        print(f"[YOLO-World] plant classes: {WORLD_PLANT_CLASSES}")
        print(f"[YOLO-World] veg classes:   {WORLD_VEG_CLASSES}")
        print("[YOLO-World] ready (SAM disabled).")
        return True
    except Exception as e:
        print(f"[YOLO] failed to load: {e}")
        _WORLD_MODEL = None
        return False


def _detect_and_segment(rgb_bgr):
    """Single-pass real-time detection.

    Returns (plant_mask, obstacle_mask), both (H, W) bool arrays.

    Steps (one YOLO-World call per frame):
      1. Step the CSRT tracker so the user's rectangle follows the pepper.
      2. Build only the target/object bbox:
         - plant/object mask = filled user rectangle OR YOLO-World plant box
      3. Obstacles are NOT boxed here; they are produced later from depth-gated
         image pixels.
    """
    global _TARGET_BBOX, _TRACKER, _TRACKER_FREEZE
    H, W = rgb_bgr.shape[:2]
    plant    = np.zeros((H, W), dtype=bool)
    obstacle = np.zeros((H, W), dtype=bool)

    # Keep the first frames after SPACE pinned to the saved home prompt so
    # every recording begins from the same segmentation mask.
    freeze_active = _TRACKER_FREEZE > 0
    if freeze_active:
        _TRACKER_FREEZE -= 1

    # ── 1. CSRT tracker update — skipped during the start reset window ──
    if not freeze_active and _TRACKER is not None and _TARGET_BBOX is not None:
        nb = update_tracker_bbox(rgb_bgr)
        if nb is not None:
            _TARGET_BBOX = nb

    plant_bboxes = []

    # ── 2a. Plant bbox from user rectangle (preferred) ──────────────────
    if _TARGET_BBOX is not None:
        rx0, ry0, rx1, ry1 = _TARGET_BBOX
        rx0 = int(max(0, min(W - 1, rx0)))
        rx1 = int(max(0, min(W,     rx1)))
        ry0 = int(max(0, min(H - 1, ry0)))
        ry1 = int(max(0, min(H,     ry1)))
        if rx1 > rx0 + 2 and ry1 > ry0 + 2:
            plant_bboxes.append([rx0, ry0, rx1, ry1])

    # ── 2b. YOLO pass — only use it for the target/object box. Do not use
    # vegetation boxes for the obstacle channel; those make giant rectangles.
    if _WORLD_MODEL is not None:
        try:
            det = _WORLD_MODEL.predict(rgb_bgr, conf=WORLD_CONF,
                                       device=_MODEL_DEVICE, verbose=False)
        except Exception as e:
            rospy.logwarn_throttle(5.0, f"YOLO-World failed: {e}")
            det = None
        if det and det[0].boxes is not None and len(det[0].boxes) > 0:
            arr = det[0].boxes.xyxy
            cls = det[0].boxes.cls
            try:
                arr = arr.cpu().numpy()
                cls = cls.cpu().numpy().astype(int)
            except Exception:
                arr = np.asarray(arr); cls = np.asarray(cls).astype(int)
            n_plant = _n_plant_classes()
            for i in range(len(cls)):
                c = int(cls[i])
                box = [float(v) for v in arr[i]]
                name = _class_name(_WORLD_MODEL, c)
                is_plant = (
                    c < n_plant if not name
                    else _name_matches(name, WORLD_PLANT_CLASSES,
                                       ("pepper", "fruit", "target"))
                )
                if is_plant and not plant_bboxes:   # use detection only if no user rect
                    plant_bboxes.append(box)

    # Object-of-interest segmentation is deliberately just the filled box.
    # Do not SAM-refine it: the policy should see the box area exactly.
    for box in plant_bboxes:
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        x0 = max(0, min(W - 1, x0)); x1 = max(0, min(W, x1))
        y0 = max(0, min(H - 1, y0)); y1 = max(0, min(H, y1))
        if x1 > x0 + 2 and y1 > y0 + 2:
            plant[y0:y1, x0:x1] = True

    return plant, obstacle


def _purple_mask(rgb_bgr):
    hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return ((h >= THRESHOLDS["H_low"]) & (h <= THRESHOLDS["H_high"])
            & (s >= THRESHOLDS["S_min"]) & (s <= THRESHOLDS["S_max"])
            & (v >= THRESHOLDS["V_min"]) & (v <= THRESHOLDS["V_max"]))


def _vegetation_pixel_mask(rgb_bgr):
    """Pixel-level plant/obstacle evidence. This deliberately avoids boxes.

    It is broad enough for leaves/stems but still color-based so flat walls and
    the table do not become obstacle just because they are in the depth range.
    """
    hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    green = (h >= 25) & (h <= 95) & (s >= 35) & (v >= 25)
    stem = (h >= 5) & (h <= 35) & (s >= 45) & (v >= 25) & (v <= 180)
    return green | stem


def _target_depth_mm(depth_mm):
    """Median valid depth (mm) inside the user's target rectangle.
    Returns None if no rectangle is set or there are too few valid pixels."""
    if _TARGET_BBOX is None:
        return None
    H, W = depth_mm.shape[:2]
    x0, y0, x1, y1 = _TARGET_BBOX
    x0 = max(0, min(W - 1, int(x0))); x1 = max(0, min(W, int(x1)))
    y0 = max(0, min(H - 1, int(y0))); y1 = max(0, min(H, int(y1)))
    if x1 <= x0 or y1 <= y0:
        return None
    roi = depth_mm[y0:y1, x0:x1]
    valid = roi[roi > 0]
    if valid.size < 10:
        return None
    return float(np.median(valid))


def _near_mask(depth_mm):
    """Foreground depth mask. If the user has drawn a target rectangle,
    the upper bound is the median target depth + a buffer (so the plant
    the pepper hangs from is included). Otherwise falls back to the static
    `depth_max_mm` threshold."""
    tgt = _target_depth_mm(depth_mm)
    if tgt is not None:
        # Keep pixels near the target depth plane. This uses depth to reject
        # far background and very-near table clutter while keeping nearby
        # leaves/stems around the target.
        buf = float(THRESHOLDS["depth_buffer_mm"])
        near_min = max(float(THRESHOLDS["depth_min_mm"]), tgt - 0.75 * buf)
        near_max = tgt + buf
    else:
        near_min = THRESHOLDS["depth_min_mm"]
        near_max = THRESHOLDS["depth_max_mm"]
    near = (depth_mm > near_min) & (depth_mm < near_max)
    if DEPTH_ALIGN_MARGIN_PX > 0:
        k = 2 * DEPTH_ALIGN_MARGIN_PX + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        near = cv2.dilate(near.astype(np.uint8), kernel) > 0
    return near


def segment_3class(rgb_bgr, depth_mm):
    """3-class index image: 0=bg, 1=obstacle (near-depth ∖ plant), 2=plant.
    Plant/object detection: filled bbox from the user prompt or YOLO-World.
    Obstacle is pixel-level vegetation/stem evidence gated by target-plane depth.
    Falls back to pure HSV+depth thresholds if the models aren't loaded."""
    H, W = rgb_bgr.shape[:2]
    if depth_mm.shape[:2] != (H, W):
        depth_mm = cv2.resize(depth_mm, (W, H), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((H, W), dtype=np.uint8)
    near = _near_mask(depth_mm)
    if _WORLD_MODEL is not None:
        plant, _ = _detect_and_segment(rgb_bgr)
    else:
        plant = _purple_mask(rgb_bgr) & near
    obstacle = (near & _vegetation_pixel_mask(rgb_bgr)) & ~plant
    out[obstacle] = 1
    out[plant] = 2
    return out


def encode_4ch(seg_idx, depth_mm, out_hw=ENCODE_HW):
    """Returns (4, oH, oW) float32 in [0,1]: [bg, obstacle, plant, depth_norm]."""
    oH, oW = out_hw
    H, W = seg_idx.shape
    if depth_mm.shape[:2] != (H, W):
        depth_mm = cv2.resize(depth_mm, (W, H), interpolation=cv2.INTER_NEAREST)
    ch = np.empty((4, H, W), dtype=np.uint8)
    for c in range(3):
        ch[c] = np.where(seg_idx == c, 255, 0).astype(np.uint8)
    d = np.clip(depth_mm.astype(np.float32) / DEPTH_MAX_MM, 0.0, 1.0) * 255.0
    ch[3] = d.astype(np.uint8)
    img4 = np.moveaxis(ch, 0, -1)
    if (H, W) != (oH, oW):
        img4 = cv2.resize(img4, (oW, oH), interpolation=cv2.INTER_NEAREST)
    img4 = np.moveaxis(img4, -1, 0).astype(np.float32) / 255.0
    return img4


def render_preview(rgb_bgr, depth_mm, seg_idx, cam4, is_recording, n_frames,
                   show_seg_legend=True):
    """Minimal paper-style preview: white background, thin black titles,
    four panels in a row (RGB | Seg | Depth | 4-ch Encoder Input)."""
    H_panel, W_panel = 240, 320
    title_h = 20
    pad_w   = 6
    FONT = cv2.FONT_HERSHEY_DUPLEX

    BG    = (255, 255, 255)
    TEXT  = ( 40,  40,  40)
    DIM   = (130, 130, 130)
    LINE  = (210, 210, 210)

    def panel(body, title):
        h, w = body.shape[:2]
        canvas = np.full((h + title_h, w, 3), BG, dtype=np.uint8)
        cv2.putText(canvas, title, (4, title_h - 6),
                    FONT, 0.42, TEXT, 1, cv2.LINE_AA)
        canvas[title_h:] = body
        # thin border around the body
        cv2.rectangle(canvas, (0, title_h),
                      (w - 1, h + title_h - 1), LINE, 1)
        return canvas

    # 1. RGB
    rgb_r = cv2.resize(rgb_bgr, (W_panel, H_panel))
    panels = [panel(rgb_r, "RGB")]

    # 2. Segmentation — paper-style overlay. Each class, including background,
    # gets both a region fill and a contour stroke.
    seg_panel_idx = cv2.resize(seg_idx, (W_panel, H_panel),
                               interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(rgb_r, cv2.COLOR_BGR2GRAY)
    gray_3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    seg_overlay = (gray_3.astype(np.float32) * 0.70
                   + rgb_r.astype(np.float32) * 0.30).astype(np.uint8)
    for cls_idx in (0, 1, 2):        # 0 = background, 1 = obstacle, 2 = plant
        m = seg_panel_idx == cls_idx
        if not m.any():
            continue
        color = tuple(int(v) for v in SEG_PALETTE_BGR[cls_idx])
        alpha = SEG_FILL_ALPHA[cls_idx]
        blended = ((1 - alpha) * seg_overlay.astype(np.float32)
                   + alpha     * np.array(color, dtype=np.float32)
                  ).astype(np.uint8)
        seg_overlay[m] = blended[m]

    # Background border: draw only where background meets a foreground class,
    # so the background has its own mask edge without outlining the whole panel.
    bg_mask = seg_panel_idx == 0
    non_bg = seg_panel_idx != 0
    if bg_mask.any() and non_bg.any():
        bg_edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bg_edge = (cv2.dilate(non_bg.astype(np.uint8), bg_edge_kernel) > 0) & bg_mask
        bg_edge_u8 = bg_edge.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            bg_edge_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(seg_overlay, contours, -1,
                         tuple(int(v) for v in SEG_PALETTE_BGR[0]),
                         SEG_CONTOUR_W[0], cv2.LINE_AA)

    for cls_idx in (1, 2):           # obstacle + plant contours
        m = seg_panel_idx == cls_idx
        if not m.any():
            continue
        color = tuple(int(v) for v in SEG_PALETTE_BGR[cls_idx])
        m_u8 = m.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(seg_overlay, contours, -1, color,
                         SEG_CONTOUR_W[cls_idx], cv2.LINE_AA)
    if show_seg_legend:
        # Tiny in-panel legend so a screenshot is self-explanatory.
        leg_y = H_panel - 8
        leg_x = 6
        legend_items = [(2, "plant"), (1, "obstacle"), (0, "background")]
        for cls_idx, label in legend_items:
            color = tuple(int(v) for v in SEG_PALETTE_BGR[cls_idx])
            cv2.rectangle(seg_overlay, (leg_x, leg_y - 8),
                          (leg_x + 10, leg_y + 2), color, -1, cv2.LINE_AA)
            cv2.rectangle(seg_overlay, (leg_x, leg_y - 8),
                          (leg_x + 10, leg_y + 2), (40, 40, 40), 1, cv2.LINE_AA)
            cv2.putText(seg_overlay, label, (leg_x + 16, leg_y),
                        FONT, 0.40, (245, 245, 245), 2, cv2.LINE_AA)
            cv2.putText(seg_overlay, label, (leg_x + 16, leg_y),
                        FONT, 0.40, (30, 30, 30), 1, cv2.LINE_AA)
            leg_x += 16 + cv2.getTextSize(label, FONT, 0.40, 1)[0][0] + 12
    panels.append(panel(seg_overlay, "Segmentation"))

    # 3. Depth — INFERNO single-hue ramp, auto-scaled to actual valid range
    #    so the contrast is in the area of interest, not wasted on empty space.
    d_mm = depth_mm.astype(np.float32)
    valid_mask = d_mm > 0
    if valid_mask.any():
        vmin, vmax = np.percentile(d_mm[valid_mask], (5, 95))
        if vmax - vmin < 50:        # too flat → use a sensible default span
            vmax = vmin + 500.0
    else:
        vmin, vmax = 0.0, DEPTH_MAX_MM
    d_norm = np.clip((d_mm - vmin) / max(1.0, vmax - vmin), 0, 1) * 255
    d_resized = cv2.resize(d_norm.astype(np.uint8), (W_panel, H_panel),
                           interpolation=cv2.INTER_NEAREST)
    d_color = cv2.applyColorMap(d_resized, cv2.COLORMAP_INFERNO)
    invalid = cv2.resize(((~valid_mask).astype(np.uint8) * 255),
                         (W_panel, H_panel),
                         interpolation=cv2.INTER_NEAREST) > 0
    d_color[invalid] = BG
    panels.append(panel(d_color,
        f"Depth  ({int(vmin)}-{int(vmax)} mm,  near = dark)"))

    # 4. 4-channel encoder input (2x2 channel mosaic)
    half_h, half_w = H_panel // 2, W_panel // 2
    ch_imgs = []
    for c in range(4):
        ci = (cam4[c] * 255).clip(0, 255).astype(np.uint8)
        ci = cv2.resize(ci, (half_w, half_h), interpolation=cv2.INTER_NEAREST)
        ch_imgs.append(cv2.cvtColor(ci, cv2.COLOR_GRAY2BGR))
    top    = np.concatenate([ch_imgs[0], ch_imgs[1]], axis=1)
    bottom = np.concatenate([ch_imgs[2], ch_imgs[3]], axis=1)
    enc = np.concatenate([top, bottom], axis=0)
    cv2.line(enc, (half_w, 0), (half_w, H_panel), (90, 90, 90), 1)
    cv2.line(enc, (0, half_h), (W_panel, half_h), (90, 90, 90), 1)
    panels.append(panel(enc, "4-ch Encoder Input"))

    # Compose horizontally with thin white padding
    pad = np.full((H_panel + title_h, pad_w, 3), BG, dtype=np.uint8)
    composite = panels[0]
    for p in panels[1:]:
        composite = np.concatenate([composite, pad, p], axis=1)

    # Compact status strip on top: REC / IDLE + frame count + counts
    bar_h = 22
    bar = np.full((bar_h, composite.shape[1], 3), BG, dtype=np.uint8)
    plant_px = int((seg_idx == 2).sum())
    obst_px  = int((seg_idx == 1).sum())
    if is_recording:
        cv2.circle(bar, (12, bar_h // 2), 4, (60, 72, 240), -1, cv2.LINE_AA)
        cv2.putText(bar, "REC", (22, bar_h - 7), FONT, 0.46, TEXT, 1, cv2.LINE_AA)
    else:
        cv2.putText(bar, "idle", (10, bar_h - 7), FONT, 0.44, DIM, 1, cv2.LINE_AA)
    right = f"frame {n_frames}   plant {plant_px:,}   obst {obst_px:,}"
    (tw, _), _ = cv2.getTextSize(right, FONT, 0.42, 1)
    cv2.putText(bar, right, (composite.shape[1] - tw - 8, bar_h - 7),
                FONT, 0.42, DIM, 1, cv2.LINE_AA)
    cv2.line(bar, (0, bar_h - 1), (composite.shape[1], bar_h - 1), LINE, 1)
    return np.concatenate([bar, composite], axis=0)


def save_individual_panels(rgb_bgr, depth_mm, seg_idx, cam4,
                           target_bbox_orig, out_dir):
    """Save the 7 preview-panel bodies as individual PNGs plus a clean RGB.

    Output (in `out_dir`):
        rgb.png             320x240  RGB with red target box (if set)
        rgb_clean.png       320x240  RGB only, no overlay
        segmentation.png    320x240  seg overlay + legend + red target box
        depth.png           320x240  INFERNO depth colormap (panel-style)
        encoder_bg.png      320x240  bg channel (grayscale)
        encoder_obstacle.png 320x240 obstacle channel
        encoder_plant.png   320x240  plant channel
        encoder_depth.png   320x240  depth channel (normalized to 0..1)

    Mirrors the panel-body code in render_preview so screenshots stay
    visually identical to what the operator sees on screen.
    """
    os.makedirs(out_dir, exist_ok=True)
    H_panel, W_panel = 240, 320
    FONT = cv2.FONT_HERSHEY_DUPLEX
    BG_PANEL = (255, 255, 255)

    orig_h, orig_w = rgb_bgr.shape[:2]
    def _bx(p):
        x = int(p[0] * W_panel / orig_w)
        y = int(p[1] * H_panel / orig_h)
        return x, y

    # 1. RGB (clean + boxed)
    rgb_r = cv2.resize(rgb_bgr, (W_panel, H_panel))
    cv2.imwrite(os.path.join(out_dir, "rgb_clean.png"), rgb_r)
    rgb_box = rgb_r.copy()
    if target_bbox_orig is not None:
        x0, y0 = _bx((target_bbox_orig[0], target_bbox_orig[1]))
        x1, y1 = _bx((target_bbox_orig[2], target_bbox_orig[3]))
        cv2.rectangle(rgb_box, (x0, y0), (x1, y1), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(out_dir, "rgb.png"), rgb_box)

    # 2. Segmentation overlay (with legend + red target box)
    seg_panel_idx = cv2.resize(seg_idx, (W_panel, H_panel),
                               interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(rgb_r, cv2.COLOR_BGR2GRAY)
    gray_3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    seg_overlay = (gray_3.astype(np.float32) * 0.70
                   + rgb_r.astype(np.float32) * 0.30).astype(np.uint8)
    for cls_idx in (0, 1, 2):
        m = seg_panel_idx == cls_idx
        if not m.any():
            continue
        color = tuple(int(v) for v in SEG_PALETTE_BGR[cls_idx])
        alpha = SEG_FILL_ALPHA[cls_idx]
        blended = ((1 - alpha) * seg_overlay.astype(np.float32)
                   + alpha     * np.array(color, dtype=np.float32)
                  ).astype(np.uint8)
        seg_overlay[m] = blended[m]
    bg_mask = seg_panel_idx == 0
    non_bg = seg_panel_idx != 0
    if bg_mask.any() and non_bg.any():
        bg_edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bg_edge = (cv2.dilate(non_bg.astype(np.uint8), bg_edge_kernel) > 0) & bg_mask
        bg_edge_u8 = bg_edge.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            bg_edge_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(seg_overlay, contours, -1,
                         tuple(int(v) for v in SEG_PALETTE_BGR[0]),
                         SEG_CONTOUR_W[0], cv2.LINE_AA)
    for cls_idx in (1, 2):
        m = seg_panel_idx == cls_idx
        if not m.any():
            continue
        color = tuple(int(v) for v in SEG_PALETTE_BGR[cls_idx])
        m_u8 = m.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(seg_overlay, contours, -1, color,
                         SEG_CONTOUR_W[cls_idx], cv2.LINE_AA)
    leg_y = H_panel - 8
    leg_x = 6
    for cls_idx, label in [(2, "plant"), (1, "obstacle"), (0, "background")]:
        color = tuple(int(v) for v in SEG_PALETTE_BGR[cls_idx])
        cv2.rectangle(seg_overlay, (leg_x, leg_y - 8),
                      (leg_x + 10, leg_y + 2), color, -1, cv2.LINE_AA)
        cv2.rectangle(seg_overlay, (leg_x, leg_y - 8),
                      (leg_x + 10, leg_y + 2), (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(seg_overlay, label, (leg_x + 16, leg_y),
                    FONT, 0.40, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.putText(seg_overlay, label, (leg_x + 16, leg_y),
                    FONT, 0.40, (30, 30, 30), 1, cv2.LINE_AA)
        leg_x += 16 + cv2.getTextSize(label, FONT, 0.40, 1)[0][0] + 12
    if target_bbox_orig is not None:
        x0, y0 = _bx((target_bbox_orig[0], target_bbox_orig[1]))
        x1, y1 = _bx((target_bbox_orig[2], target_bbox_orig[3]))
        cv2.rectangle(seg_overlay, (x0, y0), (x1, y1), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(out_dir, "segmentation.png"), seg_overlay)

    # 3. Depth (INFERNO, auto-scaled like the panel)
    d_mm = depth_mm.astype(np.float32)
    valid_mask = d_mm > 0
    if valid_mask.any():
        vmin, vmax = np.percentile(d_mm[valid_mask], (5, 95))
        if vmax - vmin < 50:
            vmax = vmin + 500.0
    else:
        vmin, vmax = 0.0, DEPTH_MAX_MM
    d_norm = np.clip((d_mm - vmin) / max(1.0, vmax - vmin), 0, 1) * 255
    d_resized = cv2.resize(d_norm.astype(np.uint8), (W_panel, H_panel),
                           interpolation=cv2.INTER_NEAREST)
    d_color = cv2.applyColorMap(d_resized, cv2.COLORMAP_INFERNO)
    invalid = cv2.resize(((~valid_mask).astype(np.uint8) * 255),
                         (W_panel, H_panel),
                         interpolation=cv2.INTER_NEAREST) > 0
    d_color[invalid] = BG_PANEL
    cv2.imwrite(os.path.join(out_dir, "depth.png"), d_color)

    # 4. Encoder channels — each at full panel size so they print cleanly
    for c, name in enumerate(["encoder_bg", "encoder_obstacle",
                              "encoder_plant", "encoder_depth"]):
        ci = (cam4[c] * 255).clip(0, 255).astype(np.uint8)
        ci = cv2.resize(ci, (W_panel, H_panel), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(out_dir, f"{name}.png"), ci)


class FullStateRecorder:
    def __init__(self, task_name, do_annotate=False,
                 use_yolo=True, yolo_weights=WORLD_WEIGHTS_DEFAULT,
                 sam_weights=None, yolo_device="cuda"):
        self.task_name = task_name
        self.do_annotate = do_annotate
        self.limb = intera_interface.Limb('right')

        if use_yolo:
            load_yolo(weights=yolo_weights, sam_weights=sam_weights,
                      device=yolo_device)
        else:
            print("[YOLO] disabled via --no_yolo; using HSV+depth only.")
        
        # --- CONFIGURATION ---
        self.save_root = os.path.expanduser("./data/final_data/")
        self.home_joints = [0.0273525390625, -0.77991015625, -1.30016015625, 1.673435546875, -0.903478515625, -1.5015849609375, -0.1209189453125]
        self.joint_names = ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6']
        
        # Precise RealSense Intrinsics (from your hardware)
        self.intrinsics_matrix = np.array([
            [386.02066040,   0.0,          321.55554199],
            [  0.0,          385.41479492, 240.14253235],
            [  0.0,          0.0,          1.0         ]
        ], dtype=np.float64)

        # Initialize Gripper
        try:
            self.gripper = Gripper('right_hand')
            print("[INFO] Gripper initialized.")
        except:
            self.gripper = None
            print("[WARNING] No gripper detected.")

        self.rgb_topic = "/camera/color/image_raw"
        self.depth_topic = "/camera/aligned_depth_to_color/image_raw"

        self.is_recording = False
        self.reset_buffers()
        self.has_received_rgb = False

        # Latest frames cached for the live preview thread (always populated,
        # regardless of recording state, so the segmenter can be tuned before
        # pressing SPACE).
        self.latest_rgb_bgr = None
        self.latest_depth = None
        self._stop_display = False

        # Even/odd frame alternation to prevent IR projector dots in RGB
        self.frame_count = 0
        self.emitter_on = True
        try:
            self.realsense_client = dynamic_reconfigure.client.Client(
                "/camera/stereo_module", timeout=5.0)
            print("[INFO] RealSense dynamic_reconfigure client connected.")
        except Exception as e:
            self.realsense_client = None
            print(f"[WARNING] Could not connect to RealSense dynamic_reconfigure: {e}")
            print("         Even/odd frame alternation disabled — capturing all frames normally.")

        print(f"\n[WAITING] Connecting to RealSense topics...")
        rospy.Subscriber(self.rgb_topic, Image, self.rgb_cb)
        rospy.Subscriber(self.depth_topic, Image, self.depth_cb)

        # Verify intrinsics from the actual camera_info topic (replaces the
        # hardcoded matrix when available).
        self.verify_intrinsics()

        # Connection check
        rospy.sleep(2.0)
        if not self.has_received_rgb:
             self.print_realsense_help()

        # Display loop now runs on the MAIN thread (called via rec.run()).
        # OpenCV's Qt backend won't permit setMouseCallback/imshow from a
        # background thread on Linux; we leave the keyboard listener in the
        # background instead.

        print(f"\n[READY] Task Base Name: {task_name}")
        print(f"[MODE] Annotation is {'ON' if self.do_annotate else 'OFF'}")
        print(f"[PREVIEW] dark-purple HSV defaults: "
              f"H=[{THRESHOLDS['H_low']},{THRESHOLDS['H_high']}] "
              f"S=[{THRESHOLDS['S_min']},{THRESHOLDS['S_max']}] "
              f"V=[{THRESHOLDS['V_min']},{THRESHOLDS['V_max']}]   "
              f"near-depth=[{THRESHOLDS['depth_min_mm']},{THRESHOLDS['depth_max_mm']}]mm "
              f"depth_norm=/{int(DEPTH_MAX_MM)}mm")
        print(f"          Tune live with the 'HSV Controls' window — current "
              f"values print every 5s; copy them into THRESHOLDS at the top of the file.")
        print(f"[YOLO-World] plant={WORLD_PLANT_CLASSES}")
        print(f"             veg  ={WORLD_VEG_CLASSES}   conf>={WORLD_CONF}")
        print("       Real-time pipeline: YOLO boxes only the target; "
              "obstacles are depth-gated pixels (SAM disabled).")
        print("       Draw one RECTANGLE around the target at home pose; that")
        print("       same home prompt is applied at the start of every recording.")
        print("       Press 'r' in the preview window to re-draw the rectangle anytime.")
        print("       Press 's' in the preview window to save all 7 panels + clean RGB.")
        print("       MIDDLE-click on preview = clear rectangle (keep all detections).")
        print("-> PRESS SPACEBAR: Start Recording")
        print("-> PRESS SPACEBAR AGAIN: Stop, Save & Move Home")
        print("-> ESC: Exit")

    def verify_intrinsics(self, topic="/camera/color/camera_info", timeout=3.0):
        """Query the camera's published intrinsics and compare against the
        hardcoded matrix. If they differ by more than 1px or 1 focal-unit,
        replace the hardcoded matrix with the queried one and warn."""
        try:
            ci = rospy.wait_for_message(topic, CameraInfo, timeout=timeout)
        except Exception as e:
            print(f"[INTRINSICS] WARN: no {topic} within {timeout}s ({e}). "
                  f"Falling back to hardcoded matrix.")
            return
        K_live = np.array(ci.K, dtype=np.float64).reshape(3, 3)
        fx_l, fy_l = K_live[0, 0], K_live[1, 1]
        cx_l, cy_l = K_live[0, 2], K_live[1, 2]
        fx_h, fy_h = self.intrinsics_matrix[0, 0], self.intrinsics_matrix[1, 1]
        cx_h, cy_h = self.intrinsics_matrix[0, 2], self.intrinsics_matrix[1, 2]
        diff = float(np.max(np.abs(K_live - self.intrinsics_matrix)))
        print(f"[INTRINSICS] camera_info: {ci.width}x{ci.height}  "
              f"fx={fx_l:.2f} fy={fy_l:.2f} cx={cx_l:.2f} cy={cy_l:.2f}")
        print(f"[INTRINSICS] hardcoded : fx={fx_h:.2f} fy={fy_h:.2f} "
              f"cx={cx_h:.2f} cy={cy_h:.2f}  (max abs diff = {diff:.2f})")
        if diff > 1.0:
            print("[INTRINSICS] >>> using live intrinsics from camera_info "
                  "(will be saved to head_camera_rgb_intrinsic_matrix.npy).")
            self.intrinsics_matrix = K_live
        else:
            print("[INTRINSICS] OK — hardcoded matrix matches camera_info.")

    def print_realsense_help(self):
        print("\n" + "!"*60)
        print("CRITICAL ERROR: REALSENSE NOT DETECTED")
        print("1. Check USB connection (3.0 preferred).")
        print("2. Run this in a new terminal:")
        print("   roslaunch realsense2_camera rs_camera.launch align_depth:=true")
        print("!"*60 + "\n")

    def reset_buffers(self):
        self.rs_rgb = []
        self.rs_depth = []
        self.full_state = []
        self.frame_count = 0

    def set_emitter(self, enabled):
        """Toggle the RealSense IR emitter on/off via dynamic_reconfigure."""
        if self.realsense_client is None:
            return
        try:
            self.realsense_client.update_configuration({"emitter_enabled": 1 if enabled else 0})
            self.emitter_on = enabled
        except Exception as e:
            rospy.logwarn_once(f"Failed to toggle IR emitter: {e}")

    def rgb_cb(self, msg):
        self.has_received_rgb = True
        try:
            frame = imgmsg_to_bgr8(msg)
            # Always cache the latest BGR frame for the preview thread.
            self.latest_rgb_bgr = frame
        except Exception as e:
            print(f"Error decoding RGB: {e}")
            return

        if not self.is_recording:
            return
        try:

            if self.realsense_client is not None:
                # Even/odd alternation: even frames = depth (emitter ON),
                # odd frames = RGB (emitter OFF, no IR dots)
                is_rgb_frame = (self.frame_count % 2 == 1)
                is_depth_frame = (self.frame_count % 2 == 0)

                if is_rgb_frame:
                    # Odd frame: emitter was OFF, so this RGB is clean
                    self.rs_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    # Collect robot state paired with each RGB frame
                    j_angles = self.limb.joint_angles()
                    curr_angles = [j_angles[j] for j in self.joint_names]
                    p = self.limb.endpoint_pose()
                    cartesian = [p['position'].x, p['position'].y, p['position'].z,
                                 p['orientation'].x, p['orientation'].y, p['orientation'].z, p['orientation'].w]
                    g_pos = self.gripper.get_position() if self.gripper else 0.0
                    state_vector = curr_angles + cartesian + [g_pos]
                    self.full_state.append(state_vector)

                    # Turn emitter ON for the next (even/depth) frame
                    self.set_emitter(True)
                else:
                    # Even frame: emitter was ON — skip this RGB (has IR dots)
                    # Turn emitter OFF for the next (odd/RGB) frame
                    self.set_emitter(False)

                self.frame_count += 1
            else:
                # Fallback: no dynamic_reconfigure, capture every frame normally
                self.rs_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                j_angles = self.limb.joint_angles()
                curr_angles = [j_angles[j] for j in self.joint_names]
                p = self.limb.endpoint_pose()
                cartesian = [p['position'].x, p['position'].y, p['position'].z,
                             p['orientation'].x, p['orientation'].y, p['orientation'].z, p['orientation'].w]
                g_pos = self.gripper.get_position() if self.gripper else 0.0
                state_vector = curr_angles + cartesian + [g_pos]
                self.full_state.append(state_vector)

        except Exception as e:
            print(f"Error in rgb_cb: {e}")

    def depth_cb(self, msg):
        try:
            d = imgmsg_to_depth16(msg)
            # Always cache the latest depth for the preview thread.
            self.latest_depth = d
        except Exception:
            return

        if not self.is_recording:
            return
        try:
            if self.realsense_client is not None:
                # Only store depth on even frames (emitter was ON)
                if self.emitter_on:
                    self.rs_depth.append(d)
            else:
                # Fallback: capture every depth frame
                self.rs_depth.append(d)
        except Exception:
            pass

    def _select_target_bbox(self):
        """Modal: pop up cv2.selectROI on the latest RGB frame, store the
        result as the home target-region rectangle. Cancelling (ESC /
        zero-size) clears the bbox so YOLO uses all of its detections in the
        WORLD_PLANT_CLASSES list."""
        global _TARGET_BBOX, _HOME_BBOX, _TRACKER
        if self.latest_rgb_bgr is None:
            print("[YOLO] no RGB frame yet — cannot select bbox")
            return
        img = self.latest_rgb_bgr.copy()
        print("\n[YOLO] Drag a rectangle around the target pepper.")
        print("        ENTER = confirm,  c / ESC = cancel (clears bbox).")
        roi = cv2.selectROI("Select target (drag, then ENTER)",
                            img, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select target (drag, then ENTER)")
        x, y, w, h = roi
        if w > 5 and h > 5:
            _TARGET_BBOX = (int(x), int(y), int(x + w), int(y + h))
            # Treat this as the canonical "home pose" bbox — it'll be restored
            # after every recording, and persisted to disk so the next launch
            # of this script reuses it without making you redraw.
            _HOME_BBOX = _TARGET_BBOX
            _save_home_bbox_to_disk()
            print(f"[YOLO] target bbox = {_TARGET_BBOX} (saved as home bbox; "
                  f"persisted to {_HOME_BBOX_FILE})")
            init_tracker_from_bbox(img, _TARGET_BBOX)
        else:
            _TARGET_BBOX = None
            _HOME_BBOX = None
            _TRACKER = None
            _delete_home_bbox_on_disk()
            print("[YOLO] bbox cleared — keeping all detections of plant classes")

    def _display_loop(self):
        """Live segmentation preview at ~20Hz; tunes purple/obstacle/depth thresholds
        visually before pressing SPACE. Saves nothing — RGB+depth+state are still the
        on-disk format, so the same recording works for deploy_seg4ch.py (re-segments
        at inference) and deploy_grove_sawyer.py (uses raw RGB).

        Mouse / key controls on the preview window:
          MIDDLE-CLICK on the RGB panel → clear the target rectangle
          'r' key                       → re-draw the target rectangle
          's' key                       → snapshot all 7 panels + clean RGB
                                          to preview_panel_exports/snapshot_<ts>/
        """
        win = "record_full_data — 4ch preview"
        ctrl = "HSV Controls"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(ctrl, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(ctrl, 480, 320)

        # Mouse callback: only middle-click is wired up (clears the target
        # bbox). Use 'r' in the preview window to re-draw the rectangle.
        def _on_mouse(event, x, y, flags, param):
            global _TARGET_BBOX, _TRACKER, _HOME_BBOX
            if event != cv2.EVENT_MBUTTONDOWN:
                return
            _TARGET_BBOX = None
            _HOME_BBOX = None
            _TRACKER = None
            _delete_home_bbox_on_disk()
            print("[YOLO] target + home bbox cleared — keeping all detections")

        # NOTE: setMouseCallback is deferred to the main loop, after the first
        # real imshow has had a chance to pump the Qt event loop. Calling it
        # here on a never-shown window raises "NULL window handler".
        self._mouse_callback_fn = _on_mouse
        self._mouse_callback_set = False
        self._preview_win_name = win

        # Trackbar callbacks write into the shared THRESHOLDS dict; the
        # segmenter reads it on every frame. Plain int writes/reads in CPython
        # are atomic enough for this use.
        def _mk(key, max_val):
            cv2.createTrackbar(key, ctrl, int(THRESHOLDS[key]), max_val,
                               lambda v, k=key: THRESHOLDS.__setitem__(k, int(v)))
        _mk("H_low",  180)
        _mk("H_high", 180)
        _mk("S_min",  255)
        _mk("S_max",  255)
        _mk("V_min",  255)
        _mk("V_max",  255)
        _mk("depth_min_mm",    4000)
        _mk("depth_max_mm",    4000)
        _mk("depth_buffer_mm", 1000)

        # Wait for the first RGB+depth frame, then either reuse the bbox
        # persisted from the previous session, or prompt the user to draw
        # one. Press 'r' anytime to redraw, middle-click to clear.
        print("\n[YOLO] Waiting for first frame...")
        while (not rospy.is_shutdown() and not self._stop_display
               and (self.latest_rgb_bgr is None or self.latest_depth is None)):
            cv2.waitKey(50)
        if not self._stop_display and not rospy.is_shutdown():
            persisted = _load_home_bbox_from_disk()
            if persisted is not None:
                global _TARGET_BBOX, _HOME_BBOX
                _TARGET_BBOX = persisted
                _HOME_BBOX = persisted
                init_tracker_from_bbox(self.latest_rgb_bgr, persisted)
                print(f"[bbox] reused home bbox {persisted} from disk "
                      f"(press 'r' to redraw, middle-click to clear).")
            else:
                self._select_target_bbox()

        last_print = 0.0
        while not rospy.is_shutdown() and not self._stop_display:
            rgb_bgr = self.latest_rgb_bgr
            depth = self.latest_depth
            if rgb_bgr is not None and depth is not None:
                try:
                    seg = segment_3class(rgb_bgr, depth)
                    cam4 = encode_4ch(seg, depth)
                    self._last_seg = seg
                    self._last_cam4 = cam4
                    preview = render_preview(rgb_bgr, depth, seg, cam4,
                                             self.is_recording, len(self.rs_rgb))
                    # Draw the target bbox on the RGB panel of the composite.
                    bar_h, title_h = 22, 20   # must match render_preview
                    H_panel, W_panel = 240, 320
                    orig_h, orig_w = rgb_bgr.shape[:2]
                    def _to_panel(pt):
                        x, y = pt
                        px = int(x * W_panel / orig_w)
                        py = int(y * H_panel / orig_h) + bar_h + title_h
                        return (px, py)
                    if _TARGET_BBOX is not None:
                        bx0, by0 = _to_panel((_TARGET_BBOX[0], _TARGET_BBOX[1]))
                        bx1, by1 = _to_panel((_TARGET_BBOX[2], _TARGET_BBOX[3]))
                        cv2.rectangle(preview, (bx0, by0), (bx1, by1),
                                      (0, 0, 255), 2)
                        # Mirror the box onto the Segmentation panel
                        # (second panel, x-shifted by W_panel + pad_w=6).
                        seg_dx = W_panel + 6
                        cv2.rectangle(preview,
                                      (bx0 + seg_dx, by0),
                                      (bx1 + seg_dx, by1),
                                      (0, 0, 255), 2)
                    cv2.imshow(win, preview)
                    # Attach the mouse callback now that the window exists.
                    if not self._mouse_callback_set:
                        try:
                            cv2.setMouseCallback(win, self._mouse_callback_fn)
                            self._mouse_callback_set = True
                            print("[bbox] middle-click clear armed on preview window.")
                        except cv2.error:
                            # Window still not ready; try again next iteration.
                            pass
                except Exception as e:
                    rospy.logwarn_throttle(5.0, f"preview render failed: {e}")

            # Periodically print the current threshold values to the terminal
            # so the user can copy a tuned set back into the file as defaults.
            now = time.monotonic()
            if now - last_print > 5.0:
                last_print = now
                print(f"[thresholds] H=[{THRESHOLDS['H_low']},{THRESHOLDS['H_high']}] "
                      f"S=[{THRESHOLDS['S_min']},{THRESHOLDS['S_max']}] "
                      f"V=[{THRESHOLDS['V_min']},{THRESHOLDS['V_max']}] "
                      f"depth=[{THRESHOLDS['depth_min_mm']},{THRESHOLDS['depth_max_mm']}]mm  "
                      f"buf={THRESHOLDS['depth_buffer_mm']}mm")
            # Capture keys from the preview window: 'r' = re-select bbox.
            key_pressed = cv2.waitKey(50) & 0xFF
            if key_pressed == ord('r'):
                self._select_target_bbox()
            elif key_pressed == ord('s'):
                last_seg = getattr(self, "_last_seg", None)
                last_cam4 = getattr(self, "_last_cam4", None)
                if (rgb_bgr is None or depth is None
                        or last_seg is None or last_cam4 is None):
                    print("[snapshot] no frame yet — wait for the preview "
                          "to start, then press 's' again.")
                else:
                    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
                    snap_dir = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "preview_panel_exports",
                        f"snapshot_{ts}",
                    )
                    save_individual_panels(
                        rgb_bgr, depth, last_seg, last_cam4,
                        _TARGET_BBOX, snap_dir,
                    )
                    print(f"[snapshot] saved 8 panels to {snap_dir}")
        try:
            cv2.destroyWindow(win)
            cv2.destroyWindow(ctrl)
        except Exception:
            pass

    def get_next_folder_path(self):
        """Finds the next available folder index to avoid overwriting."""
        idx = 1
        while True:
            folder_name = f"{self.task_name}_{idx:03d}"
            full_path = os.path.join(self.save_root, folder_name)
            if not os.path.exists(full_path):
                return full_path
            idx += 1

    def perform_annotation(self, folder_path, rgb_frame):
        """Opens a window to draw the segmentation mask immediately."""
        print("\n--- ANNOTATION MODE ---")
        print("Draw a box around the object and press ENTER. Press 'c' to cancel.")
        
        # Convert back to BGR for OpenCV display
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        roi = cv2.selectROI("Annotate Object", bgr, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Annotate Object")
        
        x, y, w, h = roi
        if w > 0 and h > 0:
            mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
            mask[y:y+h, x:x+w] = 1 # 1 = Object, 0 = Background
            
            save_path = os.path.join(folder_path, "head_camera_ws_segmap.npy")
            np.save(save_path, mask)
            print(f"✅ Saved mask to {save_path}")
        else:
            print("⚠️ No mask selected. You will need to generate this later.")

    def save_data(self):
        folder = self.get_next_folder_path()
        os.makedirs(folder, exist_ok=True)
        
        rgb_len = min(len(self.rs_rgb), len(self.full_state))
        depth_len = len(self.rs_depth)
        length = min(rgb_len, depth_len)
        if length == 0:
            print("FAILURE: No data captured.")
            return

        if self.realsense_client is not None:
            print(f"Saving {length} frames to: {folder}... (even/odd alternation: {rgb_len} RGB, {depth_len} depth)")
        else:
            print(f"Saving {length} frames to: {folder}...")

        # --- 1. Save Core Video Data ---
        # Note: Preprocessor expects 'head_camera_depth.npy', not 'depth_to_rgb'
        np.save(os.path.join(folder, "head_camera_rgb.npy"), np.array(self.rs_rgb[:length]))
        np.save(os.path.join(folder, "head_camera_depth.npy"), np.array(self.rs_depth[:length]))
        
        # --- 2. Save Robot States (Master File) ---
        state_stack = np.array(self.full_state[:length])
        np.save(os.path.join(folder, "robot_states.npy"), state_stack)

        # --- 3. Save Bottleneck (Smart Min-Z Logic) ---
        # Indices 7,8,9 are x,y,z in our 15-dim vector
        z_values = state_stack[:, 9] 
        bottleneck_idx = np.argmin(z_values) # Find lowest point
        
        # Indices 7-14 are the 7D Pose [x,y,z,qx,qy,qz,qw]
        bottleneck_pose = state_stack[bottleneck_idx, 7:14]
        
        np.save(os.path.join(folder, "bottleneck_posevec.npy"), bottleneck_pose)
        np.save(os.path.join(folder, "bottleneck_pose.npy"), bottleneck_pose) # Duplicate for safety

        # --- 4. Save Intrinsics ---
        np.save(os.path.join(folder, "head_camera_rgb_intrinsic_matrix.npy"), self.intrinsics_matrix)

        # --- 5. Save Workspace Image (For visualization/debugging) ---
        ws_rgb = cv2.cvtColor(self.rs_rgb[0], cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(folder, "head_camera_ws_rgb.png"), ws_rgb)

        print(f"✅ Data Saved.")

        # --- 6. Post-hoc segmentation pass ---
        # Replays the CSRT tracker from frame 0 (using the rectangle that was
        # active when recording started), reruns segment_3class + encode_4ch on
        # every saved frame, and writes:
        #   head_camera_seg.npy   (T, H, W) uint8           — 3-class mask
        #   head_camera_cam4.npy  (T, 4, 240, 320) float16  — policy input
        # so downstream tools don't have to re-run YOLO.
        try:
            self._save_segmentation(folder, length)
        except Exception as e:
            print(f"[seg] post-hoc segmentation failed (continuing): {e}")

        # --- 7. Optional Annotation ---
        if self.do_annotate:
            self.perform_annotation(folder, self.rs_rgb[0])

    def _save_segmentation(self, folder, length):
        """Run segment_3class + encode_4ch on every saved frame and dump the
        results next to the RGB/depth arrays. Deterministic: CSRT tracker is
        re-initialised from `recording_start_bbox` so every recording starts
        from the same home segmentation prompt."""
        if _WORLD_MODEL is None:
            print("[seg] models not loaded — skipping post-hoc segmentation")
            return

        global _TARGET_BBOX, _TRACKER, _TRACKER_FREEZE
        saved_bbox    = _TARGET_BBOX
        saved_tracker = _TRACKER
        saved_freeze  = _TRACKER_FREEZE

        start_bbox = getattr(self, "recording_start_bbox", None)
        try:
            if start_bbox is not None:
                first_bgr = cv2.cvtColor(self.rs_rgb[0], cv2.COLOR_RGB2BGR)
                _TARGET_BBOX = start_bbox
                init_tracker_from_bbox(first_bgr, start_bbox)
                _TRACKER_FREEZE = 15
            else:
                _TARGET_BBOX = None
                _TRACKER = None

            seg_buf, cam4_buf = [], []
            print(f"[seg] running post-hoc segmentation on {length} frames...")
            t0 = time.monotonic()
            for i in range(length):
                bgr_i   = cv2.cvtColor(self.rs_rgb[i], cv2.COLOR_RGB2BGR)
                depth_i = self.rs_depth[i]
                seg_i   = segment_3class(bgr_i, depth_i)
                cam4_i  = encode_4ch(seg_i, depth_i)
                seg_buf.append(seg_i)
                cam4_buf.append(cam4_i.astype(np.float16))
            dt = time.monotonic() - t0

            seg_arr  = np.stack(seg_buf,  axis=0)              # (T, H, W) uint8
            cam4_arr = np.stack(cam4_buf, axis=0)              # (T, 4, 240, 320) f16
            np.save(os.path.join(folder, "head_camera_seg.npy"),  seg_arr)
            np.save(os.path.join(folder, "head_camera_cam4.npy"), cam4_arr)
            print(f"[seg] saved head_camera_seg.npy {seg_arr.shape} {seg_arr.dtype}  "
                  f"and head_camera_cam4.npy {cam4_arr.shape} {cam4_arr.dtype}  "
                  f"({dt:.1f}s, {dt/max(1,length)*1000:.0f} ms/frame)")
        finally:
            _TARGET_BBOX = saved_bbox
            _TRACKER = saved_tracker
            _TRACKER_FREEZE = saved_freeze

    def on_press(self, key):
        global _TARGET_BBOX, _TRACKER_FREEZE
        if key == keyboard.Key.space:
            if not self.is_recording:
                # Mandatory bbox: if none is set, force the operator to draw
                # one before recording can start.
                if _HOME_BBOX is None:
                    if self.latest_rgb_bgr is None:
                        print("[bbox] no camera frame yet — wait a moment "
                              "before pressing SPACE")
                        return
                    print("[bbox] no home bbox set — draw one now to start "
                          "recording.")
                    self._select_target_bbox()
                    if _HOME_BBOX is None:
                        print("[bbox] selection cancelled — recording aborted")
                        return

                self.reset_buffers()
                # Reset the live bbox + tracker to the home bbox before any
                # frames are recorded. This applies the same initial
                # segmentation prompt at the start of every take.
                _TARGET_BBOX = _HOME_BBOX
                if self.latest_rgb_bgr is not None:
                    init_tracker_from_bbox(self.latest_rgb_bgr, _HOME_BBOX)
                _TRACKER_FREEZE = 15
                print(f"[bbox] applied home segmentation prompt {_HOME_BBOX} "
                      f"for this recording")
                # Snapshot the bbox at record-start so save_data can replay
                # the same starting bbox prompt from frame 0.
                self.recording_start_bbox = _TARGET_BBOX
                self.is_recording = True
                print("● RECORDING STARTED...")
            else:
                self.is_recording = False
                print("■ STOPPED. Processing...")
                # Restore emitter to ON so depth works normally outside recording
                self.set_emitter(True)
                self.save_data()
                # Move home after save
                self.limb.move_to_joint_positions(dict(zip(self.joint_names, self.home_joints)))
                # Robot is now back at the same pose as when the user drew the
                # rectangle, so the next take starts from the same home prompt.
                if _HOME_BBOX is not None:
                    time.sleep(0.3)   # let the camera settle a beat
                    _TARGET_BBOX = _HOME_BBOX
                    if self.latest_rgb_bgr is not None:
                        init_tracker_from_bbox(self.latest_rgb_bgr, _HOME_BBOX)
                    print(f"[bbox] home pose reached — bbox reset to "
                          f"{_HOME_BBOX}; ready for next take")
        if key == keyboard.Key.esc:
            self._stop_display = True
            return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotate", action="store_true",
                        help="Draw segmentation mask after recording")
    parser.add_argument("--no_yolo", dest="use_yolo", action="store_false",
                        help="Disable YOLO-World; fall back to HSV+depth.")
    parser.add_argument("--yolo_weights", default=WORLD_WEIGHTS_DEFAULT,
                        help=f"YOLO-World weights (default: {WORLD_WEIGHTS_DEFAULT}). "
                             "ultralytics auto-downloads on first use.")
    parser.add_argument("--sam_weights", default=None,
                        help="Ignored. Kept for compatibility; SAM is disabled.")
    parser.add_argument("--yolo_device", default="cuda",
                        help="Device (default: cuda; use 'cpu' if no GPU).")
    parser.add_argument("--yolo_classes", nargs="+", default=None,
                        help="Override YOLO-World PLANT text classes. "
                             f"Default: {WORLD_PLANT_CLASSES}")
    parser.add_argument("--veg_classes", nargs="+", default=None,
                        help="Override YOLO-World VEGETATION text classes. "
                             f"Default: {WORLD_VEG_CLASSES}")
    parser.add_argument("--yolo_conf", type=float, default=WORLD_CONF,
                        help=f"YOLO-World confidence threshold "
                             f"(default: {WORLD_CONF} — open-vocab scores are lower than COCO)")
    parser.set_defaults(use_yolo=True)
    args = parser.parse_args()

    if args.yolo_classes is not None:
        WORLD_PLANT_CLASSES = list(args.yolo_classes)
    if args.veg_classes is not None:
        WORLD_VEG_CLASSES = list(args.veg_classes)
    WORLD_CONF = float(args.yolo_conf)

    if keyboard is None:
        raise SystemExit(
            "pynput is required for recording keyboard controls. "
            "Install it in this environment or run deployment without using "
            "record_full_data.py as a recorder."
        )
    if intera_interface is None or Image is None or CameraInfo is None:
        raise SystemExit(
            "ROS/Intera Python packages are required for recording. Source the "
            "Intera workspace before running record_full_data.py."
        )

    rospy.init_node('full_pipeline_recorder')

    # NOTE: Change 'demo_task_full' to whatever base name you want
    rec = FullStateRecorder("demo_task_full",
                            do_annotate=args.annotate,
                            use_yolo=args.use_yolo,
                            yolo_weights=args.yolo_weights,
                            sam_weights=args.sam_weights,
                            yolo_device=args.yolo_device)

    # Keyboard listener runs in the background; the OpenCV GUI (preview window,
    # trackbars, mouse callback) must run on the main thread for Qt.
    listener = keyboard.Listener(on_press=rec.on_press)
    listener.start()
    try:
        rec._display_loop()
    finally:
        rec._stop_display = True
        listener.stop()
