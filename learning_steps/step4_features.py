"""
Step 4 — Shoulder-width-normalised features.

Uses the new mediapipe Tasks API (mediapipe >= 1.0).
Requires: pose_landmarker_lite.task in the project root.

Goal:
  - From the 4 key points, compute two numbers that are
    SCALE-INVARIANT (don't change when you sit closer/farther).
  - Feature 1 — ear-mid-to-shoulder-mid drop (vertical lean)
  - Feature 2 — left-right ear symmetry (lateral tilt / rotation)

Both are divided by the inter-shoulder distance so the value is the same
regardless of how far from the camera the person sits.

Run:
    python learning_steps/step4_features.py --source 0 --duration 20

Then watch the numbers: they should be small and stable when you sit
upright, and jump when you slouch or tilt.
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core import base_options as mp_base
import numpy as np

DEFAULT_MODEL = str(Path(__file__).parent.parent / "pose_landmarker_lite.task")
_LM = mp_vision.PoseLandmark


def parse_args():
    p = argparse.ArgumentParser(description="Step 4: scale-invariant features")
    p.add_argument("--source", default="0")
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--model", default=DEFAULT_MODEL)
    return p.parse_args()


def open_source(s):
    try:
        return cv2.VideoCapture(int(s), cv2.CAP_DSHOW)
    except ValueError:
        return cv2.VideoCapture(s)


def build_landmarker(model_path: str):
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_base.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)


def extract_points(result) -> dict[str, np.ndarray] | None:
    if not result.pose_landmarks:
        return None
    lm = result.pose_landmarks[0]
    return {
        "left_ear":       np.array([lm[_LM.LEFT_EAR].x,       lm[_LM.LEFT_EAR].y]),
        "right_ear":      np.array([lm[_LM.RIGHT_EAR].x,      lm[_LM.RIGHT_EAR].y]),
        "left_shoulder":  np.array([lm[_LM.LEFT_SHOULDER].x,  lm[_LM.LEFT_SHOULDER].y]),
        "right_shoulder": np.array([lm[_LM.RIGHT_SHOULDER].x, lm[_LM.RIGHT_SHOULDER].y]),
    }


def compute_features(pts: dict) -> dict | None:
    """
    Returns a dict with two normalised features.

    shoulder_width  = distance between shoulder midpoints (our scale ruler)
    ear_mid         = midpoint of the two ears
    shoulder_mid    = midpoint of the two shoulders

    vertical_drop   = (ear_mid.y - shoulder_mid.y) / shoulder_width
                      Normally negative (ears above shoulders in image coords).
                      Becomes less negative when you hunch (ears drop toward shoulders).

    ear_asymmetry   = |left_ear.y - right_ear.y| / shoulder_width
                      Near 0 when head is level; rises when you tilt left/right.
    """
    le = pts["left_ear"]
    re = pts["right_ear"]
    ls = pts["left_shoulder"]
    rs = pts["right_shoulder"]

    shoulder_width = np.linalg.norm(ls - rs)
    if shoulder_width < 1e-6:
        return None   # degenerate (person side-on or only partially visible)

    ear_mid = (le + re) / 2
    shoulder_mid = (ls + rs) / 2

    vertical_drop = float((ear_mid[1] - shoulder_mid[1]) / shoulder_width)
    ear_asymmetry = float(abs(le[1] - re[1]) / shoulder_width)

    return {
        "vertical_drop": vertical_drop,
        "ear_asymmetry": ear_asymmetry,
        "shoulder_width_px_frac": float(shoulder_width),
    }


def main():
    args = parse_args()
    if not Path(args.model).exists():
        print(f"ERROR: model file not found: {args.model}", file=sys.stderr)
        sys.exit(2)

    cap = open_source(args.source)
    if not cap.isOpened():
        print(f"ERROR: cannot open '{args.source}'", file=sys.stderr)
        sys.exit(2)

    frames_with_person = 0
    start = time.monotonic()

    with build_landmarker(args.model) as landmarker:
        while (time.monotonic() - start) < args.duration:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            elapsed_ms = int((time.monotonic() - start) * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, elapsed_ms)

            pts = extract_points(result)
            if pts is None:
                continue

            feat = compute_features(pts)
            if feat is None:
                continue

            frames_with_person += 1
            t = time.monotonic() - start
            print(
                f"t={t:5.2f}s  "
                f"vert_drop={feat['vertical_drop']:+.4f}  "
                f"ear_asym={feat['ear_asymmetry']:.4f}  "
                f"sh_width={feat['shoulder_width_px_frac']:.4f}"
            )

    cap.release()
    if frames_with_person == 0:
        print("ERROR: nobody in frame", file=sys.stderr)
        sys.exit(3)

    print(f"\nTotal frames with person: {frames_with_person}")
    sys.exit(0)


if __name__ == "__main__":
    main()
