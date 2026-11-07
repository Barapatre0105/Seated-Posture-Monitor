"""
Step 3 — MediaPipe Pose landmarks, exit 3 when nobody is in frame.

Uses the new mediapipe Tasks API (mediapipe >= 1.0).
Requires: pose_landmarker_lite.task in the project root.

Download the model once:
    python -c "
    import urllib.request
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
        'pose_landmarker_lite.task')
    print('Done')
    "

Goal:
  - Run MediaPipe PoseLandmarker on every good frame.
  - Extract LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER.
  - If, over the whole session, we never detect a person at all → exit 3.
  - Print one line per frame with the raw (x, y) coordinates.

Run:
    python learning_steps/step3_landmarks.py --source 0 --duration 5

Landmarks used (PoseLandmark enum):
    LEFT_EAR       = 7
    RIGHT_EAR      = 8
    LEFT_SHOULDER  = 11
    RIGHT_SHOULDER = 12
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core import base_options as mp_base

# Default model path (relative to project root; run from there)
DEFAULT_MODEL = str(Path(__file__).parent.parent / "pose_landmarker_lite.task")

_LM = mp_vision.PoseLandmark


def parse_args():
    p = argparse.ArgumentParser(description="Step 3: MediaPipe landmarks")
    p.add_argument("--source", default="0")
    p.add_argument("--duration", type=float, default=5.0)
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="Path to pose_landmarker_*.task model file")
    return p.parse_args()


def open_source(source_arg: str) -> cv2.VideoCapture:
    try:
        cap = cv2.VideoCapture(int(source_arg), cv2.CAP_DSHOW)
    except ValueError:
        cap = cv2.VideoCapture(source_arg)
    return cap


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


def extract_landmarks(result):
    """Return dict {name: (x, y)} for our four key points, or None."""
    if not result.pose_landmarks:
        return None
    lm = result.pose_landmarks[0]   # first (only) pose
    return {
        "left_ear":       (lm[_LM.LEFT_EAR].x,       lm[_LM.LEFT_EAR].y),
        "right_ear":      (lm[_LM.RIGHT_EAR].x,      lm[_LM.RIGHT_EAR].y),
        "left_shoulder":  (lm[_LM.LEFT_SHOULDER].x,  lm[_LM.LEFT_SHOULDER].y),
        "right_shoulder": (lm[_LM.RIGHT_SHOULDER].x, lm[_LM.RIGHT_SHOULDER].y),
    }


def main():
    args = parse_args()
    if not Path(args.model).exists():
        print(f"ERROR: model file not found: {args.model}\n"
              f"Download it first — see the docstring at the top of this file.",
              file=sys.stderr)
        sys.exit(2)

    cap = open_source(args.source)
    if not cap.isOpened():
        print(f"ERROR: cannot open '{args.source}'", file=sys.stderr)
        sys.exit(2)

    frames_with_person = 0
    start = time.monotonic()
    frame_idx = 0

    with build_landmarker(args.model) as landmarker:
        while (time.monotonic() - start) < args.duration:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            elapsed_ms = int((time.monotonic() - start) * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, elapsed_ms)
            frame_idx += 1

            pts = extract_landmarks(result)
            if pts is None:
                print(f"  t={time.monotonic()-start:.2f}s  no person detected")
                continue

            frames_with_person += 1
            print(
                f"  t={time.monotonic()-start:.2f}s  "
                f"L_ear={pts['left_ear'][0]:.3f},{pts['left_ear'][1]:.3f}  "
                f"R_ear={pts['right_ear'][0]:.3f},{pts['right_ear'][1]:.3f}  "
                f"L_sh={pts['left_shoulder'][0]:.3f},{pts['left_shoulder'][1]:.3f}  "
                f"R_sh={pts['right_shoulder'][0]:.3f},{pts['right_shoulder'][1]:.3f}"
            )

    cap.release()
    if frames_with_person == 0:
        print("ERROR: nobody detected in frame during the session", file=sys.stderr)
        sys.exit(3)

    print(f"\nFrames with person: {frames_with_person}")
    sys.exit(0)


if __name__ == "__main__":
    main()
