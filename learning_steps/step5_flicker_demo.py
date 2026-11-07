"""
Step 5 — Raw per-frame classification FLICKERS.

Uses the new mediapipe Tasks API (mediapipe >= 1.0).
Requires: pose_landmarker_lite.task in the project root.

Goal:
  Show WHY naive per-frame thresholding is bad.

  - We capture a 5-second baseline at the start (collect the mean feature
    vector while the person sits upright).
  - Then for every frame we compute deviation from that baseline.
  - We threshold it: deviation > ENTER_THRESH → "BAD", else "GOOD".
  - We count how many times the state flips per minute.

Expected output: many flips per minute, clearly unusable.

Run:
    python learning_steps/step5_flicker_demo.py --source 0 --duration 40

Watch how often the label switches even when you hold still.
Put the real N in your commit message: "Step 5: raw classification flickers N times/min"
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

BASELINE_SECS = 5          # first N seconds used to build baseline
ENTER_THRESH  = 0.08       # raw threshold (intentionally tight to show flickering)


def parse_args():
    p = argparse.ArgumentParser(description="Step 5: flicker demo")
    p.add_argument("--source", default="0")
    p.add_argument("--duration", type=float, default=40.0)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--preview", action="store_true",
                   help="Show a live OpenCV window (optional)")
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


def extract_points(result):
    if not result.pose_landmarks:
        return None
    lm = result.pose_landmarks[0]
    return {
        "left_ear":       np.array([lm[_LM.LEFT_EAR].x,       lm[_LM.LEFT_EAR].y]),
        "right_ear":      np.array([lm[_LM.RIGHT_EAR].x,      lm[_LM.RIGHT_EAR].y]),
        "left_shoulder":  np.array([lm[_LM.LEFT_SHOULDER].x,  lm[_LM.LEFT_SHOULDER].y]),
        "right_shoulder": np.array([lm[_LM.RIGHT_SHOULDER].x, lm[_LM.RIGHT_SHOULDER].y]),
    }


def compute_features(pts):
    le, re = pts["left_ear"], pts["right_ear"]
    ls, rs = pts["left_shoulder"], pts["right_shoulder"]
    sw = np.linalg.norm(ls - rs)
    if sw < 1e-6:
        return None
    ear_mid = (le + re) / 2
    sh_mid  = (ls + rs) / 2
    return np.array([
        (ear_mid[1] - sh_mid[1]) / sw,
        abs(le[1] - re[1]) / sw,
    ])


def main():
    args = parse_args()
    if not Path(args.model).exists():
        print(f"ERROR: model file not found: {args.model}", file=sys.stderr)
        sys.exit(2)

    cap = open_source(args.source)
    if not cap.isOpened():
        print(f"ERROR: cannot open '{args.source}'", file=sys.stderr)
        sys.exit(2)

    baseline_vecs = []
    baseline_mean = None
    state = "GOOD"
    flips = 0
    start = time.monotonic()
    frame_rows = []

    with build_landmarker(args.model) as landmarker:
        while (time.monotonic() - start) < args.duration:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            t = time.monotonic() - start
            elapsed_ms = int(t * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, elapsed_ms)

            pts = extract_points(result)
            if pts is None:
                continue

            feat = compute_features(pts)
            if feat is None:
                continue

            # ── Baseline phase ───────────────────────────────────────────
            if t < BASELINE_SECS:
                baseline_vecs.append(feat)
                print(f"  t={t:5.2f}s  [BASELINE] feat={feat}")
                continue

            if baseline_mean is None:
                if not baseline_vecs:
                    print("ERROR: no frames during baseline", file=sys.stderr)
                    sys.exit(3)
                baseline_mean = np.mean(baseline_vecs, axis=0)
                print(f"\n  Baseline captured: {baseline_mean}\n")

            # ── Per-frame raw classification ─────────────────────────────
            deviation = float(np.linalg.norm(feat - baseline_mean))
            new_state = "BAD" if deviation > ENTER_THRESH else "GOOD"

            if new_state != state:
                flips += 1
                state = new_state
                flip_marker = " *** FLIP ***"
            else:
                flip_marker = ""

            frame_rows.append({"t": t, "deviation": deviation, "state": state})
            print(f"  t={t:5.2f}s  dev={deviation:.4f}  {state}{flip_marker}")

            if args.preview:
                label = f"{state} dev={deviation:.3f} flips={flips}"
                colour = (0, 255, 0) if state == "GOOD" else (0, 0, 255)
                cv2.putText(frame, label, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
                cv2.imshow("Flicker Demo", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    if args.preview:
        cv2.destroyAllWindows()

    elapsed = time.monotonic() - start
    if baseline_mean is None or not frame_rows:
        print("ERROR: not enough data", file=sys.stderr)
        sys.exit(3)

    flips_per_min = flips / elapsed * 60
    print(f"\n{'='*50}")
    print(f"Duration: {elapsed:.1f}s  |  Total flips: {flips}  |  "
          f"Flips/min: {flips_per_min:.1f}")
    print("→ Put this number in your commit message!")
    sys.exit(0)


if __name__ == "__main__":
    main()
