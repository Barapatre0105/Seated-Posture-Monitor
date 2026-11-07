"""
posture_monitor.py -- Seated Posture Monitor
============================================
Reads a video source (webcam or file), tracks the seated person's upper body
with MediaPipe PoseLandmarker (Tasks API), classifies posture over time using
a personal baseline + median smoothing + hysteresis, and writes a JSON summary.

Requirements
------------
    pip install mediapipe numpy          # opencv bundled by mediapipe

Model file (download once, ~6 MB):
    python -c "
    import urllib.request
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/'
        'pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
        'pose_landmarker_lite.task')
    "

Usage
-----
    python posture_monitor.py --source 0 --duration 60 --out posture.json
    python posture_monitor.py --source 0 --duration 60 --out posture.json --preview
    python posture_monitor.py --source 0 --duration 60 --out posture.json \\
        --debug-csv debug.csv

Exit codes
----------
    0  usable observation written
    1  unexpected runtime error
    2  source could not be opened / no frames / model file missing
    3  nobody in frame during the session

Design decisions (see README for detailed rationale)
----------------------------------------------------
1.  Features
    Two values, both normalised by inter-shoulder distance (scale-invariant):
      • vertical_drop   = (ear_mid_y − shoulder_mid_y) / shoulder_width
        A healthy upright posture has a consistent "ear above shoulder" ratio.
        When you hunch, your ears drop toward your shoulders → less negative.
      • ear_asymmetry   = |left_ear_y − right_ear_y| / shoulder_width
        Head tilt / rotation shows up here.
    combined_deviation = Euclidean distance of [vertical_drop, ear_asymmetry]
                         from the personal baseline captured at session start.

2.  Baseline
    First BASELINE_SECS seconds are assumed good posture.
    We store the MEDIAN (not mean) of feature vectors from that window so
    one early dropped/corrupt reading cannot skew it.

3.  Stabilisation — the main design decision
    Raw per-frame deviation flickers (Step 5 showed 10–30 flips/min).
    We apply TWO layers:

    a) Rolling-median buffer (MEDIAN_WINDOW frames, default 15 ≈ 0.5 s at 30fps)
       The median is far more resistant to spikes than a moving average.
       A single bad frame with deviation=10 does not raise the median if the
       surrounding 14 frames are healthy.

    b) Hysteresis with min-hold timer (ENTER_THRESH / EXIT_THRESH / MIN_HOLD_S)
       State only switches GOOD→BAD after smoothed deviation exceeds
       ENTER_THRESH for at least MIN_HOLD_S consecutive seconds.
       State switches back BAD→GOOD only when smoothed deviation drops below
       EXIT_THRESH (< ENTER_THRESH) for at least MIN_HOLD_S seconds.

4.  Thresholds
    See README "Thresholds" section for the data-driven derivation.
      ENTER_THRESH = 0.12   (from debug.csv: good < 0.07, slouch > 0.18)
      EXIT_THRESH  = 0.07
      MIN_HOLD_S   = 2.0

5.  Score (0–100)
    score = round(100 × fraction_of_classified_time_in_GOOD_state)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core import base_options as mp_base

# Force UTF-8 output on Windows so Unicode chars don't crash the pipeline
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Landmark enum shortcuts ────────────────────────────────────────────────
_LM = mp_vision.PoseLandmark

# ── Defaults (overridable via CLI) ─────────────────────────────────────────
DEFAULT_MODEL         = "pose_landmarker_lite.task"
DEFAULT_MODEL_URL     = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
DEFAULT_BASELINE_SECS = 10
DEFAULT_MEDIAN_WINDOW = 15
DEFAULT_ENTER_THRESH  = 0.12
DEFAULT_EXIT_THRESH   = 0.07
DEFAULT_MIN_HOLD_S    = 2.0


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Seated Posture Monitor — MediaPipe-based headless CLI tool"
    )
    p.add_argument("--source", default="0",
                   help="Webcam index (0, 1, …) or path to a video file")
    p.add_argument("--duration", type=float, default=60.0,
                   help="Recording duration in seconds (default 60)")
    p.add_argument("--out", default="posture.json",
                   help="Path to write the JSON summary (default posture.json)")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Path to the .task model file (default {DEFAULT_MODEL})")
    p.add_argument("--baseline", type=float, default=DEFAULT_BASELINE_SECS,
                   help=f"Baseline capture window in seconds (default {DEFAULT_BASELINE_SECS})")
    p.add_argument("--median-window", type=int, default=DEFAULT_MEDIAN_WINDOW,
                   help=f"Rolling-median buffer size in frames (default {DEFAULT_MEDIAN_WINDOW})")
    p.add_argument("--enter", type=float, default=DEFAULT_ENTER_THRESH,
                   help=f"Deviation threshold to enter BAD state (default {DEFAULT_ENTER_THRESH})")
    p.add_argument("--exit", dest="exit_thresh", type=float, default=DEFAULT_EXIT_THRESH,
                   help=f"Deviation threshold to exit BAD state (default {DEFAULT_EXIT_THRESH})")
    p.add_argument("--min-hold", type=float, default=DEFAULT_MIN_HOLD_S,
                   help=f"Seconds to hold candidate state before committing (default {DEFAULT_MIN_HOLD_S})")
    p.add_argument("--preview", action="store_true",
                   help="Show a live OpenCV window (optional, not headless)")
    p.add_argument("--debug-csv", default="",
                   help="Write per-frame debug data to this CSV file")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Model setup
# ═══════════════════════════════════════════════════════════════════════════

def ensure_model(model_path: str) -> None:
    """Auto-download the model file if it is missing."""
    p = Path(model_path)
    if p.exists():
        return
    print(f"Model file '{model_path}' not found — downloading (~6 MB)…",
          file=sys.stderr)
    try:
        urllib.request.urlretrieve(DEFAULT_MODEL_URL, model_path)
        print(f"Model downloaded to '{model_path}'.", file=sys.stderr)
    except Exception as exc:
        print(f"ERROR: could not download model: {exc}\n"
              f"Download manually:\n  {DEFAULT_MODEL_URL}", file=sys.stderr)
        sys.exit(2)


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


# ═══════════════════════════════════════════════════════════════════════════
# Video source
# ═══════════════════════════════════════════════════════════════════════════

def open_source(source_arg: str) -> cv2.VideoCapture:
    try:
        cap = cv2.VideoCapture(int(source_arg), cv2.CAP_DSHOW)
    except ValueError:
        cap = cv2.VideoCapture(source_arg)
    return cap


# ═══════════════════════════════════════════════════════════════════════════
# Feature extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_points(result) -> dict[str, np.ndarray] | None:
    """Return {name: np.array([x, y])} for the four key landmarks, or None."""
    if not result.pose_landmarks:
        return None
    lm = result.pose_landmarks[0]   # first (only) pose
    return {
        "left_ear":       np.array([lm[_LM.LEFT_EAR].x,       lm[_LM.LEFT_EAR].y],       dtype=np.float64),
        "right_ear":      np.array([lm[_LM.RIGHT_EAR].x,      lm[_LM.RIGHT_EAR].y],      dtype=np.float64),
        "left_shoulder":  np.array([lm[_LM.LEFT_SHOULDER].x,  lm[_LM.LEFT_SHOULDER].y],  dtype=np.float64),
        "right_shoulder": np.array([lm[_LM.RIGHT_SHOULDER].x, lm[_LM.RIGHT_SHOULDER].y], dtype=np.float64),
    }


def compute_features(pts: dict[str, np.ndarray]) -> np.ndarray | None:
    """
    Compute two scale-invariant features normalised by inter-shoulder width.

    Returns np.array([vertical_drop, ear_asymmetry]) or None if degenerate.
    """
    le = pts["left_ear"]
    re = pts["right_ear"]
    ls = pts["left_shoulder"]
    rs = pts["right_shoulder"]

    shoulder_width = np.linalg.norm(ls - rs)
    if shoulder_width < 1e-6:
        return None      # degenerate (person fully side-on, partially visible)

    ear_mid      = (le + re) * 0.5
    shoulder_mid = (ls + rs) * 0.5

    # Positive y = down in image coordinates.
    # Normally ear_mid.y < shoulder_mid.y  → vertical_drop is negative.
    # When you hunch, ears come closer to shoulders → value becomes less negative.
    vertical_drop = float((ear_mid[1] - shoulder_mid[1]) / shoulder_width)
    ear_asymmetry = float(abs(le[1] - re[1]) / shoulder_width)

    return np.array([vertical_drop, ear_asymmetry], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# Stabilisation: rolling-median + hysteresis state machine
# ═══════════════════════════════════════════════════════════════════════════

class PostureStateMachine:
    """
    Converts a stream of per-frame deviation values into a stable GOOD/BAD label.

    Two-layer stabilisation:
      1. Rolling median over the last N deviation values (removes spiky frames).
      2. Hysteresis with a min-hold timer (prevents rapid state toggling).
    """

    def __init__(
        self,
        enter_thresh: float,
        exit_thresh: float,
        min_hold_s: float,
        median_window: int,
    ):
        self.enter_thresh  = enter_thresh
        self.exit_thresh   = exit_thresh
        self.min_hold_s    = min_hold_s
        self.buf: deque[float] = deque(maxlen=median_window)

        self.state             = "GOOD"
        self.candidate         = None
        self.candidate_since   = None
        self.smoothed          = 0.0

    def update(self, deviation: float, now: float) -> str:
        """Feed one frame's deviation; return current stable state label."""
        self.buf.append(deviation)
        self.smoothed = float(np.median(self.buf)) if self.buf else deviation

        if self.state == "GOOD":
            crossing      = self.smoothed > self.enter_thresh
            new_candidate = "BAD"
        else:
            crossing      = self.smoothed < self.exit_thresh
            new_candidate = "GOOD"

        if crossing:
            if self.candidate == new_candidate:
                if (now - self.candidate_since) >= self.min_hold_s:
                    self.state           = new_candidate
                    self.candidate       = None
                    self.candidate_since = None
            else:
                self.candidate       = new_candidate
                self.candidate_since = now
        else:
            self.candidate       = None
            self.candidate_since = None

        return self.state


# ═══════════════════════════════════════════════════════════════════════════
# Episode tracker
# ═══════════════════════════════════════════════════════════════════════════

class EpisodeTracker:
    """Accumulates {state, start, end} episodes from the state stream."""

    def __init__(self):
        self.episodes: list[dict] = []
        self._current_state: str | None = None
        self._current_start: float | None = None

    def update(self, state: str, t: float) -> None:
        if state != self._current_state:
            if self._current_state is not None:
                self.episodes.append({
                    "state":      self._current_state,
                    "start_s":    round(self._current_start, 3),
                    "end_s":      round(t, 3),
                    "duration_s": round(t - self._current_start, 3),
                })
            self._current_state = state
            self._current_start = t

    def close(self, t: float) -> None:
        if self._current_state is not None:
            self.episodes.append({
                "state":      self._current_state,
                "start_s":    round(self._current_start, 3),
                "end_s":      round(t, 3),
                "duration_s": round(t - self._current_start, 3),
            })
            self._current_state = None


# ═══════════════════════════════════════════════════════════════════════════
# JSON summary builder
# ═══════════════════════════════════════════════════════════════════════════

def build_summary(
    args: argparse.Namespace,
    episodes: list[dict],
    baseline_mean: np.ndarray,
    baseline_std: np.ndarray,
    total_frames: int,
    detected_frames: int,
    classified_frames: int,
    start_wall: float,
) -> dict:
    total_classified_time = sum(e["duration_s"] for e in episodes)
    good_time = sum(e["duration_s"] for e in episodes if e["state"] == "GOOD")
    bad_time  = sum(e["duration_s"] for e in episodes if e["state"] == "BAD")

    fraction_good = good_time / total_classified_time if total_classified_time > 0 else 0.0
    score = round(fraction_good * 100)

    bad_episodes = [e for e in episodes if e["state"] == "BAD"]
    num_breaks   = len(bad_episodes)
    avg_break_s  = (sum(e["duration_s"] for e in bad_episodes) / num_breaks
                    if num_breaks else 0.0)

    return {
        "schema_version": "1.0",
        "session": {
            "source":               args.source,
            "requested_duration_s": args.duration,
            "actual_duration_s":    round(time.time() - start_wall, 1),
            "total_frames":         total_frames,
            "detected_frames":      detected_frames,
            "classified_frames":    classified_frames,
            "detection_rate":       round(detected_frames / total_frames, 3)
                                    if total_frames else 0,
        },
        "baseline": {
            "duration_s":             args.baseline,
            "vertical_drop_mean":     round(float(baseline_mean[0]), 5),
            "ear_asymmetry_mean":     round(float(baseline_mean[1]), 5),
            "vertical_drop_std":      round(float(baseline_std[0]), 5),
            "ear_asymmetry_std":      round(float(baseline_std[1]), 5),
        },
        "thresholds": {
            "enter":                args.enter,
            "exit":                 args.exit_thresh,
            "min_hold_s":           args.min_hold,
            "median_window_frames": args.median_window,
        },
        "summary": {
            "score":              score,
            "fraction_good":      round(fraction_good, 4),
            "good_time_s":        round(good_time, 2),
            "bad_time_s":         round(bad_time, 2),
            "total_classified_s": round(total_classified_time, 2),
            "num_bad_episodes":   num_breaks,
            "avg_bad_episode_s":  round(avg_break_s, 2),
        },
        "episodes": episodes,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:  # noqa: C901
    args = parse_args()
    start_wall = time.time()

    # ── Ensure model is available (auto-download if missing) ─────────────
    ensure_model(args.model)

    # ── Open source ──────────────────────────────────────────────────────
    cap = open_source(args.source)
    if not cap.isOpened():
        print(f"ERROR: cannot open source '{args.source}'", file=sys.stderr)
        sys.exit(2)

    ok, probe = cap.read()
    if not ok or probe is None:
        cap.release()
        print(f"ERROR: source opened but no frames from '{args.source}'", file=sys.stderr)
        sys.exit(2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── Optional debug CSV ───────────────────────────────────────────────
    csv_file   = None
    csv_writer = None
    if args.debug_csv:
        csv_file   = open(args.debug_csv, "w", newline="")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=["t", "vertical_drop", "ear_asymmetry",
                        "deviation", "smoothed", "state", "phase"],
        )
        csv_writer.writeheader()

    # ── State machine + episode tracker ─────────────────────────────────
    sm = PostureStateMachine(
        enter_thresh=args.enter,
        exit_thresh=args.exit_thresh,
        min_hold_s=args.min_hold,
        median_window=args.median_window,
    )
    tracker = EpisodeTracker()

    # ── Counters ─────────────────────────────────────────────────────────
    baseline_vecs: list[np.ndarray] = []
    baseline_mean: np.ndarray | None = None
    baseline_std:  np.ndarray | None = None

    total_frames      = 0
    detected_frames   = 0
    classified_frames = 0

    start_mono = time.monotonic()

    with build_landmarker(args.model) as landmarker:
        while True:
            elapsed = time.monotonic() - start_mono
            if elapsed >= args.duration:
                break

            ok, frame = cap.read()
            total_frames += 1

            # ── Dropped / corrupt frame — skip gracefully ────────────────
            if not ok or frame is None:
                continue

            # ── MediaPipe inference ──────────────────────────────────────
            elapsed_ms = int(elapsed * 1000)
            rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result     = landmarker.detect_for_video(mp_image, elapsed_ms)

            pts = extract_points(result)

            if pts is None:
                if csv_writer:
                    csv_writer.writerow({
                        "t": round(elapsed, 3), "vertical_drop": "", "ear_asymmetry": "",
                        "deviation": "", "smoothed": "", "state": "NO_PERSON",
                        "phase": "baseline" if elapsed < args.baseline else "classify",
                    })
                if args.preview:
                    cv2.putText(frame, "No person detected", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (128, 128, 128), 2)
                    cv2.imshow("Posture Monitor", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            detected_frames += 1
            feat = compute_features(pts)
            if feat is None:
                continue   # degenerate geometry

            # ── Baseline phase ───────────────────────────────────────────
            if elapsed < args.baseline:
                baseline_vecs.append(feat)
                if csv_writer:
                    csv_writer.writerow({
                        "t": round(elapsed, 3),
                        "vertical_drop": round(float(feat[0]), 6),
                        "ear_asymmetry": round(float(feat[1]), 6),
                        "deviation": "", "smoothed": "", "state": "BASELINE",
                        "phase": "baseline",
                    })
                if args.preview:
                    pct = elapsed / args.baseline * 100
                    cv2.putText(frame, f"Baseline {pct:.0f}%  (sit upright)", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2)
                    cv2.imshow("Posture Monitor", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            # ── Build baseline on first post-baseline frame ──────────────
            if baseline_mean is None:
                if not baseline_vecs:
                    cap.release()
                    if csv_file:
                        csv_file.close()
                    print("ERROR: no landmarks detected during baseline window",
                          file=sys.stderr)
                    sys.exit(3)

                arr           = np.stack(baseline_vecs)
                baseline_mean = np.median(arr, axis=0)
                baseline_std  = arr.std(axis=0)
                print(f"Baseline: vert_drop={baseline_mean[0]:.4f}  "
                      f"ear_asym={baseline_mean[1]:.4f}",
                      file=sys.stderr)

            # ── Classification phase ─────────────────────────────────────
            deviation   = float(np.linalg.norm(feat - baseline_mean))
            now         = time.monotonic()
            state_label = sm.update(deviation, now)
            smoothed    = sm.smoothed

            classified_frames  += 1
            classified_elapsed  = elapsed - args.baseline

            tracker.update(state_label, classified_elapsed)

            if csv_writer:
                csv_writer.writerow({
                    "t":             round(elapsed, 3),
                    "vertical_drop": round(float(feat[0]), 6),
                    "ear_asymmetry": round(float(feat[1]), 6),
                    "deviation":     round(deviation, 6),
                    "smoothed":      round(smoothed, 6),
                    "state":         state_label,
                    "phase":         "classify",
                })

            if args.preview:
                colour = (0, 220, 60) if state_label == "GOOD" else (30, 30, 220)
                cv2.putText(frame,
                            f"{state_label}  dev={deviation:.3f}  smooth={smoothed:.3f}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
                cv2.putText(frame,
                            f"t={elapsed:.1f}s",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                cv2.imshow("Posture Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    # ── Cleanup ──────────────────────────────────────────────────────────
    cap.release()
    if args.preview:
        cv2.destroyAllWindows()
    if csv_file:
        csv_file.close()

    actual_duration = time.monotonic() - start_mono
    tracker.close(actual_duration - args.baseline)

    # ── Failure modes ────────────────────────────────────────────────────
    if detected_frames == 0:
        print("ERROR: nobody detected in frame during the entire session",
              file=sys.stderr)
        sys.exit(3)

    if baseline_mean is None:
        print("ERROR: baseline window ended with no valid detections",
              file=sys.stderr)
        sys.exit(3)

    if classified_frames == 0:
        print("ERROR: no classified frames (session too short or constant no-detection)",
              file=sys.stderr)
        sys.exit(3)

    # ── Build & write JSON ────────────────────────────────────────────────
    summary = build_summary(
        args=args,
        episodes=tracker.episodes,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        total_frames=total_frames,
        detected_frames=detected_frames,
        classified_frames=classified_frames,
        start_wall=start_wall,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    score      = summary["summary"]["score"]
    good_pct   = summary["summary"]["fraction_good"] * 100
    num_breaks = summary["summary"]["num_bad_episodes"]
    print(
        f"Session complete -- score={score}/100  "
        f"good={good_pct:.1f}%  breaks={num_breaks}  "
        f"-> {out_path}"
    )
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"FATAL: {exc}", file=sys.stderr)
        sys.exit(1)
