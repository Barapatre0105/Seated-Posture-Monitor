"""
Step 2 — Frame loop that skips dropped / corrupt frames.

Goal:
  - Run for --duration seconds (wall-clock).
  - Count total vs. successfully decoded frames.
  - Skip (don't crash on) frames where cap.read() returns ok=False.
  - Report FPS at the end.

Run:
    python learning_steps/step2_frames.py --source 0 --duration 5
"""

import argparse
import sys
import time
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Step 2: frame loop")
    parser.add_argument("--source", default="0")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="How many seconds to record (default 5)")
    return parser.parse_args()


def open_source(source_arg: str) -> cv2.VideoCapture:
    try:
        cap = cv2.VideoCapture(int(source_arg), cv2.CAP_DSHOW)
    except ValueError:
        cap = cv2.VideoCapture(source_arg)
    return cap


def main():
    args = parse_args()
    cap = open_source(args.source)

    if not cap.isOpened():
        print(f"ERROR: cannot open '{args.source}'", file=sys.stderr)
        sys.exit(2)

    total_attempts = 0
    good_frames = 0
    start = time.monotonic()

    while (time.monotonic() - start) < args.duration:
        ok, frame = cap.read()
        total_attempts += 1

        if not ok or frame is None:
            # Dropped frame — skip, don't crash
            continue

        good_frames += 1

    cap.release()
    elapsed = time.monotonic() - start
    fps = good_frames / elapsed if elapsed > 0 else 0

    print(f"Ran {elapsed:.1f}s | attempts={total_attempts} "
          f"good={good_frames} fps={fps:.1f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
