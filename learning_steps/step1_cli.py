"""
Step 1 — CLI args, open source, exit code 2 on failure.

Goal: Accept --source (int or file path), try to open it with OpenCV,
      print success or fail, and exit with the right code.

Run:
    python learning_steps/step1_cli.py --source 0
    python learning_steps/step1_cli.py --source missing.mp4
"""

import argparse
import sys
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Step 1: open a video source")
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index (0, 1, …) or path to a video file",
    )
    return parser.parse_args()


def open_source(source_arg: str) -> cv2.VideoCapture:
    """Convert the CLI string to an int (webcam) or keep it as a path."""
    try:
        idx = int(source_arg)
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)   # CAP_DSHOW is faster on Windows
    except ValueError:
        cap = cv2.VideoCapture(source_arg)
    return cap


def main():
    args = parse_args()
    cap = open_source(args.source)

    if not cap.isOpened():
        print(f"ERROR: could not open source '{args.source}'", file=sys.stderr)
        sys.exit(2)

    # Read one frame to confirm real data is flowing
    ok, _frame = cap.read()
    cap.release()

    if not ok:
        print(f"ERROR: source opened but no frames from '{args.source}'", file=sys.stderr)
        sys.exit(2)

    print(f"source opened OK  ({args.source})")
    sys.exit(0)


if __name__ == "__main__":
    main()
