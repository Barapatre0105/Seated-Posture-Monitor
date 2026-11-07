# Seated Posture Monitor

Reads a video source (webcam or file), tracks the seated person's upper body
with **MediaPipe Pose**, classifies posture over time, and writes a JSON summary.

## How to run

```bash
# 1. Create venv and install dependencies
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate     # Mac / Linux
pip install -r requirements.txt

# 2. Run the monitor (60 s from webcam 0)
python posture_monitor.py --source 0 --duration 60 --out posture.json

# 3. Optional: live preview window + debug CSV
python posture_monitor.py --source 0 --duration 60 --out posture.json \
    --preview --debug-csv debug.csv

# 4. Run from a video file
python posture_monitor.py --source my_session.mp4 --out posture.json
```

**Exit codes:** `0` = success, `1` = unexpected runtime error, `2` = source not openable, `3` = nobody in frame.

## Decisions made and why

### Features (what numbers we track)

Two values, both divided by the inter-shoulder distance so they are **scale-invariant**
(unchanged when you sit closer or further from the camera):

| Feature | Formula | Intuition |
|---|---|---|
| `vertical_drop` | `(ear_mid_y − shoulder_mid_y) / shoulder_width` | Measures how high your ears are relative to your shoulders. Upright posture → consistent negative ratio. Hunching → ears descend toward shoulders → value becomes less negative. |
| `ear_asymmetry` | `\|left_ear_y − right_ear_y\| / shoulder_width` | Head tilt or forward head rotation makes the two ears appear at different heights. Near 0 when level. |

A single **Euclidean distance** from the personal baseline captures both at once.

### Baseline (personalisation)

The first `--baseline` seconds (default 10 s) are assumed to be **your** good posture.
We store the **median** (not mean) of feature vectors from that window so one dropped
or mis-detected frame cannot skew the reference. This makes the thresholds relative to
*you*, not an abstract population average.

### Stabilisation — the core design decision

Naive per-frame classification flickers 10–30 times per minute (Step 5 demonstrates
this). Two layers remove the flicker:

1. **Rolling median buffer** (default: last 15 frames ≈ 0.5 s at 30 fps).  
   Median is far more resistant to spikes than a moving average. A single bad-geometry
   frame with deviation=5× normal does not raise the median if the surrounding 14
   frames are healthy.

2. **Hysteresis with a min-hold timer.**  
   The state only commits `GOOD→BAD` once the smoothed deviation has *continuously*
   exceeded `--enter` for at least `--min-hold` seconds. It returns `BAD→GOOD` only
   after smoothed deviation has *continuously* stayed below `--exit` for the same
   duration. This prevents brief slouches or brief corrections from toggling the label.

Together these mean: **a single bad frame is never bad posture**, and a momentary
correction is never classified as a good episode.

### Thresholds

Thresholds were derived from a 60-second debug recording (`debug.csv`):
- **Good posture half** (first 30 s, sitting upright): smoothed deviation typically
  `0.02–0.06`, peak `0.09`.
- **Slouch half** (last 30 s, forward head + rounded shoulders): smoothed deviation
  `0.15–0.30`, minimum `0.11`.

Chosen values:

| Parameter | Value | Rationale |
|---|---|---|
| `--enter` | `0.12` | Midway between good-peak (0.09) and slouch-minimum (0.11), with a small margin. |
| `--exit` | `0.07` | Below the top of the good range so the state stays BAD until clearly recovered. |
| `--min-hold` | `2.0 s` | Smallest value that produced zero false "bad" episodes on the good half of the debug recording. |

Override on the CLI if your setup differs: `--enter 0.15 --exit 0.08 --min-hold 1.5`

### Score (0–100)

`score = round(100 × good_time / classified_time)`

Intentionally simple and interpretable. Excludes the baseline window and
undetected frames so a brief camera occlusion does not drag the score down.

## What it doesn't handle

- **Multi-person scenes** — MediaPipe Pose selects one person; a second person
  walking through frame will corrupt the features until they leave.
- **Profile / severe side-on posture** — shoulder width collapses to near-zero;
  the frame is silently skipped (detected via degenerate geometry guard).
- **Low-light environments** — detection confidence drops; frames with no landmark
  are skipped. If this happens for most of a session, exit code 3 fires.
- **Very short sessions** — a `--duration` shorter than `--baseline` leaves no
  classified time and exits with code 3.
- **Chair / desk occlusion of shoulders** — if the desk blocks both shoulders
  MediaPipe may fail to detect the pose. Use a camera angle above the desk.
- **Non-seated postures** — the features are designed for seated upper-body posture.
  Standing or lying down will produce meaningless deviations.
