# START HERE: how to build this project from zero

## The idea in plain words

A camera watches you sit. MediaPipe finds your ears and shoulders in each video frame. We turn those points into a few numbers, compare them to how YOU sit at the start (your "good posture"), and decide good or bad over time. At the end we save a summary (JSON file). The hard part is not flickering between good and bad on every frame.

```
camera frame -> find ears+shoulders -> numbers -> compare to your baseline
            -> smooth over time -> good / bad timeline -> JSON summary
```

## Folder layout

```
posture-monitor/
  START_HERE.md        <- this file
  requirements.txt     <- libraries to install
  posture_monitor.py   <- the FINAL program (you submit this)
  README.md            <- the 1-page README you submit
  learning_steps/      <- 5 small programs that grow into the final one
    step1_cli.py
    step2_frames.py
    step3_landmarks.py
    step4_features.py
    step5_flicker_demo.py
```

## Part A: set up (once)

Install Python 3.10 or 3.11 (MediaPipe may not support the very newest Python).

Open a terminal inside the posture-monitor folder.

Create a virtual environment and install libraries:

**Windows:**
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Mac/Linux:**
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Test the camera works:
```
python learning_steps/step1_cli.py --source 0
```
It should print `source opened OK`.

## Part B: make the GitHub repo

On github.com click **New repository**, name it `posture-monitor`, set it **Public**, create it.

In your terminal:
```
git init
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/posture-monitor.git
```

## Part C: build in steps and commit after each

| Step | Run it | Commit message |
|------|--------|----------------|
| Setup | (nothing) | `Add requirements and gitignore` |
| 1 | `python learning_steps/step1_cli.py --source 0` | `Step 1: CLI args, open source, exit code 2 on failure` |
| 2 | `python learning_steps/step2_frames.py --source 0 --duration 5` | `Step 2: frame loop that skips dropped frames` |
| 3 | `python learning_steps/step3_landmarks.py --source 0 --duration 5` | `Step 3: MediaPipe landmarks, exit 3 when nobody in frame` |
| 4 | `python learning_steps/step4_features.py --source 0 --duration 20` | `Step 4: shoulder-width-normalised features` |
| 5 | `python learning_steps/step5_flicker_demo.py --source 0 --duration 40` | `Step 5: raw per-frame classification flickers N times/min` |
| 6 | `python posture_monitor.py --source 0 --duration 60 --out posture.json --debug-csv debug.csv` | `Add baseline, median smoothing, hysteresis, JSON output` |
| 7 | choose thresholds (Part D) | `Tune thresholds from my recording` |
| 8 | commit posture.json | `Add posture.json from real run` |
| 9 | edit README | `Write README` |

Commit commands each time:
```
git add .
git commit -m "your message from the table"
```

Finally: `git push -u origin main`

> If something did not work, commit that too and say so in the message. That is valued.

## Part D: choose thresholds from your own data

1. Record yourself 40–60 s: sit well first half, slouch second half.
2. Run:
   ```
   python posture_monitor.py --source 0 --duration 60 --debug-csv debug.csv
   ```
3. Open `debug.csv` in Excel. Look at the `deviation` column:
   - first half (good) values: note the highest typical value (say 0.07)
   - second half (slouch) values: note the typical value (say 0.18)
4. Pick `--enter` between them, e.g. `0.12`. Pick `--exit` about half of `--enter`.
5. Try `--min-hold 0, 1, 2` seconds on the good half; keep the smallest that gives zero false "bad" episodes.
6. Write those numbers and why in README under "Thresholds".

## Part E: test the failure cases

```powershell
python posture_monitor.py --source 5          # → exit 2 (no such camera)
python posture_monitor.py --source missing.mp4 # → exit 2
# cover the camera or face a wall             # → exit 3
echo $LASTEXITCODE                            # check exit code (PowerShell)
```

## If you get stuck

- `ModuleNotFoundError: mediapipe` → activate the virtual environment, reinstall.
- mediapipe will not install → use Python 3.10 or 3.11.
- Camera does not open on Windows → close Zoom/Teams, or try `--source 1`.
