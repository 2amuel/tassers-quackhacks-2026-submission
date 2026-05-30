# MediaPipe Webcam Pose Template

Small Python template for capturing a short webcam video, running Google's
MediaPipe Pose and Hand Landmarker models, and exporting graph artifacts.

## What It Outputs

By default, a run creates a timestamped folder under `outputs/` containing:

- `raw_capture.mp4`: the unmodified webcam clip.
- `annotated_capture.mp4`: the clip with MediaPipe pose landmarks drawn on top.
- `pose_landmarks.csv`: per-frame pose landmark coordinates and visibility.
- `hand_landmarks.csv`: per-frame 21-point hand landmarks for each detected hand.
- `pose_results.json`: structured pose landmarks for every processed frame.
- `pose_graph.png`: a 2D pose skeleton graph from the middle detected frame.
- `hand_graph.png`: a 2D hand skeleton graph from the middle detected hand frame.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The script uses the current MediaPipe Tasks API and downloads the default
pose and hand `.task` models into `models/` on first run.

## Run

Capture a five-second clip from the default webcam:

```bash
python src/capture_holistic_pose.py --duration 5
```

Use a different camera or output location:

```bash
python src/capture_holistic_pose.py --camera 1 --duration 8 --output-dir outputs/test-run
```

List named cameras:

```bash
python src/capture_holistic_pose.py --list-cameras
```

On macOS, the script avoids camera names like `iPhone`, `Continuity Camera`,
and `Desk View` by default. If you really do want to use one of those cameras,
pass `--allow-iphone-camera`.

Track more than two hands:

```bash
python src/capture_holistic_pose.py --num-hands 4
```

Run without a live preview window:

```bash
python src/capture_holistic_pose.py --no-preview
```

Record webcam footage and holistic landmark CSV data:

```bash
python src/record_webcam.py
```

This writes `webcam_capture.mp4`, `holistic_landmarks.csv`, and
`recording_metadata.json` into a timestamped `outputs/recording-*` directory.
It samples the live feed every `0.2` seconds by default and writes CSV rows in
real time as each sample is processed. Press `q` in the preview window to stop.

Change the sampling interval or run for a bounded test duration:

```bash
python src/record_webcam.py --sample-interval 0.1 --max-duration 5
```

## Notes

- Press `q` in the preview window to stop recording early.
- The script mirrors the webcam preview by default. Use `--no-mirror` if you want raw camera orientation.
- Pose graph coordinates are normalized MediaPipe coordinates, plotted with the image-style y-axis inverted so the skeleton reads upright.
- Hand tracking exports all 21 landmarks per hand, including each finger joint and fingertip.
